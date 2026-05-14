/**
 * Gradient Boosting Machine (GBM) Math Engine
 * Sequential ensemble of shallow decision trees fitting residuals.
 * Supports log-loss for binary classification with shrinkage, subsampling, and feature importance.
 */

import type { MetricValue } from '../registry';
import { generateClassificationData as generateData } from '../logistic-regression/math';

export interface Point { x: number; y: number; cls: number; }
export const generateClassificationData = generateData;

/* ─── Stump/Shallow Tree Node ─── */
export interface StumpNode {
  isLeaf: boolean;
  value: number; // leaf prediction (log-odds residual)
  splitFeature?: 'x' | 'y';
  splitValue?: number;
  left?: StumpNode;
  right?: StumpNode;
  samples?: number;
}

/* ─── GBM State ─── */
export interface GBMState {
  stumps: StumpNode[];
  learningRate: number;
  nEstimators: number;
  maxDepth: number;
  subsample: number;
  initialPrediction: number; // F0 = log(p/(1-p))
  lossHistory: number[];
  featureImportance: { x: number; y: number };
  stageContributions: number[]; // contribution magnitude per stage
}

/* ─── Sigmoid ─── */
function sigmoid(z: number): number {
  if (z > 500) return 1; if (z < -500) return 0;
  return 1 / (1 + Math.exp(-z));
}

/* ─── Seeded PRNG ─── */
function seededRandom(seed: number) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}

/* ─── Build regression stump (shallow tree) ─── */
function buildStump(
  xs: number[], ys: number[], residuals: number[], weights: number[],
  maxDepth: number, depth: number, numBins: number,
  featureImp: { x: number; y: number }
): StumpNode {
  const n = residuals.length;
  if (n === 0) return { isLeaf: true, value: 0, samples: 0 };

  // Weighted mean of residuals (leaf value = sum(residuals) / sum(|prev_prob * (1-prev_prob)|))
  let sumR = 0, sumW = 0;
  for (let i = 0; i < n; i++) { sumR += residuals[i]; sumW += weights[i]; }
  const leafValue = sumW > 0 ? sumR / sumW : 0;

  if (depth >= maxDepth || n < 4) {
    return { isLeaf: true, value: leafValue, samples: n };
  }

  // Find best split to minimize squared residuals
  let bestGain = 0, bestFeature: 'x' | 'y' = 'x', bestThreshold = 0;
  let bestLeftIdx: number[] = [], bestRightIdx: number[] = [];

  for (const feature of ['x', 'y'] as const) {
    const vals = feature === 'x' ? xs : ys;
    const sorted = [...new Set(vals)].sort((a, b) => a - b);
    const thresholds: number[] = [];
    if (sorted.length <= numBins) {
      for (let i = 0; i < sorted.length - 1; i++) thresholds.push((sorted[i] + sorted[i + 1]) / 2);
    } else {
      for (let i = 1; i < numBins; i++) thresholds.push(sorted[0] + (i / numBins) * (sorted[sorted.length - 1] - sorted[0]));
    }

    for (const th of thresholds) {
      const leftIdx: number[] = [], rightIdx: number[] = [];
      for (let i = 0; i < n; i++) { (vals[i] <= th ? leftIdx : rightIdx).push(i); }
      if (leftIdx.length < 2 || rightIdx.length < 2) continue;

      let lSumR = 0, lSumW = 0, rSumR = 0, rSumW = 0;
      for (const i of leftIdx) { lSumR += residuals[i]; lSumW += weights[i]; }
      for (const i of rightIdx) { rSumR += residuals[i]; rSumW += weights[i]; }

      const lVal = lSumW > 0 ? lSumR / lSumW : 0;
      const rVal = rSumW > 0 ? rSumR / rSumW : 0;

      // Gain = reduction in sum of squared residuals
      let parentSSR = 0; for (let i = 0; i < n; i++) parentSSR += (residuals[i] - leafValue) ** 2;
      let childSSR = 0;
      for (const i of leftIdx) childSSR += (residuals[i] - lVal) ** 2;
      for (const i of rightIdx) childSSR += (residuals[i] - rVal) ** 2;
      const gain = parentSSR - childSSR;

      if (gain > bestGain) {
        bestGain = gain; bestFeature = feature; bestThreshold = th;
        bestLeftIdx = leftIdx; bestRightIdx = rightIdx;
      }
    }
  }

  if (bestGain < 1e-10) return { isLeaf: true, value: leafValue, samples: n };

  featureImp[bestFeature] += bestGain;

  const leftXs = bestLeftIdx.map(i => xs[i]), leftYs = bestLeftIdx.map(i => ys[i]);
  const leftR = bestLeftIdx.map(i => residuals[i]), leftW = bestLeftIdx.map(i => weights[i]);
  const rightXs = bestRightIdx.map(i => xs[i]), rightYs = bestRightIdx.map(i => ys[i]);
  const rightR = bestRightIdx.map(i => residuals[i]), rightW = bestRightIdx.map(i => weights[i]);

  return {
    isLeaf: false, value: leafValue, splitFeature: bestFeature, splitValue: bestThreshold, samples: n,
    left: buildStump(leftXs, leftYs, leftR, leftW, maxDepth, depth + 1, numBins, featureImp),
    right: buildStump(rightXs, rightYs, rightR, rightW, maxDepth, depth + 1, numBins, featureImp),
  };
}

/* ─── Predict with stump ─── */
function stumpPredict(node: StumpNode, px: number, py: number): number {
  if (node.isLeaf) return node.value;
  const val = node.splitFeature === 'x' ? px : py;
  return val <= node.splitValue! ? stumpPredict(node.left!, px, py) : stumpPredict(node.right!, px, py);
}

/* ─── Train GBM ─── */
export function trainGBM(
  points: Point[], nEstimators: number, learningRate: number,
  maxDepth: number, subsample: number, numBins: number
): GBMState {
  const n = points.length;
  if (n < 2) return { stumps: [], learningRate, nEstimators, maxDepth, subsample, initialPrediction: 0, lossHistory: [], featureImportance: { x: 0, y: 0 }, stageContributions: [] };

  const rng = seededRandom(42 + n);
  const labels = points.map(p => p.cls);

  // F0: initial prediction (log odds of positive class)
  const posCount = labels.filter(l => l === 1).length;
  const p0 = Math.max(0.01, Math.min(0.99, posCount / n));
  const F0 = Math.log(p0 / (1 - p0));

  // Current predictions (raw log-odds scores)
  const F = new Array(n).fill(F0);
  const stumps: StumpNode[] = [];
  const lossHistory: number[] = [];
  const stageContributions: number[] = [];
  const featureImp = { x: 0, y: 0 };

  for (let m = 0; m < nEstimators; m++) {
    // Compute pseudo-residuals: r_i = y_i - p_i
    const probs = F.map(f => sigmoid(f));
    const residuals = labels.map((y, i) => y - probs[i]);
    const weights = probs.map(p => Math.max(p * (1 - p), 1e-6)); // Hessian

    // Subsample
    let sampleIdx: number[];
    if (subsample < 1.0) {
      const k = Math.max(4, Math.floor(n * subsample));
      sampleIdx = [];
      const used = new Set<number>();
      while (sampleIdx.length < k) {
        const idx = Math.floor(rng() * n);
        if (!used.has(idx)) { used.add(idx); sampleIdx.push(idx); }
      }
    } else {
      sampleIdx = Array.from({ length: n }, (_, i) => i);
    }

    const sXs = sampleIdx.map(i => points[i].x);
    const sYs = sampleIdx.map(i => points[i].y);
    const sR = sampleIdx.map(i => residuals[i]);
    const sW = sampleIdx.map(i => weights[i]);

    // Fit regression tree to residuals
    const stump = buildStump(sXs, sYs, sR, sW, maxDepth, 0, numBins, featureImp);
    stumps.push(stump);

    // Update F
    let totalContrib = 0;
    for (let i = 0; i < n; i++) {
      const update = learningRate * stumpPredict(stump, points[i].x, points[i].y);
      F[i] += update;
      totalContrib += Math.abs(update);
    }
    stageContributions.push(totalContrib / n);

    // Compute log loss
    let loss = 0;
    for (let i = 0; i < n; i++) {
      const p = sigmoid(F[i]);
      const cp = Math.max(1e-15, Math.min(1 - 1e-15, p));
      loss += -(labels[i] * Math.log(cp) + (1 - labels[i]) * Math.log(1 - cp));
    }
    lossHistory.push(loss / n);
  }

  // Normalize feature importance
  const total = featureImp.x + featureImp.y;
  if (total > 0) { featureImp.x /= total; featureImp.y /= total; }

  return { stumps, learningRate, nEstimators, maxDepth, subsample, initialPrediction: F0, lossHistory, featureImportance: featureImp, stageContributions };
}

/* ─── Raw score (sum of all stages) ─── */
export function gbmRawScore(px: number, py: number, state: GBMState, nStages?: number): number {
  let score = state.initialPrediction;
  const limit = nStages ?? state.stumps.length;
  for (let i = 0; i < limit && i < state.stumps.length; i++) {
    score += state.learningRate * stumpPredict(state.stumps[i], px, py);
  }
  return score;
}

/* ─── Probability ─── */
export function gbmPredictProbability(px: number, py: number, state: GBMState, nStages?: number): number {
  return sigmoid(gbmRawScore(px, py, state, nStages));
}

/* ─── Class prediction ─── */
export function gbmPredictSingle(px: number, py: number, state: GBMState, nStages?: number): number {
  return gbmPredictProbability(px, py, state, nStages) >= 0.5 ? 1 : 0;
}

/* ─── Per-stage contributions for a point ─── */
export function getStageContributions(px: number, py: number, state: GBMState): number[] {
  return state.stumps.map(s => state.learningRate * stumpPredict(s, px, py));
}

/* ─── Confusion Matrix ─── */
export interface ConfusionMatrix { tp: number; tn: number; fp: number; fn: number; }
export function computeConfusionMatrix(points: Point[], state: GBMState): ConfusionMatrix {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const p of points) {
    const pred = gbmPredictSingle(p.x, p.y, state);
    if (pred === 1 && p.cls === 1) tp++; if (pred === 0 && p.cls === 0) tn++;
    if (pred === 1 && p.cls === 0) fp++; if (pred === 0 && p.cls === 1) fn++;
  }
  return { tp, tn, fp, fn };
}

/* ─── ROC Curve ─── */
export interface ROCPoint { threshold: number; fpr: number; tpr: number; }
export function computeROCCurve(points: Point[], state: GBMState): { curve: ROCPoint[]; auc: number } {
  if (points.length === 0 || state.stumps.length === 0) return { curve: [], auc: 0 };
  const scored = points.map(p => ({ prob: gbmPredictProbability(p.x, p.y, state), cls: p.cls }));
  const thresholds = [1.01]; for (let t = 1.0; t >= -0.01; t -= 0.02) thresholds.push(t);
  const totalPos = scored.filter(s => s.cls === 1).length;
  const totalNeg = scored.filter(s => s.cls === 0).length;
  if (totalPos === 0 || totalNeg === 0) return { curve: [{ threshold: 0.5, fpr: 0, tpr: 0 }], auc: 0 };
  const curve: ROCPoint[] = [];
  for (const th of thresholds) {
    let tp = 0, fp = 0;
    for (const s of scored) { if (s.prob >= th) { if (s.cls === 1) tp++; else fp++; } }
    curve.push({ threshold: th, fpr: fp / totalNeg, tpr: tp / totalPos });
  }
  let auc = 0;
  for (let i = 1; i < curve.length; i++) auc += (curve[i].fpr - curve[i - 1].fpr) * (curve[i].tpr + curve[i - 1].tpr) / 2;
  return { curve, auc: Math.max(0, Math.min(1, auc)) };
}

/* ─── Metrics ─── */
export function computeMetrics(points: Point[], state: GBMState): MetricValue[] {
  if (points.length === 0 || state.stumps.length === 0) return [
    { label: 'Accuracy', value: '—', isPrimary: true }, { label: 'Precision', value: '—' },
    { label: 'Recall', value: '—' }, { label: 'F1 Score', value: '—' },
    { label: 'Log Loss', value: '—' }, { label: 'Stages', value: '—' },
  ];
  const { tp, tn, fp, fn } = computeConfusionMatrix(points, state);
  const acc = (tp + tn) / points.length;
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
  const lastLoss = state.lossHistory.length > 0 ? state.lossHistory[state.lossHistory.length - 1] : 0;
  return [
    { label: 'Accuracy', value: `${(acc * 100).toFixed(1)}%`, isPrimary: true },
    { label: 'Precision', value: `${(prec * 100).toFixed(1)}%` },
    { label: 'Recall', value: `${(rec * 100).toFixed(1)}%` },
    { label: 'F1 Score', value: `${(f1 * 100).toFixed(1)}%` },
    { label: 'Log Loss', value: lastLoss.toFixed(4) },
    { label: 'Stages', value: `${state.nEstimators}` },
  ];
}

/* ─── Data Statistics ─── */
export interface DataStats { n: number; nClass0: number; nClass1: number; meanX: number; meanY: number; }
export function computeDataStats(points: Point[]): DataStats | null {
  if (points.length < 2) return null;
  const n = points.length; const n0 = points.filter(p => p.cls === 0).length;
  let sx = 0, sy = 0; for (const p of points) { sx += p.x; sy += p.y; }
  return { n, nClass0: n0, nClass1: n - n0, meanX: sx / n, meanY: sy / n };
}
