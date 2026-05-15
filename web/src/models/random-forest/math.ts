/**
 * Random Forest Math Engine
 * Ensemble of Decision Trees with bootstrap aggregation (bagging).
 * Supports configurable n_estimators, max_features, OOB error, and feature importance.
 */

import type { MetricValue } from '../registry';
import { generateClassificationData as generateData } from '../logistic-regression/math';
import { trainDecisionTree, predictSingle, predictProbability, type DecisionTreeState } from '../decision-tree/math';

export interface Point {
  x: number;
  y: number;
  cls: number;
}

export const generateClassificationData = generateData;

/* ─── Random Forest State ─── */
export interface RFState {
  trees: DecisionTreeState[];
  oobError: number;
  oobAccuracy: number;
  featureImportance: { x: number; y: number };
  nEstimators: number;
  maxDepth: number;
  maxFeatures: string;
  bootstrapSamples: number[][]; // indices used for each tree
  treeBuildOrder: number[];     // order trees were built (for animation)
}

/* ─── Bootstrap Sampling ─── */
function bootstrapSample(n: number, rng: () => number): { indices: number[]; oobIndices: number[] } {
  const indices: number[] = [];
  const selected = new Set<number>();
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(rng() * n);
    indices.push(idx);
    selected.add(idx);
  }
  const oobIndices: number[] = [];
  for (let i = 0; i < n; i++) {
    if (!selected.has(i)) oobIndices.push(i);
  }
  return { indices, oobIndices };
}

/* ─── Seeded PRNG ─── */
function seededRandom(seed: number) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}

/* ─── Feature Subsampling ─── */
function shouldUseFeature(maxFeatures: string, _featureIdx: number, rng: () => number): boolean {
  if (maxFeatures === 'all') return true;
  if (maxFeatures === 'sqrt') {
    // sqrt(2) ≈ 1.41, so ~71% chance per feature
    return rng() < 0.71;
  }
  if (maxFeatures === 'log2') {
    // log2(2) = 1, so 50% chance per feature
    return rng() < 0.5;
  }
  return true;
}

/* ─── Train Random Forest ─── */
export function trainRandomForest(
  points: Point[],
  nEstimators: number,
  maxDepth: number,
  minSamplesSplit: number,
  minSamplesLeaf: number,
  maxFeatures: string,
  algorithm: 'id3' | 'c45' | 'cart',
  numBins: number
): RFState {
  const rng = seededRandom(42 + points.length);
  const trees: DecisionTreeState[] = [];
  const bootstrapSamples: number[][] = [];
  const allOobPredictions: Map<number, number[]> = new Map();
  const aggFeatureImp = { x: 0, y: 0 };

  for (let t = 0; t < nEstimators; t++) {
    const { indices, oobIndices } = bootstrapSample(points.length, rng);
    bootstrapSamples.push(indices);

    // Build bootstrap dataset
    const bsPoints = indices.map(i => points[i]);

    // Feature masking: for 2D we randomly drop features via column permutation
    let trainPoints = bsPoints;
    if (maxFeatures !== 'all') {
      const useX = shouldUseFeature(maxFeatures, 0, rng);
      const useY = shouldUseFeature(maxFeatures, 1, rng);
      // At least one feature must be used
      if (!useX && !useY) {
        // Use both if both would be dropped
        trainPoints = bsPoints;
      } else if (!useX) {
        // Shuffle x values to make them uninformative
        const shuffled = [...bsPoints];
        for (let i = shuffled.length - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          const tmp = shuffled[i].x;
          shuffled[i] = { ...shuffled[i], x: shuffled[j].x };
          shuffled[j] = { ...shuffled[j], x: tmp };
        }
        trainPoints = shuffled;
      } else if (!useY) {
        const shuffled = [...bsPoints];
        for (let i = shuffled.length - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          const tmp = shuffled[i].y;
          shuffled[i] = { ...shuffled[i], y: shuffled[j].y };
          shuffled[j] = { ...shuffled[j], y: tmp };
        }
        trainPoints = shuffled;
      }
    }

    const tree = trainDecisionTree(trainPoints, algorithm, maxDepth, minSamplesSplit, minSamplesLeaf, numBins);
    trees.push(tree);

    // Aggregate feature importance
    aggFeatureImp.x += tree.featureImportance.x;
    aggFeatureImp.y += tree.featureImportance.y;

    // OOB predictions
    if (tree.root) {
      for (const idx of oobIndices) {
        const p = points[idx];
        const pred = predictSingle(p.x, p.y, tree.root);
        if (!allOobPredictions.has(idx)) allOobPredictions.set(idx, []);
        allOobPredictions.get(idx)!.push(pred);
      }
    }
  }

  // Normalize feature importance
  const total = aggFeatureImp.x + aggFeatureImp.y;
  if (total > 0) { aggFeatureImp.x /= total; aggFeatureImp.y /= total; }

  // Compute OOB error
  let oobCorrect = 0, oobTotal = 0;
  for (const [idx, preds] of allOobPredictions) {
    const counts: Record<number, number> = {};
    for (const p of preds) counts[p] = (counts[p] || 0) + 1;
    let bestCls = -1, bestCount = -1;
    for (const [cls, count] of Object.entries(counts)) {
      if (count > bestCount) { bestCount = count; bestCls = parseInt(cls, 10); }
    }
    if (bestCls === points[idx].cls) oobCorrect++;
    oobTotal++;
  }
  const oobAccuracy = oobTotal > 0 ? oobCorrect / oobTotal : 0;
  const oobError = 1 - oobAccuracy;

  return {
    trees,
    oobError,
    oobAccuracy,
    featureImportance: aggFeatureImp,
    nEstimators,
    maxDepth,
    maxFeatures,
    bootstrapSamples,
    treeBuildOrder: trees.map((_, i) => i),
  };
}

/* ─── Ensemble Prediction (majority vote) ─── */
export function rfPredictSingle(px: number, py: number, state: RFState): number {
  const votes: Record<number, number> = {};
  for (const tree of state.trees) {
    if (!tree.root) continue;
    const pred = predictSingle(px, py, tree.root);
    votes[pred] = (votes[pred] || 0) + 1;
  }
  let best = -1, bestCount = -1;
  for (const [cls, count] of Object.entries(votes)) {
    if (count > bestCount) { bestCount = count; best = parseInt(cls, 10); }
  }
  return best;
}

/* ─── Ensemble Probability ─── */
export function rfPredictProbability(px: number, py: number, state: RFState): number {
  let sum = 0, count = 0;
  for (const tree of state.trees) {
    if (!tree.root) continue;
    sum += predictProbability(px, py, tree.root);
    count++;
  }
  return count > 0 ? sum / count : 0.5;
}

/* ─── Get individual tree votes ─── */
export function getTreeVotes(px: number, py: number, state: RFState): { treeIdx: number; vote: number; prob: number }[] {
  return state.trees.map((tree, i) => {
    if (!tree.root) return { treeIdx: i, vote: -1, prob: 0.5 };
    return {
      treeIdx: i,
      vote: predictSingle(px, py, tree.root),
      prob: predictProbability(px, py, tree.root),
    };
  });
}

/* ─── Confusion Matrix ─── */
export interface ConfusionMatrix { tp: number; tn: number; fp: number; fn: number; }

export function computeConfusionMatrix(points: Point[], state: RFState): ConfusionMatrix {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const p of points) {
    const pred = rfPredictSingle(p.x, p.y, state);
    if (pred === 1 && p.cls === 1) tp++;
    if (pred === 0 && p.cls === 0) tn++;
    if (pred === 1 && p.cls === 0) fp++;
    if (pred === 0 && p.cls === 1) fn++;
  }
  return { tp, tn, fp, fn };
}

/* ─── ROC Curve ─── */
export interface ROCPoint { threshold: number; fpr: number; tpr: number; }

export function computeROCCurve(points: Point[], state: RFState): { curve: ROCPoint[]; auc: number } {
  if (points.length === 0 || state.trees.length === 0) return { curve: [], auc: 0 };
  const scored = points.map(p => ({ prob: rfPredictProbability(p.x, p.y, state), cls: p.cls }));
  const thresholds = [1.01];
  for (let t = 1.0; t >= -0.01; t -= 0.02) thresholds.push(t);
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
  for (let i = 1; i < curve.length; i++) {
    auc += (curve[i].fpr - curve[i - 1].fpr) * (curve[i].tpr + curve[i - 1].tpr) / 2;
  }
  return { curve, auc: Math.max(0, Math.min(1, auc)) };
}

/* ─── Metrics ─── */
export function computeMetrics(points: Point[], state: RFState): MetricValue[] {
  if (points.length === 0 || state.trees.length === 0) return [
    { label: 'Accuracy', value: '—', isPrimary: true },
    { label: 'Precision', value: '—' }, { label: 'Recall', value: '—' },
    { label: 'F1 Score', value: '—' }, { label: 'OOB Accuracy', value: '—' },
    { label: 'Trees', value: '—' },
  ];
  const { tp, tn, fp, fn } = computeConfusionMatrix(points, state);
  const accuracy = (tp + tn) / points.length;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
  return [
    { label: 'Accuracy', value: `${(accuracy * 100).toFixed(1)}%`, isPrimary: true },
    { label: 'Precision', value: `${(precision * 100).toFixed(1)}%` },
    { label: 'Recall', value: `${(recall * 100).toFixed(1)}%` },
    { label: 'F1 Score', value: `${(f1 * 100).toFixed(1)}%` },
    { label: 'OOB Accuracy', value: `${(state.oobAccuracy * 100).toFixed(1)}%` },
    { label: 'Trees', value: `${state.nEstimators}` },
  ];
}

/* ─── Data Statistics ─── */
export interface DataStats { n: number; nClass0: number; nClass1: number; meanX: number; meanY: number; }

export function computeDataStats(points: Point[]): DataStats | null {
  if (points.length < 2) return null;
  const n = points.length;
  const nClass0 = points.filter(p => p.cls === 0).length;
  let sumX = 0, sumY = 0;
  for (const p of points) { sumX += p.x; sumY += p.y; }
  return { n, nClass0, nClass1: n - nClass0, meanX: sumX / n, meanY: sumY / n };
}
