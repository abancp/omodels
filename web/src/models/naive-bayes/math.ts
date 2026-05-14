/**
 * Naive Bayes Math Engine
 * Supports Gaussian, Multinomial, and Bernoulli Naive Bayes.
 */

import type { MetricValue } from '../registry';
import { generateClassificationData as generateData } from '../logistic-regression/math';

export interface Point {
  x: number;
  y: number;
  cls: number;
}

// Re-export dataset generator
export const generateClassificationData = generateData;

/* ─── Math Utils ─── */
const EPSILON = 1e-9;

/* ─── Naive Bayes State ─── */
export interface NBState {
  type: 'gaussian' | 'multinomial' | 'bernoulli';
  classes: number[];
  classPrior: Record<number, number>; // log prior
  
  // Gaussian
  theta?: Record<number, [number, number]>; // Mean of each feature per class
  var?: Record<number, [number, number]>; // Variance of each feature per class
  
  // Multinomial / Bernoulli
  featureLogProb?: Record<number, [number, number]>; // log probability of features per class
  
  // Bernoulli specific
  binarizeThreshold?: number;
}

/* ─── Training ─── */
export function trainNaiveBayes(
  points: Point[],
  type: 'gaussian' | 'multinomial' | 'bernoulli',
  fitPrior: boolean,
  varSmoothing: number,
  alpha: number,
  binarizeThreshold: number
): NBState {
  const classes = Array.from(new Set(points.map(p => p.cls))).sort();
  const classPrior: Record<number, number> = {};
  
  const state: NBState = {
    type,
    classes,
    classPrior,
    binarizeThreshold
  };

  if (classes.length === 0) return state;

  const n = points.length;

  classes.forEach(c => {
    const classPoints = points.filter(p => p.cls === c);
    const nc = classPoints.length;
    classPrior[c] = fitPrior ? Math.log(nc / n) : Math.log(1 / classes.length);

    if (type === 'gaussian') {
      if (!state.theta) state.theta = {};
      if (!state.var) state.var = {};
      
      const meanX = classPoints.reduce((acc, p) => acc + p.x, 0) / (nc || 1);
      const meanY = classPoints.reduce((acc, p) => acc + p.y, 0) / (nc || 1);
      
      let varX = classPoints.reduce((acc, p) => acc + Math.pow(p.x - meanX, 2), 0) / (nc || 1);
      let varY = classPoints.reduce((acc, p) => acc + Math.pow(p.y - meanY, 2), 0) / (nc || 1);
      
      // Global variance smoothing
      const globalVarX = points.reduce((acc, p) => acc + Math.pow(p.x - meanX, 2), 0) / n;
      const globalVarY = points.reduce((acc, p) => acc + Math.pow(p.y - meanY, 2), 0) / n;
      const maxVar = Math.max(globalVarX, globalVarY, EPSILON);
      
      varX += maxVar * varSmoothing;
      varY += maxVar * varSmoothing;

      state.theta[c] = [meanX, meanY];
      state.var[c] = [varX, varY];
    } else if (type === 'multinomial') {
      if (!state.featureLogProb) state.featureLogProb = {};
      
      const sumX = classPoints.reduce((acc, p) => acc + p.x, 0) + alpha;
      const sumY = classPoints.reduce((acc, p) => acc + p.y, 0) + alpha;
      const totalCount = sumX + sumY + 2 * alpha; // 2 features
      
      state.featureLogProb[c] = [
        Math.log(sumX / totalCount),
        Math.log(sumY / totalCount)
      ];
    } else if (type === 'bernoulli') {
      if (!state.featureLogProb) state.featureLogProb = {};
      
      const countX = classPoints.reduce((acc, p) => acc + (p.x > binarizeThreshold ? 1 : 0), 0) + alpha;
      const countY = classPoints.reduce((acc, p) => acc + (p.y > binarizeThreshold ? 1 : 0), 0) + alpha;
      const smoothedNc = nc + 2 * alpha;
      
      state.featureLogProb[c] = [
        Math.log(countX / smoothedNc),
        Math.log(countY / smoothedNc)
      ];
    }
  });

  return state;
}

/* ─── Prediction ─── */
export function predictProbabilities(px: number, py: number, state: NBState): Record<number, number> {
  const { type, classes, classPrior, theta, var: variance, featureLogProb, binarizeThreshold } = state;
  const logProbs: Record<number, number> = {};

  if (classes.length === 0) return {};

  classes.forEach(c => {
    let lp = classPrior[c] || 0;
    
    if (type === 'gaussian') {
      const [meanX, meanY] = theta![c];
      const [varX, varY] = variance![c];
      
      // log N(x|mu, var) = -0.5 * log(2 * pi * var) - 0.5 * (x - mu)^2 / var
      const lpX = -0.5 * Math.log(2 * Math.PI * varX) - 0.5 * Math.pow(px - meanX, 2) / varX;
      const lpY = -0.5 * Math.log(2 * Math.PI * varY) - 0.5 * Math.pow(py - meanY, 2) / varY;
      
      lp += lpX + lpY;
    } else if (type === 'multinomial') {
      const [lpX, lpY] = featureLogProb![c];
      lp += px * lpX + py * lpY;
    } else if (type === 'bernoulli') {
      const [lpX, lpY] = featureLogProb![c];
      const bx = px > binarizeThreshold! ? 1 : 0;
      const by = py > binarizeThreshold! ? 1 : 0;
      
      const probX = bx === 1 ? lpX : Math.log(1 - Math.exp(lpX) + EPSILON);
      const probY = by === 1 ? lpY : Math.log(1 - Math.exp(lpY) + EPSILON);
      
      lp += probX + probY;
    }
    
    logProbs[c] = lp;
  });

  // Convert log probs to actual probabilities using log-sum-exp trick for numerical stability
  const maxLogProb = Math.max(...Object.values(logProbs));
  let sumExp = 0;
  classes.forEach(c => {
    sumExp += Math.exp(logProbs[c] - maxLogProb);
  });
  
  const probs: Record<number, number> = {};
  classes.forEach(c => {
    probs[c] = Math.exp(logProbs[c] - maxLogProb) / sumExp;
  });

  return probs;
}

export function predict(px: number, py: number, state: NBState): number {
  const probs = predictProbabilities(px, py, state);
  let bestClass = -1;
  let bestProb = -1;
  for (const [cStr, p] of Object.entries(probs)) {
    const c = parseInt(cStr, 10);
    if (p > bestProb) {
      bestProb = p;
      bestClass = c;
    }
  }
  return bestClass;
}

/* ─── Metrics ─── */
export interface ConfusionMatrix {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

export function computeConfusionMatrix(points: Point[], state: NBState): ConfusionMatrix {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const p of points) {
    const pred = predict(p.x, p.y, state);
    if (pred === 1 && p.cls === 1) tp++;
    if (pred === 0 && p.cls === 0) tn++;
    if (pred === 1 && p.cls === 0) fp++;
    if (pred === 0 && p.cls === 1) fn++;
  }
  return { tp, tn, fp, fn };
}

export interface ROCPoint {
  threshold: number;
  fpr: number;
  tpr: number;
}

export function computeROCCurve(points: Point[], state: NBState): { curve: ROCPoint[]; auc: number } {
  if (points.length === 0 || state.classes.length === 0) return { curve: [], auc: 0 };

  const scored = points.map(p => ({
    prob: predictProbabilities(p.x, p.y, state)[1] || 0,
    cls: p.cls,
  }));

  const thresholds = [1.01];
  for (let t = 1.0; t >= -0.01; t -= 0.02) thresholds.push(t);

  const totalPos = scored.filter(s => s.cls === 1).length;
  const totalNeg = scored.filter(s => s.cls === 0).length;

  if (totalPos === 0 || totalNeg === 0) return { curve: [{ threshold: 0.5, fpr: 0, tpr: 0 }], auc: 0 };

  const curve: ROCPoint[] = [];
  for (const th of thresholds) {
    let tp = 0, fp = 0;
    for (const s of scored) {
      if (s.prob >= th) {
        if (s.cls === 1) tp++;
        else fp++;
      }
    }
    curve.push({ threshold: th, fpr: fp / totalNeg, tpr: tp / totalPos });
  }

  let auc = 0;
  for (let i = 1; i < curve.length; i++) {
    const dx = curve[i].fpr - curve[i - 1].fpr;
    const avgY = (curve[i].tpr + curve[i - 1].tpr) / 2;
    auc += dx * avgY;
  }
  auc = Math.max(0, Math.min(1, auc));

  return { curve, auc };
}

export function computeMetrics(points: Point[], state: NBState): MetricValue[] {
  if (points.length === 0 || state.classes.length === 0) return [
    { label: 'Accuracy', value: '—', isPrimary: true },
    { label: 'Precision', value: '—' },
    { label: 'Recall', value: '—' },
    { label: 'F1 Score', value: '—' },
    { label: 'AUC', value: '—' },
  ];

  const { tp, tn, fp, fn } = computeConfusionMatrix(points, state);

  const accuracy = (tp + tn) / points.length;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
  const { auc } = computeROCCurve(points, state);

  return [
    { label: 'Accuracy', value: `${(accuracy * 100).toFixed(1)}%`, isPrimary: true },
    { label: 'Precision', value: `${(precision * 100).toFixed(1)}%` },
    { label: 'Recall', value: `${(recall * 100).toFixed(1)}%` },
    { label: 'F1 Score', value: `${(f1 * 100).toFixed(1)}%` },
    { label: 'AUC', value: auc.toFixed(3) },
  ];
}

/* ─── Data Statistics ─── */
export interface DataStats {
  n: number;
  nClass0: number;
  nClass1: number;
  meanX: number;
  meanY: number;
}

export function computeDataStats(points: Point[]): DataStats | null {
  if (points.length < 2) return null;
  const n = points.length;
  const nClass0 = points.filter(p => p.cls === 0).length;
  const nClass1 = n - nClass0;
  let sumX = 0, sumY = 0;
  for (const p of points) {
    sumX += p.x; sumY += p.y;
  }
  return {
    n,
    nClass0,
    nClass1,
    meanX: sumX / n,
    meanY: sumY / n,
  };
}
