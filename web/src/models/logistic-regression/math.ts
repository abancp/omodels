/**
 * Logistic Regression Math Engine
 * Pure functions for classification: sigmoid, predict, loss, gradients, metrics.
 */

import type { MetricValue } from '../registry';

export interface Point {
  x: number;
  y: number;
  cls: number;
}

export type Weights = number[];

/* ─── Seeded PRNG ─── */
function seededRandom(seed: number) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}

/* ─── Dataset Generation ─── */
export function generateClassificationData(dataset: string, count: number, noise: number): Point[] {
  const rand = seededRandom(count * 1000 + Math.floor(noise * 100));
  
  const pts: Point[] = [];
  const half = Math.floor(count / 2);

  for (let i = 0; i < count; i++) {
    const cls = i < half ? 0 : 1;
    let x = 0, y = 0;

    if (dataset === 'blobs') {
      const cx = cls === 0 ? 0.3 : 0.7;
      const cy = cls === 0 ? 0.3 : 0.7;
      x = cx + (rand() - 0.5) * noise * 1.5;
      y = cy + (rand() - 0.5) * noise * 1.5;
    } else if (dataset === 'moons') {
      const angle = rand() * Math.PI;
      if (cls === 0) {
        x = 0.5 + Math.cos(angle) * 0.25 + (rand() - 0.5) * noise * 0.5;
        y = 0.6 - Math.sin(angle) * 0.25 + (rand() - 0.5) * noise * 0.5;
      } else {
        x = 0.5 - Math.cos(angle) * 0.25 + (rand() - 0.5) * noise * 0.5;
        y = 0.4 + Math.sin(angle) * 0.25 + (rand() - 0.5) * noise * 0.5;
      }
    } else if (dataset === 'circles') {
      const radius = cls === 0 ? 0.15 : 0.35;
      const angle = rand() * Math.PI * 2;
      x = 0.5 + Math.cos(angle) * radius + (rand() - 0.5) * noise * 0.5;
      y = 0.5 + Math.sin(angle) * radius + (rand() - 0.5) * noise * 0.5;
    } else if (dataset === 'xor') {
      // XOR-like pattern — challenging for linear boundary
      const quad = (i % 4);
      const cxMap = [0.25, 0.75, 0.25, 0.75];
      const cyMap = [0.25, 0.75, 0.75, 0.25];
      const clsMap = [0, 0, 1, 1];
      x = cxMap[quad] + (rand() - 0.5) * noise * 0.8;
      y = cyMap[quad] + (rand() - 0.5) * noise * 0.8;
      pts.push({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), cls: clsMap[quad] });
      continue;
    } else if (dataset === 'spiral') {
      const angle2 = (i / half) * Math.PI * 2;
      const r = 0.1 + (i % half) / half * 0.3;
      if (cls === 0) {
        x = 0.5 + r * Math.cos(angle2) + (rand() - 0.5) * noise * 0.3;
        y = 0.5 + r * Math.sin(angle2) + (rand() - 0.5) * noise * 0.3;
      } else {
        x = 0.5 + r * Math.cos(angle2 + Math.PI) + (rand() - 0.5) * noise * 0.3;
        y = 0.5 + r * Math.sin(angle2 + Math.PI) + (rand() - 0.5) * noise * 0.3;
      }
    } else {
      // Linear separable
      const cx = cls === 0 ? 0.35 : 0.65;
      const cy = cls === 0 ? 0.65 : 0.35;
      x = cx + (rand() - 0.5) * noise;
      y = cy + (rand() - 0.5) * noise;
    }

    pts.push({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), cls });
  }

  return pts;
}

/* ─── Sigmoid ─── */
export function sigmoid(z: number): number {
  if (z > 500) return 1;
  if (z < -500) return 0;
  return 1 / (1 + Math.exp(-z));
}

/* ─── Feature expansion ─── */
export function getFeatureVector(px: number, py: number, degree: number): number[] {
  if (degree === 1) return [1, px, py];
  return [1, px, py, px * px, py * py, px * py];
}

/* ─── Prediction (probability of class 1) ─── */
export function predict(px: number, py: number, weights: Weights): number {
  const degree = (weights.length > 3) ? 2 : 1; 
  let z = weights[0] + weights[1] * px + weights[2] * py;
  
  if (degree === 2 && weights.length === 6) {
    z += weights[3] * (px * px) + weights[4] * (py * py) + weights[5] * (px * py);
  }
  
  return sigmoid(z);
}

/* ─── Binary Cross-Entropy Loss ─── */
export function computeLoss(points: Point[], weights: Weights, regularization: string, regStrength: number, _degree: number): number {
  if (points.length === 0) return 0;
  let totalLoss = 0;
  
  for (const p of points) {
    const pred = predict(p.x, p.y, weights);
    const eps = 1e-15;
    const clampedPred = Math.max(eps, Math.min(1 - eps, pred));
    totalLoss += - (p.cls * Math.log(clampedPred) + (1 - p.cls) * Math.log(1 - clampedPred));
  }
  totalLoss /= points.length;

  // Regularization
  if (regularization === 'l2') {
    let regTerm = 0;
    for (let i = 1; i < weights.length; i++) regTerm += weights[i] * weights[i];
    totalLoss += (regStrength / 2) * regTerm;
  } else if (regularization === 'l1') {
    let regTerm = 0;
    for (let i = 1; i < weights.length; i++) regTerm += Math.abs(weights[i]);
    totalLoss += regStrength * regTerm;
  }

  return totalLoss;
}

/* ─── Gradients ─── */
export function computeGradients(points: Point[], weights: Weights, regularization: string, regStrength: number, degree: number): Weights {
  const grads = new Array(weights.length).fill(0);
  const n = points.length;
  if (n === 0) return grads;

  for (const p of points) {
    const pred = predict(p.x, p.y, weights);
    const err = pred - p.cls;
    const features = getFeatureVector(p.x, p.y, degree);

    for (let j = 0; j < weights.length; j++) {
      grads[j] += err * features[j];
    }
  }

  for (let j = 0; j < weights.length; j++) {
    grads[j] /= n;
  }

  // Regularization gradients
  if (regularization === 'l2') {
    for (let j = 1; j < weights.length; j++) grads[j] += regStrength * weights[j];
  } else if (regularization === 'l1') {
    for (let j = 1; j < weights.length; j++) grads[j] += regStrength * Math.sign(weights[j]);
  }

  return grads;
}

/* ─── Confusion Matrix ─── */
export interface ConfusionMatrix {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

export function computeConfusionMatrix(points: Point[], weights: Weights, threshold: number): ConfusionMatrix {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const p of points) {
    const pred = predict(p.x, p.y, weights) >= threshold ? 1 : 0;
    if (pred === 1 && p.cls === 1) tp++;
    if (pred === 0 && p.cls === 0) tn++;
    if (pred === 1 && p.cls === 0) fp++;
    if (pred === 0 && p.cls === 1) fn++;
  }
  return { tp, tn, fp, fn };
}

/* ─── ROC Curve ─── */
export interface ROCPoint {
  threshold: number;
  fpr: number;
  tpr: number;
}

export function computeROCCurve(points: Point[], weights: Weights): { curve: ROCPoint[]; auc: number } {
  if (points.length === 0) return { curve: [], auc: 0 };

  // Get probabilities
  const scored = points.map(p => ({
    prob: predict(p.x, p.y, weights),
    cls: p.cls,
  }));

  // Generate thresholds
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

  // AUC via trapezoidal rule
  let auc = 0;
  for (let i = 1; i < curve.length; i++) {
    const dx = curve[i].fpr - curve[i - 1].fpr;
    const avgY = (curve[i].tpr + curve[i - 1].tpr) / 2;
    auc += dx * avgY;
  }
  auc = Math.max(0, Math.min(1, auc));

  return { curve, auc };
}

/* ─── Metrics ─── */
export function computeMetrics(points: Point[], weights: Weights, threshold = 0.5): MetricValue[] {
  if (points.length === 0) return [
    { label: 'Accuracy', value: '—', isPrimary: true },
    { label: 'Precision', value: '—' },
    { label: 'Recall', value: '—' },
    { label: 'F1 Score', value: '—' },
    { label: 'AUC', value: '—' },
  ];

  const { tp, tn, fp, fn } = computeConfusionMatrix(points, weights, threshold);

  const accuracy = (tp + tn) / points.length;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
  const { auc } = computeROCCurve(points, weights);

  return [
    { label: 'Accuracy', value: `${(accuracy * 100).toFixed(1)}%`, isPrimary: true },
    { label: 'Precision', value: `${(precision * 100).toFixed(1)}%` },
    { label: 'Recall', value: `${(recall * 100).toFixed(1)}%` },
    { label: 'F1 Score', value: `${(f1 * 100).toFixed(1)}%` },
    { label: 'AUC', value: auc.toFixed(3) },
  ];
}

/* ─── Equation string ─── */
export function formatEquation(weights: Weights): string {
  if (weights.length === 0) return 'σ(0)';
  const degree = weights.length > 3 ? 2 : 1;
  const labels = degree === 1 ? ['', 'x₁', 'x₂'] : ['', 'x₁', 'x₂', 'x₁²', 'x₂²', 'x₁x₂'];
  
  let eq = '';
  for (let i = 0; i < weights.length; i++) {
    const w = weights[i];
    if (Math.abs(w) < 0.001 && i > 0) continue;
    const sign = w >= 0 ? (eq ? ' + ' : '') : (eq ? ' − ' : '-');
    const val = Math.abs(w).toFixed(2);
    const term = i === 0 ? val : `${val}${labels[i]}`;
    eq += `${sign}${term}`;
  }
  return `σ(${eq || '0'})`;
}

/* ─── Data Statistics ─── */
export interface DataStats {
  n: number;
  nClass0: number;
  nClass1: number;
  meanX: number;
  meanY: number;
  xRange: [number, number];
  yRange: [number, number];
  classBalance: number; // ratio of class 1 to total
}

export function computeDataStats(points: Point[]): DataStats | null {
  if (points.length < 2) return null;
  const n = points.length;
  const nClass0 = points.filter(p => p.cls === 0).length;
  const nClass1 = n - nClass0;
  let sumX = 0, sumY = 0;
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const p of points) {
    sumX += p.x; sumY += p.y;
    if (p.x < xMin) xMin = p.x; if (p.x > xMax) xMax = p.x;
    if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y;
  }
  return {
    n,
    nClass0,
    nClass1,
    meanX: sumX / n,
    meanY: sumY / n,
    xRange: [xMin, xMax],
    yRange: [yMin, yMax],
    classBalance: nClass1 / n,
  };
}
