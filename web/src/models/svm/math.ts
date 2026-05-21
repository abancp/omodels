import type { MetricValue } from '../registry';

export interface Point {
  x: number;
  y: number;
  cls: number; // 0 or 1
}

export type Weights = number[];

export interface SVMResult {
  cls: number;
  prob: number;
  marginDist: number;
}

/* ─── Dataset Generation ─── */
function seededRandom(seed: number) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}

export function generateSVMData(dataset: string, count: number, noise: number): Point[] {
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
    } else if (dataset === 'linear') {
      x = rand();
      y = rand();
      if (y > x + (rand() - 0.5) * noise) {
        pts.push({ x, y, cls: 0 });
      } else {
        pts.push({ x, y, cls: 1 });
      }
      continue;
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
      x = rand();
      y = rand();
      const nx = x + (rand() - 0.5) * noise * 0.5;
      const ny = y + (rand() - 0.5) * noise * 0.5;
      const isCls1 = (nx > 0.5 && ny > 0.5) || (nx < 0.5 && ny < 0.5);
      pts.push({ x: Math.max(0, Math.min(1, nx)), y: Math.max(0, Math.min(1, ny)), cls: isCls1 ? 1 : 0 });
      continue;
    } else if (dataset === 'spiral') {
      const n = count / 2;
      const r = (i % n) / n * 0.4;
      const t = 1.25 * (i % n) / n * 2 * Math.PI + (cls === 1 ? Math.PI : 0);
      x = 0.5 + r * Math.sin(t) + (rand() - 0.5) * noise * 0.1;
      y = 0.5 + r * Math.cos(t) + (rand() - 0.5) * noise * 0.1;
    }

    pts.push({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), cls });
  }

  // Linear needs truncation because of the random rejection logic
  return pts.slice(0, count);
}

/* ─── SVM Math (RFF for RBF) ─── */
function boxMuller(rand: () => number) {
  let u1 = 0, u2 = 0;
  while (u1 === 0) u1 = rand();
  while (u2 === 0) u2 = rand();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
  return { z0, z1 };
}

let rffParams: { w: number[][], b: number[] } | null = null;
const RFF_D = 200; // 200 dimensions for high-fidelity RFF approximation
const GAMMA = 20.0; // RBF bandwidth tuned for [0, 1] coordinate range

function initRFF() {
  if (rffParams) return;
  const rand = seededRandom(1337);
  const w: number[][] = [];
  const b: number[] = [];
  const stddev = Math.sqrt(2 * GAMMA);
  for (let i = 0; i < RFF_D; i++) {
    const bm = boxMuller(rand);
    w.push([bm.z0 * stddev, bm.z1 * stddev]);
    b.push(rand() * 2 * Math.PI);
  }
  rffParams = { w, b };
}

// Features mapping for non-linear boundaries via polynomial kernel equivalent or RFF
export function expandFeatures(x1: number, x2: number, kernel: string): number[] {
  if (kernel === 'poly2') return [x1, x2, x1 * x1, x2 * x2, x1 * x2];
  if (kernel === 'rbf') {
    initRFF();
    const feats = new Array(RFF_D);
    const mult = Math.sqrt(2 / RFF_D);
    for (let i = 0; i < RFF_D; i++) {
      feats[i] = mult * Math.cos(rffParams!.w[i][0] * x1 + rffParams!.w[i][1] * x2 + rffParams!.b[i]);
    }
    return feats;
  }
  return [x1, x2]; // linear default
}

/** Get the number of features for a given kernel */
export function featureCount(kernel: string): number {
  if (kernel === 'poly2') return 5;
  if (kernel === 'rbf') return RFF_D;
  return 2;
}

/** Initialize weights with random values (breaks symmetry) */
export function initWeights(kernel: string, seed = 42): Weights {
  const n = featureCount(kernel) + 1; // +1 for bias
  const rand = seededRandom(seed);
  return Array.from({ length: n }, () => (rand() - 0.5) * 2.0);
}

// Compute f(x) = w^T x + b
export function computeMargin(px: number, py: number, weights: Weights, kernel: string): number {
  if (weights.length === 0) return 0;
  const b = weights[0];
  const w = weights.slice(1);
  const feats = expandFeatures(px, py, kernel);
  let dot = b;
  for (let i = 0; i < feats.length; i++) {
    dot += w[i] * feats[i];
  }
  return dot;
}

export function predict(px: number, py: number, weights: Weights, kernel: string): SVMResult {
  const marginDist = computeMargin(px, py, weights, kernel);
  // Prob is mapped through a sigmoid just for visualizing confidence, even though SVM doesn't output true probs
  const prob = 1 / (1 + Math.exp(-marginDist));
  const cls = marginDist >= 0 ? 1 : 0;
  return { cls, prob, marginDist };
}

// Hinge Loss: L = C * sum(max(0, 1 - y_i * f(x_i))) + 0.5 * ||w||^2
// y_i in {-1, 1}
// This is the standard formulation (like scikit-learn) where C multiplies the sum.
export function computeLoss(points: Point[], weights: Weights, kernel: string, C: number): number {
  if (points.length === 0 || weights.length === 0) return 0;
  
  let hingeSum = 0;
  const w = weights.slice(1);

  for (const pt of points) {
    const yTrue = pt.cls === 1 ? 1 : -1;
    const fX = computeMargin(pt.x, pt.y, weights, kernel);
    hingeSum += Math.max(0, 1 - yTrue * fX);
  }

  // L2 Regularization term
  let l2 = 0;
  for (const val of w) l2 += val * val;

  return C * hingeSum + 0.5 * l2;
}

export function computeGradients(points: Point[], weights: Weights, kernel: string, C: number): Weights {
  if (points.length === 0 || weights.length === 0) return weights.map(() => 0);

  const grads = new Array(weights.length).fill(0);
  const w = weights.slice(1);

  // Regularization gradient: d/dw (0.5 * ||w||^2) = w (no regularization on bias)
  for (let i = 0; i < w.length; i++) {
    grads[i + 1] = w[i];
  }

  // Hinge loss gradient: d/dw (C * sum max(0, 1 - y*f(x)))
  for (const pt of points) {
    const yTrue = pt.cls === 1 ? 1 : -1;
    const fX = computeMargin(pt.x, pt.y, weights, kernel);
    
    if (1 - yTrue * fX > 0) {
      grads[0] -= C * yTrue; // bias gradient
      const feats = expandFeatures(pt.x, pt.y, kernel);
      for (let j = 0; j < feats.length; j++) {
        grads[j + 1] -= C * yTrue * feats[j];
      }
    }
  }

  // Clip gradients to prevent exploding gradients
  let normSq = 0;
  for (const g of grads) normSq += g * g;
  const norm = Math.sqrt(normSq);
  const maxNorm = 100.0;
  if (norm > maxNorm) {
    const scale = maxNorm / norm;
    for (let i = 0; i < grads.length; i++) grads[i] *= scale;
  }

  return grads;
}

/** Run a single PEGASOS-style SGD step (mutates and returns weights) */
export function trainStep(
  weights: Weights, points: Point[], kernel: string,
  C: number, lr: number, step: number
): { weights: Weights; loss: number; gradients: Weights } {
  // Gentle learning rate decay
  const effectiveLr = lr / (1 + step * 0.0005);
  const grads = computeGradients(points, weights, kernel, C);
  const newW = weights.map((w, i) => w - effectiveLr * grads[i]);
  const loss = computeLoss(points, newW, kernel, C);
  return { weights: newW, loss, gradients: grads };
}

export interface ConfusionMatrix {
  tp: number; tn: number; fp: number; fn: number;
}

export function computeConfusionMatrix(points: Point[], weights: Weights, kernel: string): ConfusionMatrix {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const pt of points) {
    const { cls } = predict(pt.x, pt.y, weights, kernel);
    if (cls === 1 && pt.cls === 1) tp++;
    if (cls === 0 && pt.cls === 0) tn++;
    if (cls === 1 && pt.cls === 0) fp++;
    if (cls === 0 && pt.cls === 1) fn++;
  }
  return { tp, tn, fp, fn };
}

export function computeMetrics(points: Point[], weights: Weights, kernel: string): MetricValue[] {
  if (points.length === 0 || weights.length === 0) return [
    { label: 'Accuracy', value: '—', isPrimary: true },
    { label: 'Precision', value: '—' },
    { label: 'Recall', value: '—' },
    { label: 'F1 Score', value: '—' },
  ];

  const { tp, tn, fp, fn } = computeConfusionMatrix(points, weights, kernel);
  const accuracy = (tp + tn) / points.length;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

  return [
    { label: 'Accuracy', value: `${(accuracy * 100).toFixed(1)}%`, isPrimary: true },
    { label: 'Precision', value: `${(precision * 100).toFixed(1)}%` },
    { label: 'Recall', value: `${(recall * 100).toFixed(1)}%` },
    { label: 'F1 Score', value: `${(f1 * 100).toFixed(1)}%` },
  ];
}

/** Count support vectors — points on or inside the margin */
export function countSupportVectors(points: Point[], weights: Weights, kernel: string): number {
  let count = 0;
  for (const pt of points) {
    const yTrue = pt.cls === 1 ? 1 : -1;
    const m = computeMargin(pt.x, pt.y, weights, kernel);
    if (1 - yTrue * m >= 0) count++;
  }
  return count;
}

export interface DataStats {
  n: number; nClass0: number; nClass1: number;
  xRange: [number, number]; yRange: [number, number];
  supportVectors: number;
}

export function computeDataStats(points: Point[], weights: Weights, kernel: string): DataStats | null {
  if (points.length === 0) return null;
  const n = points.length;
  const nClass0 = points.filter(p => p.cls === 0).length;
  const nClass1 = n - nClass0;
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  return {
    n,
    nClass0,
    nClass1,
    xRange: [Math.min(...xs), Math.max(...xs)],
    yRange: [Math.min(...ys), Math.max(...ys)],
    supportVectors: countSupportVectors(points, weights, kernel),
  };
}

export function formatEquation(weights: Weights, kernel: string): string {
  if (weights.length === 0) return 'f(x) = 0';
  const b = weights[0].toFixed(2);
  if (kernel === 'poly2') {
    return `f(x) = ${weights[1].toFixed(2)}x₁ + ${weights[2].toFixed(2)}x₂ + ${weights[3].toFixed(2)}x₁² + ${weights[4].toFixed(2)}x₂² + ${weights[5].toFixed(2)}x₁x₂ + ${b}`;
  } else if (kernel === 'rbf') {
    return `f(x) = Σ wᵢ·φ(x)ᵢ + ${b} (RBF Approx)`;
  } else {
    // linear
    return `f(x) = ${weights[1].toFixed(2)}x₁ + ${weights[2].toFixed(2)}x₂ + ${b}`;
  }
}
