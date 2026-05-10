/**
 * Polynomial Regression Math Engine
 * Pure functions for feature expansion, prediction, loss, and gradients.
 */

import type { MetricValue } from '../registry';

export interface Point {
  x: number;
  y: number;
}

export type Weights = number[];

/**
 * Generates synthetic dataset for polynomial regression.
 */
export function generatePolyData(dataset: string, count: number, noise: number): Point[] {
  // Simple seeded random generator for deterministic datasets
  let s = count * 100 + Math.floor(noise * 50);
  const rand = () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };

  const pts: Point[] = [];
  for (let i = 0; i < count; i++) {
    const x = rand();
    let y: number;

    switch (dataset) {
      case 'quadratic':
        // y = 2x² - x + 0.5
        y = 2 * x * x - x + 0.5 + (rand() - 0.5) * noise * 1.5;
        break;
      case 'cubic':
        // y = 4x³ - 6x² + 3x + 0.2
        y = 4 * x * x * x - 6 * x * x + 3 * x + 0.2 + (rand() - 0.5) * noise * 1.5;
        break;
      case 'sinusoidal':
        // y = sin(2πx) mapped to positive range
        y = Math.sin(x * Math.PI * 2) * 1.5 + 2.0 + (rand() - 0.5) * noise * 2.0;
        break;
      case 'step':
        y = (x > 0.5 ? 3.0 : 1.0) + (rand() - 0.5) * noise * 1.0;
        break;
      case 'noisy':
        y = 2 * x * x - x + 0.5 + (rand() - 0.5) * noise * 4.0;
        break;
      case 'linear':
      default:
        y = 2 * x + 0.5 + (rand() - 0.5) * noise * 1.5;
        break;
    }

    pts.push({ x, y: Math.max(0, Math.min(4.0, y)) });
  }
  return pts;
}

/**
 * Expands a single scalar x into a feature vector [1, x, x², ..., xⁿ]
 */
export function expandFeatures(x: number, degree: number): number[] {
  const features = new Array(degree + 1);
  features[0] = 1;
  let currentX = x;
  for (let i = 1; i <= degree; i++) {
    features[i] = currentX;
    currentX *= x;
  }
  return features;
}

/**
 * Predicts y for a given x using the provided weights.
 * weights[0] is the bias (w0), weights[1] is w1*x, weights[2] is w2*x^2, etc.
 */
export function predict(x: number, weights: Weights): number {
  let y = weights[0];
  let currentX = x;
  for (let i = 1; i < weights.length; i++) {
    y += weights[i] * currentX;
    currentX *= x;
  }
  return y;
}

/**
 * Computes the loss over the dataset.
 */
export function computeLoss(points: Point[], weights: Weights, lossType: string): number {
  if (points.length === 0) return 0;
  let totalLoss = 0;

  for (const p of points) {
    const pred = predict(p.x, weights);
    const err = pred - p.y;

    if (lossType === 'mae') {
      totalLoss += Math.abs(err);
    } else if (lossType === 'huber') {
      const delta = 1.0;
      if (Math.abs(err) <= delta) {
        totalLoss += 0.5 * err * err;
      } else {
        totalLoss += delta * Math.abs(err) - 0.5 * delta * delta;
      }
    } else {
      // Default to MSE
      totalLoss += err * err;
    }
  }

  return totalLoss / points.length;
}

/**
 * Computes gradients for each weight.
 */
export function computeGradients(points: Point[], weights: Weights, lossType: string): Weights {
  const degree = weights.length - 1;
  const grads = new Array(degree + 1).fill(0);
  const n = points.length;
  if (n === 0) return grads;

  for (const p of points) {
    const pred = predict(p.x, weights);
    const err = pred - p.y;
    const features = expandFeatures(p.x, degree);

    for (let j = 0; j <= degree; j++) {
      if (lossType === 'mae') {
        grads[j] += (err > 0 ? 1 : -1) * features[j];
      } else if (lossType === 'huber') {
        const delta = 1.0;
        if (Math.abs(err) <= delta) {
          grads[j] += err * features[j];
        } else {
          grads[j] += delta * (err > 0 ? 1 : -1) * features[j];
        }
      } else {
        // Default to MSE derivative: 2 * err * x^j
        grads[j] += 2 * err * features[j];
      }
    }
  }

  for (let j = 0; j <= degree; j++) {
    grads[j] /= n;
  }

  return grads;
}

/**
 * Formats weights array into a readable polynomial equation string.
 */
export function formatEquation(weights: Weights): string {
  if (weights.length === 0) return 'y = 0';

  let eq = '';
  for (let i = weights.length - 1; i >= 0; i--) {
    const w = weights[i];
    if (Math.abs(w) < 0.001 && i !== 0 && weights.length > 1) continue; // Skip near-zero terms unless it's the bias or the only term

    const sign = w >= 0 ? (eq ? ' + ' : '') : (eq ? ' − ' : '-');
    const val = Math.abs(w).toFixed(2);
    
    let term = '';
    if (i === 0) term = val;
    else if (i === 1) term = `${val}x`;
    else term = `${val}x^${i}`; // Using ^ for UI brevity, though superscripts are nicer if HTML allowed

    eq += `${sign}${term}`;
  }
  
  return eq ? `y = ${eq}` : 'y = 0';
}

/**
 * Computes standard metrics (R², MSE, MAE).
 */
export function computeMetrics(points: Point[], weights: Weights): MetricValue[] {
  if (points.length < 2) return [
    { label: 'R²', value: '—', isPrimary: true },
    { label: 'MSE', value: '—' },
    { label: 'MAE', value: '—' },
    { label: 'Degree', value: (weights.length - 1).toString() },
    { label: 'Equation', value: '—' },
  ];

  const yMean = points.reduce((s, p) => s + p.y, 0) / points.length;
  let ssTot = 0, ssRes = 0, mse = 0, mae = 0;

  for (const p of points) {
    const pred = predict(p.x, weights);
    const err = p.y - pred;
    ssRes += err * err;
    ssTot += (p.y - yMean) ** 2;
    mse += err * err;
    mae += Math.abs(err);
  }

  mse /= points.length;
  mae /= points.length;
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;

  return [
    { label: 'R²', value: r2.toFixed(4), isPrimary: true },
    { label: 'MSE', value: mse.toFixed(4) },
    { label: 'MAE', value: mae.toFixed(4) },
    { label: 'Degree', value: (weights.length - 1).toString() },
    { label: 'Equation', value: formatEquation(weights) },
  ];
}
