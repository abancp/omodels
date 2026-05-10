/**
 * K-Nearest Neighbors Math Engine
 * Pure functions for distance calculation, classification, and data generation.
 */

import type { MetricValue } from '../registry';

export interface Point {
  x: number;
  y: number;
  cls: number;
}

export interface KNNResult {
  cls: number;
  prob: number;
  neighbors: { point: Point; dist: number }[];
}

/* ─── Seeded PRNG ─── */
function seededRandom(seed: number) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}

/* ─── Dataset Generation ─── */
export function generateKNNData(dataset: string, count: number, noise: number): Point[] {
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
    }

    pts.push({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), cls });
  }

  return pts;
}

/* ─── Distances ─── */
export function computeDistance(ax: number, ay: number, bx: number, by: number, metric: string, p = 2): number {
  switch (metric) {
    case 'manhattan':
      return Math.abs(ax - bx) + Math.abs(ay - by);
    case 'chebyshev':
      return Math.max(Math.abs(ax - bx), Math.abs(ay - by));
    case 'minkowski':
      return Math.pow(Math.pow(Math.abs(ax - bx), p) + Math.pow(Math.abs(ay - by), p), 1 / p);
    case 'euclidean':
    default:
      return Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2);
  }
}

/* ─── Classification ─── */
export function classifyKNN(px: number, py: number, points: Point[], k: number, metric: string, weightType: string, p = 2): KNNResult {
  if (points.length === 0) return { cls: 0, prob: 0, neighbors: [] };

  const distances = points.map(pt => ({
    point: pt,
    dist: computeDistance(px, py, pt.x, pt.y, metric, p)
  })).sort((a, b) => a.dist - b.dist).slice(0, k);

  let w0 = 0, w1 = 0;
  for (const n of distances) {
    let weight = 1;
    if (weightType === 'distance') {
      weight = n.dist === 0 ? 1000 : 1 / n.dist;
    }
    if (n.point.cls === 0) w0 += weight;
    else w1 += weight;
  }

  const total = w0 + w1;
  const prob = total > 0 ? w1 / total : 0;
  const cls = w0 >= w1 ? 0 : 1;

  return { cls, prob, neighbors: distances };
}

/* ─── Confusion Matrix ─── */
export interface ConfusionMatrix {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

export function computeKNNConfusionMatrix(points: Point[], k: number, metric: string, weightType: string, p = 2): ConfusionMatrix {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  
  // Use leave-one-out for accuracy approximation to avoid always getting 100% on k=1
  for (let i = 0; i < points.length; i++) {
    const pt = points[i];
    const trainPoints = points.slice(0, i).concat(points.slice(i + 1));
    const res = classifyKNN(pt.x, pt.y, trainPoints, k, metric, weightType, p);
    
    if (res.cls === 1 && pt.cls === 1) tp++;
    if (res.cls === 0 && pt.cls === 0) tn++;
    if (res.cls === 1 && pt.cls === 0) fp++;
    if (res.cls === 0 && pt.cls === 1) fn++;
  }
  
  return { tp, tn, fp, fn };
}

/* ─── Metrics ─── */
export function computeKNNMetrics(points: Point[], k: number, metric: string, weightType: string, p = 2): MetricValue[] {
  if (points.length < 2) return [
    { label: 'Accuracy', value: '—', isPrimary: true },
    { label: 'Precision', value: '—' },
    { label: 'Recall', value: '—' },
    { label: 'F1 Score', value: '—' },
  ];

  const { tp, tn, fp, fn } = computeKNNConfusionMatrix(points, k, metric, weightType, p);

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

/* ─── Data Statistics ─── */
export interface DataStats {
  n: number;
  nClass0: number;
  nClass1: number;
  xRange: [number, number];
  yRange: [number, number];
}

export function computeDataStats(points: Point[]): DataStats | null {
  if (points.length < 2) return null;
  const n = points.length;
  const nClass0 = points.filter(p => p.cls === 0).length;
  const nClass1 = n - nClass0;
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const p of points) {
    if (p.x < xMin) xMin = p.x; if (p.x > xMax) xMax = p.x;
    if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y;
  }
  return { n, nClass0, nClass1, xRange: [xMin, xMax], yRange: [yMin, yMax] };
}
