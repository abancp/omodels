/**
 * Decision Tree Math Engine
 * Supports ID3 (Entropy/Information Gain), C4.5 (Gain Ratio), and CART (Gini Impurity).
 * Handles 2D continuous features with configurable splitting, depth, and pruning.
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

/* ─── Tree Node ─── */
export interface TreeNode {
  id: number;
  isLeaf: boolean;
  predictedClass: number;
  classCounts: Record<number, number>;
  samples: number;
  depth: number;

  // Split info (non-leaf)
  splitFeature?: 'x' | 'y';
  splitValue?: number;
  impurity?: number;          // impurity of this node
  informationGain?: number;
  left?: TreeNode;            // <= splitValue
  right?: TreeNode;           // > splitValue
}

export interface DecisionTreeState {
  algorithm: 'id3' | 'c45' | 'cart';
  root: TreeNode | null;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  nodeCount: number;
  leafCount: number;
  treeDepth: number;
  featureImportance: { x: number; y: number };
  buildSteps: BuildStep[];    // for animated building
}

/* ─── Build animation steps ─── */
export interface BuildStep {
  nodeId: number;
  action: 'split' | 'leaf';
  depth: number;
  splitFeature?: 'x' | 'y';
  splitValue?: number;
  impurity: number;
  samples: number;
  predictedClass: number;
}

/* ─── Impurity Functions ─── */

function entropy(counts: Record<number, number>, total: number): number {
  if (total === 0) return 0;
  let e = 0;
  for (const c of Object.values(counts)) {
    if (c === 0) continue;
    const p = c / total;
    e -= p * Math.log2(p);
  }
  return e;
}

function giniImpurity(counts: Record<number, number>, total: number): number {
  if (total === 0) return 0;
  let sum = 0;
  for (const c of Object.values(counts)) {
    const p = c / total;
    sum += p * p;
  }
  return 1 - sum;
}

function impurityFn(algorithm: 'id3' | 'c45' | 'cart', counts: Record<number, number>, total: number): number {
  if (algorithm === 'cart') return giniImpurity(counts, total);
  return entropy(counts, total);
}

function classCounts(points: Point[]): Record<number, number> {
  const counts: Record<number, number> = {};
  for (const p of points) {
    counts[p.cls] = (counts[p.cls] || 0) + 1;
  }
  return counts;
}

function majorityClass(points: Point[]): number {
  const counts = classCounts(points);
  let best = -1, bestCount = -1;
  for (const [cls, count] of Object.entries(counts)) {
    if (count > bestCount) {
      bestCount = count;
      best = parseInt(cls, 10);
    }
  }
  return best;
}

/* ─── Find Best Split ─── */
interface SplitCandidate {
  feature: 'x' | 'y';
  value: number;
  gain: number;
  gainRatio?: number;  // for C4.5
  leftPoints: Point[];
  rightPoints: Point[];
  leftCounts: Record<number, number>;
  rightCounts: Record<number, number>;
}

function findBestSplit(
  points: Point[],
  algorithm: 'id3' | 'c45' | 'cart',
  minSamplesLeaf: number,
  numBins: number
): SplitCandidate | null {
  const n = points.length;
  const parentCounts = classCounts(points);
  const parentImpurity = impurityFn(algorithm, parentCounts, n);

  let bestSplit: SplitCandidate | null = null;
  let bestScore = -Infinity;

  for (const feature of ['x', 'y'] as const) {
    const vals = points.map(p => p[feature]).sort((a, b) => a - b);
    const minVal = vals[0];
    const maxVal = vals[vals.length - 1];
    if (maxVal - minVal < 1e-10) continue;

    // Generate candidate thresholds
    const thresholds: number[] = [];
    const binSize = (maxVal - minVal) / numBins;
    for (let i = 1; i < numBins; i++) {
      thresholds.push(minVal + i * binSize);
    }

    // Also add midpoints between sorted unique values for better splits
    const unique = [...new Set(vals)];
    if (unique.length <= 50) {
      for (let i = 0; i < unique.length - 1; i++) {
        thresholds.push((unique[i] + unique[i + 1]) / 2);
      }
    }

    for (const threshold of thresholds) {
      const leftPts = points.filter(p => p[feature] <= threshold);
      const rightPts = points.filter(p => p[feature] > threshold);

      if (leftPts.length < minSamplesLeaf || rightPts.length < minSamplesLeaf) continue;

      const leftCounts = classCounts(leftPts);
      const rightCounts = classCounts(rightPts);

      const leftImpurity = impurityFn(algorithm, leftCounts, leftPts.length);
      const rightImpurity = impurityFn(algorithm, rightCounts, rightPts.length);

      const weightedImpurity = (leftPts.length / n) * leftImpurity + (rightPts.length / n) * rightImpurity;
      const gain = parentImpurity - weightedImpurity;

      let score = gain;

      // C4.5 uses gain ratio
      if (algorithm === 'c45') {
        const splitInfo = -(
          (leftPts.length / n) * Math.log2(leftPts.length / n + 1e-12) +
          (rightPts.length / n) * Math.log2(rightPts.length / n + 1e-12)
        );
        score = splitInfo > 1e-10 ? gain / splitInfo : 0;
      }

      if (score > bestScore) {
        bestScore = score;
        bestSplit = {
          feature,
          value: threshold,
          gain,
          gainRatio: algorithm === 'c45' ? score : undefined,
          leftPoints: leftPts,
          rightPoints: rightPts,
          leftCounts,
          rightCounts,
        };
      }
    }
  }

  return bestSplit;
}

/* ─── Build Tree ─── */
let globalNodeId = 0;

function buildNode(
  points: Point[],
  depth: number,
  algorithm: 'id3' | 'c45' | 'cart',
  maxDepth: number,
  minSamplesSplit: number,
  minSamplesLeaf: number,
  numBins: number,
  steps: BuildStep[],
  featureImportance: { x: number; y: number }
): TreeNode {
  const id = globalNodeId++;
  const counts = classCounts(points);
  const predicted = majorityClass(points);
  const imp = impurityFn(algorithm, counts, points.length);

  // Stop conditions
  const uniqueClasses = Object.keys(counts).length;
  if (
    uniqueClasses <= 1 ||
    depth >= maxDepth ||
    points.length < minSamplesSplit ||
    imp < 1e-10
  ) {
    steps.push({
      nodeId: id,
      action: 'leaf',
      depth,
      impurity: imp,
      samples: points.length,
      predictedClass: predicted,
    });
    return {
      id,
      isLeaf: true,
      predictedClass: predicted,
      classCounts: counts,
      samples: points.length,
      depth,
      impurity: imp,
    };
  }

  const split = findBestSplit(points, algorithm, minSamplesLeaf, numBins);

  if (!split || split.gain < 1e-10) {
    steps.push({
      nodeId: id,
      action: 'leaf',
      depth,
      impurity: imp,
      samples: points.length,
      predictedClass: predicted,
    });
    return {
      id,
      isLeaf: true,
      predictedClass: predicted,
      classCounts: counts,
      samples: points.length,
      depth,
      impurity: imp,
    };
  }

  // Track feature importance (weighted by sample proportion)
  featureImportance[split.feature] += split.gain * points.length;

  steps.push({
    nodeId: id,
    action: 'split',
    depth,
    splitFeature: split.feature,
    splitValue: split.value,
    impurity: imp,
    samples: points.length,
    predictedClass: predicted,
  });

  const left = buildNode(split.leftPoints, depth + 1, algorithm, maxDepth, minSamplesSplit, minSamplesLeaf, numBins, steps, featureImportance);
  const right = buildNode(split.rightPoints, depth + 1, algorithm, maxDepth, minSamplesSplit, minSamplesLeaf, numBins, steps, featureImportance);

  return {
    id,
    isLeaf: false,
    predictedClass: predicted,
    classCounts: counts,
    samples: points.length,
    depth,
    splitFeature: split.feature,
    splitValue: split.value,
    impurity: imp,
    informationGain: split.gain,
    left,
    right,
  };
}

/* ─── Count Nodes ─── */
function countNodes(node: TreeNode): { total: number; leaves: number; maxDepth: number } {
  if (node.isLeaf) return { total: 1, leaves: 1, maxDepth: node.depth };
  const l = node.left ? countNodes(node.left) : { total: 0, leaves: 0, maxDepth: 0 };
  const r = node.right ? countNodes(node.right) : { total: 0, leaves: 0, maxDepth: 0 };
  return {
    total: 1 + l.total + r.total,
    leaves: l.leaves + r.leaves,
    maxDepth: Math.max(l.maxDepth, r.maxDepth),
  };
}

/* ─── Train ─── */
export function trainDecisionTree(
  points: Point[],
  algorithm: 'id3' | 'c45' | 'cart',
  maxDepth: number,
  minSamplesSplit: number,
  minSamplesLeaf: number,
  numBins: number
): DecisionTreeState {
  globalNodeId = 0;
  const steps: BuildStep[] = [];
  const featureImportance = { x: 0, y: 0 };

  if (points.length < 2) {
    return {
      algorithm,
      root: null,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      nodeCount: 0,
      leafCount: 0,
      treeDepth: 0,
      featureImportance: { x: 0, y: 0 },
      buildSteps: [],
    };
  }

  const root = buildNode(points, 0, algorithm, maxDepth, minSamplesSplit, minSamplesLeaf, numBins, steps, featureImportance);
  const { total, leaves, maxDepth: depth } = countNodes(root);

  // Normalize feature importance
  const totalImp = featureImportance.x + featureImportance.y;
  if (totalImp > 0) {
    featureImportance.x /= totalImp;
    featureImportance.y /= totalImp;
  }

  return {
    algorithm,
    root,
    maxDepth,
    minSamplesSplit,
    minSamplesLeaf,
    nodeCount: total,
    leafCount: leaves,
    treeDepth: depth,
    featureImportance,
    buildSteps: steps,
  };
}

/* ─── Prediction ─── */
export function predictSingle(px: number, py: number, root: TreeNode): number {
  let node: TreeNode = root;
  while (!node.isLeaf) {
    if (!node.splitFeature || node.splitValue === undefined) break;
    const val = node.splitFeature === 'x' ? px : py;
    node = val <= node.splitValue ? node.left! : node.right!;
  }
  return node.predictedClass;
}

export function predictProbability(px: number, py: number, root: TreeNode): number {
  let node: TreeNode = root;
  while (!node.isLeaf) {
    if (!node.splitFeature || node.splitValue === undefined) break;
    const val = node.splitFeature === 'x' ? px : py;
    node = val <= node.splitValue ? node.left! : node.right!;
  }
  const total = Object.values(node.classCounts).reduce((a, b) => a + b, 0);
  return (node.classCounts[1] || 0) / (total || 1);
}

/* ─── Get Decision Path ─── */
export function getDecisionPath(px: number, py: number, root: TreeNode): TreeNode[] {
  const path: TreeNode[] = [];
  let node: TreeNode = root;
  path.push(node);
  while (!node.isLeaf) {
    if (!node.splitFeature || node.splitValue === undefined) break;
    const val = node.splitFeature === 'x' ? px : py;
    node = val <= node.splitValue ? node.left! : node.right!;
    path.push(node);
  }
  return path;
}

/* ─── Confusion Matrix ─── */
export interface ConfusionMatrix {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

export function computeConfusionMatrix(points: Point[], state: DecisionTreeState): ConfusionMatrix {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  if (!state.root) return { tp, tn, fp, fn };
  for (const p of points) {
    const pred = predictSingle(p.x, p.y, state.root);
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

export function computeROCCurve(points: Point[], state: DecisionTreeState): { curve: ROCPoint[]; auc: number } {
  if (points.length === 0 || !state.root) return { curve: [], auc: 0 };

  const scored = points.map(p => ({
    prob: predictProbability(p.x, p.y, state.root!),
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

/* ─── Metrics ─── */
export function computeMetrics(points: Point[], state: DecisionTreeState): MetricValue[] {
  if (points.length === 0 || !state.root) return [
    { label: 'Accuracy', value: '—', isPrimary: true },
    { label: 'Precision', value: '—' },
    { label: 'Recall', value: '—' },
    { label: 'F1 Score', value: '—' },
    { label: 'Tree Depth', value: '—' },
    { label: 'Nodes', value: '—' },
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
    { label: 'Tree Depth', value: `${state.treeDepth}` },
    { label: 'Nodes', value: `${state.nodeCount} (${state.leafCount} leaves)` },
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

/* ─── Tree Layout for Visualization ─── */
export interface LayoutNode {
  id: number;
  x: number;
  y: number;
  isLeaf: boolean;
  predictedClass: number;
  classCounts: Record<number, number>;
  samples: number;
  depth: number;
  splitFeature?: 'x' | 'y';
  splitValue?: number;
  impurity?: number;
  informationGain?: number;
  parentX?: number;
  parentY?: number;
}

export function layoutTree(root: TreeNode | null): LayoutNode[] {
  if (!root) return [];

  const nodes: LayoutNode[] = [];
  const positions = new Map<number, { x: number; minX: number; maxX: number }>();

  // Assign horizontal positions using in-order traversal
  let xCounter = 0;

  function assignX(node: TreeNode): void {
    if (node.left) assignX(node.left);
    positions.set(node.id, { x: xCounter, minX: xCounter, maxX: xCounter });
    xCounter++;
    if (node.right) assignX(node.right);
  }
  assignX(root);

  // Calculate subtree bounds for centering parent above children
  function calcBounds(node: TreeNode): { min: number; max: number } {
    if (node.isLeaf) {
      const pos = positions.get(node.id)!;
      return { min: pos.x, max: pos.x };
    }
    const lBounds = node.left ? calcBounds(node.left) : { min: Infinity, max: -Infinity };
    const rBounds = node.right ? calcBounds(node.right) : { min: Infinity, max: -Infinity };
    const min = Math.min(lBounds.min, rBounds.min);
    const max = Math.max(lBounds.max, rBounds.max);
    const centered = (min + max) / 2;
    positions.set(node.id, { x: centered, minX: min, maxX: max });
    return { min, max };
  }
  calcBounds(root);

  // Build layout nodes
  function traverse(node: TreeNode, parentX?: number, parentY?: number): void {
    const pos = positions.get(node.id)!;
    nodes.push({
      id: node.id,
      x: pos.x,
      y: node.depth,
      isLeaf: node.isLeaf,
      predictedClass: node.predictedClass,
      classCounts: node.classCounts,
      samples: node.samples,
      depth: node.depth,
      splitFeature: node.splitFeature,
      splitValue: node.splitValue,
      impurity: node.impurity,
      informationGain: node.informationGain,
      parentX,
      parentY,
    });

    if (node.left) traverse(node.left, pos.x, node.depth);
    if (node.right) traverse(node.right, pos.x, node.depth);
  }
  traverse(root);

  return nodes;
}
