/**
 * Pipeline Execution Engine
 * Handles: validation, topological sort, schema propagation, train/eval pass
 * Includes full custom client-side ML implementations for standard preprocessors & models.
 */

import type {
  Pipeline, PipelineNode, PipelineEdge, ValidationResult, ValidationError,
  DataSchema, NodeExecutionResult, DataSnapshot, DataFrame
} from './PipelineTypes';
import * as tf from '@tensorflow/tfjs';

// ─── Topological Sort (Kahn's algorithm) ──────────────────────────────────────

export function topologicalSort(nodes: PipelineNode[], edges: PipelineEdge[]): string[] | null {
  const inDegree: Record<string, number> = {};
  const adj: Record<string, string[]> = {};
  nodes.forEach(n => { inDegree[n.id] = 0; adj[n.id] = []; });
  edges.forEach(e => {
    adj[e.source].push(e.target);
    inDegree[e.target] = (inDegree[e.target] || 0) + 1;
  });
  const queue = nodes.filter(n => inDegree[n.id] === 0).map(n => n.id);
  const result: string[] = [];
  while (queue.length) {
    const cur = queue.shift()!;
    result.push(cur);
    for (const next of adj[cur]) {
      inDegree[next]--;
      if (inDegree[next] === 0) queue.push(next);
    }
  }
  return result.length === nodes.length ? result : null;
}

// ─── Validation ────────────────────────────────────────────────────────────────

export function validatePipeline(pipeline: Pipeline): ValidationResult {
  const errors: ValidationError[] = [];
  const { nodes, edges } = pipeline;

  if (!topologicalSort(nodes, edges)) {
    errors.push({ message: 'Pipeline contains a cycle — must be a DAG', type: 'error' });
  }

  const sources = nodes.filter(n => n.family === 'datasource');
  if (sources.length === 0) errors.push({ message: 'Pipeline needs exactly one Data Source node', type: 'error' });
  if (sources.length > 1) errors.push({ message: 'Pipeline can only have one Data Source node', type: 'error' });

  const targets = new Set(edges.map(e => e.target));
  nodes.filter(n => n.family !== 'datasource').forEach(n => {
    if (!targets.has(n.id)) {
      errors.push({ nodeId: n.id, message: `"${n.label}" has no incoming connection`, type: 'error' });
    }
  });

  nodes.filter(n => n.family === 'ensemble').forEach(n => {
    const incoming = edges.filter(e => e.target === n.id).length;
    if (incoming < 2) {
      errors.push({ nodeId: n.id, message: `"${n.label}" needs at least 2 inputs`, type: 'error' });
    }
  });

  sources.forEach(n => {
    const cfg = n.config as any;
    if (!cfg.targetColumn) {
      errors.push({ nodeId: n.id, message: 'Data Source: target column must be specified', type: 'warning' });
    }
  });

  return { valid: errors.filter(e => e.type === 'error').length === 0, errors };
}

// ─── Schema Propagation ────────────────────────────────────────────────────────

export function inferOutputSchema(node: PipelineNode, inputSchema?: DataSchema): DataSchema | undefined {
  if (!inputSchema) return undefined;
  const cfg = node.config as any;

  switch (node.type) {
    case 'StandardScaler':
    case 'MinMaxScaler':
    case 'RobustScaler':
    case 'SimpleImputer':
      return { ...inputSchema };

    case 'PCA': {
      const n = cfg.params?.nComponents ?? 2;
      return {
        columns: Array.from({ length: n }, (_, i) => ({ name: `pc${i + 1}`, type: 'numeric' as const })),
        shape: [inputSchema.shape[0], n],
        target: inputSchema.target,
        taskType: inputSchema.taskType,
      };
    }

    case 'SelectKBest': {
      const k = Math.min(cfg.params?.k ?? 10, inputSchema.shape[1]);
      return {
        columns: inputSchema.columns.slice(0, k),
        shape: [inputSchema.shape[0], k],
        target: inputSchema.target,
        taskType: inputSchema.taskType,
      };
    }

    case 'PolynomialFeatures': {
      const d = cfg.params?.degree ?? 2;
      const n = inputSchema.shape[1];
      const newN = Math.round(Math.pow(n + d, d) / Math.pow(d, d));
      return {
        columns: Array.from({ length: newN }, (_, i) => ({ name: `poly_${i}`, type: 'numeric' as const })),
        shape: [inputSchema.shape[0], newN],
        target: inputSchema.target,
        taskType: inputSchema.taskType,
      };
    }

    default:
      return { ...inputSchema };
  }
}

// ─── JS ML Helper functions ───────────────────────────────────────────────────

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr: number[], m: number): number {
  if (arr.length === 0) return 1;
  const variance = arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length;
  return Math.sqrt(variance + 1e-8);
}

// ─── JS Preprocessor Implementations ──────────────────────────────────────────

export class StandardScalerEngine {
  means: number[] = [];
  stds: number[] = [];

  fit(X: number[][]) {
    const cols = X[0]?.length ?? 0;
    this.means = [];
    this.stds = [];
    for (let j = 0; j < cols; j++) {
      const col = X.map(r => r[j]);
      const m = mean(col);
      this.means.push(m);
      this.stds.push(std(col, m));
    }
  }
  transform(X: number[][]): number[][] {
    return X.map(row => row.map((v, j) => (v - this.means[j]) / (this.stds[j] || 1)));
  }
}

export class MinMaxScalerEngine {
  mins: number[] = [];
  maxs: number[] = [];

  fit(X: number[][]) {
    const cols = X[0]?.length ?? 0;
    this.mins = [];
    this.maxs = [];
    for (let j = 0; j < cols; j++) {
      const col = X.map(r => r[j]);
      this.mins.push(Math.min(...col));
      this.maxs.push(Math.max(...col));
    }
  }
  transform(X: number[][]): number[][] {
    return X.map(row => row.map((v, j) => {
      const range = this.maxs[j] - this.mins[j];
      return range === 0 ? 0 : (v - this.mins[j]) / range;
    }));
  }
}

export class SimpleImputerEngine {
  values: number[] = [];
  strategy: string;
  constructor(strategy: string = 'mean') { this.strategy = strategy; }

  fit(X: number[][]) {
    const cols = X[0]?.length ?? 0;
    this.values = [];
    for (let j = 0; j < cols; j++) {
      const col = X.map(r => r[j]).filter(v => !isNaN(v) && isFinite(v));
      if (this.strategy === 'mean') this.values.push(mean(col));
      else if (this.strategy === 'median') {
        const sorted = [...col].sort((a, b) => a - b);
        this.values.push(sorted[Math.floor(sorted.length / 2)] ?? 0);
      } else this.values.push(col[0] ?? 0);
    }
  }
  transform(X: number[][]): number[][] {
    return X.map(row => row.map((v, j) => (isNaN(v) || !isFinite(v)) ? this.values[j] : v));
  }
}
export class LabelEncoderEngine {
  classes: Record<number, Record<string, number>> = {};

  fit(X: any[][]) {
    if (X.length === 0) return;
    const cols = X[0].length;
    for (let j = 0; j < cols; j++) {
      this.classes[j] = {};
      let id = 0;
      for (let i = 0; i < X.length; i++) {
        const val = String(X[i][j]);
        if (this.classes[j][val] === undefined) {
          this.classes[j][val] = id++;
        }
      }
    }
  }

  transform(X: any[][]): number[][] {
    return X.map(row => 
      row.map((val, j) => {
         const strVal = String(val);
         return this.classes[j] && this.classes[j][strVal] !== undefined 
            ? this.classes[j][strVal] 
            : 0;
      })
    );
  }
}

export class OneHotEncoderEngine {
  classes: Record<number, string[]> = {};
  dropFirst: boolean;

  constructor(dropFirst = false) {
    this.dropFirst = dropFirst;
  }
  
  fit(X: any[][]) {
    if (X.length === 0) return;
    const cols = X[0].length;
    for (let j = 0; j < cols; j++) {
      const unique = new Set<string>();
      for (let i = 0; i < X.length; i++) {
        unique.add(String(X[i][j]));
      }
      let arr = Array.from(unique).sort();
      if (this.dropFirst && arr.length > 0) arr = arr.slice(1);
      this.classes[j] = arr;
    }
  }

  transform(X: any[][]): number[][] {
    return X.map(row => {
      const newRow: number[] = [];
      row.forEach((val, j) => {
        const strVal = String(val);
        const categories = this.classes[j] || [];
        categories.forEach(cat => {
          newRow.push(strVal === cat ? 1 : 0);
        });
      });
      return newRow;
    });
  }
}

export class DropColumnsEngine {
  fit() {}
  transform(X: any[][]) {
    return X.map(() => []);
  }
}

export class PolynomialFeaturesEngine {
  degree: number;
  interactionOnly: boolean;
  
  constructor(degree = 2, interactionOnly = false) {
    this.degree = degree;
    this.interactionOnly = interactionOnly;
  }
  
  fit() {}
  
  transform(X: any[][]) {
    return X.map(row => {
      const numRow = row.map(v => Number(v) || 0);
      const out: number[] = [];
      const n = numRow.length;
      
      if (this.degree >= 2) {
        for (let i = 0; i < n; i++) {
          for (let j = i; j < n; j++) {
            if (this.interactionOnly && i === j) continue;
            out.push(numRow[i] * numRow[j]);
          }
        }
      }
      return out;
    });
  }
}

export class SelectKBestEngine {
  k: number;
  scores: number[] = [];
  selectedIndices: number[] = [];
  
  constructor(k = 10) {
    this.k = k;
  }
  
  fit(X: any[][], y?: any[]) {
    if (X.length === 0 || !y || y.length === 0) return;
    const nFeatures = X[0].length;
    this.scores = Array(nFeatures).fill(0);
    
    // Very simple variance-based scoring as a basic proxy
    for (let j = 0; j < nFeatures; j++) {
      let sum = 0;
      let sumSq = 0;
      for (let i = 0; i < X.length; i++) {
        const val = Number(X[i][j]) || 0;
        sum += val;
        sumSq += val * val;
      }
      const mean = sum / X.length;
      const variance = (sumSq / X.length) - (mean * mean);
      this.scores[j] = variance;
    }
    
    const indexedScores = this.scores.map((score, idx) => ({ score, idx }));
    indexedScores.sort((a, b) => b.score - a.score);
    this.selectedIndices = indexedScores.slice(0, this.k).map(item => item.idx);
    this.selectedIndices.sort((a, b) => a - b);
  }
  
  transform(X: any[][]) {
    if (this.selectedIndices.length === 0) return X;
    return X.map(row => this.selectedIndices.map(idx => Number(row[idx]) || 0));
  }
}

export class PCAEngine {
  nComponents: number;
  whiten: boolean;
  means: number[] = [];
  components: number[][] = [];
  explainedVariance: number[] = [];

  constructor(nComponents = 2, whiten = false) {
    this.nComponents = nComponents;
    this.whiten = whiten;
  }

  fit(X: number[][]) {
    if (X.length === 0) return;
    const n = X.length;
    const m = X[0].length;
    this.nComponents = Math.min(this.nComponents, m);

    // Compute means and center data
    this.means = [];
    for (let j = 0; j < m; j++) {
      this.means.push(mean(X.map(r => r[j])));
    }

    const X_centered = X.map(row => row.map((v, j) => v - this.means[j]));

    // Covariance matrix
    const cov: number[][] = Array(m).fill(0).map(() => Array(m).fill(0));
    for (let i = 0; i < m; i++) {
      for (let j = i; j < m; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += X_centered[k][i] * X_centered[k][j];
        }
        const val = sum / (n - 1 || 1);
        cov[i][j] = val;
        cov[j][i] = val;
      }
    }

    // Power iteration with deflation
    this.components = [];
    for (let c = 0; c < this.nComponents; c++) {
      let v = Array(m).fill(0).map(() => Math.random() - 0.5);
      let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      v = v.map(x => x / (norm || 1));

      for (let iter = 0; iter < 80; iter++) {
        for (const prev of this.components) {
          const dot = v.reduce((s, val, idx) => s + val * prev[idx], 0);
          v = v.map((val, idx) => val - dot * prev[idx]);
        }
        norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        v = v.map(x => x / (norm || 1));

        const w = Array(m).fill(0);
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < m; j++) {
            w[i] += cov[i][j] * v[j];
          }
        }

        const wNorm = Math.sqrt(w.reduce((s, x) => s + x * x, 0));
        v = w.map(x => x / (wNorm || 1));
      }

      this.components.push(v);
      this.explainedVariance.push(norm);
    }
  }

  transform(X: number[][]): number[][] {
    const X_centered = X.map(row => row.map((v, j) => v - (this.means[j] ?? 0)));
    return X_centered.map(row => {
      return this.components.map(comp => {
        return row.reduce((sum, v, idx) => sum + v * comp[idx], 0);
      });
    });
  }
}

// ─── JS Model Implementations ─────────────────────────────────────────────────

export class KNNEngine {
  X_train: number[][] = [];
  y_train: number[] = [];
  k: number;
  isRegression: boolean;

  constructor(k = 5, isRegression = false) {
    this.k = k;
    this.isRegression = isRegression;
  }

  fit(X: number[][], y: number[]) {
    this.X_train = X;
    this.y_train = y;
  }

  predict(X: number[][]): number[] {
    return X.map(row => {
      const dists = this.X_train.map((tr, i) => ({
        dist: Math.sqrt(tr.reduce((s, v, j) => s + (v - row[j]) ** 2, 0)),
        label: this.y_train[i],
      })).sort((a, b) => a.dist - b.dist).slice(0, this.k);

      if (this.isRegression) {
        return dists.reduce((s, d) => s + d.label, 0) / dists.length;
      }
      const votes: Record<number, number> = {};
      dists.forEach(d => { votes[d.label] = (votes[d.label] || 0) + 1; });
      return +Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
    });
  }
}

export class TFLinearModelEngine {
  model: tf.Sequential | null = null;
  lr: number;
  epochs: number;
  batchSize: number;
  optimizerName: string;
  regularization: string;
  lambda: number;
  isClassification: boolean;
  weights: number[] = [];
  bias: number = 0;

  constructor(
    lr = 0.01,
    epochs = 100,
    optimizer = 'adam',
    batchSize = 32,
    regularization = 'none',
    lambda = 0.001,
    isClassification = false
  ) {
    this.lr = lr;
    this.epochs = epochs;
    this.optimizerName = optimizer;
    this.batchSize = batchSize;
    this.regularization = regularization;
    this.lambda = lambda;
    this.isClassification = isClassification;
  }

  async fit(X: number[][], y: number[], onProgress?: (msg: string) => void) {
    if (X.length === 0) return;
    const numFeatures = X[0].length;
    
    this.model = tf.sequential();
    
    let kernelRegularizer = undefined;
    if (this.regularization === 'l1') kernelRegularizer = tf.regularizers.l1({ l1: this.lambda });
    else if (this.regularization === 'l2') kernelRegularizer = tf.regularizers.l2({ l2: this.lambda });
    else if (this.regularization === 'l1l2') kernelRegularizer = tf.regularizers.l1l2({ l1: this.lambda, l2: this.lambda });

    this.model.add(tf.layers.dense({
      units: 1,
      inputShape: [numFeatures],
      activation: this.isClassification ? 'sigmoid' : 'linear',
      kernelRegularizer
    }));

    let optimizer: tf.Optimizer;
    if (this.optimizerName === 'sgd') optimizer = tf.train.sgd(this.lr);
    else if (this.optimizerName === 'rmsprop') optimizer = tf.train.rmsprop(this.lr);
    else optimizer = tf.train.adam(this.lr);

    this.model.compile({
      optimizer,
      loss: this.isClassification ? 'binaryCrossentropy' : 'meanSquaredError',
      metrics: this.isClassification ? ['accuracy'] : ['mse']
    });

    const xTensor = tf.tensor2d(X);
    const yTensor = tf.tensor2d(y, [y.length, 1]);

    await this.model.fit(xTensor, yTensor, {
      epochs: this.epochs,
      batchSize: this.batchSize,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (onProgress && epoch % Math.ceil(this.epochs / 10) === 0) {
            onProgress(`Epoch ${epoch + 1}/${this.epochs} - loss: ${logs?.loss.toFixed(4)}`);
          }
        }
      }
    });

    const layer = this.model.layers[0];
    const weightsTensor = layer.getWeights();
    if (weightsTensor.length > 0) {
      this.weights = Array.from(weightsTensor[0].dataSync());
      if (weightsTensor.length > 1) {
        this.bias = weightsTensor[1].dataSync()[0];
      }
    }

    xTensor.dispose();
    yTensor.dispose();
  }

  predict(X: number[][]): number[] {
    if (!this.model) {
      return X.map(row => {
        const score = row.reduce((s, v, j) => s + v * (this.weights[j] || 0), this.bias);
        if (this.isClassification) return score >= 0 ? 1 : 0;
        return score;
      });
    }
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      const arr = Array.from(preds.dataSync());
      if (this.isClassification) return arr.map(v => v >= 0.5 ? 1 : 0);
      return arr;
    });
  }

  predictProba(X: number[][]): number[] {
    if (!this.model) return X.map(() => 0.5);
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      return Array.from(preds.dataSync());
    });
  }
}

export class SVMEngine {
  C: number;
  weights: number[] = [];
  bias = 0;

  constructor(C = 1) {
    this.C = C;
  }

  fit(X: number[][], y: number[]) {
    const n = X.length;
    if (n === 0) return;
    const m = X[0].length;
    this.weights = Array(m).fill(0);
    this.bias = 0;

    // Convert classification targets to -1 and 1
    const y_svm = y.map(v => v === 0 ? -1 : v);
    const epochs = 100;
    const lambda = 1 / (this.C * n);
    let t = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < n; i++) {
        t++;
        const eta = 1 / (lambda * t);
        const score = X[i].reduce((sum, v, j) => sum + v * this.weights[j], this.bias);
        if (y_svm[i] * score < 1) {
          for (let j = 0; j < m; j++) {
            this.weights[j] = (1 - eta * lambda) * this.weights[j] + eta * y_svm[i] * X[i][j];
          }
          this.bias += eta * y_svm[i];
        } else {
          for (let j = 0; j < m; j++) {
            this.weights[j] = (1 - eta * lambda) * this.weights[j];
          }
        }
      }
    }
  }

  predict(X: number[][]): number[] {
    return X.map(row => {
      const score = row.reduce((sum, v, j) => sum + v * this.weights[j], this.bias);
      return score >= 0 ? 1 : 0;
    });
  }
}

export class DecisionTreeEngine {
  maxDepth: number;
  minSamplesSplit: number;
  isRegression: boolean;
  criterion: string;
  root: any = null;

  constructor(maxDepth = 5, minSamplesSplit = 2, isRegression = false, criterion = 'gini') {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.isRegression = isRegression;
    this.criterion = criterion;
  }

  fit(X: number[][], y: number[]) {
    this.root = this.buildTree(X, y, 0);
  }

  private buildTree(X: number[][], y: number[], depth: number): any {
    const numSamples = X.length;
    if (numSamples === 0) return null;
    const numFeatures = X[0].length;

    const isConstant = y.every(val => val === y[0]);
    if (isConstant || depth >= this.maxDepth || numSamples < this.minSamplesSplit) {
      return {
        isLeaf: true,
        value: this.isRegression ? mean(y) : this.majorityVote(y),
        samples: numSamples,
      };
    }

    let bestGain = -1;
    let bestCriteria: { featIdx: number; threshold: number } | null = null;
    let bestSplits: { leftIdxs: number[]; rightIdxs: number[] } | null = null;

    for (let featIdx = 0; featIdx < numFeatures; featIdx++) {
      const featVals = X.map(r => r[featIdx]);
      const thresholds = [...new Set(featVals)].sort((a, b) => a - b);
      const candidates = thresholds.length > 15
        ? Array.from({ length: 15 }, (_, i) => thresholds[Math.floor(i * thresholds.length / 15)])
        : thresholds;

      for (const th of candidates) {
        const leftIdxs: number[] = [];
        const rightIdxs: number[] = [];
        for (let i = 0; i < numSamples; i++) {
          if (featVals[i] <= th) leftIdxs.push(i);
          else rightIdxs.push(i);
        }

        if (leftIdxs.length === 0 || rightIdxs.length === 0) continue;

        const gain = this.calculateGain(y, leftIdxs, rightIdxs);
        if (gain > bestGain) {
          bestGain = gain;
          bestCriteria = { featIdx, threshold: th };
          bestSplits = { leftIdxs, rightIdxs };
        }
      }
    }

    if (bestGain <= 0 || !bestCriteria || !bestSplits) {
      return {
        isLeaf: true,
        value: this.isRegression ? mean(y) : this.majorityVote(y),
        samples: numSamples,
      };
    }

    const leftX = bestSplits.leftIdxs.map(i => X[i]);
    const leftY = bestSplits.leftIdxs.map(i => y[i]);
    const rightX = bestSplits.rightIdxs.map(i => X[i]);
    const rightY = bestSplits.rightIdxs.map(i => y[i]);

    return {
      isLeaf: false,
      featIdx: bestCriteria.featIdx,
      threshold: bestCriteria.threshold,
      samples: numSamples,
      left: this.buildTree(leftX, leftY, depth + 1),
      right: this.buildTree(rightX, rightY, depth + 1),
    };
  }

  private calculateGain(y: number[], leftIdxs: number[], rightIdxs: number[]): number {
    const parentImp = this.calculateImpurity(y, Array.from({ length: y.length }, (_, i) => i));
    const leftImp = this.calculateImpurity(y, leftIdxs);
    const rightImp = this.calculateImpurity(y, rightIdxs);
    const n = y.length;
    return parentImp - (leftIdxs.length / n) * leftImp - (rightIdxs.length / n) * rightImp;
  }

  private calculateImpurity(y: number[], idxs: number[]): number {
    const n = idxs.length;
    if (n === 0) return 0;
    const subset = idxs.map(i => y[i]);

    if (this.isRegression) {
      const avg = mean(subset);
      return subset.reduce((acc, v) => acc + (v - avg) ** 2, 0) / n;
    }

    const counts: Record<number, number> = {};
    subset.forEach(val => { counts[val] = (counts[val] || 0) + 1; });

    if (this.criterion === 'gini') {
      let sumSq = 0;
      for (const c of Object.values(counts)) {
        const p = c / n;
        sumSq += p * p;
      }
      return 1 - sumSq;
    } else {
      let ent = 0;
      for (const c of Object.values(counts)) {
        const p = c / n;
        ent -= p * Math.log2(p);
      }
      return ent;
    }
  }

  private majorityVote(y: number[]): number {
    const counts: Record<number, number> = {};
    y.forEach(val => { counts[val] = (counts[val] || 0) + 1; });
    return +Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  }

  predict(X: number[][]): number[] {
    if (!this.root) return Array(X.length).fill(0);
    return X.map(row => this.predictRow(row, this.root));
  }

  private predictRow(row: number[], node: any): number {
    if (node.isLeaf) return node.value;
    if (row[node.featIdx] <= node.threshold) {
      return this.predictRow(row, node.left);
    } else {
      return this.predictRow(row, node.right);
    }
  }
}

export class RandomForestEngine {
  nEstimators: number;
  maxDepth: number;
  isRegression: boolean;
  trees: DecisionTreeEngine[] = [];

  constructor(nEstimators = 20, maxDepth = 5, isRegression = false) {
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.isRegression = isRegression;
  }

  fit(X: number[][], y: number[]) {
    this.trees = [];
    const n = X.length;
    if (n === 0) return;
    for (let i = 0; i < this.nEstimators; i++) {
      const bX: number[][] = [];
      const bY: number[] = [];
      for (let j = 0; j < n; j++) {
        const idx = Math.floor(Math.random() * n);
        bX.push(X[idx]);
        bY.push(y[idx]);
      }
      const tree = new DecisionTreeEngine(this.maxDepth, 2, this.isRegression);
      tree.fit(bX, bY);
      this.trees.push(tree);
    }
  }

  predict(X: number[][]): number[] {
    if (this.trees.length === 0) return Array(X.length).fill(0);
    const allPreds = this.trees.map(t => t.predict(X));
    return X.map((_, idx) => {
      const rowPreds = allPreds.map(preds => preds[idx] ?? 0);
      if (this.isRegression) {
        return mean(rowPreds);
      } else {
        const counts: Record<number, number> = {};
        rowPreds.forEach(p => { counts[p] = (counts[p] || 0) + 1; });
        return +Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
      }
    });
  }
}

export class GaussianNaiveBayesEngine {
  classes: number[] = [];
  priors: Record<number, number> = {};
  means: Record<number, number[]> = {};
  vars: Record<number, number[]> = {};

  fit(X: number[][], y: number[]) {
    this.classes = [...new Set(y)].sort((a, b) => a - b);
    const n = X.length;
    if (n === 0) return;

    this.classes.forEach(c => {
      const classRows = X.filter((_, i) => y[i] === c);
      const nc = classRows.length;
      this.priors[c] = nc / n;

      const numFeatures = X[0].length;
      const cMeans = Array(numFeatures).fill(0);
      const cVars = Array(numFeatures).fill(0);

      for (let j = 0; j < numFeatures; j++) {
        const col = classRows.map(r => r[j]);
        const m = mean(col);
        cMeans[j] = m;
        cVars[j] = std(col, m) ** 2 + 1e-9;
      }
      this.means[c] = cMeans;
      this.vars[c] = cVars;
    });
  }

  predict(X: number[][]): number[] {
    if (this.classes.length === 0) return Array(X.length).fill(0);
    return X.map(row => {
      let bestClass = this.classes[0];
      let maxLogProb = -Infinity;

      this.classes.forEach(c => {
        let logProb = Math.log(this.priors[c] || 1e-5);
        const cMeans = this.means[c] ?? [];
        const cVars = this.vars[c] ?? [];

        for (let j = 0; j < row.length; j++) {
          const x = row[j];
          const m = cMeans[j] ?? 0;
          const v = cVars[j] ?? 1e-9;
          const lp = -0.5 * Math.log(2 * Math.PI * v) - 0.5 * ((x - m) ** 2) / v;
          logProb += lp;
        }

        if (logProb > maxLogProb) {
          maxLogProb = logProb;
          bestClass = c;
        }
      });

      return bestClass;
    });
  }
}

export class TFMLPEngine {
  model: tf.Sequential | null = null;
  hiddenLayers: number[] = [];
  learningRate: number;
  epochs: number;
  optimizerName: string;
  batchSize: number;
  classes: number[] = [];
  lossHistory: number[] = [];
  weights: number[][][] = [];
  biases: number[][] = [];

  constructor(hiddenLayersStr = '8,4', learningRate = 0.05, epochs = 100, optimizer = 'adam', batchSize = 32) {
    this.hiddenLayers = hiddenLayersStr.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n) && n > 0);
    if (this.hiddenLayers.length === 0) this.hiddenLayers = [8, 4];
    this.learningRate = learningRate;
    this.epochs = epochs;
    this.optimizerName = optimizer;
    this.batchSize = batchSize;
  }

  async fit(X: number[][], y: number[], onProgress?: (msg: string) => void) {
    if (X.length === 0) return;
    this.classes = [...new Set(y)].sort((a, b) => a - b);
    const numFeatures = X[0].length;
    const isBinary = this.classes.length <= 2;
    const numOutputs = isBinary ? 1 : this.classes.length;

    this.model = tf.sequential();
    
    for (let i = 0; i < this.hiddenLayers.length; i++) {
      this.model.add(tf.layers.dense({
        units: this.hiddenLayers[i],
        inputShape: i === 0 ? [numFeatures] : undefined,
        activation: 'relu'
      }));
    }

    this.model.add(tf.layers.dense({
      units: numOutputs,
      activation: isBinary ? 'sigmoid' : 'softmax'
    }));

    let optimizer: tf.Optimizer;
    if (this.optimizerName === 'sgd') optimizer = tf.train.sgd(this.learningRate);
    else if (this.optimizerName === 'rmsprop') optimizer = tf.train.rmsprop(this.learningRate);
    else optimizer = tf.train.adam(this.learningRate);

    this.model.compile({
      optimizer,
      loss: isBinary ? 'binaryCrossentropy' : 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });

    const xTensor = tf.tensor2d(X);
    const yTensor = isBinary ? tf.tensor2d(y, [y.length, 1]) : tf.tensor1d(y, 'int32');

    this.lossHistory = [];

    await this.model.fit(xTensor, yTensor, {
      epochs: this.epochs,
      batchSize: this.batchSize,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (logs) this.lossHistory.push(logs.loss);
          if (onProgress && epoch % Math.ceil(this.epochs / 10) === 0) {
            onProgress(`Epoch ${epoch + 1}/${this.epochs} - loss: ${logs?.loss.toFixed(4)}`);
          }
        }
      }
    });

    xTensor.dispose();
    yTensor.dispose();
  }

  predict(X: number[][]): number[] {
    if (!this.model) return Array(X.length).fill(0);
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      const isBinary = this.classes.length <= 2;
      
      if (isBinary) {
        return Array.from(preds.dataSync()).map(v => v >= 0.5 ? 1 : 0);
      } else {
        const argmax = preds.argMax(-1);
        const indices = Array.from(argmax.dataSync());
        return indices.map(idx => this.classes[idx] ?? idx);
      }
    });
  }

  predictProba(X: number[][]): number[][] | number[] {
    if (!this.model) return [];
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      const isBinary = this.classes.length <= 2;
      if (isBinary) return Array.from(preds.dataSync());
      
      const arr = Array.from(preds.dataSync());
      const res: number[][] = [];
      const numClasses = this.classes.length;
      for (let i = 0; i < arr.length; i += numClasses) {
        res.push(arr.slice(i, i + numClasses));
      }
      return res;
    });
  }
}

export class KMeansEngine {
  nClusters: number = 3;
  maxIterations: number = 50;
  centroids: number[][] = [];

  constructor(nClusters: number = 3, maxIterations: number = 50) {
    this.nClusters = nClusters;
    this.maxIterations = maxIterations;
  }

  fit(X: number[][], _y?: number[]) {
    const n = X.length;
    if (n === 0) return;
    const d = X[0].length;

    // Pick K starting centroids randomly from data points
    this.centroids = [];
    const usedIndices = new Set<number>();
    while (this.centroids.length < Math.min(this.nClusters, n)) {
      const idx = Math.floor(Math.random() * n);
      if (!usedIndices.has(idx)) {
        usedIndices.add(idx);
        this.centroids.push([...X[idx]]);
      }
    }

    let iterations = 0;
    let converged = false;

    while (iterations < this.maxIterations && !converged) {
      const clusters: number[][] = Array.from({ length: this.centroids.length }, () => []);
      
      // 1. Assign points to nearest centroid
      for (let i = 0; i < n; i++) {
        const row = X[i];
        let bestIdx = 0;
        let minDist = Infinity;

        for (let c = 0; c < this.centroids.length; c++) {
          const centroid = this.centroids[c];
          let dist = 0;
          for (let j = 0; j < d; j++) {
            dist += (row[j] - centroid[j]) ** 2;
          }
          if (dist < minDist) {
            minDist = dist;
            bestIdx = c;
          }
        }
        clusters[bestIdx].push(i);
      }

      // 2. Update centroids
      let maxShift = 0;
      const nextCentroids: number[][] = [];

      for (let c = 0; c < this.centroids.length; c++) {
        const indices = clusters[c];
        if (indices.length === 0) {
          // Keep centroid unchanged if no points assigned
          nextCentroids.push([...this.centroids[c]]);
          continue;
        }

        const newCentroid = Array(d).fill(0);
        for (const idx of indices) {
          const row = X[idx];
          for (let j = 0; j < d; j++) {
            newCentroid[j] += row[j];
          }
        }

        for (let j = 0; j < d; j++) {
          newCentroid[j] /= indices.length;
        }

        // Shift distance calculation
        let shift = 0;
        for (let j = 0; j < d; j++) {
          shift += (newCentroid[j] - this.centroids[c][j]) ** 2;
        }
        maxShift = Math.max(maxShift, shift);
        nextCentroids.push(newCentroid);
      }

      this.centroids = nextCentroids;
      converged = maxShift < 1e-6;
      iterations++;
    }
  }

  predict(X: number[][]): number[] {
    if (this.centroids.length === 0) return Array(X.length).fill(0);

    return X.map(row => {
      let bestIdx = 0;
      let minDist = Infinity;

      for (let c = 0; c < this.centroids.length; c++) {
        const centroid = this.centroids[c];
        let dist = 0;
        for (let j = 0; j < row.length; j++) {
          dist += (row[j] - centroid[j]) ** 2;
        }
        if (dist < minDist) {
          minDist = dist;
          bestIdx = c;
        }
      }
      return bestIdx;
    });
  }
}

// ─── Metrics Computations ─────────────────────────────────────────────────────

export function computeClassificationMetrics(y_true: number[], y_pred: number[]) {
  const n = y_true.length;
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < n; i++) {
    if (y_true[i] === 1 && y_pred[i] === 1) tp++;
    else if (y_true[i] === 0 && y_pred[i] === 1) fp++;
    else if (y_true[i] === 1 && y_pred[i] === 0) fn++;
    else tn++;
  }
  const accuracy = (tp + tn) / (n || 1);
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
  return { accuracy, precision, recall, f1, tp, fp, fn, tn };
}

export function computeRegressionMetrics(y_true: number[], y_pred: number[]) {
  const n = y_true.length;
  if (n === 0) return { mse: 0, rmse: 0, mae: 0, r2: 0 };
  const mse = y_true.reduce((s, v, i) => s + (v - y_pred[i]) ** 2, 0) / n;
  const mae = y_true.reduce((s, v, i) => s + Math.abs(v - y_pred[i]), 0) / n;
  const meanY = mean(y_true);
  const ss_tot = y_true.reduce((s, v) => s + (v - meanY) ** 2, 0);
  const ss_res = y_true.reduce((s, v, i) => s + (v - y_pred[i]) ** 2, 0);
  const r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : 0;
  return { mse, rmse: Math.sqrt(mse), mae, r2 };
}

// ─── Main execution pass ──────────────────────────────────────────────────────

// ─── Main execution pass (DataFrame-based) ────────────────────────────────────

export async function executePipeline(
  pipeline: Pipeline,
  onNodeStart: (id: string) => void,
  onNodeDone: (id: string, result: NodeExecutionResult) => void,
  onProgress: (msg: string) => void,
): Promise<{ results: Record<string, NodeExecutionResult>; fittedEngines: Record<string, any>; featureNames: string[] }> {
  const { nodes, edges } = pipeline;
  const order = topologicalSort(nodes, edges);
  if (!order) throw new Error('Pipeline has a cycle');

  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const results: Record<string, NodeExecutionResult> = {};
  const fittedEngines: Record<string, any> = {};

  // DataFrame flow: each node's output is stored here
  const nodeOutputs: Record<string, { train: DataFrame, test: DataFrame }> = {};

  const targetCol = pipeline.schema?.target ?? '';
  const trainDF: DataFrame = pipeline.data?.train ?? { columns: [], rows: [] };
  const testDF: DataFrame = pipeline.data?.test ?? { columns: [], rows: [] };

  for (const nodeId of order) {
    const node = nodeMap.get(nodeId)!;
    onNodeStart(nodeId);
    onProgress(`Processing: ${node.label}…`);
    await sleep(60);

    // Get input DataFrames — from parent node's output, or from datasource
    let inputTrain: DataFrame;
    let inputTest: DataFrame;

    if (node.family === 'datasource') {
      inputTrain = trainDF;
      inputTest = testDF;
    } else {
      const parentEdges = edges.filter(e => e.target === nodeId);
      const parentIds = parentEdges.map(e => e.source);
      if (parentIds.length > 0 && nodeOutputs[parentIds[0]]) {
        inputTrain = nodeOutputs[parentIds[0]].train;
        inputTest = nodeOutputs[parentIds[0]].test;
      } else {
        inputTrain = { columns: [], rows: [] };
        inputTest = { columns: [], rows: [] };
      }
    }

    const inputSnapshot: DataSnapshot = {
      shape: [inputTrain.rows.length, inputTrain.columns.length],
      sample: inputTrain.rows.slice(0, 5),
      featureNames: [...inputTrain.columns],
    };

    try {
      const cfg = node.config as any;
      let outputTrain: DataFrame = inputTrain;
      let outputTest: DataFrame = inputTest;
      let result: NodeExecutionResult = { nodeId, success: true };

      // ─── DATASOURCE ────────────────────────────────────────────────
      if (node.family === 'datasource') {
        outputTrain = inputTrain;
        outputTest = inputTest;
      }

      // ─── PREPROCESSOR ──────────────────────────────────────────────
      else if (node.family === 'preprocessor') {
        const preprocessResult = executePreprocessor(node, inputTrain, inputTest, targetCol);
        outputTrain = preprocessResult.train;
        outputTest = preprocessResult.test;
        result.fittedParams = preprocessResult.fittedParams;
        if (preprocessResult.engine) {
          fittedEngines[nodeId] = preprocessResult.engine;
        }
      }

      // ─── MODEL ─────────────────────────────────────────────────────
      else if (node.family === 'model') {
        const modelResult = await executeModel(node, inputTrain, inputTest, targetCol, cfg, onProgress);
        result.fittedParams = modelResult.fitted;
        result.predictions = modelResult.predictions;
        if (modelResult.engine) {
          fittedEngines[nodeId] = modelResult.engine;
        }
        outputTrain = inputTrain;
        outputTest = inputTest;
      }

      // ─── EVALUATION ────────────────────────────────────────────────
      else if (node.family === 'evaluation') {
        const parentEdges = edges.filter(e => e.target === nodeId);
        const parentIds = parentEdges.map(e => e.source);
        const modelResult = parentIds.map(id => results[id]).find(r => r?.predictions && r.predictions.length > 0);
        const preds = modelResult?.predictions ?? [];
        const tIdx = inputTest.columns.indexOf(targetCol);
        const y_test = tIdx >= 0 ? inputTest.rows.map(r => Number(r[tIdx])) : [];
        const tIdxTrain = inputTrain.columns.indexOf(targetCol);
        const evalTarget = y_test.length > 0 ? y_test : (tIdxTrain >= 0 ? inputTrain.rows.map(r => Number(r[tIdxTrain])) : []);
        if (preds.length > 0 && evalTarget.length > 0) {
          const isReg = evalTarget.some(v => v !== 0 && v !== 1);
          const aligned = evalTarget.slice(0, preds.length);
          result.metrics = (isReg || node.type === 'RegressionMetrics')
            ? computeRegressionMetrics(aligned, preds)
            : computeClassificationMetrics(aligned, preds);
        } else {
          result.metrics = {};
        }
      }

      // ─── ENSEMBLE ──────────────────────────────────────────────────
      else if (node.family === 'ensemble') {
        const parentEdges = edges.filter(e => e.target === nodeId);
        const parentIds = parentEdges.map(e => e.source);
        const incomingPreds = parentIds
          .map(id => results[id])
          .filter(r => r?.predictions && r.predictions.length > 0)
          .map(r => r!.predictions!);
        if (incomingPreds.length > 0) {
          result.predictions = incomingPreds[0].map((_, i) =>
            Math.round(incomingPreds.reduce((s, p) => s + (p[i] ?? 0), 0) / incomingPreds.length)
          );
        }
      }

      // ─── OUTPUT ────────────────────────────────────────────────────
      else if (node.family === 'output') {
        if (node.type === 'CSVExport') {
          const parentEdges = edges.filter(e => e.target === nodeId);
          const parentIds = parentEdges.map(e => e.source);
          const preds = parentIds.map(id => results[id]).find(r => r?.predictions)?.predictions ?? [];
          if (preds.length > 0) {
            const csvStr = 'prediction\n' + preds.join('\n');
            const blob = new Blob([csvStr], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = cfg.filename ?? 'predictions.csv'; a.click();
            URL.revokeObjectURL(url);
          }
        }
      }

      // Store node output
      nodeOutputs[nodeId] = { train: outputTrain, test: outputTest };

      // Build output snapshot
      const outputSnapshot: DataSnapshot = (node.family === 'model')
        ? {
            shape: [result.predictions?.length ?? 0, 1],
            sample: result.predictions?.slice(0, 5).map(v => [v]) ?? [],
            featureNames: ['prediction'],
          }
        : {
            shape: [outputTrain.rows.length, outputTrain.columns.length],
            sample: outputTrain.rows.slice(0, 5),
            featureNames: [...outputTrain.columns],
          };

      const testInputSnapshot: DataSnapshot | undefined = (node.family === 'model')
        ? {
            shape: [inputTest.rows.length, inputTest.columns.length],
            sample: inputTest.rows.slice(0, 5),
            featureNames: [...inputTest.columns],
          }
        : undefined;

      results[nodeId] = { ...result, inputSnapshot, outputSnapshot, testInputSnapshot };
      onNodeDone(nodeId, results[nodeId]);
    } catch (e: any) {
      const errResult: NodeExecutionResult = { nodeId, success: false, error: e.message, inputSnapshot };
      results[nodeId] = errResult;
      onNodeDone(nodeId, errResult);
      throw e;
    }
  }

  const featureNames = pipeline.schema?.columns.map(c => c.name) ?? [];
  return { results, fittedEngines, featureNames };
}

// ─── Helper: Extract X and y from DataFrame ─────────────────────────────────

function dfToXY(df: DataFrame, targetCol: string): { X: number[][], y: number[] } {
  const targetIdx = df.columns.indexOf(targetCol);
  if (targetIdx === -1) return { X: df.rows.map(r => r.map(Number)), y: [] };
  const X = df.rows.map(row => row.filter((_, i) => i !== targetIdx).map(Number));
  const y = df.rows.map(row => Number(row[targetIdx]));
  return { X, y };
}

// ─── Preprocessor Execution (DataFrame → DataFrame) ─────────────────────────

function executePreprocessor(
  node: PipelineNode,
  trainDF: DataFrame,
  testDF: DataFrame,
  targetCol: string,
): { train: DataFrame; test: DataFrame; fittedParams?: Record<string, any>; engine?: any } {
  const cfg = node.config as any;
  
  let engine: any = null;
  if (node.type === 'StandardScaler') engine = new StandardScalerEngine();
  else if (node.type === 'MinMaxScaler') engine = new MinMaxScalerEngine();
  else if (node.type === 'SimpleImputer') engine = new SimpleImputerEngine(cfg.params?.strategy);
  else if (node.type === 'PCA') engine = new PCAEngine(cfg.params?.nComponents ?? 2, cfg.params?.whiten ?? false);
  else if (node.type === 'LabelEncoder') engine = new LabelEncoderEngine();
  else if (node.type === 'OneHotEncoder') engine = new OneHotEncoderEngine(cfg.params?.dropFirst ?? false);
  else if (node.type === 'DropColumns') engine = new DropColumnsEngine();
  else if (node.type === 'PolynomialFeatures') engine = new PolynomialFeaturesEngine(cfg.params?.degree ?? 2, cfg.params?.interactionOnly ?? false);
  else if (node.type === 'SelectKBest') engine = new SelectKBestEngine(cfg.params?.k ?? 10);

  if (!engine || trainDF.rows.length === 0) return { train: trainDF, test: testDF };

  // Determine which columns to transform
  const selectedCols: string[] = cfg.targetColumns || [];
  let colIndices: number[];

  if (selectedCols.length > 0) {
    colIndices = selectedCols
      .map((col: string) => trainDF.columns.indexOf(col))
      .filter((idx: number) => idx !== -1);
  } else {
    // Default: all columns except target
    colIndices = trainDF.columns
      .map((_, i) => i)
      .filter(i => trainDF.columns[i] !== targetCol);
  }

  if (colIndices.length === 0) return { train: trainDF, test: testDF };

  // DropColumns: remove columns entirely
  if (node.type === 'DropColumns') {
    const keepIndices = trainDF.columns.map((_, i) => i).filter(i => !colIndices.includes(i));
    const newCols = keepIndices.map(i => trainDF.columns[i]);
    return {
      train: { columns: newCols, rows: trainDF.rows.map(row => keepIndices.map(i => row[i])) },
      test: { columns: newCols, rows: testDF.rows.map(row => keepIndices.map(i => row[i])) },
      fittedParams: { dropped: colIndices.length },
      engine,
    };
  }

  // Extract sub-matrices for selected columns
  const extractCols = (rows: any[][]) => rows.map(row => colIndices.map(i => row[i]));
  const subTrain = extractCols(trainDF.rows);
  const subTest = testDF.rows.length > 0 ? extractCols(testDF.rows) : [];

  const { y } = dfToXY(trainDF, targetCol);
  engine.fit(subTrain, y);
  const newSubTrain = engine.transform(subTrain);
  const newSubTest = subTest.length > 0 ? engine.transform(subTest) : [];

  const sameSize = newSubTrain[0]?.length === colIndices.length;
  let newColumns: string[];
  let stitchTrain: any[][];
  let stitchTest: any[][];

  if (sameSize) {
    newColumns = [...trainDF.columns];
    stitchTrain = trainDF.rows.map((row, r) => {
      const newRow = [...row];
      for (let i = 0; i < colIndices.length; i++) newRow[colIndices[i]] = newSubTrain[r][i];
      return newRow;
    });
    stitchTest = testDF.rows.map((row, r) => {
      const newRow = [...row];
      if (newSubTest[r]) {
        for (let i = 0; i < colIndices.length; i++) newRow[colIndices[i]] = newSubTest[r][i];
      }
      return newRow;
    });
  } else {
    const keepIndices = trainDF.columns.map((_, i) => i).filter(i => !colIndices.includes(i));
    const keptNames = keepIndices.map(i => trainDF.columns[i]);
    const newNames = Array.from({ length: newSubTrain[0]?.length ?? 0 }, (_, i) => {
      if (node.type === 'PCA') return `PC${i + 1}`;
      if (node.type === 'OneHotEncoder') return `OHE_${i}`;
      if (node.type === 'PolynomialFeatures') return `Poly_${i}`;
      if (node.type === 'SelectKBest') return `KBest_${i}`;
      return `${node.type}_${i}`;
    });
    newColumns = [...keptNames, ...newNames];
    stitchTrain = trainDF.rows.map((row, r) => [...keepIndices.map(i => row[i]), ...newSubTrain[r]]);
    stitchTest = testDF.rows.map((row, r) => [...keepIndices.map(i => row[i]), ...(newSubTest[r] ?? [])]);
  }

  let fittedParams: Record<string, any> = {};
  if (node.type === 'StandardScaler') fittedParams = { means: engine.means, stds: engine.stds };
  else if (node.type === 'MinMaxScaler') fittedParams = { mins: engine.mins, maxs: engine.maxs };
  else if (node.type === 'SimpleImputer') fittedParams = { values: engine.values, strategy: engine.strategy };
  else if (node.type === 'PCA') fittedParams = { explainedVariance: engine.explainedVariance, components: engine.components };
  else if (node.type === 'LabelEncoder' || node.type === 'OneHotEncoder') fittedParams = { classes: engine.classes };
  else if (node.type === 'SelectKBest') fittedParams = { scores: engine.scores, selectedIndices: engine.selectedIndices };

  return {
    train: { columns: newColumns, rows: stitchTrain },
    test: { columns: newColumns, rows: stitchTest },
    fittedParams,
    engine,
  };
}

// ─── Model Execution (DataFrame → predictions) ──────────────────────────────

async function executeModel(
  node: PipelineNode,
  trainDF: DataFrame,
  testDF: DataFrame,
  targetCol: string,
  cfg: any,
  onProgress?: (msg: string) => void
): Promise<{ predictions: number[]; fitted: Record<string, unknown>; engine: any }> {
  const { X: X_train, y: y_train } = dfToXY(trainDF, targetCol);
  const { X: X_test } = dfToXY(testDF, targetCol);

  if (X_train.length === 0) {
    throw new Error('No training data — upload a CSV and configure the Data Source node target column');
  }

  const isRegression = node.type === 'LinearRegression' || !y_train.every(v => v === 0 || v === 1);
  let fittedEngine: any = null;

  if (node.type === 'KNN') fittedEngine = new KNNEngine(cfg.params?.k ?? 5, isRegression);
  else if (node.type === 'DecisionTree') fittedEngine = new DecisionTreeEngine(cfg.params?.maxDepth ?? 5, cfg.params?.minSamplesSplit ?? 2, isRegression, cfg.params?.criterion ?? 'gini');
  else if (node.type === 'RandomForest' || node.type === 'GradientBoosting') fittedEngine = new RandomForestEngine(cfg.params?.nEstimators ?? 20, cfg.params?.maxDepth ?? 5, isRegression);
  else if (node.type === 'NaiveBayes') fittedEngine = new GaussianNaiveBayesEngine();
  else if (node.type === 'MLPClassifier') fittedEngine = new MLPClassifierEngine(cfg.params?.hiddenLayers ?? '8,4', cfg.params?.learningRate ?? 0.05, cfg.params?.maxEpochs ?? 100);
  else if (node.type === 'KMeans') fittedEngine = new KMeansEngine(cfg.params?.nClusters ?? 3, cfg.params?.maxIterations ?? 50);
  else if (node.type === 'SVM') fittedEngine = new SVMEngine(cfg.params?.C ?? 1);
  else if (isRegression || node.type === 'LinearRegression' || node.type === 'RidgeRegression' || node.type === 'LassoRegression') fittedEngine = new LinearRegressionEngine(cfg.params?.learningRate ?? 0.01, 300);
  else fittedEngine = new LogisticRegressionEngine(cfg.params?.learningRate ?? 0.01, 300);

  // Load pretrained weights
  if (cfg.source === 'loaded' && cfg.fittedParams && cfg.mode !== 'retrain') {
    if (node.type === 'KNN' && cfg.fittedParams.k) fittedEngine.k = cfg.fittedParams.k;
    else if (node.type === 'DecisionTree' && cfg.fittedParams.root) fittedEngine.root = cfg.fittedParams.root;
    else if ((node.type === 'RandomForest' || node.type === 'GradientBoosting') && cfg.fittedParams.trees) fittedEngine.trees = cfg.fittedParams.trees;
    else if (node.type === 'NaiveBayes') { if (cfg.fittedParams.classes) fittedEngine.classes = cfg.fittedParams.classes; if (cfg.fittedParams.priors) fittedEngine.priors = cfg.fittedParams.priors; if (cfg.fittedParams.means) fittedEngine.means = cfg.fittedParams.means; }
    else if (node.type === 'MLPClassifier') { if (cfg.fittedParams.weights) fittedEngine.weights = cfg.fittedParams.weights; if (cfg.fittedParams.biases) fittedEngine.biases = cfg.fittedParams.biases; }
    else if (node.type === 'KMeans' && cfg.fittedParams.centroids) fittedEngine.centroids = cfg.fittedParams.centroids;
    else { if (cfg.fittedParams.weights) fittedEngine.weights = [...cfg.fittedParams.weights]; if (cfg.fittedParams.bias !== undefined) fittedEngine.bias = cfg.fittedParams.bias; }
  }

  if (cfg.source !== 'loaded' || cfg.mode === 'retrain' || cfg.mode === 'finetune') {
    await fittedEngine.fit(X_train, y_train, onProgress);
  }

  const predictions = fittedEngine.predict(X_test.length > 0 ? X_test : X_train);

  let fitted: Record<string, unknown> = {};
  if (node.type === 'KNN') fitted = { k: fittedEngine.k, isRegression: fittedEngine.isRegression };
  else if (node.type === 'DecisionTree') fitted = { maxDepth: fittedEngine.maxDepth, criterion: fittedEngine.criterion, root: fittedEngine.root };
  else if (node.type === 'RandomForest' || node.type === 'GradientBoosting') fitted = { nEstimators: fittedEngine.nEstimators, maxDepth: fittedEngine.maxDepth, trees: fittedEngine.trees, treeCount: fittedEngine.trees.length };
  else if (node.type === 'NaiveBayes') fitted = { classes: fittedEngine.classes, priors: fittedEngine.priors, means: fittedEngine.means };
  else if (node.type === 'MLPClassifier') fitted = { hiddenLayers: fittedEngine.hiddenLayers, learningRate: fittedEngine.learningRate, lossHistory: fittedEngine.lossHistory, weights: fittedEngine.weights, biases: fittedEngine.biases };
  else if (node.type === 'KMeans') fitted = { nClusters: fittedEngine.nClusters, centroids: fittedEngine.centroids };
  else if (node.type === 'SVM') fitted = { weights: fittedEngine.weights, bias: fittedEngine.bias };
  else fitted = { weights: fittedEngine.weights ?? null, bias: fittedEngine.bias ?? null };

  return { predictions, fitted, engine: fittedEngine };
}

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }

// ─── CSV Parsing ───────────────────────────────────────────────────────────────

export function parseCSV(text: string): { headers: string[]; rows: any[][]; rawRows: string[][] } {
  const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return { headers: [], rows: [], rawRows: [] };

  const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
  const rows: any[][] = [];
  const rawRows: string[][] = [];

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map(c => c.trim().replace(/^["']|["']$/g, ''));
    rawRows.push(cols);
    const parsed = cols.map(c => {
      const val = parseFloat(c);
      return isNaN(val) ? c : val;
    });
    rows.push(parsed);
  }

  return { headers, rows, rawRows };
}

export function trainTestSplit(X: any[][], y: any[], testSplit = 0.2) {
  const n = X.length;
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor((i + 1) * 0.582) % (i + 1);
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const splitIdx = Math.floor(n * (1 - testSplit));
  return {
    X_train: indices.slice(0, splitIdx).map(i => X[i]),
    X_test: indices.slice(splitIdx).map(i => X[i]),
    y_train: indices.slice(0, splitIdx).map(i => y[i]),
    y_test: indices.slice(splitIdx).map(i => y[i]),
  };
}
