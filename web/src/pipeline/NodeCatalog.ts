/**
 * Node Catalog — Focused pipeline node types with connection constraints.
 * 
 * Each entry defines:
 * - What families can connect TO this node (acceptsFrom)
 * - What families this node can connect TO (outputsTo)
 */

import type { NodeCatalogEntry, NodeFamily } from './PipelineTypes';
import { CONNECTION_RULES } from './PipelineTypes';

export const NODE_CATALOG: NodeCatalogEntry[] = [
  // ─── DATASET ───────────────────────────────────────────────────────────────
  {
    family: 'dataset',
    type: 'Dataset',
    label: 'Dataset',
    icon: 'dataset',
    description: 'Upload a CSV file and configure the target column. Multiple datasets allowed.',
    color: '#6366f1',
    defaultConfig: {
      targetColumn: '',
      taskType: 'auto',
    },
    acceptsFrom: [],
    outputsTo: ['split', 'preprocessor', 'model'],
    maxIncoming: 0,
  },

  // ─── SPLIT ─────────────────────────────────────────────────────────────────
  {
    family: 'split',
    type: 'TrainTestSplit',
    label: 'Train/Test Split',
    icon: 'call_split',
    description: 'Split data into training and testing subsets with configurable ratio.',
    color: '#818cf8',
    defaultConfig: {
      testSplit: 0.2,
      stratified: true,
      shuffle: true,
    },
    acceptsFrom: ['dataset'],
    outputsTo: ['preprocessor', 'model'],
    maxIncoming: 1,
  },

  // ─── PREPROCESSORS ─────────────────────────────────────────────────────────
  {
    family: 'preprocessor',
    type: 'StandardScaler',
    label: 'Standard Scaler',
    icon: 'tune',
    description: 'Zero mean, unit variance normalization (z-score).',
    color: '#10b981',
    defaultConfig: { type: 'StandardScaler', params: {} },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['preprocessor', 'model'],
    maxIncoming: 1,
  },
  {
    family: 'preprocessor',
    type: 'MinMaxScaler',
    label: 'MinMax Scaler',
    icon: 'compress',
    description: 'Scale features to [0, 1] range.',
    color: '#10b981',
    defaultConfig: { type: 'MinMaxScaler', params: {} },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['preprocessor', 'model'],
    maxIncoming: 1,
  },
  {
    family: 'preprocessor',
    type: 'SimpleImputer',
    label: 'Imputer',
    icon: 'healing',
    description: 'Fill missing values with mean, median, or mode.',
    color: '#10b981',
    defaultConfig: { type: 'SimpleImputer', params: { strategy: 'mean' } },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['preprocessor', 'model'],
    maxIncoming: 1,
  },
  {
    family: 'preprocessor',
    type: 'LabelEncoder',
    label: 'Label Encoder',
    icon: 'font_download',
    description: 'Encode categorical strings into sequential integers.',
    color: '#10b981',
    defaultConfig: { type: 'LabelEncoder', params: {} },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['preprocessor', 'model'],
    maxIncoming: 1,
  },
  {
    family: 'preprocessor',
    type: 'OneHotEncoder',
    label: 'One-Hot Encoder',
    icon: 'grid_on',
    description: 'Encode categorical features as binary columns.',
    color: '#10b981',
    defaultConfig: { type: 'OneHotEncoder', params: { dropFirst: false } },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['preprocessor', 'model'],
    maxIncoming: 1,
  },
  {
    family: 'preprocessor',
    type: 'DropColumns',
    label: 'Drop Columns',
    icon: 'delete_sweep',
    description: 'Remove specific feature columns from the dataset flow.',
    color: '#10b981',
    defaultConfig: { type: 'DropColumns', params: {}, targetColumns: [] },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['preprocessor', 'model'],
    maxIncoming: 1,
  },


  // ─── MODELS ────────────────────────────────────────────────────────────────
  {
    family: 'model',
    type: 'LinearRegression',
    label: 'Linear Regression',
    icon: 'trending_up',
    description: 'OLS linear model — powered by TensorFlow.js.',
    color: '#f59e0b',
    defaultConfig: {
      type: 'LinearRegression',
      params: { learningRate: 0.01, epochs: 100, optimizer: 'adam', batchSize: 32 },
    },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['evaluation', 'output'],
    maxIncoming: 1,
  },
  {
    family: 'model',
    type: 'LogisticRegression',
    label: 'Logistic Regression',
    icon: 'show_chart',
    description: 'Binary/multiclass classifier — powered by TensorFlow.js.',
    color: '#f59e0b',
    defaultConfig: {
      type: 'LogisticRegression',
      params: { learningRate: 0.01, epochs: 100, optimizer: 'adam', batchSize: 32 },
    },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['evaluation', 'output'],
    maxIncoming: 1,
  },
  {
    family: 'model',
    type: 'MLPClassifier',
    label: 'MLP Neural Network',
    icon: 'hub',
    description: 'Multi-Layer Perceptron with backpropagation — powered by TensorFlow.js.',
    color: '#f59e0b',
    defaultConfig: {
      type: 'MLPClassifier',
      params: { hiddenLayers: '8,4', learningRate: 0.05, epochs: 100, optimizer: 'adam', batchSize: 32 },
    },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['evaluation', 'output'],
    maxIncoming: 1,
  },
  {
    family: 'model',
    type: 'KNN',
    label: 'K-Nearest Neighbors',
    icon: 'people',
    description: 'Non-parametric instance-based learner.',
    color: '#f59e0b',
    defaultConfig: {
      type: 'KNN',
      params: { k: 5 },
    },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['evaluation', 'output'],
    maxIncoming: 1,
  },
  {
    family: 'model',
    type: 'DecisionTree',
    label: 'Decision Tree',
    icon: 'account_tree',
    description: 'Recursive binary partitioning tree.',
    color: '#f59e0b',
    defaultConfig: {
      type: 'DecisionTree',
      params: { maxDepth: 5, minSamplesSplit: 2, criterion: 'gini' },
    },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['evaluation', 'output'],
    maxIncoming: 1,
  },
  {
    family: 'model',
    type: 'RandomForest',
    label: 'Random Forest',
    icon: 'forest',
    description: 'Bagged ensemble of decision trees.',
    color: '#f59e0b',
    defaultConfig: {
      type: 'RandomForest',
      params: { nEstimators: 20, maxDepth: 5 },
    },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['evaluation', 'output'],
    maxIncoming: 1,
  },
  {
    family: 'model',
    type: 'SVM',
    label: 'SVM',
    icon: 'hub',
    description: 'Support Vector Machine with linear kernel.',
    color: '#f59e0b',
    defaultConfig: {
      type: 'SVM',
      params: { C: 1 },
    },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['evaluation', 'output'],
    maxIncoming: 1,
  },
  {
    family: 'model',
    type: 'NaiveBayes',
    label: 'Naive Bayes',
    icon: 'calculate',
    description: 'Gaussian probabilistic classifier.',
    color: '#f59e0b',
    defaultConfig: {
      type: 'NaiveBayes',
      params: {},
    },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['evaluation', 'output'],
    maxIncoming: 1,
  },
  {
    family: 'model',
    type: 'KMeans',
    label: 'K-Means Clustering',
    icon: 'filter_alt',
    description: 'Unsupervised partition clustering.',
    color: '#f59e0b',
    defaultConfig: {
      type: 'KMeans',
      params: { nClusters: 3, maxIterations: 50 },
    },
    acceptsFrom: ['dataset', 'split', 'preprocessor'],
    outputsTo: ['evaluation', 'output'],
    maxIncoming: 1,
  },

  // ─── EVALUATION ────────────────────────────────────────────────────────────
  {
    family: 'evaluation',
    type: 'ClassificationMetrics',
    label: 'Classification Metrics',
    icon: 'bar_chart',
    description: 'Accuracy, precision, recall, F1 score.',
    color: '#ef4444',
    defaultConfig: { type: 'ClassificationMetrics' },
    acceptsFrom: ['model'],
    outputsTo: [],
    maxIncoming: 1,
    maxOutgoing: 0,
  },
  {
    family: 'evaluation',
    type: 'RegressionMetrics',
    label: 'Regression Metrics',
    icon: 'query_stats',
    description: 'MSE, RMSE, MAE, R² score.',
    color: '#ef4444',
    defaultConfig: { type: 'RegressionMetrics' },
    acceptsFrom: ['model'],
    outputsTo: [],
    maxIncoming: 1,
    maxOutgoing: 0,
  },

  // ─── OUTPUT ────────────────────────────────────────────────────────────────
  {
    family: 'output',
    type: 'CSVExport',
    label: 'CSV Export',
    icon: 'download',
    description: 'Download predictions as a CSV file.',
    color: '#06b6d4',
    defaultConfig: { type: 'CSVExport', filename: 'predictions.csv' },
    acceptsFrom: ['model'],
    outputsTo: [],
    maxIncoming: 1,
    maxOutgoing: 0,
  },
];

// ─── Family Metadata ─────────────────────────────────────────────────────────

export const FAMILY_META: Record<string, { label: string; color: string; icon: string }> = {
  dataset:      { label: 'Dataset',       color: '#6366f1', icon: 'storage' },
  split:        { label: 'Split',         color: '#818cf8', icon: 'call_split' },
  preprocessor: { label: 'Preprocessing', color: '#10b981', icon: 'auto_fix_high' },
  model:        { label: 'Models',        color: '#f59e0b', icon: 'psychology' },
  evaluation:   { label: 'Evaluation',    color: '#ef4444', icon: 'assessment' },
  output:       { label: 'Output',        color: '#06b6d4', icon: 'output' },
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

export function getCatalogByFamily(family: string): NodeCatalogEntry[] {
  return NODE_CATALOG.filter(n => n.family === family);
}

export function getCatalogEntry(family: string, type: string): NodeCatalogEntry | undefined {
  return NODE_CATALOG.find(n => n.family === family && n.type === type);
}

/** Check if a connection from sourceFamily to targetFamily is valid */
export function isConnectionValid(sourceFamily: NodeFamily, targetFamily: NodeFamily): boolean {
  const allowed = CONNECTION_RULES[targetFamily];
  return allowed ? allowed.includes(sourceFamily) : false;
}
