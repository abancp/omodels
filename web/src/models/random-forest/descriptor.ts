import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import RandomForestVisualization from './RandomForestVisualization';

const descriptor: ModelDescriptor = {
  id: 'random-forest',
  name: 'Random Forest',
  shortName: 'RF',
  vizLabel: 'Decision Boundary',
  category: 'Classification',
  categoryIcon: 'category',
  trainable: true,

  params: [
    /* ─── Basic ─── */
    {
      type: 'slider',
      key: 'nEstimators',
      label: 'Number of Trees',
      min: 1,
      max: 50,
      step: 1,
      defaultValue: 10,
      formatValue: (v) => String(v),
    },
    {
      type: 'slider',
      key: 'maxDepth',
      label: 'Max Depth',
      min: 1,
      max: 15,
      step: 1,
      defaultValue: 5,
      formatValue: (v) => String(v),
    },
    {
      type: 'select',
      key: 'maxFeatures',
      label: 'Max Features',
      options: [
        { value: 'sqrt', label: 'sqrt (Recommended)' },
        { value: 'log2', label: 'log2' },
        { value: 'all', label: 'All Features' },
      ],
      defaultValue: 'sqrt',
    },

    /* ─── Advanced ─── */
    {
      type: 'select',
      key: 'algorithm',
      label: 'Split Criterion',
      options: [
        { value: 'cart', label: 'Gini Impurity' },
        { value: 'id3', label: 'Entropy' },
        { value: 'c45', label: 'Gain Ratio' },
      ],
      defaultValue: 'cart',
      level: 'advanced',
    },
    {
      type: 'slider',
      key: 'minSamplesSplit',
      label: 'Min Samples to Split',
      min: 2,
      max: 30,
      step: 1,
      defaultValue: 2,
      formatValue: (v) => String(v),
      level: 'advanced',
    },
    {
      type: 'slider',
      key: 'minSamplesLeaf',
      label: 'Min Samples per Leaf',
      min: 1,
      max: 20,
      step: 1,
      defaultValue: 1,
      formatValue: (v) => String(v),
      level: 'advanced',
    },
    {
      type: 'slider',
      key: 'numBins',
      label: 'Feature Bins',
      min: 5,
      max: 40,
      step: 5,
      defaultValue: 15,
      formatValue: (v) => String(v),
      level: 'advanced',
    },
    {
      type: 'toggle',
      key: 'showTreeVotes',
      label: 'Show Tree Votes',
      defaultValue: true,
      level: 'advanced',
    },
  ],

  dataset: {
    options: [
      { value: 'blobs', label: 'Blobs' },
      { value: 'moons', label: 'Moons' },
      { value: 'circles', label: 'Circles' },
      { value: 'xor', label: 'XOR' },
      { value: 'spiral', label: 'Spiral' },
      { value: 'custom', label: 'Custom', icon: 'edit' },
      { value: 'import', label: 'Import', icon: 'upload' },
    ],
    defaultDataset: 'moons',
    params: [
      {
        type: 'slider', key: 'points', label: 'Points',
        min: 20, max: 500, step: 10, defaultValue: 150,
      },
      {
        type: 'slider', key: 'noise', label: 'Noise',
        min: 0, max: 1, step: 0.05, defaultValue: 0.2,
        formatValue: (v) => v.toFixed(2),
      },
    ],
  },

  codeSnippets: [
    {
      language: 'Python',
      lines: [
        { text: 'from sklearn.ensemble import RandomForestClassifier', highlights: [{ start: 0, end: 4, type: 'keyword' }, { start: 30, end: 36, type: 'keyword' }] },
        { text: '' },
        { text: 'clf = RandomForestClassifier(' },
        { text: '    n_estimators=100,', highlights: [{ start: 17, end: 20, type: 'number' }] },
        { text: '    max_depth=5,', highlights: [{ start: 14, end: 15, type: 'number' }] },
        { text: '    max_features="sqrt",', highlights: [{ start: 18, end: 24, type: 'string' }] },
        { text: ')' },
        { text: 'clf.fit(X_train, y_train)' },
        { text: 'print(f"Accuracy = {clf.score(X_test, y_test):.4f}")', highlights: [{ start: 0, end: 5, type: 'keyword' }] },
      ],
    },
    {
      language: 'JavaScript',
      lines: [
        { text: "import { RandomForest } from 'omodels';", highlights: [{ start: 0, end: 6, type: 'keyword' }, { start: 28, end: 36, type: 'keyword' }] },
        { text: '' },
        { text: "const clf = new RandomForest({ nEstimators: 100 });", highlights: [{ start: 0, end: 5, type: 'keyword' }] },
        { text: 'clf.fit(xTrain, yTrain);' },
        { text: 'const pred = clf.predict(xTest);' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'Accuracy', value: '—', isPrimary: true },
    { label: 'Precision', value: '—' },
    { label: 'Recall', value: '—' },
    { label: 'F1 Score', value: '—' },
    { label: 'OOB Accuracy', value: '—' },
    { label: 'Trees', value: '—' },
  ],

  VisualizationComponent: RandomForestVisualization,
};

export function registerRandomForest() {
  registerModel(descriptor);
}
