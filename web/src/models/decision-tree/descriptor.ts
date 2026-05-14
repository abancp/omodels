import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import DecisionTreeVisualization from './DecisionTreeVisualization';

const descriptor: ModelDescriptor = {
  id: 'decision-tree',
  name: 'Decision Tree',
  shortName: 'DT',
  vizLabel: 'Decision Boundary',
  category: 'Classification',
  categoryIcon: 'category',
  trainable: true,

  params: [
    /* ─── Basic ─── */
    {
      type: 'select',
      key: 'algorithm',
      label: 'Algorithm',
      options: [
        { value: 'id3', label: 'ID3 (Entropy)' },
        { value: 'c45', label: 'C4.5 (Gain Ratio)' },
        { value: 'cart', label: 'CART (Gini)' },
      ],
      defaultValue: 'cart',
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
      type: 'slider',
      key: 'minSamplesSplit',
      label: 'Min Samples to Split',
      min: 2,
      max: 50,
      step: 1,
      defaultValue: 2,
      formatValue: (v) => String(v),
    },

    /* ─── Advanced ─── */
    {
      type: 'slider',
      key: 'minSamplesLeaf',
      label: 'Min Samples per Leaf',
      min: 1,
      max: 25,
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
      max: 50,
      step: 5,
      defaultValue: 20,
      formatValue: (v) => String(v),
      level: 'advanced',
    },
    {
      type: 'toggle',
      key: 'showDecisionPath',
      label: 'Show Decision Path',
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
    ],
    defaultDataset: 'moons',
    params: [
      {
        type: 'slider',
        key: 'points',
        label: 'Points',
        min: 10,
        max: 500,
        step: 10,
        defaultValue: 120,
      },
      {
        type: 'slider',
        key: 'noise',
        label: 'Noise',
        min: 0,
        max: 1,
        step: 0.05,
        defaultValue: 0.2,
        formatValue: (v) => v.toFixed(2),
      },
    ],
  },

  codeSnippets: [
    {
      language: 'Python',
      lines: [
        {
          text: 'from sklearn.tree import DecisionTreeClassifier',
          highlights: [
            { start: 0, end: 4, type: 'keyword' },
            { start: 22, end: 28, type: 'keyword' },
          ],
        },
        { text: '' },
        {
          text: 'clf = DecisionTreeClassifier(',
        },
        {
          text: '    criterion="gini",  # or "entropy"',
          highlights: [{ start: 15, end: 21, type: 'string' }],
        },
        {
          text: '    max_depth=5,',
          highlights: [{ start: 14, end: 15, type: 'number' }],
        },
        { text: ')' },
        { text: 'clf.fit(X_train, y_train)' },
        {
          text: 'print(f"Accuracy = {clf.score(X_test, y_test):.4f}")',
          highlights: [{ start: 0, end: 5, type: 'keyword' }],
        },
      ],
    },
    {
      language: 'JavaScript',
      lines: [
        {
          text: "import { DecisionTree } from 'omodels';",
          highlights: [
            { start: 0, end: 6, type: 'keyword' },
            { start: 28, end: 34, type: 'keyword' },
          ],
        },
        { text: '' },
        {
          text: "const clf = new DecisionTree({ criterion: 'gini' });",
          highlights: [
            { start: 0, end: 5, type: 'keyword' },
          ],
        },
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
    { label: 'Tree Depth', value: '—' },
    { label: 'Nodes', value: '—' },
  ],

  VisualizationComponent: DecisionTreeVisualization,
};

export function registerDecisionTree() {
  registerModel(descriptor);
}
