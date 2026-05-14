import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import NaiveBayesVisualization from './NaiveBayesVisualization';

const descriptor: ModelDescriptor = {
  id: 'naive-bayes',
  name: 'Naive Bayes',
  shortName: 'NB',
  vizLabel: 'Decision Boundary',
  category: 'Classification',
  categoryIcon: 'category',
  trainable: true,

  params: [
    /* ─── Basic ─── */
    {
      type: 'select',
      key: 'nbType',
      label: 'Model Type',
      options: [
        { value: 'gaussian', label: 'Gaussian' },
        { value: 'multinomial', label: 'Multinomial' },
        { value: 'bernoulli', label: 'Bernoulli' },
      ],
      defaultValue: 'gaussian',
    },
    {
      type: 'toggle',
      key: 'fitPrior',
      label: 'Learn Class Prior',
      defaultValue: true,
    },

    /* ─── Advanced ─── */
    {
      type: 'slider',
      key: 'varSmoothing',
      label: 'Var Smoothing (Gaussian)',
      min: 1e-9,
      max: 1e-1,
      step: 0.001,
      defaultValue: 1e-9,
      formatValue: (v) => v.toExponential(1),
      level: 'advanced',
    },
    {
      type: 'slider',
      key: 'alpha',
      label: 'Alpha (Smoothing)',
      min: 0.0,
      max: 10.0,
      step: 0.1,
      defaultValue: 1.0,
      formatValue: (v) => v.toFixed(1),
      level: 'advanced',
    },
    {
      type: 'slider',
      key: 'binarizeThreshold',
      label: 'Binarize Threshold (Bernoulli)',
      min: 0.0,
      max: 1.0,
      step: 0.05,
      defaultValue: 0.5,
      formatValue: (v) => v.toFixed(2),
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
    defaultDataset: 'blobs',
    params: [
      {
        type: 'slider',
        key: 'points',
        label: 'Points',
        min: 10,
        max: 500,
        step: 10,
        defaultValue: 100,
      },
      {
        type: 'slider',
        key: 'noise',
        label: 'Noise',
        min: 0,
        max: 1,
        step: 0.05,
        defaultValue: 0.15,
        formatValue: (v) => v.toFixed(2),
      },
    ],
  },

  codeSnippets: [
    {
      language: 'Python',
      lines: [
        {
          text: 'from sklearn.naive_bayes import GaussianNB',
          highlights: [
            { start: 0, end: 4, type: 'keyword' },
            { start: 32, end: 38, type: 'keyword' },
          ],
        },
        { text: '' },
        { text: 'model = GaussianNB()' },
        { text: 'model.fit(X_train, y_train)' },
        {
          text: 'print(f"Accuracy = {model.score(X_test, y_test):.4f}")',
          highlights: [{ start: 0, end: 5, type: 'keyword' }],
        },
      ],
    },
    {
      language: 'JavaScript',
      lines: [
        {
          text: "import { GaussianNB } from 'omodels';",
          highlights: [
            { start: 0, end: 6, type: 'keyword' },
            { start: 27, end: 31, type: 'keyword' },
          ],
        },
        { text: '' },
        {
          text: 'const model = new GaussianNB();',
          highlights: [
            { start: 0, end: 5, type: 'keyword' },
          ],
        },
        { text: 'model.fit(xTrain, yTrain);' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'Accuracy', value: '—', isPrimary: true },
    { label: 'Precision', value: '—' },
    { label: 'Recall', value: '—' },
    { label: 'F1 Score', value: '—' },
    { label: 'AUC', value: '—' },
  ],

  VisualizationComponent: NaiveBayesVisualization,
};

export function registerNaiveBayes() {
  registerModel(descriptor);
}
