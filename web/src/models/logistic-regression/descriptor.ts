import { registerModel } from '../registry';
import LogisticRegressionVisualization from './LogisticRegressionVisualization';
import type { ModelDescriptor } from '../registry';

const descriptor: ModelDescriptor = {
  id: 'logistic-regression',
  name: 'Logistic Regression',
  shortName: 'LogReg',
  vizLabel: 'Classification fit',
  category: 'Classification',
  categoryIcon: 'category', // Using material icon equivalent or generic
  trainable: true,
  params: [
    {
      type: 'select',
      key: 'degree',
      label: 'Feature Degree',
      defaultValue: '1',
      options: [
        { value: '1', label: 'Linear Boundary (1)' },
        { value: '2', label: 'Conic Boundary (2)' }
      ]
    },
    {
      type: 'slider',
      key: 'learningRate',
      label: 'Learning Rate',
      min: 0.01,
      max: 1.0,
      step: 0.01,
      defaultValue: 0.1,
      formatValue: (v) => v.toFixed(2)
    },
    {
      type: 'slider',
      key: 'epochs',
      label: 'Epochs',
      min: 50,
      max: 1000,
      step: 50,
      defaultValue: 200
    },
    {
      type: 'select',
      key: 'regularization',
      label: 'Regularization',
      defaultValue: 'none',
      options: [
        { value: 'none', label: 'None' },
        { value: 'l1', label: 'L1 (Lasso)' },
        { value: 'l2', label: 'L2 (Ridge)' }
      ],
      level: 'advanced'
    },
    {
      type: 'slider',
      key: 'regStrength',
      label: 'Reg. Strength (λ)',
      min: 0.001,
      max: 0.1,
      step: 0.001,
      defaultValue: 0.01,
      formatValue: (v) => v.toFixed(3),
      level: 'advanced'
    },
    {
      type: 'slider',
      key: 'threshold',
      label: 'Decision Threshold',
      min: 0.1,
      max: 0.9,
      step: 0.05,
      defaultValue: 0.5,
      formatValue: (v) => v.toFixed(2)
    }
  ],
  dataset: {
    defaultDataset: 'blobs',
    options: [
      { value: 'blobs', label: 'Clusters' },
      { value: 'moons', label: 'Moons' },
      { value: 'circles', label: 'Circles' },
      { value: 'linear', label: 'Linearly Separable' },
      { value: 'xor', label: 'XOR Pattern' },
      { value: 'spiral', label: 'Spiral' }
    ],
    params: [
      {
        type: 'slider',
        key: 'points',
        label: 'Number of Points',
        min: 20,
        max: 300,
        step: 10,
        defaultValue: 100
      },
      {
        type: 'slider',
        key: 'noise',
        label: 'Noise Level',
        min: 0,
        max: 1,
        step: 0.05,
        defaultValue: 0.15
      }
    ]
  },
  defaultMetrics: [
    { label: 'Accuracy', value: '—', isPrimary: true },
    { label: 'Precision', value: '—' },
    { label: 'Recall', value: '—' },
    { label: 'F1 Score', value: '—' },
    { label: 'AUC', value: '—' }
  ],
  codeSnippets: [
    {
      language: 'Python',
      lines: [
        { text: 'from sklearn.linear_model import LogisticRegression', highlights: [{ start: 5, end: 12, type: 'keyword' }, { start: 33, end: 51, type: 'keyword' }] },
        { text: '' },
        { text: 'model = LogisticRegression(penalty="l2", C=1.0)' },
        { text: 'model.fit(X_train, y_train)' },
        { text: 'y_pred = model.predict(X_test)' },
        { text: 'print(f"Accuracy: {model.score(X_test, y_test):.4f}")' }
      ]
    },
    {
      language: 'JavaScript',
      lines: [
        { text: '// Basic Logistic Regression implementation' },
        { text: 'function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }' },
        { text: 'function predict(x, w, b) {' },
        { text: '  let z = b;' },
        { text: '  for(let i=0; i<x.length; i++) z += w[i]*x[i];' },
        { text: '  return sigmoid(z);' },
        { text: '}' }
      ]
    }
  ],
  VisualizationComponent: LogisticRegressionVisualization
};

export function registerLogisticRegression() {
  registerModel(descriptor);
}
