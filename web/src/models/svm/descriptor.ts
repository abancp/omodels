import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import SVMVisualization from './SVMVisualization';

const svmDescriptor: ModelDescriptor = {
  id: 'svm',
  name: 'Support Vector Machine',
  shortName: 'SVM',
  vizLabel: 'Margin boundary',
  category: 'Classification',
  categoryIcon: 'category',
  trainable: true,

  params: [
    {
      type: 'chip-select',
      key: 'kernel',
      label: 'Kernel Type',
      options: [
        { value: 'linear', label: 'Linear' },
        { value: 'poly2', label: 'Polynomial (2)' },
        { value: 'rbf', label: 'RBF (Gaussian)' }
      ],
      defaultValue: 'linear'
    },
    {
      type: 'slider',
      key: 'cParam',
      label: 'Regularization (C)',
      min: 0.1,
      max: 100,
      step: 0.1,
      defaultValue: 1.0,
      formatValue: (v) => v.toFixed(1)
    },
    {
      type: 'slider',
      key: 'learningRate',
      label: 'Learning Rate',
      min: 0.001,
      max: 0.5,
      step: 0.001,
      defaultValue: 0.05,
      formatValue: (v) => v.toFixed(3)
    },
    {
      type: 'slider',
      key: 'epochs',
      label: 'Epochs',
      min: 50,
      max: 2000,
      step: 50,
      defaultValue: 500
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
      { value: 'spiral', label: 'Spiral' },
      { value: 'import', label: 'Import', icon: 'upload' },
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
    { label: 'F1 Score', value: '—' }
  ],

  codeSnippets: [
    {
      language: 'Python',
      lines: [
        {
          text: 'from sklearn.svm import SVC',
          highlights: [
            { start: 0, end: 4, type: 'keyword' },
            { start: 25, end: 31, type: 'keyword' }
          ]
        },
        { text: '' },
        {
          text: 'model = SVC(C=1.0, kernel="linear")',
          highlights: [
            { start: 14, end: 17, type: 'number' },
            { start: 26, end: 34, type: 'string' }
          ]
        },
        { text: 'model.fit(X_train, y_train)' }
      ]
    },
    {
      language: 'JavaScript',
      lines: [
        {
          text: 'import { SVM } from "omodels";',
          highlights: [
            { start: 0, end: 6, type: 'keyword' },
            { start: 20, end: 29, type: 'string' }
          ]
        },
        { text: '' },
        {
          text: 'const model = new SVM({ c: 1.0, kernel: "linear" });',
          highlights: [
            { start: 0, end: 5, type: 'keyword' },
            { start: 14, end: 17, type: 'keyword' }
          ]
        },
        { text: 'await model.train(data, { epochs: 500 });' }
      ]
    }
  ],

  VisualizationComponent: SVMVisualization
};

export function registerSVM() {
  registerModel(svmDescriptor);
}
