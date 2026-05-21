import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import PerceptronVisualization from './PerceptronVisualization';

const descriptor: ModelDescriptor = {
  id: 'perceptron',
  name: 'Perceptron',
  shortName: 'Perceptron',
  vizLabel: 'Single Neuron',
  category: 'Neural Networks',
  categoryIcon: 'route',
  trainable: true,

  params: [
    { type: 'slider', key: 'inputNodes', label: 'Input Nodes (Features)', min: 2, max: 8, step: 1, defaultValue: 2, formatValue: (v) => String(v) },
    { type: 'slider', key: 'numPerceptrons', label: 'Hidden Neurons (Perceptrons)', min: 1, max: 8, step: 1, defaultValue: 1, formatValue: (v) => String(v) },
    {
      type: 'select', key: 'activation', label: 'Hidden Activation', options: [
        { value: 'step', label: 'Step (Binary)' },
        { value: 'sigmoid', label: 'Sigmoid' },
        { value: 'tanh', label: 'Tanh' },
        { value: 'relu', label: 'ReLU' },
        { value: 'linear', label: 'Linear' },
      ], defaultValue: 'step'
    },
    {
      type: 'select', key: 'outAct', label: 'Output Activation', options: [
        { value: 'step', label: 'Step (Binary)' },
        { value: 'sigmoid', label: 'Sigmoid (Binary)' },
        { value: 'linear', label: 'Linear (Regression)' },
      ], defaultValue: 'sigmoid'
    },
    { type: 'slider', key: 'learningRate', label: 'Learning Rate (η)', min: 0.01, max: 1.0, step: 0.01, defaultValue: 0.1, formatValue: (v) => v.toFixed(2) },
    { type: 'slider', key: 'maxEpochs', label: 'Max Epochs', min: 10, max: 1000, step: 10, defaultValue: 100, formatValue: (v) => String(v) },
  ],

  dataset: {
    options: [
      { value: 'linearly_separable', label: 'Linearly Separable' },
      { value: 'xor', label: 'XOR (Not Separable)' },
      { value: 'circles', label: 'Circles (Not Separable)' },
      { value: 'uniform', label: 'Uniform' },
      { value: 'custom', label: 'Custom', icon: 'edit' },
      { value: 'import', label: 'Import', icon: 'upload' },
    ],
    defaultDataset: 'linearly_separable',
    params: [
      { type: 'slider', key: 'points', label: 'Points', min: 50, max: 400, step: 50, defaultValue: 100 },
      { type: 'slider', key: 'noise', label: 'Noise', min: 0.0, max: 1.0, step: 0.1, defaultValue: 0.2, formatValue: (v) => v.toFixed(1) },
    ],
  },

  codeSnippets: [
    {
      language: 'Python', lines: [
        { text: 'from sklearn.linear_model import Perceptron', highlights: [{ start: 0, end: 4, type: 'keyword' }, { start: 33, end: 43, type: 'keyword' }] },
        { text: '' },
        { text: "clf = Perceptron(eta0=0.1, max_iter=100)", highlights: [{ start: 22, end: 25, type: 'number' }, { start: 36, end: 39, type: 'number' }] },
        { text: 'clf.fit(X, y)' },
        { text: 'y_pred = clf.predict(X)' },
      ],
    },
    {
      language: 'JavaScript', lines: [
        { text: "import { Perceptron } from 'omodels';", highlights: [{ start: 0, end: 6, type: 'keyword' }] },
        { text: '' },
        { text: "const p = new Perceptron({ learningRate: 0.1, activation: 'step' });", highlights: [{ start: 0, end: 5, type: 'keyword' }] },
        { text: 'p.fit(data, labels);' },
        { text: 'const y_pred = p.predict(data);' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'Accuracy', value: '—', isPrimary: true },
    { label: 'Loss (MSE)', value: '—' },
    { label: 'Epochs', value: '—' },
    { label: 'Converged', value: '—' },
    { label: 'Precision', value: '—' },
    { label: 'Recall', value: '—' },
    { label: 'F1 Score', value: '—' },
  ],

  VisualizationComponent: PerceptronVisualization,
};

export function registerPerceptron() { registerModel(descriptor); }
