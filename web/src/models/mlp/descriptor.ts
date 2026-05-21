import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import MLPVisualization from './MLPVisualization';

const descriptor: ModelDescriptor = {
  id: 'mlp',
  name: 'Multi-Layer Perceptron',
  shortName: 'MLP',
  vizLabel: 'Deep Network',
  category: 'Neural Networks',
  categoryIcon: 'route',
  trainable: true,

  params: [
    { type: 'slider', key: 'inputNodes', label: 'Input Features', min: 2, max: 8, step: 1, defaultValue: 2, formatValue: (v) => String(v) },
    { type: 'slider', key: 'learningRate', label: 'Learning Rate', min: 0.001, max: 1.0, step: 0.001, defaultValue: 0.01, formatValue: (v) => v.toFixed(3) },
    { type: 'slider', key: 'maxEpochs', label: 'Max Epochs', min: 50, max: 5000, step: 50, defaultValue: 500, formatValue: (v) => String(v) },
    { type: 'slider', key: 'numLayers', label: 'Hidden Layers', min: 1, max: 3, step: 1, defaultValue: 2, formatValue: (v) => String(v) },
    // Layer 1
    { type: 'slider', key: 'l1Nodes', label: 'L1 Neurons', min: 1, max: 16, step: 1, defaultValue: 4, formatValue: (v) => String(v), level: 'advanced' },
    { type: 'select', key: 'l1Act', label: 'L1 Activation', options: [{ value: 'relu', label: 'ReLU' }, { value: 'tanh', label: 'Tanh' }, { value: 'sigmoid', label: 'Sigmoid' }, { value: 'linear', label: 'Linear' }], defaultValue: 'relu', level: 'advanced' },
    // Layer 2
    { type: 'slider', key: 'l2Nodes', label: 'L2 Neurons', min: 1, max: 16, step: 1, defaultValue: 4, formatValue: (v) => String(v), level: 'advanced' },
    { type: 'select', key: 'l2Act', label: 'L2 Activation', options: [{ value: 'relu', label: 'ReLU' }, { value: 'tanh', label: 'Tanh' }, { value: 'sigmoid', label: 'Sigmoid' }, { value: 'linear', label: 'Linear' }], defaultValue: 'relu', level: 'advanced' },
    // Layer 3
    { type: 'slider', key: 'l3Nodes', label: 'L3 Neurons', min: 1, max: 16, step: 1, defaultValue: 4, formatValue: (v) => String(v), level: 'advanced' },
    { type: 'select', key: 'l3Act', label: 'L3 Activation', options: [{ value: 'relu', label: 'ReLU' }, { value: 'tanh', label: 'Tanh' }, { value: 'sigmoid', label: 'Sigmoid' }, { value: 'linear', label: 'Linear' }], defaultValue: 'relu', level: 'advanced' },
    // Output Layer
    { type: 'slider', key: 'outNodes', label: 'Output Neurons', min: 1, max: 10, step: 1, defaultValue: 1, formatValue: (v) => String(v), level: 'advanced' },
    { type: 'select', key: 'outAct', label: 'Output Activation', options: [{ value: 'sigmoid', label: 'Sigmoid (Binary)' }, { value: 'linear', label: 'Linear (Regression)' }, { value: 'softmax', label: 'Softmax (Multi-class)' }], defaultValue: 'sigmoid', level: 'advanced' },
  ],

  dataset: {
    options: [
      { value: 'spirals', label: 'Spirals (Complex)' },
      { value: 'moons', label: 'Moons (Non-Linear)' },
      { value: 'circles', label: 'Circles' },
      { value: 'xor', label: 'XOR' },
      { value: 'custom', label: 'Custom', icon: 'edit' },
      { value: 'import', label: 'Import', icon: 'upload' },
    ],
    defaultDataset: 'moons',
    params: [
      { type: 'slider', key: 'points', label: 'Points', min: 50, max: 500, step: 50, defaultValue: 200 },
      { type: 'slider', key: 'noise', label: 'Noise', min: 0.0, max: 1.0, step: 0.05, defaultValue: 0.1, formatValue: (v) => v.toFixed(2) },
    ],
  },

  codeSnippets: [
    {
      language: 'Python', lines: [
        { text: 'from sklearn.neural_network import MLPClassifier', highlights: [{ start: 0, end: 4, type: 'keyword' }] },
        { text: "clf = MLPClassifier(hidden_layer_sizes=(4, 4), activation='relu')" },
        { text: 'clf.fit(X, y)' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'Loss (MSE)', value: '—', isPrimary: true },
    { label: 'Epochs', value: '—' },
    { label: 'Converged', value: '—' },
  ],

  VisualizationComponent: MLPVisualization,
};

export function registerMLP() { registerModel(descriptor); }
