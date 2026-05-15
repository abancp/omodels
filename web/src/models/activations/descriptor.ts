import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import ActivationsVisualization from './ActivationsVisualization';

const descriptor: ModelDescriptor = {
  id: 'activations',
  name: 'Activation Functions',
  shortName: 'Activations',
  vizLabel: 'Function Plot',
  category: 'Neural Networks',
  categoryIcon: 'route',
  trainable: false,

  params: [
    { type: 'select', key: 'function', label: 'Function', options: [
      { value: 'step', label: 'Step' },
      { value: 'linear', label: 'Linear' },
      { value: 'sigmoid', label: 'Sigmoid' },
      { value: 'tanh', label: 'Tanh' },
      { value: 'relu', label: 'ReLU' },
      { value: 'leaky_relu', label: 'Leaky ReLU' },
      { value: 'elu', label: 'ELU' },
      { value: 'swish', label: 'Swish' },
    ], defaultValue: 'sigmoid' },
    { type: 'toggle', key: 'showDerivative', label: 'Show Derivative', defaultValue: true }
  ],

  dataset: {
    options: [],
    defaultDataset: '',
    params: []
  },

  codeSnippets: [
    {
      language: 'Python', lines: [
        { text: 'import torch.nn as nn', highlights: [{ start: 0, end: 6, type: 'keyword' }, { start: 16, end: 18, type: 'keyword' }] },
        { text: '' },
        { text: 'activation = nn.Sigmoid()' },
        { text: 'y = activation(x)' },
      ],
    },
    {
      language: 'JavaScript', lines: [
        { text: 'function sigmoid(x) {', highlights: [{ start: 0, end: 8, type: 'keyword' }] },
        { text: '  return 1 / (1 + Math.exp(-x));', highlights: [{ start: 2, end: 8, type: 'keyword' }] },
        { text: '}' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'Equation', value: '—', isPrimary: true },
  ],

  VisualizationComponent: ActivationsVisualization,
};

export function registerActivations() { registerModel(descriptor); }
