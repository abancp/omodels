import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import LinearRegressionVisualization from './LinearRegressionVisualization';

const descriptor: ModelDescriptor = {
  id: 'linear-regression',
  name: 'Linear Regression',
  shortName: 'LinReg',
  vizLabel: 'Regression fit',
  category: 'Classical ML',
  categoryIcon: 'settings_input_component',
  trainable: true,

  params: [
    /* ─── Basic ─── */
    {
      type: 'slider',
      key: 'learningRate',
      label: 'Learning Rate',
      min: 0.001,
      max: 0.1,
      step: 0.001,
      defaultValue: 0.01,
      formatValue: (v) => v.toFixed(3),
    },
    {
      type: 'slider',
      key: 'epochs',
      label: 'Epochs',
      min: 10,
      max: 500,
      step: 10,
      defaultValue: 100,
    },
    {
      type: 'select',
      key: 'lossFunction',
      label: 'Loss Function',
      options: [
        { value: 'mse', label: 'MSE (L2)' },
        { value: 'mae', label: 'MAE (L1)' },
        { value: 'huber', label: 'Huber' },
      ],
      defaultValue: 'mse',
    },
    {
      type: 'toggle',
      key: 'showResiduals',
      label: 'Show residuals',
      defaultValue: false,
    },

    /* ─── Advanced ─── */
    {
      type: 'select',
      key: 'optimizer',
      label: 'Optimizer',
      options: [
        { value: 'sgd', label: 'SGD' },
        { value: 'momentum', label: 'Momentum' },
        { value: 'adam', label: 'Adam' },
        { value: 'rmsprop', label: 'RMSProp' },
      ],
      defaultValue: 'sgd',
      level: 'advanced',
    },
    {
      type: 'slider',
      key: 'momentum',
      label: 'Momentum β',
      min: 0,
      max: 0.99,
      step: 0.01,
      defaultValue: 0.9,
      formatValue: (v) => v.toFixed(2),
      level: 'advanced',
    },
    {
      type: 'select',
      key: 'regularization',
      label: 'Regularization',
      options: [
        { value: 'none', label: 'None' },
        { value: 'l1', label: 'L1 (Lasso)' },
        { value: 'l2', label: 'L2 (Ridge)' },
        { value: 'elastic', label: 'Elastic Net' },
      ],
      defaultValue: 'none',
      level: 'advanced',
    },
    {
      type: 'slider',
      key: 'regStrength',
      label: 'Reg. Strength (λ)',
      min: 0,
      max: 1,
      step: 0.01,
      defaultValue: 0.01,
      formatValue: (v) => v.toFixed(2),
      level: 'advanced',
    },
    {
      type: 'slider',
      key: 'lrDecay',
      label: 'LR Decay',
      min: 0,
      max: 0.01,
      step: 0.0001,
      defaultValue: 0,
      formatValue: (v) => v.toFixed(4),
      level: 'advanced',
    },
    {
      type: 'select',
      key: 'initMethod',
      label: 'Weight Init',
      options: [
        { value: 'random', label: 'Random' },
        { value: 'zeros', label: 'Zeros' },
        { value: 'xavier', label: 'Xavier' },
      ],
      defaultValue: 'random',
      level: 'advanced',
    },
    {
      type: 'toggle',
      key: 'showConfidence',
      label: 'Show confidence band',
      defaultValue: false,
      level: 'advanced',
    },
    {
      type: 'toggle',
      key: 'gradClipping',
      label: 'Gradient clipping',
      defaultValue: false,
      level: 'advanced',
    },
  ],

  dataset: {
    options: [
      { value: 'linear', label: 'Linear' },
      { value: 'noisy', label: 'Noisy' },
      { value: 'outliers', label: 'Outliers' },
      { value: 'custom', label: 'Custom', icon: 'edit' },
    ],
    defaultDataset: 'linear',
    params: [
      {
        type: 'slider',
        key: 'points',
        label: 'Points',
        min: 10,
        max: 200,
        step: 5,
        defaultValue: 50,
      },
      {
        type: 'slider',
        key: 'noise',
        label: 'Noise',
        min: 0,
        max: 1,
        step: 0.05,
        defaultValue: 0.3,
        formatValue: (v) => v.toFixed(2),
      },
    ],
  },

  codeSnippets: [
    {
      language: 'Python',
      lines: [
        {
          text: 'from sklearn.linear_model import LinearRegression',
          highlights: [
            { start: 0, end: 4, type: 'keyword' },
            { start: 37, end: 43, type: 'keyword' },
          ],
        },
        { text: '' },
        { text: 'model = LinearRegression()' },
        { text: 'model.fit(X_train, y_train)' },
        {
          text: 'print(f"R² = {model.score(X_test, y_test):.4f}")',
          highlights: [{ start: 0, end: 5, type: 'keyword' }],
        },
      ],
    },
    {
      language: 'JavaScript',
      lines: [
        {
          text: "import { LinearRegression } from 'omodels';",
          highlights: [
            { start: 0, end: 6, type: 'keyword' },
            { start: 28, end: 32, type: 'keyword' },
          ],
        },
        { text: '' },
        {
          text: 'const model = new LinearRegression({ lr: 0.01 });',
          highlights: [
            { start: 0, end: 5, type: 'keyword' },
            { start: 42, end: 46, type: 'number' },
          ],
        },
        { text: 'model.fit(xTrain, yTrain);' },
      ],
    },
    {
      language: 'C++',
      lines: [
        {
          text: '#include "linear_regression.h"',
          highlights: [{ start: 0, end: 8, type: 'keyword' }],
        },
        { text: '' },
        {
          text: 'auto lr = LinearRegression(0.01, 100);',
          highlights: [
            { start: 27, end: 31, type: 'number' },
            { start: 33, end: 36, type: 'number' },
          ],
        },
        { text: 'lr.fit(X_train, y_train);' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'R²', value: '—', isPrimary: true },
    { label: 'MSE', value: '—' },
    { label: 'MAE', value: '—' },
    { label: 'Equation', value: '—' },
  ],

  VisualizationComponent: LinearRegressionVisualization,
};

export function registerLinearRegression() {
  registerModel(descriptor);
}
