import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import PolynomialRegressionVisualization from './PolynomialRegressionVisualization';

const descriptor: ModelDescriptor = {
  id: 'polynomial-regression',
  name: 'Polynomial Regression',
  shortName: 'PolyReg',
  vizLabel: 'Polynomial fit',
  category: 'Regression',
  categoryIcon: 'show_chart',
  trainable: true,

  params: [
    /* ─── Basic ─── */
    {
      type: 'slider',
      key: 'degree',
      label: 'Polynomial Degree',
      min: 1,
      max: 8,
      step: 1,
      defaultValue: 2,
    },
    {
      type: 'slider',
      key: 'learningRate',
      label: 'Learning Rate',
      min: 0.0001,
      max: 0.1,
      step: 0.0001,
      defaultValue: 0.005,
      formatValue: (v) => v.toFixed(4),
    },
    {
      type: 'slider',
      key: 'epochs',
      label: 'Epochs',
      min: 10,
      max: 500,
      step: 10,
      defaultValue: 150,
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
    {
      type: 'toggle',
      key: 'showTrueCurve',
      label: 'Show true curve',
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
      type: 'toggle',
      key: 'featureNormalize',
      label: 'Normalize features',
      defaultValue: true,
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
      defaultValue: true,
      level: 'advanced',
    },
  ],

  dataset: {
    options: [
      { value: 'quadratic', label: 'Quadratic' },
      { value: 'cubic', label: 'Cubic' },
      { value: 'sinusoidal', label: 'Sinusoidal' },
      { value: 'step', label: 'Step Function' },
      { value: 'noisy', label: 'Noisy Quadratic' },
      { value: 'custom', label: 'Custom', icon: 'edit' },
    ],
    defaultDataset: 'quadratic',
    params: [
      {
        type: 'slider',
        key: 'points',
        label: 'Points',
        min: 10,
        max: 200,
        step: 5,
        defaultValue: 60,
      },
      {
        type: 'slider',
        key: 'noise',
        label: 'Noise',
        min: 0,
        max: 1,
        step: 0.05,
        defaultValue: 0.25,
        formatValue: (v) => v.toFixed(2),
      },
    ],
  },

  codeSnippets: [
    {
      language: 'Python',
      lines: [
        {
          text: 'import numpy as np',
          highlights: [{ start: 0, end: 6, type: 'keyword' }],
        },
        {
          text: 'from sklearn.preprocessing import PolynomialFeatures',
          highlights: [{ start: 0, end: 4, type: 'keyword' }, { start: 27, end: 33, type: 'keyword' }],
        },
        {
          text: 'from sklearn.linear_model import LinearRegression',
          highlights: [{ start: 0, end: 4, type: 'keyword' }, { start: 26, end: 32, type: 'keyword' }],
        },
        { text: '' },
        { text: 'poly = PolynomialFeatures(degree=2)' },
        { text: 'X_poly = poly.fit_transform(X_train)' },
        { text: '' },
        { text: 'model = LinearRegression()' },
        { text: 'model.fit(X_poly, y_train)' },
      ],
    },
    {
      language: 'JavaScript',
      lines: [
        {
          text: "import { PolynomialRegression } from 'omodels';",
          highlights: [
            { start: 0, end: 6, type: 'keyword' },
            { start: 32, end: 36, type: 'keyword' },
          ],
        },
        { text: '' },
        {
          text: 'const model = new PolynomialRegression({ degree: 2, lr: 0.005 });',
          highlights: [
            { start: 0, end: 5, type: 'keyword' },
            { start: 14, end: 17, type: 'keyword' },
            { start: 49, end: 50, type: 'number' },
            { start: 56, end: 61, type: 'number' },
          ],
        },
        { text: 'model.fit(xTrain, yTrain);' },
      ],
    },
    {
      language: 'C++',
      lines: [
        {
          text: '#include "polynomial_regression.h"',
          highlights: [{ start: 0, end: 8, type: 'keyword' }],
        },
        { text: '' },
        {
          text: 'auto poly = PolynomialRegression(2, 0.005, 150);',
          highlights: [
            { start: 0, end: 4, type: 'keyword' },
            { start: 33, end: 34, type: 'number' },
            { start: 36, end: 41, type: 'number' },
            { start: 43, end: 46, type: 'number' },
          ],
        },
        { text: 'poly.fit(X_train, y_train);' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'R²', value: '—', isPrimary: true },
    { label: 'MSE', value: '—' },
    { label: 'MAE', value: '—' },
    { label: 'Degree', value: '2' },
    { label: 'Equation', value: '—' },
  ],

  VisualizationComponent: PolynomialRegressionVisualization,
};

export function registerPolynomialRegression() {
  registerModel(descriptor);
}
