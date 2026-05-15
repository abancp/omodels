import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import GMMVisualization from './GMMVisualization';

const descriptor: ModelDescriptor = {
  id: 'gmm',
  name: 'Gaussian Mixture',
  shortName: 'GMM',
  vizLabel: 'Soft Clustering',
  category: 'Clustering',
  categoryIcon: 'blur_circular',
  trainable: true,

  params: [
    { type: 'slider', key: 'k', label: 'Components (K)', min: 2, max: 10, step: 1, defaultValue: 3, formatValue: (v) => String(v) },
    { type: 'select', key: 'covarianceType', label: 'Covariance Type', options: [
      { value: 'full', label: 'Full (Ellipses)' },
      { value: 'diag', label: 'Diagonal (Axis-aligned)' },
      { value: 'spherical', label: 'Spherical (Circles)' },
    ], defaultValue: 'full' },
    { type: 'slider', key: 'maxIter', label: 'Max Iterations', min: 10, max: 200, step: 10, defaultValue: 50, formatValue: (v) => String(v) },

    { type: 'select', key: 'initMethod', label: 'Initialization', options: [
      { value: 'kmeans++', label: 'K-Means++' },
      { value: 'random', label: 'Random' },
    ], defaultValue: 'kmeans++', level: 'advanced' },
    { type: 'toggle', key: 'showCovariance', label: 'Show Covariance Ellipses', defaultValue: true, level: 'advanced' },
    { type: 'toggle', key: 'colorMixing', label: 'Soft Assignment Colors', defaultValue: true, level: 'advanced' },
  ],

  dataset: {
    options: [
      { value: 'blobs', label: 'Blobs' }, { value: 'anisotropic', label: 'Anisotropic' },
      { value: 'moons', label: 'Moons' }, { value: 'circles', label: 'Circles' },
      { value: 'uniform', label: 'Uniform' }, { value: 'custom', label: 'Custom', icon: 'edit' },
    ],
    defaultDataset: 'blobs',
    params: [
      { type: 'slider', key: 'points', label: 'Points', min: 50, max: 800, step: 50, defaultValue: 300 },
      { type: 'slider', key: 'noise', label: 'Noise', min: 0.1, max: 2.0, step: 0.1, defaultValue: 1.0, formatValue: (v) => v.toFixed(1) },
    ],
  },

  codeSnippets: [
    {
      language: 'Python', lines: [
        { text: 'from sklearn.mixture import GaussianMixture', highlights: [{ start: 0, end: 4, type: 'keyword' }, { start: 28, end: 34, type: 'keyword' }] },
        { text: '' },
        { text: "gmm = GaussianMixture(n_components=3, covariance_type='full')", highlights: [{ start: 35, end: 36, type: 'number' }, { start: 54, end: 60, type: 'string' }] },
        { text: 'gmm.fit(X)' },
        { text: 'labels = gmm.predict(X)' },
        { text: 'probs = gmm.predict_proba(X)' },
      ],
    },
    {
      language: 'JavaScript', lines: [
        { text: "import { GMM } from 'omodels';", highlights: [{ start: 0, end: 6, type: 'keyword' }] },
        { text: '' },
        { text: "const gmm = new GMM({ k: 3, covarianceType: 'full' });", highlights: [{ start: 0, end: 5, type: 'keyword' }] },
        { text: 'gmm.fit(data);' },
        { text: 'const probs = gmm.predictProba(data);' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'Log-Likelihood', value: '—', isPrimary: true },
    { label: 'Iterations', value: '—' },
    { label: 'Converged', value: '—' },
    { label: 'AIC', value: '—' }
  ],

  VisualizationComponent: GMMVisualization,
};

export function registerGMM() { registerModel(descriptor); }
