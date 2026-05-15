import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import DBSCANVisualization from './DBSCANVisualization';

const descriptor: ModelDescriptor = {
  id: 'dbscan',
  name: 'DBSCAN',
  shortName: 'DB',
  vizLabel: 'Density Reachability',
  category: 'Clustering',
  categoryIcon: 'scatter_plot',
  trainable: true,

  params: [
    { type: 'slider', key: 'eps', label: 'Epsilon (eps)', min: 0.01, max: 0.5, step: 0.01, defaultValue: 0.1, formatValue: (v) => v.toFixed(2) },
    { type: 'slider', key: 'minPts', label: 'Min Points (minPts)', min: 2, max: 20, step: 1, defaultValue: 4, formatValue: (v) => String(v) },
    { type: 'select', key: 'metric', label: 'Distance Metric', options: [
      { value: 'euclidean', label: 'Euclidean (L2)' },
      { value: 'manhattan', label: 'Manhattan (L1)' },
      { value: 'chebyshev', label: 'Chebyshev (L∞)' },
    ], defaultValue: 'euclidean' },

    { type: 'toggle', key: 'showEpsCircles', label: 'Show Epsilon Neighborhoods', defaultValue: true, level: 'advanced' },
    { type: 'toggle', key: 'showPointTypes', label: 'Show Point Types (Core/Border/Noise)', defaultValue: true, level: 'advanced' },
  ],

  dataset: {
    options: [
      { value: 'moons', label: 'Moons' }, { value: 'circles', label: 'Circles' },
      { value: 'blobs', label: 'Blobs' }, { value: 'uniform', label: 'Uniform' },
      { value: 'anisotropic', label: 'Anisotropic' }, { value: 'custom', label: 'Custom', icon: 'edit' },
    ],
    defaultDataset: 'moons',
    params: [
      { type: 'slider', key: 'points', label: 'Points', min: 50, max: 800, step: 50, defaultValue: 300 },
      { type: 'slider', key: 'noise', label: 'Noise', min: 0.01, max: 2.0, step: 0.01, defaultValue: 0.05, formatValue: (v) => v.toFixed(2) },
    ],
  },

  codeSnippets: [
    {
      language: 'Python', lines: [
        { text: 'from sklearn.cluster import DBSCAN', highlights: [{ start: 0, end: 4, type: 'keyword' }, { start: 28, end: 34, type: 'keyword' }] },
        { text: '' },
        { text: "db = DBSCAN(eps=0.1, min_samples=4)", highlights: [{ start: 16, end: 19, type: 'number' }, { start: 33, end: 34, type: 'number' }] },
        { text: 'labels = db.fit_predict(X)' },
        { text: 'n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)' },
        { text: 'n_noise_ = list(labels).count(-1)' },
      ],
    },
    {
      language: 'JavaScript', lines: [
        { text: "import { DBSCAN } from 'omodels';", highlights: [{ start: 0, end: 6, type: 'keyword' }] },
        { text: '' },
        { text: 'const db = new DBSCAN({ eps: 0.1, minPts: 4 });', highlights: [{ start: 0, end: 5, type: 'keyword' }] },
        { text: 'const labels = db.fitPredict(data);' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'Clusters', value: '—', isPrimary: true },
    { label: 'Core Pts', value: '—' },
    { label: 'Border Pts', value: '—' },
    { label: 'Noise Pts', value: '—' },
    { label: '% Noise', value: '—' },
  ],

  VisualizationComponent: DBSCANVisualization,
};

export function registerDBSCAN() { registerModel(descriptor); }
