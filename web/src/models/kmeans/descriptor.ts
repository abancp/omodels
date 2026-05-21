import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import KMeansVisualization from './KMeansVisualization';

const descriptor: ModelDescriptor = {
  id: 'kmeans',
  name: 'K-Means',
  shortName: 'KM',
  vizLabel: 'Cluster Map',
  category: 'Clustering',
  categoryIcon: 'hub',
  trainable: true,

  params: [
    { type: 'slider', key: 'k', label: 'Number of Clusters (K)', min: 2, max: 10, step: 1, defaultValue: 3, formatValue: (v) => String(v) },
    { type: 'slider', key: 'maxIter', label: 'Max Iterations', min: 5, max: 200, step: 5, defaultValue: 50, formatValue: (v) => String(v) },
    { type: 'select', key: 'initMethod', label: 'Initialization', options: [
      { value: 'kmeans++', label: 'K-Means++' },
      { value: 'random', label: 'Random' },
    ], defaultValue: 'kmeans++' },

    { type: 'toggle', key: 'showVoronoi', label: 'Show Voronoi Regions', defaultValue: true, level: 'advanced' },
    { type: 'toggle', key: 'showCentroidPath', label: 'Show Centroid Path', defaultValue: true, level: 'advanced' },
    { type: 'slider', key: 'elbowMaxK', label: 'Elbow Max K', min: 2, max: 12, step: 1, defaultValue: 8, formatValue: (v) => String(v), level: 'advanced' },
  ],

  dataset: {
    options: [
      { value: 'blobs', label: 'Blobs' }, { value: 'moons', label: 'Moons' },
      { value: 'circles', label: 'Circles' }, { value: 'uniform', label: 'Uniform' },
      { value: 'anisotropic', label: 'Anisotropic' }, { value: 'custom', label: 'Custom', icon: 'edit' },
      { value: 'import', label: 'Import', icon: 'upload' },
    ],
    defaultDataset: 'blobs',
    params: [
      { type: 'slider', key: 'points', label: 'Points', min: 20, max: 500, step: 10, defaultValue: 150 },
      { type: 'slider', key: 'noise', label: 'Noise', min: 0.1, max: 2.0, step: 0.1, defaultValue: 1.0, formatValue: (v) => v.toFixed(1) },
    ],
  },

  codeSnippets: [
    {
      language: 'Python', lines: [
        { text: 'from sklearn.cluster import KMeans', highlights: [{ start: 0, end: 4, type: 'keyword' }, { start: 28, end: 34, type: 'keyword' }] },
        { text: '' },
        { text: "km = KMeans(n_clusters=3, init='k-means++')", highlights: [{ start: 22, end: 23, type: 'number' }] },
        { text: 'km.fit(X)' },
        { text: 'labels = km.predict(X)' },
        { text: 'print(f"Inertia = {km.inertia_:.4f}")', highlights: [{ start: 0, end: 5, type: 'keyword' }] },
      ],
    },
    {
      language: 'JavaScript', lines: [
        { text: "import { KMeans } from 'omodels';", highlights: [{ start: 0, end: 6, type: 'keyword' }] },
        { text: '' },
        { text: 'const km = new KMeans({ k: 3 });', highlights: [{ start: 0, end: 5, type: 'keyword' }] },
        { text: 'km.fit(data);' },
        { text: 'const labels = km.predict(data);' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'Inertia', value: '—', isPrimary: true }, { label: 'Silhouette', value: '—' },
    { label: 'Iterations', value: '—' }, { label: 'K', value: '—' },
    { label: 'Converged', value: '—' }, { label: 'Clusters', value: '—' },
  ],

  VisualizationComponent: KMeansVisualization,
};

export function registerKMeans() { registerModel(descriptor); }
