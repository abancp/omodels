import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import KNNVisualization from './KNNVisualization';

const knnDescriptor: ModelDescriptor = {
  id: 'knn',
  name: 'K-Nearest Neighbors',
  shortName: 'KNN',
  vizLabel: 'Decision boundary',
  category: 'Classification',
  categoryIcon: 'category',

  params: [
    /* ─── Basic ─── */
    {
      type: 'slider',
      key: 'k',
      label: 'K Neighbors',
      min: 1,
      max: 15,
      step: 1,
      defaultValue: 5,
    },
    {
      type: 'select',
      key: 'metric',
      label: 'Distance Metric',
      options: [
        { value: 'euclidean', label: 'Euclidean' },
        { value: 'manhattan', label: 'Manhattan' },
        { value: 'chebyshev', label: 'Chebyshev' },
        { value: 'minkowski', label: 'Minkowski' },
      ],
      defaultValue: 'euclidean',
    },
    {
      type: 'chip-select',
      key: 'weights',
      label: 'Weights',
      options: [
        { value: 'uniform', label: 'Uniform' },
        { value: 'distance', label: 'Distance' },
      ],
      defaultValue: 'uniform',
    },
    {
      type: 'toggle',
      key: 'showBoundaries',
      label: 'Show boundaries',
      defaultValue: true,
    },
    {
      type: 'toggle',
      key: 'showNeighbors',
      label: 'Show neighbors',
      defaultValue: false,
    },

    /* ─── Advanced ─── */
    {
      type: 'select',
      key: 'algorithm',
      label: 'Algorithm',
      options: [
        { value: 'auto', label: 'Auto' },
        { value: 'ball_tree', label: 'Ball Tree' },
        { value: 'kd_tree', label: 'KD Tree' },
        { value: 'brute', label: 'Brute Force' },
      ],
      defaultValue: 'auto',
      level: 'advanced',
    },
    {
      type: 'slider',
      key: 'leafSize',
      label: 'Leaf Size',
      min: 10,
      max: 100,
      step: 5,
      defaultValue: 30,
      level: 'advanced',
    },
    {
      type: 'slider',
      key: 'p',
      label: 'P (Minkowski)',
      min: 1,
      max: 5,
      step: 0.5,
      defaultValue: 2,
      level: 'advanced',
    },
    {
      type: 'slider',
      key: 'boundaryRes',
      label: 'Boundary Resolution',
      min: 2,
      max: 12,
      step: 1,
      defaultValue: 6,
      level: 'advanced',
    },
    {
      type: 'toggle',
      key: 'showVoronoi',
      label: 'Show Voronoi edges',
      defaultValue: false,
      level: 'advanced',
    },
    {
      type: 'toggle',
      key: 'showCentroids',
      label: 'Show class centroids',
      defaultValue: false,
      level: 'advanced',
    },
  ],

  dataset: {
    options: [
      { value: 'blobs', label: 'Blobs' },
      { value: 'moons', label: 'Moons' },
      { value: 'circles', label: 'Circles' },
      { value: 'custom', label: 'Custom', icon: 'upload' },
      { value: 'import', label: 'Import', icon: 'upload' },
    ],
    defaultDataset: 'blobs',
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
        defaultValue: 0.2,
        formatValue: (v) => v.toFixed(2),
      },
    ],
  },

  codeSnippets: [
    {
      language: 'Python',
      lines: [
        {
          text: 'from sklearn.neighbors import KNeighborsClassifier',
          highlights: [
            { start: 0, end: 4, type: 'keyword' },
            { start: 37, end: 43, type: 'keyword' },
          ],
        },
        { text: '' },
        {
          text: 'model = KNeighborsClassifier(n_neighbors=5)',
          highlights: [{ start: 41, end: 42, type: 'number' }],
        },
        { text: 'model.fit(X_train, y_train)' },
      ],
    },
    {
      language: 'C++',
      lines: [
        { text: '#include "knn.h"', highlights: [{ start: 0, end: 8, type: 'keyword' }] },
        { text: '' },
        { text: 'auto knn = KNN(5, "euclidean");', highlights: [{ start: 15, end: 16, type: 'number' }] },
        { text: 'knn.fit(X_train, y_train);' },
      ],
    },
    {
      language: 'JavaScript',
      lines: [
        {
          text: "import { KNN } from 'omodels';",
          highlights: [
            { start: 0, end: 6, type: 'keyword' },
            { start: 15, end: 19, type: 'keyword' },
          ],
        },
        { text: '' },
        {
          text: "const model = new KNN({ k: 5, metric: 'euclidean' });",
          highlights: [
            { start: 0, end: 5, type: 'keyword' },
            { start: 27, end: 28, type: 'number' },
          ],
        },
        { text: 'model.fit(xTrain, yTrain);' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'Accuracy', value: '94.2%', isPrimary: true },
    { label: 'Precision', value: '91.7%' },
    { label: 'Recall', value: '93.1%' },
    { label: 'F1 Score', value: '92.4%' },
  ],

  VisualizationComponent: KNNVisualization,
};

export function registerKNN() {
  registerModel(knnDescriptor);
}
