import { registerModel } from '../registry';
import type { ModelDescriptor } from '../registry';
import GBMVisualization from './GBMVisualization';

const descriptor: ModelDescriptor = {
  id: 'gradient-boosting',
  name: 'Gradient Boosting',
  shortName: 'GBM',
  vizLabel: 'Decision Boundary',
  category: 'Classification',
  categoryIcon: 'category',
  trainable: true,

  params: [
    { type: 'slider', key: 'nEstimators', label: 'Number of Stages', min: 1, max: 100, step: 1, defaultValue: 30, formatValue: (v) => String(v) },
    { type: 'slider', key: 'learningRate', label: 'Learning Rate', min: 0.01, max: 1.0, step: 0.01, defaultValue: 0.1, formatValue: (v) => v.toFixed(2) },
    { type: 'slider', key: 'maxDepth', label: 'Max Tree Depth', min: 1, max: 8, step: 1, defaultValue: 3, formatValue: (v) => String(v) },

    { type: 'slider', key: 'subsample', label: 'Subsample Ratio', min: 0.3, max: 1.0, step: 0.05, defaultValue: 0.8, formatValue: (v) => v.toFixed(2), level: 'advanced' },
    { type: 'slider', key: 'numBins', label: 'Feature Bins', min: 5, max: 40, step: 5, defaultValue: 15, formatValue: (v) => String(v), level: 'advanced' },
    { type: 'toggle', key: 'showStageSlider', label: 'Stage Slider', defaultValue: true, level: 'advanced' },
  ],

  dataset: {
    options: [
      { value: 'blobs', label: 'Blobs' }, { value: 'moons', label: 'Moons' },
      { value: 'circles', label: 'Circles' }, { value: 'xor', label: 'XOR' },
      { value: 'spiral', label: 'Spiral' }, { value: 'custom', label: 'Custom', icon: 'edit' },
      { value: 'import', label: 'Import', icon: 'upload' },
    ],
    defaultDataset: 'moons',
    params: [
      { type: 'slider', key: 'points', label: 'Points', min: 20, max: 500, step: 10, defaultValue: 150 },
      { type: 'slider', key: 'noise', label: 'Noise', min: 0, max: 1, step: 0.05, defaultValue: 0.2, formatValue: (v) => v.toFixed(2) },
    ],
  },

  codeSnippets: [
    {
      language: 'Python',
      lines: [
        { text: 'from sklearn.ensemble import GradientBoostingClassifier', highlights: [{ start: 0, end: 4, type: 'keyword' }, { start: 30, end: 36, type: 'keyword' }] },
        { text: '' },
        { text: 'clf = GradientBoostingClassifier(' },
        { text: '    n_estimators=100, learning_rate=0.1,', highlights: [{ start: 17, end: 20, type: 'number' }, { start: 37, end: 40, type: 'number' }] },
        { text: '    max_depth=3, subsample=0.8,', highlights: [{ start: 14, end: 15, type: 'number' }, { start: 27, end: 30, type: 'number' }] },
        { text: ')' },
        { text: 'clf.fit(X_train, y_train)' },
        { text: 'print(f"Accuracy = {clf.score(X_test, y_test):.4f}")', highlights: [{ start: 0, end: 5, type: 'keyword' }] },
      ],
    },
    {
      language: 'JavaScript',
      lines: [
        { text: "import { GradientBoosting } from 'omodels';", highlights: [{ start: 0, end: 6, type: 'keyword' }] },
        { text: '' },
        { text: 'const clf = new GradientBoosting({ nEstimators: 100 });', highlights: [{ start: 0, end: 5, type: 'keyword' }] },
        { text: 'clf.fit(xTrain, yTrain);' },
      ],
    },
  ],

  defaultMetrics: [
    { label: 'Accuracy', value: '—', isPrimary: true }, { label: 'Precision', value: '—' },
    { label: 'Recall', value: '—' }, { label: 'F1 Score', value: '—' },
    { label: 'Log Loss', value: '—' }, { label: 'Stages', value: '—' },
  ],

  VisualizationComponent: GBMVisualization,
};

export function registerGBM() { registerModel(descriptor); }
