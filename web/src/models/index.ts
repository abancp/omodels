export { registerKNN } from './knn/descriptor';
export { registerLinearRegression } from './linear-regression/descriptor';
export { registerPolynomialRegression } from './polynomial-regression/descriptor';
export { registerLogisticRegression } from './logistic-regression/descriptor';
export { registerSVM } from './svm/descriptor';
export { registerNaiveBayes } from './naive-bayes/descriptor';
export { registerDecisionTree } from './decision-tree/descriptor';
export { registerRandomForest } from './random-forest/descriptor';
export { registerGBM } from './gradient-boosting/descriptor';
export { registerKMeans } from './kmeans/descriptor';
export { registerDBSCAN } from './dbscan/descriptor';
export { registerGMM } from './gmm/descriptor';
export { registerPerceptron } from './perceptron/descriptor';
export { registerMLP } from './mlp/descriptor';
export { registerActivations } from './activations/descriptor';
export {
  registerModel,
  getModel,
  getAllModels,
  getCategories,
  getModelsByCategory,
} from './registry';
export type {
  ModelDescriptor,
  ParamDescriptor,
  ParamLevel,
  SliderParam,
  SelectParam,
  ToggleParam,
  ChipSelectParam,
  CodeSnippet,
  CodeLine,
  MetricValue,
  VisualizationProps,
} from './registry';
