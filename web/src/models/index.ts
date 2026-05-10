export { registerKNN } from './knn/descriptor';
export { registerLinearRegression } from './linear-regression/descriptor';
export { registerPolynomialRegression } from './polynomial-regression/descriptor';
export { registerLogisticRegression } from './logistic-regression/descriptor';
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
