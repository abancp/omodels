export { registerKNN } from './knn/descriptor';
export { registerLinearRegression } from './linear-regression/descriptor';
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
  MetricValue,
  VisualizationProps,
} from './registry';
