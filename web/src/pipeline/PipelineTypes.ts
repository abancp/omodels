/**
 * Pipeline Types — Core type system for ML Pipeline Builder
 */

// ─── Node Families ───────────────────────────────────────────────────────────

export type NodeFamily =
  | 'datasource'
  | 'preprocessor'
  | 'model'
  | 'ensemble'
  | 'evaluation'
  | 'output';

export type NodeState = 'idle' | 'training' | 'trained' | 'stale' | 'error';

export type PipelineState = 'draft' | 'validating' | 'training' | 'trained' | 'stale' | 'error';

// ─── Schema ───────────────────────────────────────────────────────────────────

export interface ColumnSchema {
  name: string;
  type: 'numeric' | 'categorical' | 'boolean';
}

export interface DataSchema {
  columns: ColumnSchema[];
  shape: [number, number]; // [rows, features]
  target?: string;
  taskType?: 'classification' | 'regression';
}

// ─── DataFrame — the unit of data flowing through edges ───────────────────────

export interface DataFrame {
  columns: string[];    // column names (ALL columns including target)
  rows: any[][];        // row × column matrix
}

// ─── Node Config Types ────────────────────────────────────────────────────────

export interface DataSourceConfig {
  datasetName: string;
  datasetSource: 'manual' | 'file' | 'datastudio';
  testSplit: number; // 0-1
  stratified: boolean;
  splitMode: 'holdout' | 'kfold';
  kFolds: number;
  targetColumn: string;
  taskType: 'classification' | 'regression' | 'auto';
}

export interface PreprocessorConfig {
  type: string;
  params: Record<string, unknown>;
}

export interface ModelConfig {
  type: string;
  source: 'new' | 'loaded';
  params: Record<string, unknown>;
  loadedModelId?: string;
  loadedModelFile?: string;
  mode?: 'frozen' | 'retrain' | 'finetune';
  fittedParams?: Record<string, unknown>;
  trainingSchema?: DataSchema;
}

export interface EnsembleConfig {
  type: string;
  params: Record<string, unknown>;
}

export interface EvaluationConfig {
  type: string;
  metrics?: string[];
}

export interface OutputConfig {
  type: string;
  filename?: string;
}

export type NodeConfig =
  | DataSourceConfig
  | PreprocessorConfig
  | ModelConfig
  | EnsembleConfig
  | EvaluationConfig
  | OutputConfig;

// ─── Pipeline Node ────────────────────────────────────────────────────────────

export interface PipelineNode {
  id: string;
  family: NodeFamily;
  type: string;
  label: string;
  config: NodeConfig;
  fittedParams?: Record<string, unknown>;
  state: NodeState;
  error?: string;
  // Layout
  position: { x: number; y: number };
  // Schema inference
  inputSchema?: DataSchema;
  outputSchema?: DataSchema;
  // Metrics (for evaluation nodes)
  metrics?: Record<string, number>;
}

// ─── Pipeline Edge ────────────────────────────────────────────────────────────

export interface PipelineEdge {
  id: string;
  source: string; // node id
  target: string; // node id
  sourceHandle?: string;
  targetHandle?: string;
}

// ─── Pipeline ─────────────────────────────────────────────────────────────────

export interface Pipeline {
  id: string;
  name: string;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  state: PipelineState;
  metrics?: Record<string, number>;
  schema?: DataSchema;
  createdAt: string;
  updatedAt: string;
  // Runtime data (not serialized)
  data?: {
    train: DataFrame;
    test: DataFrame;
  };
}

// ─── Node Catalog Entry ───────────────────────────────────────────────────────

export interface NodeCatalogEntry {
  family: NodeFamily;
  type: string;
  label: string;
  icon: string;
  description: string;
  color: string; // accent color
  defaultConfig: Partial<NodeConfig>;
  maxIncoming?: number;
  maxOutgoing?: number;
  minIncoming?: number;
}

// ─── Validation Result ────────────────────────────────────────────────────────

export interface ValidationError {
  nodeId?: string;
  edgeId?: string;
  message: string;
  type: 'error' | 'warning';
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

// ─── Data Snapshot (per-node preview) ────────────────────────────────────────

export interface DataSnapshot {
  shape: [number, number];
  sample: any[][];
  featureNames: string[];
}

// ─── Execution Result ─────────────────────────────────────────────────────────

export interface NodeExecutionResult {
  nodeId: string;
  success: boolean;
  fittedParams?: Record<string, unknown>;
  metrics?: Record<string, number>;
  outputSchema?: DataSchema;
  error?: string;
  predictions?: number[];
  probabilities?: number[][];
  inputSnapshot?: DataSnapshot;
  outputSnapshot?: DataSnapshot;
  testInputSnapshot?: DataSnapshot;
}

// ─── Drag state for node palette ─────────────────────────────────────────────

export interface DragNodeInfo {
  family: NodeFamily;
  type: string;
  label: string;
}
