/**
 * Pipeline Types — Core type system for ML Pipeline Builder
 * 
 * Architecture: Each node is fully isolated. Data flows only through
 * explicit connections via a standard DataPacket protocol.
 */

// ─── Port Types ──────────────────────────────────────────────────────────────

/** Port types determine what kind of data flows through connections */
export type PortType = 'data' | 'predictions';

export interface PortDefinition {
  type: PortType;
  label: string;
  multiple?: boolean; // allow multiple connections to this port
}

// ─── Node Families ───────────────────────────────────────────────────────────

export type NodeFamily =
  | 'dataset'
  | 'split'
  | 'preprocessor'
  | 'model'
  | 'evaluation'
  | 'output';

export type NodeState = 'idle' | 'training' | 'trained' | 'stale' | 'error';

export type PipelineState = 'draft' | 'validating' | 'training' | 'trained' | 'stale' | 'error';

// ─── Schema ──────────────────────────────────────────────────────────────────

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

// ─── DataFrame — the raw data unit ───────────────────────────────────────────

export interface DataFrame {
  columns: string[];    // column names (ALL columns including target)
  rows: any[][];        // row × column matrix
}

// ─── DataPacket — the standard unit flowing through edges ────────────────────

export interface DataPacket {
  train: DataFrame;
  test: DataFrame;
  meta: DataPacketMeta;
  /** Set by model nodes after training */
  predictions?: number[];
  /** Set by model nodes that support probability output */
  probabilities?: number[][];
}

export interface DataPacketMeta {
  targetColumn: string;
  taskType: 'classification' | 'regression' | 'auto';
  featureColumns: string[];
  schema: DataSchema;
}

// ─── Connection Rules ────────────────────────────────────────────────────────

/** Defines which families a node family can accept input from */
export const CONNECTION_RULES: Record<NodeFamily, NodeFamily[]> = {
  dataset:      [],                                         // no inputs
  split:        ['dataset'],                                // only from dataset
  preprocessor: ['dataset', 'split', 'preprocessor'],      // data chain
  model:        ['dataset', 'split', 'preprocessor'],      // data in
  evaluation:   ['model'],                                  // predictions in
  output:       ['model'],                                  // predictions in
};

/** Defines which families a node family can send output to */
export const OUTPUT_RULES: Record<NodeFamily, NodeFamily[]> = {
  dataset:      ['split', 'preprocessor', 'model'],
  split:        ['preprocessor', 'model'],
  preprocessor: ['preprocessor', 'model'],
  model:        ['evaluation', 'output'],
  evaluation:   [],                                         // terminal
  output:       [],                                         // terminal
};

// ─── Node Config Types ───────────────────────────────────────────────────────

export interface DatasetConfig {
  targetColumn: string;
  taskType: 'classification' | 'regression' | 'auto';
  /** CSV data stored per-node — each Dataset node owns its own data */
  csvText?: string;
  csvFilename?: string;
  csvHeaders?: string[];
  csvRowCount?: number;
}

export interface SplitConfig {
  testSplit: number;
  stratified: boolean;
  shuffle: boolean;
}

export interface PreprocessorConfig {
  type: string;
  params: Record<string, unknown>;
  /** Optional: specific columns to transform (empty = all feature columns) */
  targetColumns?: string[];
}

export interface ModelConfig {
  type: string;
  params: Record<string, unknown>;
}

export interface EvaluationConfig {
  type: string;
}

export interface OutputConfig {
  type: string;
  filename?: string;
}

export type NodeConfig =
  | DatasetConfig
  | SplitConfig
  | PreprocessorConfig
  | ModelConfig
  | EvaluationConfig
  | OutputConfig;

// ─── Pipeline Node ───────────────────────────────────────────────────────────

export interface PipelineNode {
  id: string;
  family: NodeFamily;
  type: string;
  label: string;
  config: NodeConfig;
  state: NodeState;
  error?: string;
  // Layout
  position: { x: number; y: number };
  // Fitted state from training
  fittedParams?: Record<string, unknown>;
  // Metrics (for evaluation nodes)
  metrics?: Record<string, number>;
  // Schema inferred during execution
  inputSchema?: DataSchema;
  outputSchema?: DataSchema;
}

// ─── Pipeline Edge ───────────────────────────────────────────────────────────

export interface PipelineEdge {
  id: string;
  source: string; // node id
  target: string; // node id
  sourceHandle?: string;
  targetHandle?: string;
}

// ─── Pipeline ────────────────────────────────────────────────────────────────

export interface Pipeline {
  id: string;
  name: string;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  state: PipelineState;
  metrics?: Record<string, number>;
  createdAt: string;
  updatedAt: string;
}

// ─── Node Catalog Entry ──────────────────────────────────────────────────────

export interface NodeCatalogEntry {
  family: NodeFamily;
  type: string;
  label: string;
  icon: string;
  description: string;
  color: string;
  defaultConfig: Partial<NodeConfig>;
  /** Which families can connect TO this node */
  acceptsFrom: NodeFamily[];
  /** Which families this node can connect TO */
  outputsTo: NodeFamily[];
  /** Max incoming connections (0 = none, undefined = 1) */
  maxIncoming?: number;
  /** Max outgoing connections (0 = none, undefined = unlimited) */
  maxOutgoing?: number;
}

// ─── Validation ──────────────────────────────────────────────────────────────

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

// ─── Execution Result ────────────────────────────────────────────────────────

export interface NodeExecutionResult {
  nodeId: string;
  success: boolean;
  fittedParams?: any;
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
