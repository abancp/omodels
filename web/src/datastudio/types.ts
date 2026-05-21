/* ═══════════════════════════════════════
   Data Studio — Core Types
   ═══════════════════════════════════════ */

export type ColType = 'numeric' | 'categorical' | 'boolean' | 'datetime' | 'unknown';

export interface ColumnSchema {
  name: string;
  type: ColType;
  uniqueCount: number;
  nullCount: number;
}

export interface Schema {
  columns: ColumnSchema[];
  rowCount: number;
  colCount: number;
}

export interface NumericStats {
  min: number;
  max: number;
  mean: number;
  median: number;
  stdDev: number;
  q1: number;
  q3: number;
  skewness: number;
  histogram: { bin: number; count: number }[];
}

export interface CategoricalStats {
  uniqueCount: number;
  mode: string;
  topN: { value: string; count: number }[];
}

export interface ColumnStats {
  name: string;
  type: ColType;
  stats: NumericStats | CategoricalStats;
}

export type TransformKind = 
  | 'drop-column' 
  | 'rename-column' 
  | 'drop-nulls' 
  | 'fill-nulls' 
  | 'min-max-scale' 
  | 'z-score' 
  | 'log-transform' 
  | 'one-hot-encode' 
  | 'label-encode' 
  | 'drop-duplicates' 
  | 'clip-outliers' 
  | 'bin-numeric' 
  | 'sort-by' 
  | 'filter-rows';

export interface Transform {
  id: string;
  kind: TransformKind;
  params: Record<string, any>;
  label: string;
  active: boolean;
}

export interface SplitConfig {
  trainRatio: number;
  testRatio: number;
  valRatio: number;
  stratifyColumn?: string;
  shuffle: boolean;
  seed: number;
}

export type VizType = 'histogram' | 'scatter' | 'correlation' | 'box-plot' | 'bar-chart' | 'missing-map';

export interface DataState {
  raw: Record<string, any>[];
  working: Record<string, any>[];
  schema: Schema | null;
  columnStats: Record<string, ColumnStats>;
  transforms: Transform[];
  splitConfig: SplitConfig;
  split: { train: any[]; test: any[]; val?: any[] } | null;
  fileName: string;
  source: string;
  activePanel: 'transform' | 'visualize' | 'split' | 'schema' | 'edit';
  activeViz: VizType;
  selectedColumn: string | null;
  loading: { active: boolean; message: string };
  error: string | null;
}
