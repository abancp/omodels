import Papa from 'papaparse';

/* ─── Types ─── */

export interface ParseResult {
  data: Record<string, any>[];
  columns: string[];
  numericColumns: string[];
  categoricalColumns: string[];
  fileName: string;
}

export type ModelDataType = 'regression' | 'classification' | 'clustering';

export interface ValidationResult {
  valid: boolean;
  error?: string;
  info: string;
}

export interface RegressionPoint { x: number; y: number; }
export interface ClassificationPoint { x: number; y: number; cls: number; }
export interface ClusteringPoint { x: number; y: number; }

export interface ImportMapping {
  x: string;
  y: string;
  label?: string; // for classification
}

/* ─── Engine ─── */

export const PlaygroundDataEngine = {

  /** Parse CSV / JSON / JSONL / TSV */
  async parseFile(file: File): Promise<ParseResult> {
    return new Promise((resolve, reject) => {
      const name = file.name.toLowerCase();

      if (name.endsWith('.json') || name.endsWith('.jsonl')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const content = e.target?.result as string;
            let data: any[];
            if (name.endsWith('.jsonl')) {
              data = content.split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
            } else {
              const parsed = JSON.parse(content);
              data = Array.isArray(parsed) ? parsed : [parsed];
            }
            resolve({ ...this.processData(data), fileName: file.name });
          } catch { reject(new Error('Failed to parse JSON')); }
        };
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsText(file);

      } else {
        // CSV / TSV
        Papa.parse(file, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (r) => {
            try {
              resolve({ ...this.processData(r.data as Record<string, any>[]), fileName: file.name });
            } catch (err: any) { reject(err); }
          },
          error: (err) => reject(new Error(err.message)),
        });
      }
    });
  },

  /** Identify columns and numeric types */
  processData(data: Record<string, any>[]): Omit<ParseResult, 'fileName'> {
    if (!data.length) throw new Error('Dataset is empty');
    const columns = Object.keys(data[0]);
    const numericColumns = columns.filter(col => {
      let numCount = 0;
      const samples = data.slice(0, Math.min(10, data.length));
      for (const row of samples) {
        const v = row[col];
        if (v == null || v === '') continue;
        if (typeof v === 'number' || !isNaN(Number(v))) numCount++;
      }
      return numCount >= samples.length * 0.7;
    });
    const categoricalColumns = columns.filter(col => !numericColumns.includes(col));
    return { data, columns, numericColumns, categoricalColumns };
  },

  /** Determine model data type from model id and params */
  getModelDataType(modelId: string, params?: Record<string, any>): ModelDataType {
    if (['linear-regression', 'polynomial-regression'].includes(modelId)) return 'regression';
    if (['mlp', 'perceptron'].includes(modelId) && params?.outAct === 'linear') return 'regression';
    if (['kmeans', 'dbscan', 'gmm'].includes(modelId)) return 'clustering';
    return 'classification';
  },

  /** Validate dataset compatibility with a model type */
  validate(result: ParseResult, modelType: ModelDataType): ValidationResult {
    const { numericColumns, data, columns } = result;
    const info = `${data.length} rows · ${columns.length} cols · ${numericColumns.length} numeric`;

    switch (modelType) {
      case 'regression':
        if (numericColumns.length < 2) return { valid: false, error: 'Regression needs at least 2 numeric columns (X, Y)', info };
        return { valid: true, info };

      case 'classification':
        if (numericColumns.length < 2) return { valid: false, error: 'Classification needs at least 2 numeric feature columns', info };
        // Label column can be numeric or categorical
        return { valid: true, info };

      case 'clustering':
        if (numericColumns.length < 2) return { valid: false, error: 'Clustering needs at least 2 numeric feature columns', info };
        return { valid: true, info };
    }
  },

  /** Auto-suggest default column mapping */
  suggestMapping(result: ParseResult, modelType: ModelDataType): ImportMapping {
    const { numericColumns, columns } = result;
    const mapping: ImportMapping = { x: numericColumns[0] || columns[0], y: numericColumns[1] || columns[1] };

    if (modelType === 'classification') {
      // Try to find a label/class/target column
      const labelCandidates = columns.filter(c =>
        /^(label|class|cls|target|category|species|type|output|y)$/i.test(c)
      );
      mapping.label = labelCandidates[0] || columns[columns.length - 1];
    }

    return mapping;
  },

  /** Convert parsed data to regression points */
  toRegressionPoints(data: Record<string, any>[], mapping: ImportMapping): RegressionPoint[] {
    return data
      .map(r => ({ x: Number(r[mapping.x]), y: Number(r[mapping.y]) }))
      .filter(p => isFinite(p.x) && isFinite(p.y));
  },

  /** Convert parsed data to classification points (x, y, cls as 0 or 1) */
  toClassificationPoints(data: Record<string, any>[], mapping: ImportMapping): ClassificationPoint[] {
    if (!mapping.label) return [];
    // Find unique labels and map to 0, 1, 2, ...
    const uniqueLabels = [...new Set(data.map(r => r[mapping.label!]))].filter(v => v != null);
    const labelMap = new Map(uniqueLabels.map((l, i) => [l, i]));
    return data
      .map(r => ({
        x: Number(r[mapping.x]),
        y: Number(r[mapping.y]),
        cls: labelMap.get(r[mapping.label!]) ?? 0,
      }))
      .filter(p => isFinite(p.x) && isFinite(p.y));
  },

  /** Convert parsed data to clustering points (just x, y) */
  toClusteringPoints(data: Record<string, any>[], mapping: ImportMapping): ClusteringPoint[] {
    return data
      .map(r => ({ x: Number(r[mapping.x]), y: Number(r[mapping.y]) }))
      .filter(p => isFinite(p.x) && isFinite(p.y));
  },

  /** Normalize points to 0-1 range for models that expect it */
  normalizePoints<T extends { x: number; y: number }>(pts: T[]): { points: T[]; stats: { xMin: number; xMax: number; yMin: number; yMax: number } } {
    if (pts.length === 0) return { points: pts, stats: { xMin: 0, xMax: 1, yMin: 0, yMax: 1 } };
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;
    const points = pts.map(p => ({ ...p, x: (p.x - xMin) / xRange, y: (p.y - yMin) / yRange }));
    return { points, stats: { xMin, xMax, yMin, yMax } };
  },

  /** One-hot encode a categorical column, returning expanded numeric columns */
  oneHotEncode(data: Record<string, any>[], col: string): { data: Record<string, any>[]; newCols: string[] } {
    const categories = [...new Set(data.map(r => String(r[col] ?? '')))].sort();
    const newCols = categories.map(c => `${col}_${c}`);
    const expanded = data.map(row => {
      const r = { ...row };
      for (const cat of categories) {
        r[`${col}_${cat}`] = String(row[col] ?? '') === cat ? 1 : 0;
      }
      delete r[col];
      return r;
    });
    return { data: expanded, newCols };
  },

  /** Label-encode a categorical column to integers 0..N-1 */
  labelEncode(data: Record<string, any>[], col: string): { data: Record<string, any>[]; labelMap: Record<string, number> } {
    const categories = [...new Set(data.map(r => String(r[col] ?? '')))].sort();
    const labelMap: Record<string, number> = {};
    categories.forEach((c, i) => { labelMap[c] = i; });
    const encoded = data.map(row => ({
      ...row,
      [col]: labelMap[String(row[col] ?? '')] ?? 0,
    }));
    return { data: encoded, labelMap };
  },
};
