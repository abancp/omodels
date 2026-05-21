/* ═══════════════════════════════════════
   Data Studio — Data Engine
   Parsing, Schema Detection, Stats, Transforms
   ═══════════════════════════════════════ */
import Papa from 'papaparse';
import type { Schema, ColumnSchema, ColumnStats, ColType } from './types';

/** Parse CSV or JSON */
export async function parseData(file: File | string): Promise<Record<string, any>[]> {
  return new Promise((resolve, reject) => {
    const config = {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results: any) => resolve(results.data),
      error: (err: any) => reject(err),
    };
    if (typeof file === 'string') Papa.parse(file, config);
    else Papa.parse(file, config);
  });
}

/** Detect Schema */
export function detectSchema(data: Record<string, any>[]): Schema {
  if (!data.length) return { columns: [], rowCount: 0, colCount: 0 };
  const keys = Object.keys(data[0]);
  const columns: ColumnSchema[] = keys.map(name => {
    const vals = data.map(r => r[name]).filter(v => v != null && v !== '');
    let type: ColType = 'unknown';
    if (vals.length > 0) {
      // Check multiple samples to be more robust
      const samples = vals.slice(0, 10);
      const types = samples.map(s => {
        if (typeof s === 'number') return 'numeric';
        if (typeof s === 'boolean') return 'boolean';
        const str = String(s).trim();
        if (str.toLowerCase() === 'true' || str.toLowerCase() === 'false') return 'boolean';
        if (!isNaN(Number(str)) && str.length > 0) return 'numeric';
        // Stricter date check: must contain separators like -, /, or :
        if ((str.includes('-') || str.includes('/') || str.includes(':')) && !isNaN(Date.parse(str))) return 'datetime';
        return 'categorical';
      });
      // Majority vote
      const counts: Record<string, number> = {};
      types.forEach(t => counts[t] = (counts[t] || 0) + 1);
      type = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0] as ColType;
    }
    return { name, type, uniqueCount: new Set(vals).size, nullCount: data.length - vals.length };
  });
  return { columns, rowCount: data.length, colCount: keys.length };
}

/** Compute Column Stats */
export function computeStats(data: Record<string, any>[], schema: Schema): Record<string, ColumnStats> {
  const stats: Record<string, ColumnStats> = {};
  schema.columns.forEach(col => {
    const vals = data.map(r => r[col.name]).filter(v => v != null && v !== '');
    if (col.type === 'numeric') {
      const nums = vals.map(v => Number(v)).filter(v => !isNaN(v)).sort((a, b) => a - b);
      if (!nums.length) return;
      const sum = nums.reduce((a, b) => a + b, 0);
      const mean = sum / nums.length;
      const median = nums[Math.floor(nums.length / 2)];
      const stdDev = Math.sqrt(nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / nums.length);
      const bins = 20;
      const min = nums[0], max = nums[nums.length - 1];
      const step = (max - min) / bins;
      const hist = Array.from({ length: bins }, (_, i) => ({
        bin: min + i * step,
        count: nums.filter(v => v >= min + i * step && v < (i === bins - 1 ? max + 1 : min + (i + 1) * step)).length,
      }));
      stats[col.name] = { 
        name: col.name, type: 'numeric', 
        stats: { min, max, mean, median, stdDev, q1: nums[Math.floor(nums.length * .25)], q3: nums[Math.floor(nums.length * .75)], skewness: 0, histogram: hist } 
      };
    } else {
      const counts: Record<string, number> = {};
      vals.forEach(v => { counts[String(v)] = (counts[String(v)] || 0) + 1; });
      const topN = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 20).map(([value, count]) => ({ value, count }));
      stats[col.name] = { name: col.name, type: 'categorical', stats: { uniqueCount: col.uniqueCount, mode: topN[0]?.value || '', topN } };
    }
  });
  return stats;
}

/** Apply Transform (The Core 14) */
export function applyTransform(data: Record<string, any>[], kind: string, params: any): Record<string, any>[] {
  let next = [...data];
  const col = params.column;

  switch (kind) {
    case 'drop-column': return next.map(r => { const row = { ...r }; delete row[col]; return row; });
    case 'rename-column': return next.map(r => { const row = { ...r }; row[params.to] = row[params.from]; delete row[params.from]; return row; });
    case 'drop-nulls': return next.filter(r => Object.values(r).every(v => v != null && v !== ''));
    case 'drop-duplicates': return Array.from(new Set(next.map(r => JSON.stringify(r)))).map(s => JSON.parse(s));
    case 'fill-nulls': {
      const fillVal = params.method === 'custom' ? params.value : 0; // simplified
      return next.map(r => ({ ...r, [col]: (r[col] == null || r[col] === '') ? fillVal : r[col] }));
    }
    case 'sort-by': {
      const { ascending } = params;
      return next.sort((a, b) => {
        const va = a[col], vb = b[col];
        if (va < vb) return ascending ? -1 : 1;
        if (va > vb) return ascending ? 1 : -1;
        return 0;
      });
    }
    case 'filter-rows': {
      const { op, value } = params;
      return next.filter(r => {
        const v = r[col];
        if (op === '==') return v == value;
        if (op === '!=') return v != value;
        if (op === '>') return v > value;
        if (op === '<') return v < value;
        if (op === 'contains') return String(v).includes(String(value));
        return true;
      });
    }
    case 'min-max-scale': {
      const vals = next.map(r => Number(r[col])).filter(v => !isNaN(v));
      const min = Math.min(...vals), max = Math.max(...vals), range = max - min || 1;
      return next.map(r => ({ ...r, [col]: !isNaN(Number(r[col])) ? (Number(r[col]) - min) / range : r[col] }));
    }
    case 'z-score': {
      const vals = next.map(r => Number(r[col])).filter(v => !isNaN(v));
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const std = Math.sqrt(vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / vals.length) || 1;
      return next.map(r => ({ ...r, [col]: !isNaN(Number(r[col])) ? (Number(r[col]) - mean) / std : r[col] }));
    }
    case 'log-transform': return next.map(r => ({ ...r, [col]: Math.log(Number(r[col]) + 1) }));
    case 'clip-outliers': {
      const vals = next.map(r => Number(r[col])).filter(v => !isNaN(v)).sort((a, b) => a - b);
      const q1 = vals[Math.floor(vals.length * 0.25)], q3 = vals[Math.floor(vals.length * 0.75)], iqr = q3 - q1;
      const lower = q1 - 1.5 * iqr, upper = q3 + 1.5 * iqr;
      return next.map(r => ({ ...r, [col]: Math.max(lower, Math.min(upper, Number(r[col]))) }));
    }
    case 'bin-numeric': {
      const bins = params.bins || 5;
      const vals = next.map(r => Number(r[col])).filter(v => !isNaN(v));
      const min = Math.min(...vals), max = Math.max(...vals), step = (max - min) / bins;
      return next.map(r => ({ ...r, [col]: `Bin ${Math.floor((Number(r[col]) - min) / step)}` }));
    }
    case 'one-hot-encode': {
      const unique = Array.from(new Set(next.map(r => r[col])));
      return next.map(r => {
        const row = { ...r };
        unique.forEach(val => { row[`${col}_${val}`] = r[col] === val ? 1 : 0; });
        delete row[col];
        return row;
      });
    }
    case 'label-encode': {
      const mapping: Record<string, number> = {};
      Array.from(new Set(next.map(r => r[col]))).forEach((v, i) => mapping[String(v)] = i);
      return next.map(r => ({ ...r, [col]: mapping[String(r[col])] }));
    }
    default: return next;
  }
}

export function computeCorrelationMatrix(data: Record<string, any>[], cols: string[]) {
  const n = cols.length;
  const matrix = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) { matrix[i][j] = 1; continue; }
      const x = data.map(r => Number(r[cols[i]])).filter(v => !isNaN(v));
      const y = data.map(r => Number(r[cols[j]])).filter(v => !isNaN(v));
      const mx = x.reduce((a, b) => a + b, 0) / x.length;
      const my = y.reduce((a, b) => a + b, 0) / y.length;
      const num = x.reduce((a, v, k) => a + (v - mx) * (y[k] - my), 0);
      const den = Math.sqrt(x.reduce((a, v) => a + Math.pow(v - mx, 2), 0) * y.reduce((a, v) => a + Math.pow(v - my, 2), 0));
      matrix[i][j] = den === 0 ? 0 : num / den;
    }
  }
  return { matrix, cols };
}
