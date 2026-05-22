/**
 * Preprocessor Engines — Pure JS implementations
 * Each engine: fit(X, y?) → void, transform(X) → X'
 */

function mean(arr: number[]): number {
  return arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr: number[], m: number): number {
  if (arr.length === 0) return 1;
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length + 1e-8);
}

export class StandardScalerEngine {
  means: number[] = [];
  stds: number[] = [];
  fit(X: number[][]) {
    const cols = X[0]?.length ?? 0;
    this.means = []; this.stds = [];
    for (let j = 0; j < cols; j++) {
      const col = X.map(r => r[j]);
      const m = mean(col);
      this.means.push(m);
      this.stds.push(std(col, m));
    }
  }
  transform(X: number[][]): number[][] {
    return X.map(row => row.map((v, j) => (v - this.means[j]) / (this.stds[j] || 1)));
  }
}

export class MinMaxScalerEngine {
  mins: number[] = [];
  maxs: number[] = [];
  fit(X: number[][]) {
    const cols = X[0]?.length ?? 0;
    this.mins = []; this.maxs = [];
    for (let j = 0; j < cols; j++) {
      const col = X.map(r => r[j]);
      this.mins.push(Math.min(...col));
      this.maxs.push(Math.max(...col));
    }
  }
  transform(X: number[][]): number[][] {
    return X.map(row => row.map((v, j) => {
      const range = this.maxs[j] - this.mins[j];
      return range === 0 ? 0 : (v - this.mins[j]) / range;
    }));
  }
}

export class SimpleImputerEngine {
  values: number[] = [];
  strategy: string;
  constructor(strategy = 'mean') { this.strategy = strategy; }
  fit(X: number[][]) {
    const cols = X[0]?.length ?? 0;
    this.values = [];
    for (let j = 0; j < cols; j++) {
      const col = X.map(r => r[j]).filter(v => !isNaN(v) && isFinite(v));
      if (this.strategy === 'median') {
        const sorted = [...col].sort((a, b) => a - b);
        this.values.push(sorted[Math.floor(sorted.length / 2)] ?? 0);
      } else {
        this.values.push(mean(col));
      }
    }
  }
  transform(X: number[][]): number[][] {
    return X.map(row => row.map((v, j) => (isNaN(v) || !isFinite(v)) ? this.values[j] : v));
  }
}

export class LabelEncoderEngine {
  classes: Record<number, Record<string, number>> = {};
  fit(X: any[][]) {
    if (X.length === 0) return;
    for (let j = 0; j < X[0].length; j++) {
      this.classes[j] = {};
      let id = 0;
      for (let i = 0; i < X.length; i++) {
        const val = String(X[i][j]);
        if (this.classes[j][val] === undefined) this.classes[j][val] = id++;
      }
    }
  }
  transform(X: any[][]): number[][] {
    return X.map(row => row.map((val, j) => {
      const strVal = String(val);
      return this.classes[j]?.[strVal] ?? 0;
    }));
  }
}

export class OneHotEncoderEngine {
  classes: Record<number, string[]> = {};
  dropFirst: boolean;
  constructor(dropFirst = false) { this.dropFirst = dropFirst; }
  fit(X: any[][]) {
    if (X.length === 0) return;
    for (let j = 0; j < X[0].length; j++) {
      const unique = new Set<string>();
      for (let i = 0; i < X.length; i++) unique.add(String(X[i][j]));
      let arr = Array.from(unique).sort();
      if (this.dropFirst && arr.length > 0) arr = arr.slice(1);
      this.classes[j] = arr;
    }
  }
  transform(X: any[][]): number[][] {
    return X.map(row => {
      const out: number[] = [];
      row.forEach((val, j) => {
        const strVal = String(val);
        (this.classes[j] || []).forEach(cat => out.push(strVal === cat ? 1 : 0));
      });
      return out;
    });
  }
}

export class DropColumnsEngine {
  fit(_X: any[][]) {
    // Structural preprocessor; no fitting needed.
  }
  transform(X: any[][]): number[][] {
    return X.map(() => []);
  }
}

