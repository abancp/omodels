/**
 * Pure JS Model Engines — KNN, DecisionTree, RandomForest, SVM, NaiveBayes, KMeans
 * Each engine: fit(X, y) → void, predict(X) → number[]
 */

function mean(arr: number[]): number {
  return arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;
}
function std(arr: number[], m: number): number {
  if (arr.length === 0) return 1;
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length + 1e-8);
}

export class KNNEngine {
  X_train: number[][] = []; y_train: number[] = []; k: number; isRegression: boolean;
  constructor(k = 5, isRegression = false) { this.k = k; this.isRegression = isRegression; }
  fit(X: number[][], y: number[]) { this.X_train = X; this.y_train = y; }
  predict(X: number[][]): number[] {
    return X.map(row => {
      const dists = this.X_train.map((tr, i) => ({
        dist: Math.sqrt(tr.reduce((s, v, j) => s + (v - row[j]) ** 2, 0)),
        label: this.y_train[i],
      })).sort((a, b) => a.dist - b.dist).slice(0, this.k);
      if (this.isRegression) return dists.reduce((s, d) => s + d.label, 0) / dists.length;
      const votes: Record<number, number> = {};
      dists.forEach(d => { votes[d.label] = (votes[d.label] || 0) + 1; });
      return +Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
    });
  }
}

export class DecisionTreeEngine {
  maxDepth: number; minSamplesSplit: number; isRegression: boolean; criterion: string;
  root: any = null;
  constructor(maxDepth = 5, minSamples = 2, isRegression = false, criterion = 'gini') {
    this.maxDepth = maxDepth; this.minSamplesSplit = minSamples;
    this.isRegression = isRegression; this.criterion = criterion;
  }
  fit(X: number[][], y: number[]) { this.root = this._build(X, y, 0); }
  predict(X: number[][]): number[] {
    if (!this.root) return Array(X.length).fill(0);
    return X.map(row => this._predictRow(row, this.root));
  }
  private _predictRow(row: number[], node: any): number {
    if (node.isLeaf) return node.value;
    return row[node.featIdx] <= node.threshold
      ? this._predictRow(row, node.left) : this._predictRow(row, node.right);
  }
  private _build(X: number[][], y: number[], depth: number): any {
    const n = X.length;
    if (n === 0) return null;
    if (y.every(v => v === y[0]) || depth >= this.maxDepth || n < this.minSamplesSplit) {
      return { isLeaf: true, value: this.isRegression ? mean(y) : this._majority(y), samples: n };
    }
    let bestGain = -1, bestCrit: any = null, bestSplit: any = null;
    for (let f = 0; f < X[0].length; f++) {
      const vals = [...new Set(X.map(r => r[f]))].sort((a, b) => a - b);
      const cands = vals.length > 15 ? Array.from({ length: 15 }, (_, i) => vals[Math.floor(i * vals.length / 15)]) : vals;
      for (const th of cands) {
        const li: number[] = [], ri: number[] = [];
        for (let i = 0; i < n; i++) { (X[i][f] <= th ? li : ri).push(i); }
        if (!li.length || !ri.length) continue;
        const gain = this._gain(y, li, ri);
        if (gain > bestGain) { bestGain = gain; bestCrit = { featIdx: f, threshold: th }; bestSplit = { li, ri }; }
      }
    }
    if (bestGain <= 0 || !bestCrit || !bestSplit)
      return { isLeaf: true, value: this.isRegression ? mean(y) : this._majority(y), samples: n };
    return {
      isLeaf: false, ...bestCrit, samples: n,
      left: this._build(bestSplit.li.map((i: number) => X[i]), bestSplit.li.map((i: number) => y[i]), depth + 1),
      right: this._build(bestSplit.ri.map((i: number) => X[i]), bestSplit.ri.map((i: number) => y[i]), depth + 1),
    };
  }
  private _gain(y: number[], li: number[], ri: number[]): number {
    const pI = this._impurity(y, Array.from({ length: y.length }, (_, i) => i));
    return pI - (li.length / y.length) * this._impurity(y, li) - (ri.length / y.length) * this._impurity(y, ri);
  }
  private _impurity(y: number[], idxs: number[]): number {
    const n = idxs.length; if (!n) return 0;
    const sub = idxs.map(i => y[i]);
    if (this.isRegression) { const avg = mean(sub); return sub.reduce((a, v) => a + (v - avg) ** 2, 0) / n; }
    const counts: Record<number, number> = {};
    sub.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
    if (this.criterion === 'gini') { let s = 0; for (const c of Object.values(counts)) { s += (c / n) ** 2; } return 1 - s; }
    let e = 0; for (const c of Object.values(counts)) { const p = c / n; e -= p * Math.log2(p); } return e;
  }
  private _majority(y: number[]): number {
    const c: Record<number, number> = {};
    y.forEach(v => { c[v] = (c[v] || 0) + 1; });
    return +Object.entries(c).sort((a, b) => b[1] - a[1])[0][0];
  }
}

export class RandomForestEngine {
  nEstimators: number; maxDepth: number; isRegression: boolean;
  trees: DecisionTreeEngine[] = [];
  constructor(nEst = 20, maxDepth = 5, isRegression = false) {
    this.nEstimators = nEst; this.maxDepth = maxDepth; this.isRegression = isRegression;
  }
  fit(X: number[][], y: number[]) {
    this.trees = [];
    const n = X.length; if (!n) return;
    for (let i = 0; i < this.nEstimators; i++) {
      const bX: number[][] = [], bY: number[] = [];
      for (let j = 0; j < n; j++) { const idx = Math.floor(Math.random() * n); bX.push(X[idx]); bY.push(y[idx]); }
      const tree = new DecisionTreeEngine(this.maxDepth, 2, this.isRegression);
      tree.fit(bX, bY); this.trees.push(tree);
    }
  }
  predict(X: number[][]): number[] {
    if (!this.trees.length) return Array(X.length).fill(0);
    const all = this.trees.map(t => t.predict(X));
    return X.map((_, i) => {
      const preds = all.map(p => p[i] ?? 0);
      if (this.isRegression) return mean(preds);
      const c: Record<number, number> = {};
      preds.forEach(p => { c[p] = (c[p] || 0) + 1; });
      return +Object.entries(c).sort((a, b) => b[1] - a[1])[0][0];
    });
  }
}

export class SVMEngine {
  C: number; weights: number[] = []; bias = 0;
  constructor(C = 1) { this.C = C; }
  fit(X: number[][], y: number[]) {
    const n = X.length; if (!n) return;
    const m = X[0].length;
    this.weights = Array(m).fill(0); this.bias = 0;
    const ySvm = y.map(v => v === 0 ? -1 : v);
    const lambda = 1 / (this.C * n); let t = 0;
    for (let ep = 0; ep < 100; ep++) {
      for (let i = 0; i < n; i++) {
        t++;
        const eta = 1 / (lambda * t);
        const score = X[i].reduce((s, v, j) => s + v * this.weights[j], this.bias);
        if (ySvm[i] * score < 1) {
          for (let j = 0; j < m; j++) this.weights[j] = (1 - eta * lambda) * this.weights[j] + eta * ySvm[i] * X[i][j];
          this.bias += eta * ySvm[i];
        } else {
          for (let j = 0; j < m; j++) this.weights[j] = (1 - eta * lambda) * this.weights[j];
        }
      }
    }
  }
  predict(X: number[][]): number[] {
    return X.map(row => row.reduce((s, v, j) => s + v * this.weights[j], this.bias) >= 0 ? 1 : 0);
  }
}

export class NaiveBayesEngine {
  classes: number[] = [];
  priors: Record<number, number> = {};
  means: Record<number, number[]> = {};
  vars: Record<number, number[]> = {};
  fit(X: number[][], y: number[]) {
    this.classes = [...new Set(y)].sort((a, b) => a - b);
    const n = X.length; if (!n) return;
    this.classes.forEach(c => {
      const rows = X.filter((_, i) => y[i] === c);
      this.priors[c] = rows.length / n;
      const nF = X[0].length;
      const cM = Array(nF).fill(0), cV = Array(nF).fill(0);
      for (let j = 0; j < nF; j++) {
        const col = rows.map(r => r[j]);
        const m = mean(col); cM[j] = m; cV[j] = std(col, m) ** 2 + 1e-9;
      }
      this.means[c] = cM; this.vars[c] = cV;
    });
  }
  predict(X: number[][]): number[] {
    if (!this.classes.length) return Array(X.length).fill(0);
    return X.map(row => {
      let best = this.classes[0], maxLP = -Infinity;
      this.classes.forEach(c => {
        let lp = Math.log(this.priors[c] || 1e-5);
        for (let j = 0; j < row.length; j++) {
          const m = this.means[c]?.[j] ?? 0, v = this.vars[c]?.[j] ?? 1e-9;
          lp += -0.5 * Math.log(2 * Math.PI * v) - 0.5 * ((row[j] - m) ** 2) / v;
        }
        if (lp > maxLP) { maxLP = lp; best = c; }
      });
      return best;
    });
  }
}

export class KMeansEngine {
  nClusters: number; maxIter: number; centroids: number[][] = [];
  constructor(k = 3, maxIter = 50) { this.nClusters = k; this.maxIter = maxIter; }
  fit(X: number[][], _y?: number[]) {
    const n = X.length; if (!n) return;
    const d = X[0].length;
    this.centroids = [];
    const used = new Set<number>();
    while (this.centroids.length < Math.min(this.nClusters, n)) {
      const idx = Math.floor(Math.random() * n);
      if (!used.has(idx)) { used.add(idx); this.centroids.push([...X[idx]]); }
    }
    for (let iter = 0; iter < this.maxIter; iter++) {
      const clusters: number[][] = Array.from({ length: this.centroids.length }, () => []);
      for (let i = 0; i < n; i++) {
        let best = 0, minD = Infinity;
        for (let c = 0; c < this.centroids.length; c++) {
          let dist = 0;
          for (let j = 0; j < d; j++) dist += (X[i][j] - this.centroids[c][j]) ** 2;
          if (dist < minD) { minD = dist; best = c; }
        }
        clusters[best].push(i);
      }
      let maxShift = 0;
      const next: number[][] = [];
      for (let c = 0; c < this.centroids.length; c++) {
        if (!clusters[c].length) { next.push([...this.centroids[c]]); continue; }
        const nc = Array(d).fill(0);
        for (const idx of clusters[c]) for (let j = 0; j < d; j++) nc[j] += X[idx][j];
        for (let j = 0; j < d; j++) nc[j] /= clusters[c].length;
        let shift = 0;
        for (let j = 0; j < d; j++) shift += (nc[j] - this.centroids[c][j]) ** 2;
        maxShift = Math.max(maxShift, shift);
        next.push(nc);
      }
      this.centroids = next;
      if (maxShift < 1e-6) break;
    }
  }
  predict(X: number[][]): number[] {
    if (!this.centroids.length) return Array(X.length).fill(0);
    return X.map(row => {
      let best = 0, minD = Infinity;
      for (let c = 0; c < this.centroids.length; c++) {
        let dist = 0;
        for (let j = 0; j < row.length; j++) dist += (row[j] - this.centroids[c][j]) ** 2;
        if (dist < minD) { minD = dist; best = c; }
      }
      return best;
    });
  }
}
