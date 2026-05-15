/**
 * K-Means Clustering Math Engine
 * Supports K-Means, K-Means++ initialization, iterative Lloyd's algorithm.
 * Computes inertia, silhouette score, cluster assignments, and centroid history.
 */

import type { MetricValue } from '../registry';

export interface Point { x: number; y: number; }
export interface LabeledPoint extends Point { cluster: number; }

/* ─── K-Means State ─── */
export interface KMeansState {
  centroids: Point[];
  assignments: number[];
  k: number;
  iteration: number;
  converged: boolean;
  inertia: number;
  silhouetteScore: number;
  inertiaHistory: number[];
  centroidHistory: Point[][]; // centroids at each iteration
  clusterSizes: number[];
}

/* ─── Seeded PRNG ─── */
function seededRandom(seed: number) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}

/* ─── Generate clustering datasets ─── */
export function generateClusteringData(type: string, n: number, noise: number): Point[] {
  const rng = seededRandom(42);
  const gauss = () => { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v); };

  switch (type) {
    case 'blobs': {
      const centers = [[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]];
      return Array.from({ length: n }, () => {
        const c = centers[Math.floor(rng() * centers.length)];
        return { x: c[0] + gauss() * noise * 0.15, y: c[1] + gauss() * noise * 0.15 };
      });
    }
    case 'moons': {
      return Array.from({ length: n }, (_, i) => {
        const isMoon2 = i >= n / 2;
        const angle = (i % (n / 2)) / (n / 2) * Math.PI;
        const cx = isMoon2 ? 0.5 + 0.3 * Math.cos(angle) : 0.5 - 0.3 * Math.cos(angle);
        const cy = isMoon2 ? 0.55 - 0.3 * Math.sin(angle) : 0.45 + 0.3 * Math.sin(angle);
        return { x: cx + gauss() * noise * 0.08, y: cy + gauss() * noise * 0.08 };
      });
    }
    case 'circles': {
      return Array.from({ length: n }, (_, i) => {
        const isOuter = i >= n / 2;
        const angle = rng() * Math.PI * 2;
        const r = isOuter ? 0.35 : 0.15;
        return { x: 0.5 + r * Math.cos(angle) + gauss() * noise * 0.04, y: 0.5 + r * Math.sin(angle) + gauss() * noise * 0.04 };
      });
    }
    case 'uniform': {
      return Array.from({ length: n }, () => ({ x: rng(), y: rng() }));
    }
    case 'anisotropic': {
      const centers = [[0.3, 0.3], [0.7, 0.7], [0.3, 0.7]];
      const stretches = [[2, 0.5], [0.5, 2], [1.5, 1]];
      return Array.from({ length: n }, () => {
        const ci = Math.floor(rng() * 3);
        const c = centers[ci]; const s = stretches[ci];
        return { x: c[0] + gauss() * noise * 0.1 * s[0], y: c[1] + gauss() * noise * 0.1 * s[1] };
      });
    }
    default: return generateClusteringData('blobs', n, noise);
  }
}

/* ─── Distance ─── */
function dist2(a: Point, b: Point): number { return (a.x - b.x) ** 2 + (a.y - b.y) ** 2; }
function dist(a: Point, b: Point): number { return Math.sqrt(dist2(a, b)); }

/* ─── K-Means++ initialization ─── */
function kMeansPPInit(points: Point[], k: number, rng: () => number): Point[] {
  const n = points.length;
  const centroids: Point[] = [];
  // Pick first centroid randomly
  centroids.push({ ...points[Math.floor(rng() * n)] });

  for (let c = 1; c < k; c++) {
    // Compute D(x)^2 for each point
    const dists = points.map(p => {
      let minD = Infinity;
      for (const cent of centroids) minD = Math.min(minD, dist2(p, cent));
      return minD;
    });
    const totalD = dists.reduce((a, b) => a + b, 0);
    if (totalD === 0) { centroids.push({ ...points[Math.floor(rng() * n)] }); continue; }
    // Weighted random sampling
    let r = rng() * totalD, cumSum = 0;
    for (let i = 0; i < n; i++) {
      cumSum += dists[i];
      if (cumSum >= r) { centroids.push({ ...points[i] }); break; }
    }
    if (centroids.length <= c) centroids.push({ ...points[Math.floor(rng() * n)] });
  }
  return centroids;
}

/* ─── Random initialization ─── */
function randomInit(points: Point[], k: number, rng: () => number): Point[] {
  const shuffled = [...points].sort(() => rng() - 0.5);
  return shuffled.slice(0, k).map(p => ({ ...p }));
}

/* ─── Assign clusters ─── */
function assignClusters(points: Point[], centroids: Point[]): number[] {
  return points.map(p => {
    let minD = Infinity, best = 0;
    for (let c = 0; c < centroids.length; c++) {
      const d = dist2(p, centroids[c]);
      if (d < minD) { minD = d; best = c; }
    }
    return best;
  });
}

/* ─── Update centroids ─── */
function updateCentroids(points: Point[], assignments: number[], k: number): Point[] {
  const sums = Array.from({ length: k }, () => ({ x: 0, y: 0, count: 0 }));
  for (let i = 0; i < points.length; i++) {
    const c = assignments[i];
    sums[c].x += points[i].x; sums[c].y += points[i].y; sums[c].count++;
  }
  return sums.map((s) => s.count > 0 ? { x: s.x / s.count, y: s.y / s.count } : { x: 0.5, y: 0.5 });
}

/* ─── Compute inertia ─── */
function computeInertia(points: Point[], assignments: number[], centroids: Point[]): number {
  let inertia = 0;
  for (let i = 0; i < points.length; i++) inertia += dist2(points[i], centroids[assignments[i]]);
  return inertia;
}

/* ─── Silhouette score ─── */
function computeSilhouette(points: Point[], assignments: number[], k: number): number {
  if (k < 2 || points.length < k) return 0;
  const n = points.length;
  let totalSil = 0;
  for (let i = 0; i < n; i++) {
    const ci = assignments[i];
    // a(i) = avg distance to same cluster
    let aSum = 0, aCount = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i && assignments[j] === ci) { aSum += dist(points[i], points[j]); aCount++; }
    }
    const a = aCount > 0 ? aSum / aCount : 0;
    // b(i) = min avg distance to other clusters
    let minB = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === ci) continue;
      let bSum = 0, bCount = 0;
      for (let j = 0; j < n; j++) {
        if (assignments[j] === c) { bSum += dist(points[i], points[j]); bCount++; }
      }
      if (bCount > 0) minB = Math.min(minB, bSum / bCount);
    }
    if (minB === Infinity) minB = 0;
    const s = Math.max(a, minB) > 0 ? (minB - a) / Math.max(a, minB) : 0;
    totalSil += s;
  }
  return totalSil / n;
}

/* ─── Run single iteration of K-Means ─── */
export function kMeansStep(points: Point[], state: KMeansState): KMeansState {
  const newAssignments = assignClusters(points, state.centroids);
  const newCentroids = updateCentroids(points, newAssignments, state.k);
  const inertia = computeInertia(points, newAssignments, newCentroids);
  const converged = state.centroids.every((c, i) => dist2(c, newCentroids[i]) < 1e-10);
  const clusterSizes = Array.from({ length: state.k }, (_, c) => newAssignments.filter(a => a === c).length);

  return {
    centroids: newCentroids,
    assignments: newAssignments,
    k: state.k,
    iteration: state.iteration + 1,
    converged,
    inertia,
    silhouetteScore: state.k >= 2 ? computeSilhouette(points, newAssignments, state.k) : 0,
    inertiaHistory: [...state.inertiaHistory, inertia],
    centroidHistory: [...state.centroidHistory, newCentroids.map(c => ({ ...c }))],
    clusterSizes,
  };
}

/* ─── Train K-Means to convergence ─── */
export function trainKMeans(
  points: Point[], k: number, maxIter: number, initMethod: string
): KMeansState {
  if (points.length < k || k < 1) {
    return { centroids: [], assignments: [], k, iteration: 0, converged: true, inertia: 0, silhouetteScore: 0, inertiaHistory: [], centroidHistory: [], clusterSizes: [] };
  }

  const rng = seededRandom(42 + points.length + k);
  const initCentroids = initMethod === 'kmeans++' ? kMeansPPInit(points, k, rng) : randomInit(points, k, rng);
  const initAssignments = assignClusters(points, initCentroids);
  const initInertia = computeInertia(points, initAssignments, initCentroids);
  const clusterSizes = Array.from({ length: k }, (_, c) => initAssignments.filter(a => a === c).length);

  let state: KMeansState = {
    centroids: initCentroids, assignments: initAssignments, k, iteration: 0,
    converged: false, inertia: initInertia, silhouetteScore: 0,
    inertiaHistory: [initInertia],
    centroidHistory: [initCentroids.map(c => ({ ...c }))],
    clusterSizes,
  };

  for (let i = 0; i < maxIter; i++) {
    state = kMeansStep(points, state);
    if (state.converged) break;
  }

  return state;
}

/* ─── Elbow method: run K-Means for k=1..maxK ─── */
export function computeElbow(points: Point[], maxK: number, initMethod: string): { k: number; inertia: number }[] {
  const results: { k: number; inertia: number }[] = [];
  for (let k = 1; k <= maxK; k++) {
    const st = trainKMeans(points, k, 50, initMethod);
    results.push({ k, inertia: st.inertia });
  }
  return results;
}

/* ─── Predict cluster for a point ─── */
export function predictCluster(px: number, py: number, centroids: Point[]): number {
  let minD = Infinity, best = 0;
  for (let c = 0; c < centroids.length; c++) {
    const d = dist2({ x: px, y: py }, centroids[c]);
    if (d < minD) { minD = d; best = c; }
  }
  return best;
}

/* ─── Metrics ─── */
export function computeMetrics(state: KMeansState): MetricValue[] {
  if (state.centroids.length === 0) return [
    { label: 'Inertia', value: '—', isPrimary: true }, { label: 'Silhouette', value: '—' },
    { label: 'Iterations', value: '—' }, { label: 'K', value: '—' },
    { label: 'Converged', value: '—' }, { label: 'Clusters', value: '—' },
  ];
  return [
    { label: 'Inertia', value: state.inertia.toFixed(3), isPrimary: true },
    { label: 'Silhouette', value: state.silhouetteScore.toFixed(3) },
    { label: 'Iterations', value: String(state.iteration) },
    { label: 'K', value: String(state.k) },
    { label: 'Converged', value: state.converged ? 'Yes' : 'No' },
    { label: 'Clusters', value: state.clusterSizes.join(', ') },
  ];
}

/* ─── Data Statistics ─── */
export interface DataStats { n: number; meanX: number; meanY: number; stdX: number; stdY: number; }
export function computeDataStats(points: Point[]): DataStats | null {
  if (points.length < 2) return null;
  const n = points.length;
  let sx = 0, sy = 0; for (const p of points) { sx += p.x; sy += p.y; }
  const mx = sx / n, my = sy / n;
  let vx = 0, vy = 0; for (const p of points) { vx += (p.x - mx) ** 2; vy += (p.y - my) ** 2; }
  return { n, meanX: mx, meanY: my, stdX: Math.sqrt(vx / n), stdY: Math.sqrt(vy / n) };
}

/* ─── Cluster colors ─── */
export const CLUSTER_COLORS = [
  '#a855f7', '#e7c365', '#4ade80', '#f87171', '#38bdf8',
  '#fb923c', '#c084fc', '#22d3ee', '#fbbf24', '#a3e635',
];
