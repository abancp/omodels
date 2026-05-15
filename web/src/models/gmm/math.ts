import type { MetricValue } from '../registry';
import { generateClusteringData, type Point } from '../kmeans/math';

export { generateClusteringData, type Point };

export interface GMMState {
  k: number;
  means: Point[];
  covariances: { vxx: number; vyy: number; vxy: number }[]; // 2x2 symmetric matrices
  weights: number[]; // mixture weights, sum to 1
  
  responsibilities: number[][]; // N x K
  assignments: number[]; // Hard assignments (argmax)
  
  iteration: number;
  logLikelihood: number;
  logLikelihoodHistory: number[];
  converged: boolean;
  
  covarianceType: 'full' | 'diag' | 'spherical';
}

/* ─── 2x2 Matrix Utilities ─── */
function det2x2(vxx: number, vyy: number, vxy: number): number {
  return Math.max(vxx * vyy - vxy * vxy, 1e-10); // add epsilon to avoid singular
}

function inv2x2(vxx: number, vyy: number, vxy: number) {
  const d = det2x2(vxx, vyy, vxy);
  return {
    ixx: vyy / d,
    iyy: vxx / d,
    ixy: -vxy / d,
    det: d
  };
}

/* ─── Bivariate Normal PDF ─── */
function gaussianPDF(p: Point, mean: Point, cov: {vxx: number, vyy: number, vxy: number}): number {
  const { ixx, iyy, ixy, det } = inv2x2(cov.vxx, cov.vyy, cov.vxy);
  const dx = p.x - mean.x;
  const dy = p.y - mean.y;
  
  // (x-mu)^T * Sigma^-1 * (x-mu)
  const mahalanobis2 = dx * (ixx * dx + ixy * dy) + dy * (ixy * dx + iyy * dy);
  
  const normConst = 1.0 / (2 * Math.PI * Math.sqrt(det));
  return normConst * Math.exp(-0.5 * mahalanobis2);
}

/* ─── Initialization ─── */
export function initGMM(points: Point[], k: number, covType: string, initMethod: string): GMMState {
  const n = points.length;
  let means: Point[] = [];
  
  if (initMethod === 'random') {
    // Random sample from points
    const shuffled = [...points].sort(() => Math.random() - 0.5);
    means = shuffled.slice(0, k).map(p => ({ ...p }));
  } else {
    // K-Means++ style init
    means.push({ ...points[Math.floor(Math.random() * n)] });
    for (let c = 1; c < k; c++) {
      const dists = points.map(p => {
        let minD = Infinity;
        for (const cent of means) minD = Math.min(minD, (p.x-cent.x)**2 + (p.y-cent.y)**2);
        return minD;
      });
      const totalD = dists.reduce((a, b) => a + b, 0);
      let r = Math.random() * totalD, cumSum = 0;
      for (let i = 0; i < n; i++) {
        cumSum += dists[i];
        if (cumSum >= r) { means.push({ ...points[i] }); break; }
      }
      if (means.length <= c) means.push({ ...points[Math.floor(Math.random() * n)] });
    }
  }

  // Initial cov is identity * global variance
  let vx = 0, vy = 0;
  if (n > 1) {
    const mx = points.reduce((s, p) => s + p.x, 0) / n;
    const my = points.reduce((s, p) => s + p.y, 0) / n;
    vx = points.reduce((s, p) => s + (p.x - mx)**2, 0) / n;
    vy = points.reduce((s, p) => s + (p.y - my)**2, 0) / n;
  }
  const initVar = Math.max((vx + vy) / 2, 0.01);
  
  const covariances = Array(k).fill(0).map(() => ({ vxx: initVar, vyy: initVar, vxy: 0 }));
  const weights = Array(k).fill(1 / k);
  
  return {
    k,
    means,
    covariances,
    weights,
    responsibilities: Array(n).fill(0).map(() => Array(k).fill(1/k)),
    assignments: Array(n).fill(0),
    iteration: 0,
    logLikelihood: -Infinity,
    logLikelihoodHistory: [],
    converged: false,
    covarianceType: covType as any
  };
}

/* ─── E-Step ─── */
function eStep(points: Point[], state: GMMState): { resp: number[][], ll: number } {
  const n = points.length;
  const k = state.k;
  const resp: number[][] = Array(n).fill(0).map(() => Array(k).fill(0));
  let logLikelihood = 0;

  for (let i = 0; i < n; i++) {
    const p = points[i];
    let rowSum = 0;
    
    // Calculate P(x | z) * P(z)
    for (let j = 0; j < k; j++) {
      const pdf = gaussianPDF(p, state.means[j], state.covariances[j]);
      const wPdf = state.weights[j] * pdf;
      resp[i][j] = wPdf;
      rowSum += wPdf;
    }
    
    // Normalize and add to log likelihood
    if (rowSum > 0) {
      logLikelihood += Math.log(rowSum);
      for (let j = 0; j < k; j++) {
        resp[i][j] /= rowSum;
      }
    } else {
      // Degenerate case fallback
      for (let j = 0; j < k; j++) resp[i][j] = 1 / k;
    }
  }
  
  return { resp, ll: logLikelihood };
}

/* ─── M-Step ─── */
function mStep(points: Point[], resp: number[][], state: GMMState): { means: Point[], covs: {vxx: number, vyy: number, vxy: number}[], weights: number[] } {
  const n = points.length;
  const k = state.k;
  
  const means: Point[] = [];
  const covs: {vxx: number, vyy: number, vxy: number}[] = [];
  const weights: number[] = [];
  
  // Nk is the effective number of points assigned to cluster k
  const Nk = Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) Nk[j] += resp[i][j];
  }
  
  for (let j = 0; j < k; j++) {
    const effN = Math.max(Nk[j], 1e-10); // avoid div by zero
    weights.push(Nk[j] / n);
    
    // Mean
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) {
      mx += resp[i][j] * points[i].x;
      my += resp[i][j] * points[i].y;
    }
    mx /= effN; my /= effN;
    means.push({ x: mx, y: my });
    
    // Covariance
    let vxx = 0, vyy = 0, vxy = 0;
    for (let i = 0; i < n; i++) {
      const dx = points[i].x - mx;
      const dy = points[i].y - my;
      const r = resp[i][j];
      vxx += r * dx * dx;
      vyy += r * dy * dy;
      vxy += r * dx * dy;
    }
    vxx /= effN; vyy /= effN; vxy /= effN;
    
    // Regularization to prevent collapse
    const reg = 1e-6;
    vxx += reg; vyy += reg;
    
    if (state.covarianceType === 'spherical') {
      const avg = (vxx + vyy) / 2;
      covs.push({ vxx: avg, vyy: avg, vxy: 0 });
    } else if (state.covarianceType === 'diag') {
      covs.push({ vxx, vyy, vxy: 0 });
    } else { // full
      covs.push({ vxx, vyy, vxy });
    }
  }
  
  return { means, covs, weights };
}

/* ─── Single GMM Step ─── */
export function gmmStep(points: Point[], state: GMMState): GMMState {
  if (state.converged) return state;
  
  const { resp, ll } = eStep(points, state);
  const { means, covs, weights } = mStep(points, resp, state);
  
  const assignments = resp.map(r => r.indexOf(Math.max(...r)));
  
  const converged = state.iteration > 0 && Math.abs(ll - state.logLikelihood) < 1e-4;
  
  return {
    ...state,
    means,
    covariances: covs,
    weights,
    responsibilities: resp,
    assignments,
    iteration: state.iteration + 1,
    logLikelihood: ll,
    logLikelihoodHistory: [...state.logLikelihoodHistory, ll],
    converged
  };
}

/* ─── Prediction ─── */
export function predictGMM(p: Point, state: GMMState): number[] {
  const k = state.k;
  const probs = Array(k).fill(0);
  let sum = 0;
  for (let j = 0; j < k; j++) {
    const pdf = gaussianPDF(p, state.means[j], state.covariances[j]);
    probs[j] = state.weights[j] * pdf;
    sum += probs[j];
  }
  if (sum > 0) {
    for (let j = 0; j < k; j++) probs[j] /= sum;
  }
  return probs;
}

/* ─── Eigen Decomposition for 2x2 Matrix (Ellipse Drawing) ─── */
export function getCovarianceEllipse(cov: {vxx: number, vyy: number, vxy: number}) {
  const tr = cov.vxx + cov.vyy;
  const det = cov.vxx * cov.vyy - cov.vxy * cov.vxy;
  
  const gap = Math.sqrt(Math.max((tr * tr) / 4 - det, 0));
  const lambda1 = tr / 2 + gap;
  const lambda2 = tr / 2 - gap;
  
  let angle = 0;
  if (Math.abs(cov.vxy) > 1e-8) {
    angle = Math.atan2(lambda1 - cov.vxx, cov.vxy);
  } else if (cov.vxx < cov.vyy) {
    angle = Math.PI / 2;
  }
  
  return {
    angle,
    r1: Math.sqrt(Math.max(lambda1, 1e-10)), // semi-major axis (1 std dev)
    r2: Math.sqrt(Math.max(lambda2, 1e-10))  // semi-minor axis (1 std dev)
  };
}

export function computeMetrics(state: GMMState): MetricValue[] {
  if (state.iteration === 0) return [
    { label: 'Log-Likelihood', value: '—', isPrimary: true },
    { label: 'Iterations', value: '—' },
    { label: 'Converged', value: '—' },
    { label: 'AIC', value: '—' }
  ];
  
  // AIC = 2k - 2ln(L), where k is num parameters
  let pPerCov = 3; // full
  if (state.covarianceType === 'diag') pPerCov = 2;
  if (state.covarianceType === 'spherical') pPerCov = 1;
  const numParams = (state.k - 1) + (state.k * 2) + (state.k * pPerCov); // weights + means + covs
  const aic = 2 * numParams - 2 * state.logLikelihood;
  
  return [
    { label: 'Log-Likelihood', value: state.logLikelihood.toFixed(2), isPrimary: true },
    { label: 'Iterations', value: String(state.iteration) },
    { label: 'Converged', value: state.converged ? 'Yes' : 'No' },
    { label: 'AIC', value: aic.toFixed(1) }
  ];
}
