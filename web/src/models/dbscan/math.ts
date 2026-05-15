import type { MetricValue } from '../registry';
import { generateClusteringData, type Point } from '../kmeans/math';

export { generateClusteringData, type Point };

export interface DBSCANState {
  eps: number;
  minPts: number;
  metric: string;
  
  assignments: number[]; // -1 for noise, >= 0 for cluster, -2 for unassigned
  pointTypes: ('core' | 'border' | 'noise' | 'unassigned')[];
  
  phase: 'SEARCH' | 'EXPAND' | 'DONE';
  currentPoint: number;
  currentCluster: number;
  searchQueue: number[];
  
  coreCount: number;
  borderCount: number;
  noiseCount: number;
}

export function initDBSCAN(n: number, eps: number, minPts: number, metric: string): DBSCANState {
  return {
    eps, minPts, metric,
    assignments: Array(n).fill(-2),
    pointTypes: Array(n).fill('unassigned'),
    phase: 'SEARCH',
    currentPoint: 0,
    currentCluster: 0,
    searchQueue: [],
    coreCount: 0,
    borderCount: 0,
    noiseCount: 0,
  };
}

export function getDist(p1: Point, p2: Point, metric: string): number {
  const dx = Math.abs(p1.x - p2.x);
  const dy = Math.abs(p1.y - p2.y);
  if (metric === 'manhattan') return dx + dy;
  if (metric === 'chebyshev') return Math.max(dx, dy);
  return Math.sqrt(dx*dx + dy*dy);
}

export function dbscanStep(points: Point[], state: DBSCANState, steps: number = 1): DBSCANState {
  if (state.phase === 'DONE') return state;
  
  const s = { ...state, assignments: [...state.assignments], pointTypes: [...state.pointTypes], searchQueue: [...state.searchQueue] };
  
  const getNeighbors = (pIdx: number) => {
    const n = [];
    for (let i = 0; i < points.length; i++) {
      if (getDist(points[pIdx], points[i], s.metric) <= s.eps) n.push(i);
    }
    return n;
  };

  let stepsRemaining = steps;

  while (stepsRemaining > 0 && s.phase !== 'DONE') {
    if (s.phase === 'SEARCH') {
      while (s.currentPoint < points.length && s.pointTypes[s.currentPoint] !== 'unassigned') {
        s.currentPoint++;
      }
      
      if (s.currentPoint >= points.length) {
        s.phase = 'DONE';
        break;
      }
      
      const p = s.currentPoint;
      const neighbors = getNeighbors(p);
      
      if (neighbors.length < s.minPts) {
        s.pointTypes[p] = 'noise';
        s.assignments[p] = -1;
        s.noiseCount++;
        s.currentPoint++;
      } else {
        s.pointTypes[p] = 'core';
        s.assignments[p] = s.currentCluster;
        s.coreCount++;
        s.searchQueue = neighbors.filter(n => n !== p);
        s.phase = 'EXPAND';
      }
      stepsRemaining--;
    } else if (s.phase === 'EXPAND') {
      if (s.searchQueue.length === 0) {
        s.currentCluster++;
        s.currentPoint++;
        s.phase = 'SEARCH';
        continue; // don't consume step for state transition
      }
      
      const q = s.searchQueue.shift()!;
      
      if (s.pointTypes[q] === 'noise') {
        s.pointTypes[q] = 'border';
        s.assignments[q] = s.currentCluster;
        s.noiseCount--;
        s.borderCount++;
      } else if (s.pointTypes[q] === 'unassigned') {
        s.assignments[q] = s.currentCluster;
        const neighbors = getNeighbors(q);
        if (neighbors.length >= s.minPts) {
          s.pointTypes[q] = 'core';
          s.coreCount++;
          for (const n of neighbors) {
             if (s.pointTypes[n] === 'unassigned' && !s.searchQueue.includes(n)) {
                 s.searchQueue.push(n);
             }
          }
        } else {
          s.pointTypes[q] = 'border';
          s.borderCount++;
        }
      }
      stepsRemaining--;
    }
  }
  
  return s;
}

export function trainDBSCAN(points: Point[], eps: number, minPts: number, metric: string): DBSCANState {
   let state = initDBSCAN(points.length, eps, minPts, metric);
   while (state.phase !== 'DONE') {
      state = dbscanStep(points, state, 1000);
   }
   return state;
}

export function computeMetrics(state: DBSCANState, totalPoints: number): MetricValue[] {
  if (state.phase === 'SEARCH' && state.currentPoint === 0) return [
    { label: 'Clusters', value: '—', isPrimary: true },
    { label: 'Core Pts', value: '—' },
    { label: 'Border Pts', value: '—' },
    { label: 'Noise Pts', value: '—' },
    { label: '% Noise', value: '—' },
  ];
  return [
    { label: 'Clusters', value: String(state.currentCluster + (state.phase==='EXPAND'?1:0)), isPrimary: true },
    { label: 'Core Pts', value: String(state.coreCount) },
    { label: 'Border Pts', value: String(state.borderCount) },
    { label: 'Noise Pts', value: String(state.noiseCount) },
    { label: '% Noise', value: ((state.noiseCount / totalPoints) * 100).toFixed(1) + '%' },
  ];
}
