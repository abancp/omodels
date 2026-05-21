export interface Point {
  features: number[];
  label: number;
}

export type ActivationType = 'step' | 'sigmoid' | 'tanh' | 'relu' | 'linear' | 'softmax';

export interface MLPLayer {
  nodes: number;
  activation: ActivationType;
  weights: number[][]; // [node_idx][input_idx]
  biases: number[];    // [node_idx]
}

export interface MLPState {
  numInputs: number;
  layers: MLPLayer[];
  learningRate: number;
  epoch: number;
  maxEpochs: number;
  lossHistory: number[];
  converged: boolean;
  
  // Real-time backtracking tracker metrics
  layerGradients?: number[]; // Average absolute delta per layer
  layerWeightsMean?: number[]; // Average absolute weight per layer
  
  // Rolling 50-epoch history for sparklines
  layerGradientsHistory?: number[][]; 
  layerWeightsMeanHistory?: number[][];
}

export function applyActivation(sum: number, type: ActivationType): number {
  switch (type) {
    case 'step': return sum >= 0 ? 1 : 0;
    case 'sigmoid': return 1 / (1 + Math.exp(-sum));
    case 'tanh': return Math.tanh(sum);
    case 'relu': return Math.max(0, sum);
    case 'linear': return sum;
    default: return sum;
  }
}

export function activationDerivative(output: number, type: ActivationType): number {
  switch (type) {
    case 'step': return 1;
    case 'sigmoid': return output * (1 - output);
    case 'tanh': return 1 - output * output;
    case 'relu': return output > 0 ? 1 : 0;
    case 'linear': return 1;
    default: return 1;
  }
}

export function applyLayerActivation(sums: number[], type: ActivationType): number[] {
  if (type === 'softmax') {
    const max = Math.max(...sums);
    const exps = sums.map(s => Math.exp(s - max));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumExps);
  }
  return sums.map(s => applyActivation(s, type));
}

export function predictMLP(features: number[], state: MLPState): { acts: number[][], pred: number[] } {
  const acts: number[][] = [];
  let currentIn = features || [0.5, 0.5];

  for (let l = 0; l < state.layers.length; l++) {
    const layer = state.layers[l];
    const sums = new Array(layer.nodes);
    for (let k = 0; k < layer.nodes; k++) {
      let sum = layer.biases[k];
      for (let j = 0; j < currentIn.length; j++) {
        sum += currentIn[j] * (layer.weights[k]?.[j] ?? 0);
      }
      sums[k] = sum;
    }
    const nextActs = applyLayerActivation(sums, layer.activation);
    acts.push(nextActs);
    currentIn = nextActs;
  }
  
  return { acts, pred: currentIn };
}

export function trainStep(points: Point[], state: MLPState): MLPState {
  if (state.converged || state.epoch >= state.maxEpochs) return state;

  let mse = 0;
  
  // Clone layers for update
  const newLayers = state.layers.map(layer => ({
    ...layer,
    weights: layer.weights.map(w => [...w]),
    biases: [...layer.biases]
  }));

  const L = newLayers.length;
  let lastDeltas: number[][] = [];

  for (const p of points) {
    const { acts } = predictMLP(p.features, { ...state, layers: newLayers });
    
    // Deltas: array of layers, each has delta per node
    const deltas: number[][] = new Array(L);
    
    // Output layer (L-1)
    const outLayer = newLayers[L - 1];
    deltas[L - 1] = new Array(outLayer.nodes);
    
    // For simplicity, handle MSE loss + softmax as (pred - target) * deriv.
    // In strict cross-entropy + softmax, delta is just (pred - target). 
    // We map label (0 or 1) to a one-hot vector if there are multiple output nodes.
    const targets = new Array(outLayer.nodes).fill(0);
    if (outLayer.nodes > 1) {
      if (p.label < outLayer.nodes) targets[p.label] = 1;
    } else {
      targets[0] = p.label;
    }

    for (let k = 0; k < outLayer.nodes; k++) {
      const outAct = acts[L - 1][k];
      const err = targets[k] - outAct;
      let outDeriv = 1;
      if (outLayer.activation === 'softmax') {
        outDeriv = outAct * (1 - outAct); // Simplistic approx for diagonal of Jacobian
      } else {
        outDeriv = activationDerivative(outAct, outLayer.activation);
      }
      deltas[L - 1][k] = err * outDeriv;
      
      // Update MSE tracking. For multiple nodes, average error? Just track node 0 or sum for simplicity.
      if (k === 0 || outLayer.nodes > 1) {
          mse += err * err;
      }
    }

    // Backprop hidden layers
    for (let l = L - 2; l >= 0; l--) {
      const layer = newLayers[l];
      const nextLayer = newLayers[l + 1];
      deltas[l] = new Array(layer.nodes);

      for (let k = 0; k < layer.nodes; k++) {
        let errSum = 0;
        for (let nextK = 0; nextK < nextLayer.nodes; nextK++) {
          errSum += deltas[l + 1][nextK] * nextLayer.weights[nextK][k];
        }
        const act = acts[l][k];
        deltas[l][k] = errSum * activationDerivative(act, layer.activation);
      }
    }

    // Update weights and biases
    for (let l = 0; l < L; l++) {
      const layer = newLayers[l];
      const prevActs = l === 0 ? p.features : acts[l - 1];
      
      for (let k = 0; k < layer.nodes; k++) {
        for (let j = 0; j < prevActs.length; j++) {
          layer.weights[k][j] += state.learningRate * deltas[l][k] * prevActs[j];
        }
        layer.biases[k] += state.learningRate * deltas[l][k];
      }
    }
    lastDeltas = deltas;
  }

  mse /= points.length;

  const layerGradients: number[] = [];
  const layerWeightsMean: number[] = [];
  
  const layerGradientsHistory = (state.layerGradientsHistory ?? []).map(a => [...a]);
  const layerWeightsMeanHistory = (state.layerWeightsMeanHistory ?? []).map(a => [...a]);

  // `lastDeltas` is indexed from 0 to L-1
  for (let l = 0; l < L; l++) {
    const avgGrad = lastDeltas[l] ? lastDeltas[l].reduce((a, b) => a + Math.abs(b), 0) / lastDeltas[l].length : 0;
    layerGradients.push(avgGrad);
    
    let sumW = 0, countW = 0;
    for (const row of newLayers[l].weights) {
      for (const w of row) { sumW += Math.abs(w); countW++; }
    }
    const avgW = countW > 0 ? sumW / countW : 0;
    layerWeightsMean.push(avgW);
    
    if (!layerGradientsHistory[l]) layerGradientsHistory[l] = [];
    if (!layerWeightsMeanHistory[l]) layerWeightsMeanHistory[l] = [];
    
    layerGradientsHistory[l].push(avgGrad);
    layerWeightsMeanHistory[l].push(avgW);
    
    // Keep only last 50 epochs for the sparkline
    if (layerGradientsHistory[l].length > 50) layerGradientsHistory[l].shift();
    if (layerWeightsMeanHistory[l].length > 50) layerWeightsMeanHistory[l].shift();
  }

  return {
    ...state,
    layers: newLayers,
    epoch: state.epoch + 1,
    lossHistory: [...state.lossHistory, mse],
    converged: mse < 1e-4,
    layerGradients,
    layerWeightsMean,
    layerGradientsHistory,
    layerWeightsMeanHistory
  };
}

function seededRandom(seed: number) {
  let s = seed;
  return () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
}

export function generateData(type: string, count: number, noise: number, numFeatures: number = 2): Point[] {
  const rand = seededRandom(42 + count + Math.floor(noise * 100) + numFeatures);
  const pts: Point[] = [];
  
  if (type === 'spirals') {
    for (let i = 0; i < count; i++) {
      const label = rand() > 0.5 ? 1 : 0;
      const r = rand() * 0.4 + 0.1;
      const t = 1.5 * Math.PI * (r / 0.5) + (label ? Math.PI : 0);
      const f0 = 0.5 + r * Math.cos(t) + (rand() - 0.5) * noise * 0.2;
      const f1 = 0.5 + r * Math.sin(t) + (rand() - 0.5) * noise * 0.2;
      pts.push({ features: [f0, f1], label });
    }
  } else if (type === 'moons') {
    for (let i = 0; i < count; i++) {
      const label = rand() > 0.5 ? 1 : 0;
      const t = rand() * Math.PI;
      let f0 = 0.5 + 0.3 * Math.cos(t);
      let f1 = 0.5 + 0.3 * Math.sin(t);
      if (label === 1) {
        f0 += 0.3;
        f1 = 0.5 - 0.3 * Math.sin(t);
      } else {
        f0 -= 0.3;
      }
      f0 += (rand() - 0.5) * noise * 0.3;
      f1 += (rand() - 0.5) * noise * 0.3;
      pts.push({ features: [f0, f1], label });
    }
  } else if (type === 'xor') { 
    for (let i = 0; i < count; i++) {
      const f0 = rand() > 0.5 ? 0.8 : 0.2;
      const f1 = rand() > 0.5 ? 0.8 : 0.2;
      const label = (f0 > 0.5) !== (f1 > 0.5) ? 1 : 0;
      pts.push({ features: [f0 + (rand() - 0.5)*noise*0.3, f1 + (rand() - 0.5)*noise*0.3], label });
    }
  } else if (type === 'circles') {
    for (let i = 0; i < count; i++) {
      const r = rand(); const theta = rand() * Math.PI * 2;
      const dist = r > 0.5 ? 0.4 : 0.15;
      const label = r > 0.5 ? 1 : 0;
      const f0 = 0.5 + dist * Math.cos(theta) + (rand() - 0.5) * noise * 0.1;
      const f1 = 0.5 + dist * Math.sin(theta) + (rand() - 0.5) * noise * 0.1;
      pts.push({ features: [f0, f1], label });
    }
  } else {
    for (let i = 0; i < count; i++) {
      pts.push({ features: [rand(), rand()], label: rand() > 0.5 ? 1 : 0 });
    }
  }
  return pts;
}
