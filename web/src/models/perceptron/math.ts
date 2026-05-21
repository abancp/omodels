export interface Point {
  features: number[];
  label: number;
}

export type ActivationType = 'step' | 'sigmoid' | 'tanh' | 'relu' | 'linear';

export interface PerceptronState {
  numInputs: number;
  numPerceptrons: number;
  hiddenWeights: number[][]; // [numPerceptrons][numInputs]
  hiddenBias: number[];      // [numPerceptrons]
  outWeights: number[];      // [numPerceptrons]
  outBias: number;
  activation: ActivationType;
  outAct: ActivationType;
  learningRate: number;
  epoch: number;
  maxEpochs: number;
  lossHistory: number[];
  converged: boolean;
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
    case 'step': return 1; // approximation for backprop
    case 'sigmoid': return output * (1 - output);
    case 'tanh': return 1 - output * output;
    case 'relu': return output > 0 ? 1 : 0;
    case 'linear': return 1;
    default: return 1;
  }
}

// Predict for a single point
export function predictPerceptron(p: Point, state: PerceptronState): { hiddenActs: number[], pred: number } {
  const features = p?.features || [0.5, 0.5];
  const hiddenActs = new Array(state.numPerceptrons);
  for (let k = 0; k < state.numPerceptrons; k++) {
    let sum = state.hiddenBias[k] || 0;
    for (let j = 0; j < state.numInputs; j++) {
      sum += (features[j] ?? 0.5) * (state.hiddenWeights[k]?.[j] ?? 0);
    }
    hiddenActs[k] = applyActivation(sum, state.activation);
  }
  
  let outSum = state.outBias || 0;
  for (let k = 0; k < state.numPerceptrons; k++) {
    outSum += hiddenActs[k] * (state.outWeights[k] ?? 0);
  }
  const pred = applyActivation(outSum, state.outAct ?? 'sigmoid'); // Use output activation
  
  return { hiddenActs, pred };
}

// Single epoch step (Backpropagation)
export function trainStep(points: Point[], state: PerceptronState): PerceptronState {
  if (state.converged || state.epoch >= state.maxEpochs) return state;

  let mse = 0;
  
  // Clone weights for update
  const hiddenWeights = state.hiddenWeights.map(w => [...w]);
  const hiddenBias = [...state.hiddenBias];
  const outWeights = [...state.outWeights];
  let outBias = state.outBias;

  for (const p of points) {
    // Forward pass
    const { hiddenActs, pred } = predictPerceptron(p, { ...state, hiddenWeights, hiddenBias, outWeights, outBias });
    
    const error = p.label - pred;
    mse += error * error;

    // Backward pass
    // Output layer delta
    // Derivative of output activation
    const predDeriv = activationDerivative(pred, state.outAct ?? 'sigmoid');
    const deltaOut = error * predDeriv;

    // Hidden layer deltas
    const deltaHidden = new Array(state.numPerceptrons);
    for (let k = 0; k < state.numPerceptrons; k++) {
      const actDeriv = activationDerivative(hiddenActs[k], state.activation);
      deltaHidden[k] = deltaOut * outWeights[k] * actDeriv;
    }

    // Update Output Weights
    for (let k = 0; k < state.numPerceptrons; k++) {
      outWeights[k] += state.learningRate * deltaOut * hiddenActs[k];
    }
    outBias += state.learningRate * deltaOut;

    // Update Hidden Weights
    for (let k = 0; k < state.numPerceptrons; k++) {
      for (let j = 0; j < state.numInputs; j++) {
        hiddenWeights[k][j] += state.learningRate * deltaHidden[k] * p.features[j];
      }
      hiddenBias[k] += state.learningRate * deltaHidden[k];
    }
  }

  mse /= points.length;

  return {
    ...state,
    hiddenWeights, hiddenBias, outWeights, outBias,
    epoch: state.epoch + 1,
    lossHistory: [...state.lossHistory, mse],
    converged: mse < 1e-4
  };
}

// Data generation
function seededRandom(seed: number) {
  let s = seed;
  return () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
}

export function generateData(type: string, count: number, noise: number, numFeatures: number = 2): Point[] {
  const rand = seededRandom(42 + count + Math.floor(noise * 100) + numFeatures);
  const pts: Point[] = [];
  
  if (type === 'linearly_separable') {
    const trueW = Array.from({ length: numFeatures }, () => rand() * 2 - 1);
    const trueB = rand() * 2 - 1;

    for (let i = 0; i < count; i++) {
      const features = Array.from({ length: numFeatures }, () => rand());
      let val = trueB;
      for (let j = 0; j < numFeatures; j++) val += features[j] * trueW[j];
      const label = val + (rand() - 0.5) * noise * 2 > 0 ? 1 : 0;
      pts.push({ features, label });
    }
  } else if (type === 'xor') { 
    for (let i = 0; i < count; i++) {
      const features: number[] = Array.from({ length: numFeatures }, () => rand() > 0.5 ? 0.8 : 0.2);
      const label = (features[0] > 0.5) !== (features[1] > 0.5) ? 1 : 0;
      for (let j = 0; j < numFeatures; j++) {
        features[j] = Math.max(0, Math.min(1, features[j] + (rand() - 0.5) * noise * 0.3));
      }
      pts.push({ features, label });
    }
  } else if (type === 'circles') {
    for (let i = 0; i < count; i++) {
      const r = rand(); const theta = rand() * Math.PI * 2;
      const dist = r > 0.5 ? 0.4 : 0.15;
      const label = r > 0.5 ? 1 : 0;
      const f0 = 0.5 + dist * Math.cos(theta) + (rand() - 0.5) * noise * 0.1;
      const f1 = 0.5 + dist * Math.sin(theta) + (rand() - 0.5) * noise * 0.1;
      const features = [f0, f1];
      for (let j = 2; j < numFeatures; j++) features.push(rand());
      pts.push({ features, label });
    }
  } else {
    for (let i = 0; i < count; i++) {
      pts.push({ features: Array.from({ length: numFeatures }, () => rand()), label: rand() > 0.5 ? 1 : 0 });
    }
  }
  return pts;
}
