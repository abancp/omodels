/**
 * TensorFlow.js Model Engines
 * 
 * These engines use TF.js for GPU-accelerated training.
 * Each engine follows the fit/predict interface:
 *   - fit(X, y, onProgress?) — train the model
 *   - predict(X) — return predictions
 *   - predictProba(X) — return probabilities (classification only)
 */

import * as tf from '@tensorflow/tfjs';

// ─── Helper: Build TF optimizer ──────────────────────────────────────────────

function buildOptimizer(name: string, lr: number): tf.Optimizer {
  switch (name) {
    case 'sgd': return tf.train.sgd(lr);
    case 'rmsprop': return tf.train.rmsprop(lr);
    case 'adam':
    default: return tf.train.adam(lr);
  }
}

// ─── Linear Regression (TF.js) ──────────────────────────────────────────────

export class TFLinearRegressionEngine {
  model: tf.Sequential | null = null;
  lr: number;
  epochs: number;
  batchSize: number;
  optimizerName: string;

  constructor(lr = 0.01, epochs = 100, optimizer = 'adam', batchSize = 32) {
    this.lr = lr;
    this.epochs = epochs;
    this.optimizerName = optimizer;
    this.batchSize = batchSize;
  }

  async fit(X: number[][], y: number[], onProgress?: (msg: string) => void) {
    if (X.length === 0) return;
    const numFeatures = X[0].length;

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      units: 1,
      inputShape: [numFeatures],
      activation: 'linear',
    }));

    this.model.compile({
      optimizer: buildOptimizer(this.optimizerName, this.lr),
      loss: 'meanSquaredError',
      metrics: ['mse'],
    });

    const xTensor = tf.tensor2d(X);
    const yTensor = tf.tensor2d(y, [y.length, 1]);

    try {
      await this.model.fit(xTensor, yTensor, {
        epochs: this.epochs,
        batchSize: this.batchSize,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (onProgress && epoch % Math.max(1, Math.ceil(this.epochs / 10)) === 0) {
              onProgress(`LinearReg Epoch ${epoch + 1}/${this.epochs} — loss: ${logs?.loss?.toFixed(4)}`);
            }
          },
        },
      });
    } finally {
      xTensor.dispose();
      yTensor.dispose();
    }
  }

  predict(X: number[][]): number[] {
    if (!this.model || X.length === 0) return [];
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      return Array.from(preds.dataSync());
    });
  }
}

// ─── Logistic Regression (TF.js) ────────────────────────────────────────────

export class TFLogisticRegressionEngine {
  model: tf.Sequential | null = null;
  lr: number;
  epochs: number;
  batchSize: number;
  optimizerName: string;
  numClasses: number = 2;
  classes: number[] = [];

  constructor(lr = 0.01, epochs = 100, optimizer = 'adam', batchSize = 32) {
    this.lr = lr;
    this.epochs = epochs;
    this.optimizerName = optimizer;
    this.batchSize = batchSize;
  }

  async fit(X: number[][], y: number[], onProgress?: (msg: string) => void) {
    if (X.length === 0) return;
    this.classes = [...new Set(y)].sort((a, b) => a - b);
    this.numClasses = this.classes.length;
    const numFeatures = X[0].length;
    const isBinary = this.numClasses <= 2;

    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      units: isBinary ? 1 : this.numClasses,
      inputShape: [numFeatures],
      activation: isBinary ? 'sigmoid' : 'softmax',
    }));

    this.model.compile({
      optimizer: buildOptimizer(this.optimizerName, this.lr),
      loss: isBinary ? 'binaryCrossentropy' : 'sparseCategoricalCrossentropy',
      metrics: ['accuracy'],
    });

    const xTensor = tf.tensor2d(X);
    const yTensor = isBinary
      ? tf.tensor2d(y, [y.length, 1])
      : tf.tensor1d(y, 'int32');

    try {
      await this.model.fit(xTensor, yTensor, {
        epochs: this.epochs,
        batchSize: this.batchSize,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (onProgress && epoch % Math.max(1, Math.ceil(this.epochs / 10)) === 0) {
              onProgress(`LogisticReg Epoch ${epoch + 1}/${this.epochs} — loss: ${logs?.loss?.toFixed(4)}, acc: ${logs?.acc?.toFixed(3)}`);
            }
          },
        },
      });
    } finally {
      xTensor.dispose();
      yTensor.dispose();
    }
  }

  predict(X: number[][]): number[] {
    if (!this.model || X.length === 0) return [];
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      if (this.numClasses <= 2) {
        return Array.from(preds.dataSync()).map(v => v >= 0.5 ? 1 : 0);
      } else {
        const argmax = preds.argMax(-1);
        const indices = Array.from(argmax.dataSync());
        return indices.map(idx => this.classes[idx] ?? idx);
      }
    });
  }

  predictProba(X: number[][]): number[] {
    if (!this.model || X.length === 0) return [];
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      return Array.from(preds.dataSync());
    });
  }
}

// ─── MLP Neural Network (TF.js) ─────────────────────────────────────────────

export class TFMLPEngine {
  model: tf.Sequential | null = null;
  hiddenLayers: number[] = [];
  learningRate: number;
  epochs: number;
  optimizerName: string;
  batchSize: number;
  classes: number[] = [];
  lossHistory: number[] = [];
  isRegression: boolean = false;

  constructor(
    hiddenLayersStr = '8,4',
    learningRate = 0.05,
    epochs = 100,
    optimizer = 'adam',
    batchSize = 32
  ) {
    this.hiddenLayers = hiddenLayersStr
      .split(',')
      .map(s => parseInt(s.trim(), 10))
      .filter(n => !isNaN(n) && n > 0);
    if (this.hiddenLayers.length === 0) this.hiddenLayers = [8, 4];
    this.learningRate = learningRate;
    this.epochs = epochs;
    this.optimizerName = optimizer;
    this.batchSize = batchSize;
  }

  async fit(X: number[][], y: number[], onProgress?: (msg: string) => void) {
    if (X.length === 0) return;
    this.classes = [...new Set(y)].sort((a, b) => a - b);
    const numFeatures = X[0].length;

    // Detect if regression (continuous values) or classification
    this.isRegression = this.classes.length > 10 || y.some(v => v !== Math.floor(v));
    const isBinary = !this.isRegression && this.classes.length <= 2;
    const numOutputs = this.isRegression ? 1 : (isBinary ? 1 : this.classes.length);

    this.model = tf.sequential();

    for (let i = 0; i < this.hiddenLayers.length; i++) {
      this.model.add(tf.layers.dense({
        units: this.hiddenLayers[i],
        inputShape: i === 0 ? [numFeatures] : undefined,
        activation: 'relu',
      }));
    }

    this.model.add(tf.layers.dense({
      units: numOutputs,
      activation: this.isRegression ? 'linear' : (isBinary ? 'sigmoid' : 'softmax'),
    }));

    const loss = this.isRegression
      ? 'meanSquaredError'
      : (isBinary ? 'binaryCrossentropy' : 'sparseCategoricalCrossentropy');

    this.model.compile({
      optimizer: buildOptimizer(this.optimizerName, this.learningRate),
      loss,
      metrics: this.isRegression ? ['mse'] : ['accuracy'],
    });

    const xTensor = tf.tensor2d(X);
    const yTensor = this.isRegression
      ? tf.tensor2d(y, [y.length, 1])
      : (isBinary ? tf.tensor2d(y, [y.length, 1]) : tf.tensor1d(y, 'int32'));

    this.lossHistory = [];

    try {
      await this.model.fit(xTensor, yTensor, {
        epochs: this.epochs,
        batchSize: this.batchSize,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (logs) this.lossHistory.push(logs.loss);
            if (onProgress && epoch % Math.max(1, Math.ceil(this.epochs / 10)) === 0) {
              onProgress(`MLP Epoch ${epoch + 1}/${this.epochs} — loss: ${logs?.loss?.toFixed(4)}`);
            }
          },
        },
      });
    } finally {
      xTensor.dispose();
      yTensor.dispose();
    }
  }

  predict(X: number[][]): number[] {
    if (!this.model || X.length === 0) return [];
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      if (this.isRegression) {
        return Array.from(preds.dataSync());
      }
      const isBinary = this.classes.length <= 2;
      if (isBinary) {
        return Array.from(preds.dataSync()).map(v => v >= 0.5 ? 1 : 0);
      } else {
        const argmax = preds.argMax(-1);
        const indices = Array.from(argmax.dataSync());
        return indices.map(idx => this.classes[idx] ?? idx);
      }
    });
  }

  predictProba(X: number[][]): number[] {
    if (!this.model || X.length === 0) return [];
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      return Array.from(preds.dataSync());
    });
  }
}
