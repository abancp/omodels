import * as tf from '@tensorflow/tfjs';

export class TFLinearModelEngine {
  model: tf.Sequential | null = null;
  lr: number;
  epochs: number;
  batchSize: number;
  optimizerName: string;
  regularization: string;
  lambda: number;
  isClassification: boolean;
  weights: number[] = [];
  bias: number = 0;

  constructor(
    lr = 0.01,
    epochs = 100,
    optimizer = 'adam',
    batchSize = 32,
    regularization = 'none',
    lambda = 0.001,
    isClassification = false
  ) {
    this.lr = lr;
    this.epochs = epochs;
    this.optimizerName = optimizer;
    this.batchSize = batchSize;
    this.regularization = regularization;
    this.lambda = lambda;
    this.isClassification = isClassification;
  }

  async fit(X: number[][], y: number[], onProgress?: (msg: string) => void) {
    if (X.length === 0) return;
    const numFeatures = X[0].length;
    
    this.model = tf.sequential();
    
    let kernelRegularizer = undefined;
    if (this.regularization === 'l1') kernelRegularizer = tf.regularizers.l1({ l1: this.lambda });
    else if (this.regularization === 'l2') kernelRegularizer = tf.regularizers.l2({ l2: this.lambda });
    else if (this.regularization === 'l1l2') kernelRegularizer = tf.regularizers.l1l2({ l1: this.lambda, l2: this.lambda });

    this.model.add(tf.layers.dense({
      units: 1,
      inputShape: [numFeatures],
      activation: this.isClassification ? 'sigmoid' : 'linear',
      kernelRegularizer
    }));

    let optimizer: tf.Optimizer;
    if (this.optimizerName === 'sgd') optimizer = tf.train.sgd(this.lr);
    else if (this.optimizerName === 'rmsprop') optimizer = tf.train.rmsprop(this.lr);
    else optimizer = tf.train.adam(this.lr);

    this.model.compile({
      optimizer,
      loss: this.isClassification ? 'binaryCrossentropy' : 'meanSquaredError',
      metrics: this.isClassification ? ['accuracy'] : ['mse']
    });

    const xTensor = tf.tensor2d(X);
    const yTensor = tf.tensor2d(y, [y.length, 1]);

    await this.model.fit(xTensor, yTensor, {
      epochs: this.epochs,
      batchSize: this.batchSize,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (onProgress && epoch % Math.ceil(this.epochs / 10) === 0) {
            onProgress(`Epoch ${epoch + 1}/${this.epochs} - loss: ${logs?.loss.toFixed(4)}`);
          }
        }
      }
    });

    const layer = this.model.layers[0];
    const weightsTensor = layer.getWeights();
    if (weightsTensor.length > 0) {
      this.weights = Array.from(weightsTensor[0].dataSync());
      if (weightsTensor.length > 1) {
        this.bias = weightsTensor[1].dataSync()[0];
      }
    }

    xTensor.dispose();
    yTensor.dispose();
  }

  predict(X: number[][]): number[] {
    if (!this.model) {
      return X.map(row => {
        const score = row.reduce((s, v, j) => s + v * (this.weights[j] || 0), this.bias);
        if (this.isClassification) return score >= 0 ? 1 : 0;
        return score;
      });
    }
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      const arr = Array.from(preds.dataSync());
      if (this.isClassification) return arr.map(v => v >= 0.5 ? 1 : 0);
      return arr;
    });
  }

  predictProba(X: number[][]): number[] {
    if (!this.model) return X.map(() => 0.5);
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      return Array.from(preds.dataSync());
    });
  }
}

export class TFMLPEngine {
  model: tf.Sequential | null = null;
  hiddenLayers: number[] = [];
  learningRate: number;
  epochs: number;
  optimizerName: string;
  batchSize: number;
  classes: number[] = [];
  lossHistory: number[] = [];
  weights: number[][][] = [];
  biases: number[][] = [];

  constructor(hiddenLayersStr = '8,4', learningRate = 0.05, epochs = 100, optimizer = 'adam', batchSize = 32) {
    this.hiddenLayers = hiddenLayersStr.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n) && n > 0);
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
    const isBinary = this.classes.length <= 2;
    const numOutputs = isBinary ? 1 : this.classes.length;

    this.model = tf.sequential();
    
    for (let i = 0; i < this.hiddenLayers.length; i++) {
      this.model.add(tf.layers.dense({
        units: this.hiddenLayers[i],
        inputShape: i === 0 ? [numFeatures] : undefined,
        activation: 'relu'
      }));
    }

    this.model.add(tf.layers.dense({
      units: numOutputs,
      activation: isBinary ? 'sigmoid' : 'softmax'
    }));

    let optimizer: tf.Optimizer;
    if (this.optimizerName === 'sgd') optimizer = tf.train.sgd(this.learningRate);
    else if (this.optimizerName === 'rmsprop') optimizer = tf.train.rmsprop(this.learningRate);
    else optimizer = tf.train.adam(this.learningRate);

    this.model.compile({
      optimizer,
      loss: isBinary ? 'binaryCrossentropy' : 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });

    const xTensor = tf.tensor2d(X);
    const yTensor = isBinary ? tf.tensor2d(y, [y.length, 1]) : tf.tensor1d(y, 'int32');

    this.lossHistory = [];

    await this.model.fit(xTensor, yTensor, {
      epochs: this.epochs,
      batchSize: this.batchSize,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (logs) this.lossHistory.push(logs.loss);
          if (onProgress && epoch % Math.ceil(this.epochs / 10) === 0) {
            onProgress(`Epoch ${epoch + 1}/${this.epochs} - loss: ${logs?.loss.toFixed(4)}`);
          }
        }
      }
    });

    xTensor.dispose();
    yTensor.dispose();
  }

  predict(X: number[][]): number[] {
    if (!this.model) return Array(X.length).fill(0);
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
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

  predictProba(X: number[][]): number[][] | number[] {
    if (!this.model) return [];
    return tf.tidy(() => {
      const preds = this.model!.predict(tf.tensor2d(X)) as tf.Tensor;
      const isBinary = this.classes.length <= 2;
      if (isBinary) return Array.from(preds.dataSync());
      
      const arr = Array.from(preds.dataSync());
      const res: number[][] = [];
      const numClasses = this.classes.length;
      for (let i = 0; i < arr.length; i += numClasses) {
        res.push(arr.slice(i, i + numClasses));
      }
      return res;
    });
  }
}
