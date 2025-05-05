import * as tf from '@tensorflow/tfjs';

export default async function trainModel() {
  // 1. Create a sequential model
  const model = tf.sequential();

  // 2. Add a single dense layer
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // 3. Compile the model with optimizer and loss function
  model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

  // 4. Prepare training data
  const xs = tf.tensor2d([0, 1, 2, 3, 4], [5, 1]);  // input: x
  const ys = tf.tensor2d([1, 3, 5, 7, 9], [5, 1]);  // output: y = 2x + 1

  // 5. Train the model
  await model.fit(xs, ys, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}`);
      }
    }
  });

  // 6. Make a prediction
  const output = model.predict(tf.tensor2d([10], [1, 1]));
  output.print();  // Should be close to 11
}
