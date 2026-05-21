import { generateSVMData, initWeights, trainStep } from "./math";

const points = generateSVMData("blobs", 100, 0.15);
let w = initWeights("linear", 42);
const C = 1.0;
const lr = 0.05;

for (let i=0; i<500; i++) {
  const result = trainStep(w, points, "linear", C, lr, i);
  w = result.weights;
  if (i % 50 === 0) {
    console.log(`Step ${i}: loss=${result.loss.toFixed(4)} w=[${w.map(x=>x.toFixed(4)).join(', ')}]`);
  }
}
