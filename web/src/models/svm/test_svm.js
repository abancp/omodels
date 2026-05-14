const math = require('./math.js');
let w = math.initWeights(1, 42);
const C = "1.0";
const lr = "0.05";
const points = math.generateSVMData("blobs", 100, 0.15);
for (let i=0; i<5; i++) {
  const result = math.trainStep(w, points, 1, C, lr, i);
  w = result.weights;
  console.log(`Step ${i}: loss=${result.loss} w=[${w.join(', ')}]`);
}
