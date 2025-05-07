#include "nn.h"
#include "../utils/activations.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

Linear::Linear(int givenIps, int givenOps, int givenActivation,
               std::vector<std::vector<double>> gw, std::vector<double> gb)
    : ips(givenIps), ops(givenOps), activation(givenActivation), w(gw), b(gb) {
  // w = std::vector<std::vector<double>>(ips,std::vector<double>(ops,1.0));
  // w = {{2, 3, 4}, {4, 5, 6}};
  // b = {1, 2, 3};
}

std::vector<double> Linear::forward(std::vector<double> givenX) {
  // forward pass
  x = givenX;
  std::vector<double> z(ops);
  for (int i = 0; i < ips; i++) {
    for (int j = 0; j < ops; j++) {
      z[j] += w[i][j] * x[i];
      if (i == 0) {
        z[j] += b[j];
      }
    }
  }

  switch (activation) {
  case 0:
    return relu(z); // Relu
  case 1:
    return softmax(z); // softmax
  case 2:
    return sigmoid(z); // sigmoid
  default:
    std::cout << "Activation function not found! , exiting! given : "
              << activation;
    exit(1);
  }
}

void Linear::backward(std::vector<double> y_cap) {}

NN::NN(std::vector<std::vector<double>> config) {
  depth = config.size();
  std::vector<Linear *> layers(depth);
  for (int i = 0; i < depth; i++) {
    layers[i] = new Linear(config[i][0], config[i][1], config[i][2]);
  }
}

std::vector<double> NN::forward(std::vector<double> x) {
  for (Linear *layer : layers) {
    x = layer->forward(x);
  }
  return x;
}

// std::vector<double> Linear::backward(std::vector<double>);

// std::vector<double> Linear::backword() { return {1, 1, 1}; }
double categoricalCrossEntropy(std::vector<double> y_cap, std::vector<int> y) {
  // y_cap is prediction , y is one_hot labeled true y

  for (int i = 0; i < y.size(); i++) {
    if (y[i] == 0) {
      continue;
    }
    return -std::log(y_cap[i]);
  }
  std::cout << "Invalid one-hot label - y";
  exit(1);
}

std::vector<double> outputCCESoftmaxDZ(std::vector<double> y_cap,
                                       std::vector<int> y) {
  for (int i = 0; i < y.size(); i++) {
    if (y[i] == 1) {
      y_cap[i] -= 1;
    }
  }
  return y_cap;
}

void updateWandB(Linear *l, std::vector<std::vector<double>> dw,
                 std::vector<double> db, double lr) {
  for (int i = 0; i < l->ips; i++) {
    for (int j = 0; j < l->ops; j++) {
      l->w[i][j] -= lr * dw[i][j];
      if (i == 0) {
        l->b[j] -= lr * db[j];
      }
    }
  }
}
