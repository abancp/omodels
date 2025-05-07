#include "../src/nn.h"
#include "../utils/activations.h"
#include "../utils/functions.h"
#include <cerrno>
#include <cmath>
#include <iostream>
#include <pthread.h>
#include <vector>

int main() {
  Linear *l1 = new Linear(2, 2, 0);
  Linear *l2 = new Linear(2, 2, 1);
  double lr = 0.1;

  std::vector<double> a1 = l1->forward({1, 2});
  std::vector<double> y_cap = l2->forward(a1);
  print(a1);

  print(y_cap);
  double loss = categoricalCrossEntropy(y_cap, {1, 0});
  std::cout << "Loss : " << loss;

  // backpropagation
  std::vector<double> d2 = outputCCESoftmaxDZ(y_cap, {1, 0});
  print(d2);

  std::vector<std::vector<double>> dw2(d2.size(),
                                       std::vector<double>(a1.size()));

  for (int i = 0; i < a1.size(); i++) {
    for (int j = 0; j < d2.size(); j++) {
      dw2[i][j] = a1[i] * d2[j];
    }
  }

  updateWandB(l2, dw2, d2, lr);

  // in hidden layers (backprop)

  std::vector<double> d1(d2.size(), 0);
  for (int i = 0; i < l2->w.size(); i++) {
    for (int j = 0; j < l2->w[i].size(); j++) {
      d1[i] += l2->w[i][j] * d2[j];
    }
  }

  print(d1);

  // element wise mul with relu derivative

  for (int i = 0; i < a1.size(); i++) {
    if (a1[i] < 0) {
      d1[i] *= 0;
    } else {
      d1[i] *= 1;
    }
  }

  std::vector<std::vector<double>> dw1(l1->ips, std::vector<double>(l1->ops));
  for (int i = 0; i < l1->ips; i++) {
    for (int j = 0; j < l1->ops; j++) {
      dw1[i][j] = l1->x[i] * d1[j];
    }
  }

  updateWandB(l1, dw1, d1, lr);
  a1 = l1->forward({1, 2});
  y_cap = l2->forward(a1);
  print(a1);

  print(y_cap);
  loss = categoricalCrossEntropy(y_cap, {1, 0});
  std::cout << "Loss : " << loss;
}
