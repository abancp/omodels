#ifndef NN_H
#define NN_H

#include <vector>

class Linear {
public:
  int ips, ops;
  int activation;
  std::vector<std::vector<double>> w;
  std::vector<double> x, z, b;

  Linear(int, int, int);
  std::vector<double> forward(std::vector<double>);

  void backward(std::vector<double>);
};

class NN {
public:
  std::vector<Linear *> layers;
  int depth;
  NN(std::vector<std::vector<double>> config);
  std::vector<double> forward(std::vector<double> x);
  std::vector<double> backword(std::vector<double> y);
};

double categoricalCrossEntropy(std::vector<double>, std::vector<int>);
std::vector<double> outputCCESoftmaxDZ(std::vector<double>, std::vector<int>);
void updateWandB(Linear *l, std::vector<std::vector<double>> dw,
                 std::vector<double> db, double lr);

#endif
