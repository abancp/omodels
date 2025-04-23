#include <chrono>
#include <cmath>
#include <emscripten.h>
#include <emscripten/bind.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;

class NN {

public:
  int no_inputs, no_outputs;
  vector<vector<double>> w; // weight [from_neuron][to_neuron]
  vector<double> b;
  vector<vector<int>> inter_z;
  vector<double> z;
  double lr = 0.01;

  NN(int no_ips , int no_ops) {
    no_inputs = no_ips;
    no_outputs = no_ops;
    // w = vector<vector<int>>(no_ips,vector<int>(no_ops,1));
    w = vector<vector<double>>(no_inputs, vector<double>(no_outputs, 1));
    // b = vector<int>(no_ops,4);
    b = {2, 2, 2};
    inter_z = vector<vector<int>>(no_ips, vector<int>(no_ops));
    z = vector<double>(no_ops);
    cout << "Making Neural Network with input : " << no_ips
         << " output : " << no_ops << " Weights : " << w.size() << " x "
         << w[0].size() << endl;
  }

  void setShape(int no_ips, int no_ops) {
    no_inputs = no_ips;
    no_outputs = no_ops;
    // w = vector<vector<int>>(no_ips,vector<int>(no_ops,1));
    w = vector<vector<double>>(no_inputs, vector<double>(no_outputs, 1));
    // b = vector<int>(no_ops,4);
    b = {2, 2, 2};
    inter_z = vector<vector<int>>(no_ips, vector<int>(no_ops));
    z = vector<double>(no_ops);
    cout << "Making Neural Network with input : " << no_ips
         << " output : " << no_ops << " Weights : " << w.size() << " x "
         << w[0].size() << endl;
  }

  void print_w() {
    for (int i = 0; i < no_inputs; i++) {
      for (int j = 0; j < no_outputs; j++) {
        cout << w[i][j] << " ";
      }
      cout << endl;
    }
  }

  void print_b() {
    for (int j = 0; j < no_outputs; j++) {
      cout << b[j] << " ";
    }
  }

  void forward(vector<double> &ips) {
    int given_ips_len = ips.size();
    if (given_ips_len != no_inputs) {
      cout << "Input size mismatch given : " << given_ips_len
           << " , expected : " << no_inputs;
      return;
    }
    for (int i = 0; i < no_inputs; i++) {
      for (int j = 0; j < no_outputs; j++) {
        // cout << b[j];
        inter_z[i][j] = (ips[i] * w[i][j]);
      }
    }
    for (int i = 0; i < no_inputs; i++) {
      for (int j = 0; j < no_outputs; j++) {
        z[j] += inter_z[i][j];
      }
    }
    for (int j = 0; j < no_outputs; j++) {
      z[j] += b[j];
    }
  }

  void print_z() {
    for (int j = 0; j < no_outputs; j++) {
      cout << z[j] << " ";
    }
  }

  void softmax(vector<double> &logits) {
    double sum = 0.0;
    for (int i = 0; i < no_outputs; i++) {
      logits[i] = exp(logits[i]);
      sum += logits[i];
    }
    for (int i = 0; i < no_outputs; i++) {
      logits[i] /= sum;
    }
  }

  vector<double> find_dz(vector<double> y, int y_cap) {
    // y = [double,double,double] , y_cap = index of true class
    y[y_cap] -= 1;
    return y;
  }

  vector<vector<double>> find_dw(vector<double> dz, vector<double> &x) {
    // dz -> vector:no_ops , x = vector:no_ips
    vector<vector<double>> dw =
        vector<vector<double>>(no_inputs, vector<double>(no_outputs));
    for (int i = 0; i < no_inputs; i++) {
      for (int j = 0; j < no_outputs; j++) {
        dw[i][j] = dz[j] * x[i];
      }
    }
    return dw; // dz is now dw . its multiplied with x
  }

  void relu(vector<double> &logits) {
    for (int i = 0; i < no_outputs; i++) {
      if (logits[i] < 0) {
        logits[i] = 0.0;
      }
    }
  }

  double find_loss(vector<double> y, int y_cap) {
    return -log(y[y_cap] + 1e-8);
  }
  void backrpropagate() {
    // double loss = find_loss()
  }
};

NN *nn = new NN(2, 3);

void train(int ips, int ops, vector<vector<double>> data_x,
           vector<double> data_y, int epoches, double lr) {
  /*
    vector<vector<double>> data_x = {{-1.0, 0.5}, {-0.8, 0.3}, {-1.2, 0.1},
                                     {1.0, 0.5},  {0.8, 0.3},  {1.2, 0.1},
                                     {0.0, 0.0},  {0.2, -0.2}, {-0.2, -0.2}};
    vector<double> data_y = {0, 0, 0, 1, 1, 1, 2, 2, 2};

  */
  // training loop
  nn->setShape(ips, ops);
  for (int epoch = 0; epoch < epoches; epoch++) {
    double loss = 0.0;
    for (int i = 0; i < data_x.size(); i++) {
      nn->forward(data_x[i]);
      nn->softmax(nn->z);
      loss = nn->find_loss(nn->z, data_y[i]);
      vector<double> dz = nn->find_dz(nn->z, data_y[i]);
      // cout<<"DZ : - > "<<endl;
      vector<double> xx = {1.0, 2.0};
      vector<vector<double>> dw = nn->find_dw(dz, data_x[i]);
      // cout << "DW : ->"<<endl;
      for (int k = 0; k < nn->no_inputs; k++) {
        for (int h = 0; h < nn->no_outputs; h++) {
          // cout<<dw[k][h]<<" ";
          nn->w[k][h] -= (lr * dw[k][h]);
        }
      }
      for (int j = 0; j < nn->no_outputs; j++) {
        nn->b[j] -= (lr * dz[j]);
      }
    }
    EM_ASM_({ report_train($0, $1); }, epoch, loss);
    emscripten_sleep(0); // yield to JS

    // cout << "Epoch : " << epoch << " Loss : " << loss << endl;
  }
  cout << "\n";
}

int inference(vector<double> x) {
  nn->forward(x);
  nn->softmax(nn->z);
  int maxI = 0;
  for (int i = 0; i < nn->no_outputs; i++) {
    if (nn->z[i] > nn->z[maxI]) {
      maxI = i;
    }
  }
  return maxI;
}

EMSCRIPTEN_BINDINGS(training_bindings) {
  emscripten::function("train", &train);
  emscripten::function("inference", &inference);

  emscripten::register_vector<vector<double>>("VectorVectorDouble");
  emscripten::register_vector<double>("VectorDouble");
}
