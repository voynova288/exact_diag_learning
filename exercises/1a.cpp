#include <linear_tools.h>
#include <matplotlibcpp.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <bitset>
#include <complex>
#include <iostream>
#include <omp.h>

using namespace std;
using namespace std::complex_literals;
using namespace Eigen;
namespace plt = matplotlibcpp;

Matrix2cd sigma_x() {
  Matrix2cd result;
  result << 0, 1, 1, 0;
  return result;
}

Matrix2cd sigma_y() {
  Matrix2cd result;
  result << 0, -1.0i, 1.0i, 0;
  return result;
}

Matrix2cd sigma_z() {
  Matrix2cd result;
  result << 1, 0, 0, -1;
  return result;
}

//*将编号转换为对应的一维链的自旋态
vector<int> int_to_state(int number, int length) {
  vector<int> state(length);

  if (number < 0) {
    throw InvalidInput;
  }

  int i = 0;
  while (number > 0) {
    state[i] = number % 2;
    number /= 2;
  }

  reverse(state.begin(), state.end());

  return state;
}

//*将一维链的自旋态转换为对应的编号
int state_to_int(vector<int> state) {
  int number = 0;
  int length = state.size();

  reverse(state.begin(), state.end());

  for (int i = 0; i < length; i++) {
    number += state[i] * pow(2, i);
  }

  return number;
}

//*反转num的第i位（二进制）的0,1（即反转自旋）
int flip_bit(int num, int i) {
  int mask = 1 << (i - 1);
  return num ^ mask;
}

//* exercise 1a的哈密顿量
MatrixXd hamiltonian(double J, double h_x, int length) {
  MatrixXd result = MatrixXcd::Zero(pow(2, length), pow(2, length));

#pragma omp parallel for
  for (int i = 0; i < pow(2, length); i++) {
    for (int j = 0; j < pow(2, length); j++) {
      if ((abs(i - j) == 0)) {
        //*考虑了周期性边界条件
        double elem = 0.0;

        //*表示一维链的状态，右矢
        vector<int> state_ket(length);
        state_ket = int_to_state(j, length);

        for (int k = 0; k < length; k++) {
          elem += (state_ket[k % length] - 0.5) *
                  (state_ket[(k + 1) % length] - 0.5);
        }
        elem *= -4 * J;

        result(i, j) = elem;
      } else {
        //*磁场那一项
        for(int k=0;k<length;k++){
          if(flip_bit(j,k) == i){
            result(i,j) = -h_x;
            break;
          }
        }
      }
    }
  }

  return result;
}

vector<VectorXd> Direct_Calculation(double J, int length,
                                    vector<double> h_x_list) {
  vector<VectorXd> Eig_values;

  for (double h_x : h_x_list) {
    MatrixXcd H = hamiltonian(J, h_x, length);
    SelfAdjointEigenSolver<MatrixXd> solver(H);
    eig_h result;

    result.value = solver.eigenvalues().real();
    result.vec = solver.eigenvectors();
    sort_eig(result);

    Eig_values.push_back(result.value);
  }

  return Eig_values;
}

void plot_energy(vector<double> h_x_list, vector<VectorXd> eig_values,
                 int energy_num = 10, string title = "Energy",
                 string x_label = "$h_x$", string y_label = "$E-E_0$",
                 array<int, 2> y_lim = {0, 6}) {
  int hx_size = h_x_list.size();
  vector<double> y(hx_size);

  for (int i = 0; i < energy_num; i++) {
    for (int j = 0; j < hx_size; j++) {
      y[j] = eig_values[j][i];
    }
    plt::scatter(h_x_list, y, 5.0);
  }

  plt::title(title);
  plt::xlabel(x_label);
  plt::ylabel(y_label);
  // plt::ylim(y_lim[0], y_lim[1]);
  plt::show();
}

int main() {
  double J = 1.0;
  int length = 12;
  int point_num = 20;
  vector<double> hx_list(point_num);
  vector<VectorXd> eig_values;

  for (int i = 0; i < point_num; i++) {
    hx_list[i] = i * 2.0 / (point_num - 1);
  }

  cout << hamiltonian(J, 1.0, length) << endl;
  eig_values = Direct_Calculation(J, length, hx_list);
  plot_energy(hx_list, eig_values);

  return 0;
}