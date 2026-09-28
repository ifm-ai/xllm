#pragma once

#include <c10/macros/Macros.h>

#include <type_traits>

namespace xllm {
namespace utils {

template <typename T>
struct WelfordDecayData {
  int64_t count = 0;
  T beta1_pow = T(1);
  T beta2_pow = T(1);
  T mean = T(0);
  T var = T(0);
  bool bos_exists = false;

  C10_HOST_DEVICE WelfordDecayData<T>& operator+=(T x) {
    assert(false);
    // ++count;
    // beta1_pow *= beta1;
    // beta2_pow *= beta2;

    // mean = mean * beta1 + (1 - beta1) * x;
    // var = var * beta2 + (1 - beta2) * T(0);
    // return *this;
  }

  C10_HOST_DEVICE WelfordDecayData<T>& operator+=(WelfordDecayData<T> x) {
    if (!x.bos_exists) {
      count += x.count;
      beta1_pow *= x.beta1_pow;
      beta2_pow *= x.beta2_pow;
      if (x.count == 1) {
        mean = mean * x.beta1_pow + (1 - x.beta1_pow) * x.mean;
        var = var * x.beta2_pow + (1 - x.beta2_pow) * x.var;
      } else {
        mean = mean * x.beta1_pow + x.mean;
        var = var * x.beta2_pow + x.var;
      }
    } else {
      count = x.count;
      beta1_pow = x.beta1_pow;
      beta2_pow = x.beta2_pow;
      mean = x.mean;
      var = x.var;
      bos_exists = true;
    }
    return *this;
  }
};

template <typename T>
C10_HOST_DEVICE WelfordDecayData<T> operator+(WelfordDecayData<T> m, T x) {
  return m += x;
}

template <typename T>
C10_HOST_DEVICE WelfordDecayData<T> operator+(WelfordDecayData<T> lhs,
                                              WelfordDecayData<T> rhs) {
  return lhs += rhs;
}

}  // namespace utils
}  // namespace xllm
