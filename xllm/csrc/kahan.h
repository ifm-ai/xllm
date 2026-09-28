#pragma once

#include <c10/macros/Macros.h>
#include <thrust/pair.h>

#include "welford.h"
#include "welford_decay.h"

namespace xllm {
namespace utils {

// https://en.wikipedia.org/wiki/Kahan_summation_algorithm
template <typename T>
C10_HOST_DEVICE thrust::pair<T, T> KahanAdd(T x, T sum, T c) {
  const T y = x - c;
  const T t = sum + y;
  c = (t - sum) - y;
  return thrust::make_pair(t, c);
}

template <typename T>
class KahanWrapper {
 public:
  explicit C10_HOST_DEVICE KahanWrapper(T v) : value_(v) {}

  C10_HOST_DEVICE T operator*() const { return value_; }

  C10_HOST_DEVICE const T* operator->() const { return &value_; }

  C10_HOST_DEVICE T value() const { return value_; }

  C10_HOST_DEVICE KahanWrapper<T>& operator+=(T x) {
    thrust::tie(value_, c_) = KahanAdd(x, value_, c_);
    return *this;
  }

 protected:
  T value_ = T(0);
  T c_ = T(0);
};

template <typename T>
C10_HOST_DEVICE KahanWrapper<T> operator+(KahanWrapper<T> sum, T x) {
  return sum += x;
}

template <typename T>
class KahanWrapper<WelfordData<T>> {
 public:
  explicit C10_HOST_DEVICE KahanWrapper(WelfordData<T> v) : value_(v) {}

  C10_HOST_DEVICE WelfordData<T> operator*() const { return value_; }

  C10_HOST_DEVICE const WelfordData<T>* operator->() const { return &value_; }

  WelfordData<T> value() const { return value_; }

  C10_HOST_DEVICE KahanWrapper<WelfordData<T>>& operator+=(T x) {
    ++value_.m0;
    const T coef = T(1) / static_cast<T>(value_.m0);
    const T delta1 = x - value_.m1;
    thrust::tie(value_.m1, c1_) = KahanAdd(coef * delta1, value_.m1, c1_);
    const T delta2 = delta1 * (x - value_.m1) - value_.m2;
    thrust::tie(value_.m2, c2_) = KahanAdd(coef * delta2, value_.m2, c2_);
    return *this;
  }

  C10_HOST_DEVICE KahanWrapper<WelfordData<T>>& operator+=(WelfordData<T> x) {
    value_.m0 += x.m0;
    const T coef = value_.m0 == 0 ? T(0) : static_cast<T>(x.m0) / static_cast<T>(value_.m0);
    const T delta1 = x.m1 - value_.m1;
    const T delta2 = x.m2 + (T(1) - coef) * delta1 * delta1 - value_.m2;
    thrust::tie(value_.m1, c1_) = KahanAdd(coef * delta1, value_.m1, c1_);
    thrust::tie(value_.m2, c2_) = KahanAdd(coef * delta2, value_.m2, c2_);
    return *this;
  }

 protected:
  WelfordData<T> value_ = {};
  T c1_ = T(0);
  T c2_ = T(0);
};

template <typename T>
C10_HOST_DEVICE KahanWrapper<WelfordData<T>> operator+(KahanWrapper<WelfordData<T>> sum, T x) {
  return sum += x;
}

template <typename T>
class KahanWrapper<WelfordDecayData<T>> {
 public:
  explicit C10_HOST_DEVICE KahanWrapper(WelfordDecayData<T> v) : value_(v) {}

  C10_HOST_DEVICE WelfordDecayData<T> operator*() const { return value_; }

  C10_HOST_DEVICE const WelfordDecayData<T>* operator->() const { return &value_; }

  WelfordDecayData<T> value() const { return value_; }

  C10_HOST_DEVICE KahanWrapper<WelfordDecayData<T>>& operator+=(T x) { assert(false); }

  C10_HOST_DEVICE KahanWrapper<WelfordDecayData<T>>& operator+=(WelfordDecayData<T> x) {
    if (!x.bos_exists) {
      value_.count += x.count;
      value_.beta1_pow *= x.beta1_pow;
      value_.beta2_pow *= x.beta2_pow;
      if (x.count == 1) {
        thrust::tie(value_.mean, c1_) =
            KahanAdd((1 - x.beta1_pow) * (x.mean - value_.mean), value_.mean, c1_);
        thrust::tie(value_.var, c1_) =
            KahanAdd((1 - x.beta2_pow) * (x.var - value_.var), value_.var, c2_);
      } else {
        thrust::tie(value_.mean, c1_) =
            KahanAdd(x.mean - (1 - x.beta1_pow) * value_.mean, value_.mean, c1_);
        thrust::tie(value_.var, c2_) =
            KahanAdd(x.var - (1 - x.beta2_pow) * value_.var, value_.var, c2_);
      }
    } else {
      value_.count = x.count;
      value_.beta1_pow = x.beta1_pow;
      value_.beta2_pow = x.beta2_pow;
      if (x.count == 1) {
        value_.mean = (1 - x.beta1_pow) * x.mean;
        value_.var = (1 - x.beta2_pow) * x.var;
      } else {
        value_.mean = x.mean;
        value_.var = x.var;
      }
      value_.bos_exists = true;
      c1_ = 0;
      c2_ = 0;
    }
    return *this;
  }

 protected:
  WelfordDecayData<T> value_ = {};
  T c1_ = T(0);
  T c2_ = T(0);
};

template <typename T>
C10_HOST_DEVICE KahanWrapper<WelfordDecayData<T>> operator+(KahanWrapper<WelfordDecayData<T>> sum, T x) {
  return sum += x;
}

}  // namespace utils
}  // namespace xllm
