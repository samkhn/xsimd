#ifndef _PERF_SIMD_H_
#define _PERF_SIMD_H_

#include <cstdint>
#include <cstdio>
#include <type_traits>

#ifdef __OPTIMIZE__
#include <x86intrin.h>
#define SK_FORCE_INLINE inline __attribute__((__always_inline_))
#else
#define __OPTIMIZE__
#include <x86intrin.h>
#undef __OPTIMIZE__
#define SK_FORCE_INLINE inline
#endif

namespace X {
namespace SIMD {

// __m512 is the 512 bit floating type.
// __m512i is the 512 bit integer type.
// We'll use uint32_t as a mask and convert to __mmask16 as needed.
// We use unaligned load/stores to more precise control.

// Broadcasts `fill` (32 bit float or int) to 512 bit register
SK_FORCE_INLINE __m512 LoadValue(float fill) { return _mm512_set1_ps(fill); }

SK_FORCE_INLINE __m512i LoadValue(int32_t fill) {
  return _mm512_set1_epi32(fill);
}

SK_FORCE_INLINE __m512 LoadFrom(const float *src) {
  return _mm512_loadu_ps(src);
}

// For every bit, if mask bit is 0, we get ith bit of fill. If mask bit is 1, we
// get ith bit of src.
SK_FORCE_INLINE __m512 MaskedLoadFrom(const float *src, float fill,
                                      uint32_t mask) {
  return _mm512_mask_loadu_ps(_mm512_set1_ps(fill), (__mmask16)mask, src);
}

SK_FORCE_INLINE __m512 MaskedLoadFrom(const float *src, __m512 fill,
                                      uint32_t mask) {
  return _mm512_mask_loadu_ps(fill, (__mmask16)mask, src);
}

SK_FORCE_INLINE void StoreTo(float *dst, __m512 r) {
  _mm512_mask_storeu_ps(dst, (__mmask16)0xFFFFu, r);
}

// For every bit i, if maks bit is 0, we get the ith bit of r. If mask bit is 1,
// we get the ith bit from dst (aka whatever is already there).
SK_FORCE_INLINE void MaskedStoreTo(float *dst, __m512 r, uint32_t mask) {
  _mm512_mask_storeu_ps(dst, (__mmask16)mask, r);
}

// TODO: does std::bitset replace this?
template <unsigned A = 0, unsigned B = 0, unsigned C = 0, unsigned D = 0,
          unsigned E = 0, unsigned F = 0, unsigned G = 0, unsigned H = 0,
          unsigned I = 0, unsigned J = 0, unsigned K = 0, unsigned L = 0,
          unsigned M = 0, unsigned N = 0, unsigned O = 0, unsigned P = 0>
SK_FORCE_INLINE constexpr uint32_t MakeBitMask() {
  static_assert((A < 2) && (B < 2) && (C < 2) && (D < 2) && (E < 2) &&
                (F < 2) && (G < 2) && (H < 2) && (I < 2) && (J < 2) &&
                (K < 2) && (L < 2) && (M < 2) && (N < 2) && (O < 2) && (P < 2));
  return ((A << 0) | (B << 1) | (C << 2) | (D << 3) | (E << 4) | (F << 5) |
          (G << 6) | (H << 7) | (I << 8) | (J << 9) | (K << 10) | (L << 11) |
          (M << 12) | (N << 13) | (O << 14) | (P << 15));
}

// Where mask bit is 0, we get a. Where mask bit is 1, we get b.
SK_FORCE_INLINE __m512 Blend(__m512 a, __m512 b, uint32_t mask) {
  return _mm512_mask_blend_ps((__mmask16)mask, a, b);
}

// TODO: improve this description
// Perm contains 16 integers of 32 bits.
// Result's bit n contains r_(perm's nth integer)
// E.g. you can reverse the order of bits in register r by calling
// Permute(r, (512 bit integer containing 15...0)).
SK_FORCE_INLINE __m512 Permute(__m512 r, __m512i perm) {
  return _mm512_permutexvar_ps(perm, r);
}

// If mask is 0, r at that bit does not change
// If mask is 1, we apply Permute based on perm
SK_FORCE_INLINE __m512 MaskedPermute(__m512 a, __m512 b, __m512i perm,
                                     uint32_t mask) {
  return _mm512_mask_permutexvar_ps(a, (__mmask16)mask, perm, b);
}

template <unsigned A, unsigned B, unsigned C, unsigned D, unsigned E,
          unsigned F, unsigned G, unsigned H, unsigned I, unsigned J,
          unsigned K, unsigned L, unsigned M, unsigned N, unsigned O,
          unsigned P>
SK_FORCE_INLINE __m512i MakePermutationMap() {
  static_assert((A < 16) && (B < 16) && (C < 16) && (D < 16) && (E < 16) &&
                (F < 16) && (G < 16) && (H < 16) && (I < 16) && (J < 16) &&
                (K < 16) && (L < 16) && (M < 16) && (N < 16) && (O < 16) &&
                (P < 16));
  return _mm512_setr_epi32(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
}

template <int R>
SK_FORCE_INLINE __m512 Rotate(__m512 r) {
  if constexpr ((R % 16) == 0) {
    return r;
  }
  constexpr int S = (R > 0) ? (16 - (R % 16)) : -R;
  constexpr int A = (S + 0) % 16;
  constexpr int B = (S + 1) % 16;
  constexpr int C = (S + 2) % 16;
  constexpr int D = (S + 3) % 16;
  constexpr int E = (S + 4) % 16;
  constexpr int F = (S + 5) % 16;
  constexpr int G = (S + 6) % 16;
  constexpr int H = (S + 7) % 16;
  constexpr int I = (S + 8) % 16;
  constexpr int J = (S + 9) % 16;
  constexpr int K = (S + 10) % 16;
  constexpr int L = (S + 11) % 16;
  constexpr int M = (S + 12) % 16;
  constexpr int N = (S + 13) % 16;
  constexpr int O = (S + 14) % 16;
  constexpr int P = (S + 15) % 16;
  return _mm512_permutexvar_ps(
      _mm512_setr_epi32(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P), r);
}

template <int R>
SK_FORCE_INLINE __m512 RotateLeft(__m512 r) {
  static_assert(R >= 0);
  return Rotate<-R>(r);
}

template <int R>
SK_FORCE_INLINE __m512 RotateRight(__m512 r) {
  static_assert(R >= 0);
  return Rotate<R>(r);
}

template <int S>
SK_FORCE_INLINE uint32_t ShiftLeftBlendMask() {
  static_assert(S >= 0 && S <= 32);
  return ((uint32_t)~0u << S);
}

template <int S>
SK_FORCE_INLINE __m512 ShiftLeft(__m512 r) {
  static_assert(S >= 0 && S <= 16);
  return Blend(RotateLeft<S>(r), LoadValue(0.0f), ShiftLeftBlendMask<S>());
}

template <int S>
SK_FORCE_INLINE __m512 ShiftLeftWithCarry(__m512 a, __m512 b) {
  static_assert(S >= 0 && S <= 16);
  return Blend(RotateLeft<S>(a), RotateLeft<S>(b), ShiftLeftBlendMask<S>());
}

template <int S>
SK_FORCE_INLINE uint32_t ShiftRightBlendMask() {
  static_assert(S >= 0 && S <= 32);
  return ((uint32_t)~0u >> S);
}

template <int S>
SK_FORCE_INLINE __m512 ShiftRight(__m512 r) {
  static_assert(S >= 0 && S <= 16);
  return Blend(RotateRight<S>(r), LoadValue(0.0f), ShiftRightBlendMask<S>());
}

template <int S>
SK_FORCE_INLINE __m512 ShiftRightWithCarry(__m512 a, __m512 b) {
  static_assert(S >= 0 && S <= 16);
  return Blend(RotateRight<S>(a), RotateRight<S>(b), ShiftRightBlendMask<S>());
}

// TODO: test this works with InPlaceShiftLeftWithCarry
template <int S>
SK_FORCE_INLINE __m512i MakeShiftLeftPermutation() {
  static_assert(S >= 0 && S <= 16);
  constexpr int A = (S + 0) % 16;
  constexpr int B = (S + 1) % 16;
  constexpr int C = (S + 2) % 16;
  constexpr int D = (S + 3) % 16;
  constexpr int E = (S + 4) % 16;
  constexpr int F = (S + 5) % 16;
  constexpr int G = (S + 6) % 16;
  constexpr int H = (S + 7) % 16;
  constexpr int I = (S + 8) % 16;
  constexpr int J = (S + 9) % 16;
  constexpr int K = (S + 10) % 16;
  constexpr int L = (S + 11) % 16;
  constexpr int M = (S + 12) % 16;
  constexpr int N = (S + 13) % 16;
  constexpr int O = (S + 14) % 16;
  constexpr int P = (S + 15) % 16;
  return _mm512_setr_epi32(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
}

// Imagine a and b are contiguous in memory [A B]. This bit shifts them together
// by S such that the first S bits of B are now the tailend of where A was and
// we lose the first S bits of A.
template <int S>
SK_FORCE_INLINE void InPlaceShiftLeftWithCarry(__m512 &a, __m512 &b) {
  static_assert(S >= 0 && S <= 16);
  constexpr uint32_t zmask = (0xFFFFu >> (unsigned)S);
  constexpr uint32_t bmask = ~zmask & 0xFFFFu;
  __m512i perm = MakeShiftLeftPermutation<S>();
  a = _mm512_permutex2var_ps(a, perm, b);
  b = _mm512_maskz_permutex2var_ps((__mmask16)zmask, b, perm, b);
}

// d = ab + c
SK_FORCE_INLINE __m512 FusedMultiplyAdd(__m512 a, __m512 b, __m512 c) {
  return _mm512_fmadd_ps(a, b, c);
}

// Min and maximum compares each of the 16, 32-bit values in the 512 bit
// register.
SK_FORCE_INLINE __m512 Minimum(__m512 a, __m512 b) {
  return _mm512_min_ps(a, b);
}

SK_FORCE_INLINE __m512 Maximum(__m512 a, __m512 b) {
  return _mm512_max_ps(a, b);
}

// Vals contains the values you are comparing
// Perm specifies which elements are to be compared
// Mask specifies which of the two elements compared is "greater", where 0 means
// the lesser of the two values is max and 1 means the greater of the two valies
// is max. Takes about 4 intrinsics per invoke.
SK_FORCE_INLINE __m512 CompareWithExchange(__m512 vals, __m512i perm,
                                           uint32_t mask) {
  __m512 exchange = Permute(vals, perm);
  __m512 vmin = Minimum(vals, exchange);
  __m512 vmax = Maximum(vals, exchange);
  return Blend(vmin, vmax, mask);
}

}  // namespace SIMD
}  // namespace X

#endif  // _PERF_SIMD_H_
