// Intraregister Sorting via Bitonic merge sort
// Inspired by sorting networks, to do hardware sorting of electrical wires.
// https://en.wikipedia.org/wiki/Bitonic_sorter

#include <iostream>

#include "simd.h"

SK_FORCE_INLINE __m512 Sort(__m512 values) {
  const __m512i perm0 =
      X::SIMD::MakePermutationMap<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12,
                                  15, 14>();
  constexpr uint32_t mask0 =
      X::SIMD::MakeBitMask<0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>();

  const __m512i perm1 = X::SIMD::MakePermutationMap<3, 2, 1, 0, 7, 6, 5, 4, 11,
                                                    10, 9, 8, 15, 14, 13, 12>();
  constexpr uint32_t mask1 =
      X::SIMD::MakeBitMask<0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1>();

  const __m512i perm2 =
      X::SIMD::MakePermutationMap<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12,
                                  15, 14>();
  constexpr uint32_t mask2 =
      X::SIMD::MakeBitMask<0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>();

  const __m512i perm3 = X::SIMD::MakePermutationMap<7, 6, 5, 4, 3, 2, 1, 0, 15,
                                                    14, 13, 12, 11, 10, 9, 8>();
  constexpr uint32_t mask3 =
      X::SIMD::MakeBitMask<0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1>();

  const __m512i perm4 = X::SIMD::MakePermutationMap<2, 3, 0, 1, 6, 7, 4, 5, 10,
                                                    11, 8, 9, 14, 15, 12, 13>();
  constexpr uint32_t mask4 =
      X::SIMD::MakeBitMask<0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1>();

  const __m512i perm5 =
      X::SIMD::MakePermutationMap<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12,
                                  15, 14>();
  constexpr uint32_t mask5 =
      X::SIMD::MakeBitMask<0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>();
  values = X::SIMD::CompareWithExchange(values, perm0, mask0);
  values = X::SIMD::CompareWithExchange(values, perm1, mask1);
  values = X::SIMD::CompareWithExchange(values, perm2, mask2);
  values = X::SIMD::CompareWithExchange(values, perm3, mask3);
  values = X::SIMD::CompareWithExchange(values, perm4, mask4);
  values = X::SIMD::CompareWithExchange(values, perm5, mask5);
  return values;
}

static constexpr int kTotalTests = 1;

int main() {
  int passed = 0;
  __m512 values =
      _mm512_set_ps(15, 16, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2);
  __m512 want =
      _mm512_set_ps(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  __m512 got = Sort(values);
  if (_mm512_cmpeq_ps_mask(got, want) == 0xFFFFu) passed++;
  std::cout << "test_intraregister_sort: PASSED " << passed << "/"
            << kTotalTests << "\n";
  return 0;
}
