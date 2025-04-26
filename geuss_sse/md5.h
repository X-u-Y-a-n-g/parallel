#ifndef MD5_H
#define MD5_H

#include <string>
#include <cstdint>
// 将 arm_neon.h 替换为 SSE 相关头文件
#include <emmintrin.h>  // SSE2
#include <smmintrin.h>  // SSE4.1
#include <vector>

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef uint32_t bit32;

// MD5的一系列参数。参数是固定的
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

// 修改 SIMD 相关的宏定义
#define F_SSE(x, y, z) _mm_or_si128(_mm_and_si128((x), (y)), _mm_and_si128(_mm_andnot_si128(x, _mm_set1_epi32(-1)), (z)))
#define G_SSE(x, y, z) _mm_or_si128(_mm_and_si128((x), (z)), _mm_and_si128((y), _mm_andnot_si128(z, _mm_set1_epi32(-1))))
#define H_SSE(x, y, z) _mm_xor_si128(_mm_xor_si128((x), (y)), (z))
#define I_SSE(x, y, z) _mm_xor_si128((y), _mm_or_si128((x), _mm_andnot_si128(z, _mm_set1_epi32(-1))))

// 修改循环左移宏定义
#define ROTATELEFT_SSE(num, n) \
    _mm_or_si128(_mm_slli_epi32((num), (n)), _mm_srli_epi32((num), 32 - (n)))

// 修改向量化操作宏定义
#define FF_VEC(a, b, c, d, x, s, t) \
    a = _mm_add_epi32(a, _mm_add_epi32(F_SSE(b, c, d), _mm_add_epi32(x, _mm_set1_epi32(t)))); \
    a = ROTATELEFT_SSE(a, s); \
    a = _mm_add_epi32(a, b);

#define GG_VEC(a, b, c, d, x, s, t) \
    a = _mm_add_epi32(a, _mm_add_epi32(G_SSE(b, c, d), _mm_add_epi32(x, _mm_set1_epi32(t)))); \
    a = ROTATELEFT_SSE(a, s); \
    a = _mm_add_epi32(a, b);

#define HH_VEC(a, b, c, d, x, s, t) \
    a = _mm_add_epi32(a, _mm_add_epi32(H_SSE(b, c, d), _mm_add_epi32(x, _mm_set1_epi32(t)))); \
    a = ROTATELEFT_SSE(a, s); \
    a = _mm_add_epi32(a, b);

#define II_VEC(a, b, c, d, x, s, t) \
    a = _mm_add_epi32(a, _mm_add_epi32(I_SSE(b, c, d), _mm_add_epi32(x, _mm_set1_epi32(t)))); \
    a = ROTATELEFT_SSE(a, s); \
    a = _mm_add_epi32(a, b);

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
// 定义了一系列MD5中的具体函数
// 这四个计算函数是需要你进行SIMD并行化的
// 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
// 定义了一系列MD5中的具体函数
// 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
// 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))

#define FF(a, b, c, d, x, s, ac) { \
  (a) += F ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define GG(a, b, c, d, x, s, ac) { \
  (a) += G ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define HH(a, b, c, d, x, s, ac) { \
  (a) += H ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define II(a, b, c, d, x, s, ac) { \
  (a) += I ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5哈希值
 * @param input 输入字符串
 * @param[out] state 输出状态数组（4个32位整数）
 */
void MD5Hash(std::string input, bit32 *state);

// 添加新的函数声明
void MD5HashBatch(const std::vector<std::string>& inputs, std::vector<bit32*>& states);


#endif // MD5_H

