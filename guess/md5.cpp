#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <vector>
#include <cstring>

using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte* StringProcess(string input, int *n_byte)
{
    // 使用固定大小的预分配缓冲区
    static thread_local vector<Byte> buffer(8192);
    
    const int length = input.length();
    const int bitLength = length * 8;
    
    // 计算填充大小
    int paddingBits = ((448 - (bitLength + 1) % 512) + 512) % 512;
    int paddingBytes = (paddingBits + 1) / 8;
    int paddedLength = length + paddingBytes + 8;
    
    // 确保缓冲区足够大
    if(buffer.size() < paddedLength) {
        buffer.resize(paddedLength);
    }
    
    // 快速复制
    memcpy(buffer.data(), input.c_str(), length);
    
    // 添加填充
    buffer[length] = 0x80;
    memset(buffer.data() + length + 1, 0, paddingBytes - 1);
    
    // 添加消息长度
    uint64_t messageBits = bitLength;
    memcpy(buffer.data() + paddedLength - 8, &messageBits, 8);
    
    *n_byte = paddedLength;
    
    // 返回新分配的内存
    Byte* result = new Byte[paddedLength];
    memcpy(result, buffer.data(), paddedLength);
    return result;
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(string input, bit32 *state)
{

	Byte *paddedMessage;
	int *messageLength = new int[1];
	for (int i = 0; i < 1; i += 1)
	{
		paddedMessage = StringProcess(input, &messageLength[i]);
		// cout<<messageLength[i]<<endl;
		assert(messageLength[i] == messageLength[0]);
	}
	int n_blocks = messageLength[0] / 64;

	// bit32* state= new bit32[4];
	state[0] = 0x67452301;
	state[1] = 0xefcdab89;
	state[2] = 0x98badcfe;
	state[3] = 0x10325476;

	// 逐block地更新state
	for (int i = 0; i < n_blocks; i += 1)
	{
		bit32 x[16];

		// 下面的处理，在理解上较为复杂
		for (int i1 = 0; i1 < 16; ++i1)
		{
			x[i1] = (paddedMessage[4 * i1 + i * 64]) |
					(paddedMessage[4 * i1 + 1 + i * 64] << 8) |
					(paddedMessage[4 * i1 + 2 + i * 64] << 16) |
					(paddedMessage[4 * i1 + 3 + i * 64] << 24);
		}

		bit32 a = state[0], b = state[1], c = state[2], d = state[3];

		auto start = system_clock::now();
		/* Round 1 */
		FF(a, b, c, d, x[0], s11, 0xd76aa478);
		FF(d, a, b, c, x[1], s12, 0xe8c7b756);
		FF(c, d, a, b, x[2], s13, 0x242070db);
		FF(b, c, d, a, x[3], s14, 0xc1bdceee);
		FF(a, b, c, d, x[4], s11, 0xf57c0faf);
		FF(d, a, b, c, x[5], s12, 0x4787c62a);
		FF(c, d, a, b, x[6], s13, 0xa8304613);
		FF(b, c, d, a, x[7], s14, 0xfd469501);
		FF(a, b, c, d, x[8], s11, 0x698098d8);
		FF(d, a, b, c, x[9], s12, 0x8b44f7af);
		FF(c, d, a, b, x[10], s13, 0xffff5bb1);
		FF(b, c, d, a, x[11], s14, 0x895cd7be);
		FF(a, b, c, d, x[12], s11, 0x6b901122);
		FF(d, a, b, c, x[13], s12, 0xfd987193);
		FF(c, d, a, b, x[14], s13, 0xa679438e);
		FF(b, c, d, a, x[15], s14, 0x49b40821);

		/* Round 2 */
		GG(a, b, c, d, x[1], s21, 0xf61e2562);
		GG(d, a, b, c, x[6], s22, 0xc040b340);
		GG(c, d, a, b, x[11], s23, 0x265e5a51);
		GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GG(a, b, c, d, x[5], s21, 0xd62f105d);
		GG(d, a, b, c, x[10], s22, 0x2441453);
		GG(c, d, a, b, x[15], s23, 0xd8a1e681);
		GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GG(a, b, c, d, x[9], s21, 0x21e1cde6);
		GG(d, a, b, c, x[14], s22, 0xc33707d6);
		GG(c, d, a, b, x[3], s23, 0xf4d50d87);
		GG(b, c, d, a, x[8], s24, 0x455a14ed);
		GG(a, b, c, d, x[13], s21, 0xa9e3e905);
		GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GG(c, d, a, b, x[7], s23, 0x676f02d9);
		GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		/* Round 3 */
		HH(a, b, c, d, x[5], s31, 0xfffa3942);
		HH(d, a, b, c, x[8], s32, 0x8771f681);
		HH(c, d, a, b, x[11], s33, 0x6d9d6122);
		HH(b, c, d, a, x[14], s34, 0xfde5380c);
		HH(a, b, c, d, x[1], s31, 0xa4beea44);
		HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HH(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH(a, b, c, d, x[13], s31, 0x289b7ec6);
		HH(d, a, b, c, x[0], s32, 0xeaa127fa);
		HH(c, d, a, b, x[3], s33, 0xd4ef3085);
		HH(b, c, d, a, x[6], s34, 0x4881d05);
		HH(a, b, c, d, x[9], s31, 0xd9d4d039);
		HH(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HH(b, c, d, a, x[2], s34, 0xc4ac5665);

		/* Round 4 */
		II(a, b, c, d, x[0], s41, 0xf4292244);
		II(d, a, b, c, x[7], s42, 0x432aff97);
		II(c, d, a, b, x[14], s43, 0xab9423a7);
		II(b, c, d, a, x[5], s44, 0xfc93a039);
		II(a, b, c, d, x[12], s41, 0x655b59c3);
		II(d, a, b, c, x[3], s42, 0x8f0ccc92);
		II(c, d, a, b, x[10], s43, 0xffeff47d);
		II(b, c, d, a, x[1], s44, 0x85845dd1);
		II(a, b, c, d, x[8], s41, 0x6fa87e4f);
		II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II(c, d, a, b, x[6], s43, 0xa3014314);
		II(b, c, d, a, x[13], s44, 0x4e0811a1);
		II(a, b, c, d, x[4], s41, 0xf7537e82);
		II(d, a, b, c, x[11], s42, 0xbd3af235);
		II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		II(b, c, d, a, x[9], s44, 0xeb86d391);

		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;
	}

	// 下面的处理，在理解上较为复杂
	for (int i = 0; i < 4; i++)
	{
		uint32_t value = state[i];
		state[i] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
				   ((value & 0xff00) << 8) |	 // 将次低字节左移
				   ((value & 0xff0000) >> 8) |	 // 将次高字节右移
				   ((value & 0xff000000) >> 24); // 将最高字节移到最低位
	}

	// 输出最终的hash结果
	// for (int i1 = 0; i1 < 4; i1 += 1)
	// {
	// 	cout << std::setw(8) << std::setfill('0') << hex << state[i1];
	// }
	// cout << endl;

	// 释放动态分配的内存
	// 实现SIMD并行算法的时候，也请记得及时回收内存！
	delete[] paddedMessage;
	delete[] messageLength;
}

void MD5HashBatch(const vector<string>& inputs, vector<bit32*>& states) {
    // 增加SIMD处理宽度以提高并行度
    constexpr size_t SIMD_WIDTH = 16;
    states.resize(inputs.size());
    constexpr uint32_t INIT_STATES[4] = {
		0x67452301,
		0xEFCDAB89,
		0x98BADCFE,
		0x10325476
	};
	
    // 使用OpenMP并行化外层循环
    #pragma omp parallel for schedule(dynamic)
    for(size_t base = 0; base < inputs.size(); base += SIMD_WIDTH) {
        size_t remaining = min(SIMD_WIDTH, inputs.size() - base);
        
        // 为当前批次的每个输入分配状态
        for(size_t i = 0; i < remaining; i++) {
            states[base + i] = new bit32[4];
            states[base + i][0] = 0x67452301;
            states[base + i][1] = 0xefcdab89;
            states[base + i][2] = 0x98badcfe;
            states[base + i][3] = 0x10325476;
        }

        // 预处理消息
        vector<Byte*> paddedMessages(SIMD_WIDTH);
        vector<int> messageLengths(SIMD_WIDTH);

        // 并行预处理输入消息
        #pragma omp parallel for
        for(size_t i = 0; i < remaining; i++) {
            paddedMessages[i] = StringProcess(inputs[base + i], &messageLengths[i]);
        }
        
        // 填充不足SIMD_WIDTH的部分
        for(size_t i = remaining; i < SIMD_WIDTH; i++) {
            paddedMessages[i] = new Byte[messageLengths[0]];
            memcpy(paddedMessages[i], paddedMessages[0], messageLengths[0]);
            messageLengths[i] = messageLengths[0];
        }

        // 初始化SIMD状态向量
        uint32x4_t state_vectors[4][4];  // 4x4矩阵存储16个并行状态
        for(int j = 0; j < 4; j++) {
            for(int k = 0; k < 4; k++) {
                state_vectors[j][k] = vdupq_n_u32(INIT_STATES[j]);
            }
        }

        // 处理消息块
        int blocks = messageLengths[0] / 64;
        for(int block = 0; block < blocks; block++) {
            // 使用NEON intrinsics加载消息数据
            uint32x4_t x[16][4];  // 16x4矩阵存储消息块
            
            // 并行加载消息数据
            #pragma omp parallel for collapse(2)
            for(int j = 0; j < 16; j++) {
                for(int k = 0; k < 4; k++) {
                    uint32_t temp[4];
                    for(int m = 0; m < 4; m++) {
                        int offset = block * 64 + j * 4;
                        int msg_idx = k * 4 + m;
                        if(msg_idx < SIMD_WIDTH) {
                            temp[m] = ((uint32_t)paddedMessages[msg_idx][offset]) |
                                    ((uint32_t)paddedMessages[msg_idx][offset + 1] << 8) |
                                    ((uint32_t)paddedMessages[msg_idx][offset + 2] << 16) |
                                    ((uint32_t)paddedMessages[msg_idx][offset + 3] << 24);
                        }
                    }
                    x[j][k] = vld1q_u32(temp);
                }
            }

            // 对4组SIMD向量同时进行MD5运算
            for(int k = 0; k < 4; k++) {
                uint32x4_t a = state_vectors[0][k];
                uint32x4_t b = state_vectors[1][k];
                uint32x4_t c = state_vectors[2][k];
                uint32x4_t d = state_vectors[3][k];

                // Round 1
                FF_VEC(a, b, c, d, x[0][k], s11, 0xd76aa478);
                FF_VEC(d, a, b, c, x[1][k], s12, 0xe8c7b756);
                FF_VEC(c, d, a, b, x[2][k], s13, 0x242070db);
                FF_VEC(b, c, d, a, x[3][k], s14, 0xc1bdceee);
                FF_VEC(a, b, c, d, x[4][k], s11, 0xf57c0faf);
                FF_VEC(d, a, b, c, x[5][k], s12, 0x4787c62a);
                FF_VEC(c, d, a, b, x[6][k], s13, 0xa8304613);
                FF_VEC(b, c, d, a, x[7][k], s14, 0xfd469501);
                FF_VEC(a, b, c, d, x[8][k], s11, 0x698098d8);
                FF_VEC(d, a, b, c, x[9][k], s12, 0x8b44f7af);
                FF_VEC(c, d, a, b, x[10][k], s13, 0xffff5bb1);
                FF_VEC(b, c, d, a, x[11][k], s14, 0x895cd7be);
                FF_VEC(a, b, c, d, x[12][k], s11, 0x6b901122);
                FF_VEC(d, a, b, c, x[13][k], s12, 0xfd987193);
                FF_VEC(c, d, a, b, x[14][k], s13, 0xa679438e);
                FF_VEC(b, c, d, a, x[15][k], s14, 0x49b40821);

                // Round 2 
                GG_VEC(a, b, c, d, x[1][k], s21, 0xf61e2562);
                GG_VEC(d, a, b, c, x[6][k], s22, 0xc040b340);
                GG_VEC(c, d, a, b, x[11][k], s23, 0x265e5a51);
                GG_VEC(b, c, d, a, x[0][k], s24, 0xe9b6c7aa);
                GG_VEC(a, b, c, d, x[5][k], s21, 0xd62f105d);
                GG_VEC(d, a, b, c, x[10][k], s22, 0x2441453);
                GG_VEC(c, d, a, b, x[15][k], s23, 0xd8a1e681);
                GG_VEC(b, c, d, a, x[4][k], s24, 0xe7d3fbc8);
                GG_VEC(a, b, c, d, x[9][k], s21, 0x21e1cde6);
                GG_VEC(d, a, b, c, x[14][k], s22, 0xc33707d6);
                GG_VEC(c, d, a, b, x[3][k], s23, 0xf4d50d87);
                GG_VEC(b, c, d, a, x[8][k], s24, 0x455a14ed);
                GG_VEC(a, b, c, d, x[13][k], s21, 0xa9e3e905);
                GG_VEC(d, a, b, c, x[2][k], s22, 0xfcefa3f8);
                GG_VEC(c, d, a, b, x[7][k], s23, 0x676f02d9);
                GG_VEC(b, c, d, a, x[12][k], s24, 0x8d2a4c8a);

                // Round 3
                HH_VEC(a, b, c, d, x[5][k], s31, 0xfffa3942);
                HH_VEC(d, a, b, c, x[8][k], s32, 0x8771f681);
                HH_VEC(c, d, a, b, x[11][k], s33, 0x6d9d6122);
                HH_VEC(b, c, d, a, x[14][k], s34, 0xfde5380c);
                HH_VEC(a, b, c, d, x[1][k], s31, 0xa4beea44);
                HH_VEC(d, a, b, c, x[4][k], s32, 0x4bdecfa9);
                HH_VEC(c, d, a, b, x[7][k], s33, 0xf6bb4b60);
                HH_VEC(b, c, d, a, x[10][k], s34, 0xbebfbc70);
                HH_VEC(a, b, c, d, x[13][k], s31, 0x289b7ec6);
                HH_VEC(d, a, b, c, x[0][k], s32, 0xeaa127fa);
                HH_VEC(c, d, a, b, x[3][k], s33, 0xd4ef3085);
                HH_VEC(b, c, d, a, x[6][k], s34, 0x4881d05);
                HH_VEC(a, b, c, d, x[9][k], s31, 0xd9d4d039);
                HH_VEC(d, a, b, c, x[12][k], s32, 0xe6db99e5);
                HH_VEC(c, d, a, b, x[15][k], s33, 0x1fa27cf8);
                HH_VEC(b, c, d, a, x[2][k], s34, 0xc4ac5665);

                // Round 4
                II_VEC(a, b, c, d, x[0][k], s41, 0xf4292244);
                II_VEC(d, a, b, c, x[7][k], s42, 0x432aff97);
                II_VEC(c, d, a, b, x[14][k], s43, 0xab9423a7);
                II_VEC(b, c, d, a, x[5][k], s44, 0xfc93a039);
                II_VEC(a, b, c, d, x[12][k], s41, 0x655b59c3);
                II_VEC(d, a, b, c, x[3][k], s42, 0x8f0ccc92);
                II_VEC(c, d, a, b, x[10][k], s43, 0xffeff47d);
                II_VEC(b, c, d, a, x[1][k], s44, 0x85845dd1);
                II_VEC(a, b, c, d, x[8][k], s41, 0x6fa87e4f);
                II_VEC(d, a, b, c, x[15][k], s42, 0xfe2ce6e0);
                II_VEC(c, d, a, b, x[6][k], s43, 0xa3014314);
                II_VEC(b, c, d, a, x[13][k], s44, 0x4e0811a1);
                II_VEC(a, b, c, d, x[4][k], s41, 0xf7537e82);
                II_VEC(d, a, b, c, x[11][k], s42, 0xbd3af235);
                II_VEC(c, d, a, b, x[2][k], s43, 0x2ad7d2bb);
                II_VEC(b, c, d, a, x[9][k], s44, 0xeb86d391);

                state_vectors[0][k] = vaddq_u32(state_vectors[0][k], a);
                state_vectors[1][k] = vaddq_u32(state_vectors[1][k], b);
                state_vectors[2][k] = vaddq_u32(state_vectors[2][k], c);
                state_vectors[3][k] = vaddq_u32(state_vectors[3][k], d);
            }
            
        }
        
        // 保存计算结果
        for(int k = 0; k < 4; k++) {
            uint32_t results[4][4];
            for(int j = 0; j < 4; j++) {
                vst1q_u32(results[j], state_vectors[j][k]);
            }

            // 转换字节序并保存有效结果
            for(int i = 0; i < 4; i++) {
                if(k * 4 + i < remaining) {
                    for(int j = 0; j < 4; j++) {
                        states[base + k * 4 + i][j] = 
                            ((results[j][i] & 0xff) << 24) |
                            ((results[j][i] & 0xff00) << 8) |
                            ((results[j][i] & 0xff0000) >> 8) |
                            ((results[j][i] & 0xff000000) >> 24);
                    }
                }
            }
        }

        // 清理内存
        for(size_t i = 0; i < SIMD_WIDTH; i++) {
            delete[] paddedMessages[i];
        }
    }
}

