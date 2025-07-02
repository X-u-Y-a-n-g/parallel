#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
using namespace std;

//nvcc -std=c++11 -O3 -o pcfg_gpu guessing_gpu.cu correctness_guess.cpp md5.cpp train.cpp -lcudart

// 定义GPU处理的最小阈值
#define MIN_GPU_SIZE 1000

// CUDA错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// 优化的字符串复制函数
void copy_strings_to_gpu_optimized(const vector<string>& strings, char* d_buffer, int* d_offsets, int* d_lengths) {
    if (strings.empty()) return;

    vector<int> offsets(strings.size());
    vector<int> lengths(strings.size());

    int total_size = 0;
    for (size_t i = 0; i < strings.size(); i++) {
        offsets[i] = total_size;
        lengths[i] = strings[i].length();
        total_size += lengths[i] + 1;
    }

    // 创建连续的主机缓冲区
    char* host_buffer = new char[total_size];
    for (size_t i = 0; i < strings.size(); i++) {
        memcpy(host_buffer + offsets[i], strings[i].c_str(), lengths[i] + 1);
    }

    // 一次性传输到GPU
    CUDA_CHECK(cudaMemcpy(d_buffer, host_buffer, total_size * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), strings.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, lengths.data(), strings.size() * sizeof(int), cudaMemcpyHostToDevice));

    delete[] host_buffer;
}

__global__ void fastGenerateKernel(char* d_input, int* d_offsets, int* d_lengths,
                                   char* d_prefix, int prefix_len,
                                   char* d_output, int max_output_len, int num_items) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    // 计算输出位置
    char* output_start = d_output + idx * max_output_len;
    int write_pos = 0;

    // 快速复制前缀（如果存在）
    if (d_prefix && prefix_len > 0) {
        for (int i = 0; i < prefix_len && write_pos < max_output_len - 1; i++) {
            output_start[write_pos++] = d_prefix[i];
        }
    }

    // 快速复制值
    char* input_start = d_input + d_offsets[idx];
    int input_len = d_lengths[idx];
    for (int i = 0; i < input_len && write_pos < max_output_len - 1; i++) {
        output_start[write_pos++] = input_start[i];
    }

    // 添加终止符
    output_start[write_pos] = '\0';
}

// 全局GPU资源
static char* g_d_input = nullptr;
static char* g_d_prefix = nullptr;
static char* g_d_output = nullptr;
static int* g_d_offsets = nullptr;
static int* g_d_lengths = nullptr;
static size_t g_max_items = 100000;
static size_t g_max_input_size = 50 * 1024 * 1024;  // 50MB
static size_t g_max_output_size = 100 * 1024 * 1024; // 100MB
static bool g_gpu_initialized = false;

// 初始化GPU资源
bool initGPU() {
    if (g_gpu_initialized) return true;

    cudaError_t err1 = cudaMalloc(&g_d_input, g_max_input_size);
    cudaError_t err2 = cudaMalloc(&g_d_prefix, 1024); // 前缀最大1KB
    cudaError_t err3 = cudaMalloc(&g_d_output, g_max_output_size);
    cudaError_t err4 = cudaMalloc(&g_d_offsets, g_max_items * sizeof(int));
    cudaError_t err5 = cudaMalloc(&g_d_lengths, g_max_items * sizeof(int));

    if (err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess && 
        err4 == cudaSuccess && err5 == cudaSuccess) {
        g_gpu_initialized = true;
        return true;
    }

    // 清理失败的分配
    if (g_d_input) { cudaFree(g_d_input); g_d_input = nullptr; }
    if (g_d_prefix) { cudaFree(g_d_prefix); g_d_prefix = nullptr; }
    if (g_d_output) { cudaFree(g_d_output); g_d_output = nullptr; }
    if (g_d_offsets) { cudaFree(g_d_offsets); g_d_offsets = nullptr; }
    if (g_d_lengths) { cudaFree(g_d_lengths); g_d_lengths = nullptr; }

    return false;
}

// 高效GPU处理函数
bool processWithGPU(const vector<string>& values, const string& prefix, 
                   vector<string>& results) {
    if (!initGPU()) return false;

    size_t num_values = values.size();
    if (num_values == 0 || num_values > g_max_items) return false;

    // 计算所需空间
    size_t total_input_size = 0;
    for (const auto& val : values) {
        total_input_size += val.length() + 1; // +1 for alignment
    }

    if (total_input_size > g_max_input_size) return false;

    const int MAX_OUTPUT_LEN = 64;
    size_t output_size = num_values * MAX_OUTPUT_LEN;
    if (output_size > g_max_output_size) return false;

    // 准备主机数据
    char* h_input = (char*)malloc(total_input_size);
    int* h_offsets = (int*)malloc(num_values * sizeof(int));
    int* h_lengths = (int*)malloc(num_values * sizeof(int));
    char* h_output = (char*)malloc(output_size);

    if (!h_input || !h_offsets || !h_lengths || !h_output) {
        if (h_input) free(h_input);
        if (h_offsets) free(h_offsets);
        if (h_lengths) free(h_lengths);
        if (h_output) free(h_output);
        return false;
    }

    // 打包输入数据
    size_t input_pos = 0;
    for (size_t i = 0; i < num_values; i++) {
        h_offsets[i] = input_pos;
        h_lengths[i] = values[i].length();
        memcpy(h_input + input_pos, values[i].c_str(), values[i].length());
        input_pos += values[i].length();
        h_input[input_pos++] = '\0'; // 对齐
    }

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 异步传输数据
    cudaMemcpyAsync(g_d_input, h_input, total_input_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_d_offsets, h_offsets, num_values * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_d_lengths, h_lengths, num_values * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 传输前缀（如果有）
    char* d_prefix_ptr = nullptr;
    if (!prefix.empty() && prefix.length() < 1024) {
        cudaMemcpyAsync(g_d_prefix, prefix.c_str(), prefix.length(), cudaMemcpyHostToDevice, stream);
        d_prefix_ptr = g_d_prefix;
    }

    // 启动核函数
    const int BLOCK_SIZE = 512;
    int numBlocks = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fastGenerateKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
        g_d_input, g_d_offsets, g_d_lengths, d_prefix_ptr, prefix.length(),
        g_d_output, MAX_OUTPUT_LEN, num_values);

    // 异步复制结果
    cudaMemcpyAsync(h_output, g_d_output, output_size, cudaMemcpyDeviceToHost, stream);

    // 等待完成
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // 处理结果
    results.reserve(results.size() + num_values);
    for (size_t i = 0; i < num_values; i++) {
        char* result_ptr = h_output + i * MAX_OUTPUT_LEN;
        results.emplace_back(result_ptr);
    }

    // 清理
    free(h_input);
    free(h_offsets);
    free(h_lengths);
    free(h_output);

    return true;
}

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    const int BATCH_SIZE = 64; // 批量处理大小
    vector<PT> batch_pts;

    // 1. 收集一批PT进行批量处理
    for (int i = 0; i < BATCH_SIZE && !priority.empty(); i++) {
        batch_pts.push_back(priority.front());
        priority.erase(priority.begin());
    }

    // 2. 批量生成猜测
    if (!batch_pts.empty()) {
        BatchGenerate(batch_pts);
    }

    // 3. 为每个处理过的PT生成新的PT并插入优先级队列
    for (PT& processed_pt : batch_pts) {  // 去掉const
        vector<PT> new_pts = processed_pt.NewPTs();
        for (PT pt : new_pts) {
            CalProb(pt);
            // 按概率插入到正确位置
            bool inserted = false;
            for (auto iter = priority.begin(); iter != priority.end(); iter++) {
                if (pt.prob > iter->prob) {
                    priority.emplace(iter, pt);
                    inserted = true;
                    break;
                }
            }
            if (!inserted) {
                priority.emplace_back(pt);
            }
        }
    }
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;

    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;

        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;

            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
    return res;
}

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        int num_values = pt.max_indices[0];

        // 尝试GPU处理
        if (num_values >= MIN_GPU_SIZE) {
            vector<string> temp_results;
            if (processWithGPU(a->ordered_values, "", temp_results)) {
                // GPU成功
                for (const auto& result : temp_results) {
                    guesses.push_back(result);
                    total_guesses++;
                }
                return;
            }
        }

        // GPU失败或数据太小，使用CPU
        for (int i = 0; i < num_values; i++)
        {
            guesses.push_back(a->ordered_values[i]);
            total_guesses++;
        }
    }
    else
    {
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }

        int num_values = pt.max_indices.back();

        // 尝试GPU处理
        if (num_values >= MIN_GPU_SIZE) {
            vector<string> temp_results;
            if (processWithGPU(a->ordered_values, prefix, temp_results)) {
                // GPU成功
                for (const auto& result : temp_results) {
                    guesses.push_back(result);
                    total_guesses++;
                }
                return;
            }
        }

        // GPU失败或数据太小，使用CPU
        for (int i = 0; i < num_values; i++)
        {
            guesses.push_back(prefix + a->ordered_values[i]);
            total_guesses++;
        }
    }
}

// 在程序结束时清理GPU资源
void cleanup_gpu_resources() {
    if (g_d_input) { cudaFree(g_d_input); g_d_input = nullptr; }
    if (g_d_prefix) { cudaFree(g_d_prefix); g_d_prefix = nullptr; }
    if (g_d_output) { cudaFree(g_d_output); g_d_output = nullptr; }
    if (g_d_offsets) { cudaFree(g_d_offsets); g_d_offsets = nullptr; }
    if (g_d_lengths) { cudaFree(g_d_lengths); g_d_lengths = nullptr; }
    g_gpu_initialized = false;
}

void PriorityQueue::BatchGenerate(const vector<PT>& batch_pts) {
    // 批量处理PT列表
    for (const PT& pt : batch_pts) {
        Generate(const_cast<PT&>(pt));
    }
}