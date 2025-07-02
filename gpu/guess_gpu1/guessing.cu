#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>

using namespace std;

// 简化的GPU核函数 - 针对性能优化
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
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
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
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    const int MIN_GPU_SIZE = 1000; // 大幅降低GPU使用阈值

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
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

        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        // for (int i = 0; i < pt.max_indices[0]; i += 1)
        // {
        //     string guess = a->ordered_values[i];
        //     // cout << guess << endl;
        //     guesses.emplace_back(guess);
        //     total_guesses += 1;
        // }


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
        string prefix; // 修改变量名，避免冲突
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
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

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
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

        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        // for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        // {
        //     string temp = guess + a->ordered_values[i];
        //     // cout << temp << endl;
        //     guesses.emplace_back(temp);
        //     total_guesses += 1;
        // }


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