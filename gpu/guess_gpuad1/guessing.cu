#include "PCFG.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;




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
// {

//     // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
//     Generate(priority.front());

//     // 然后需要根据即将出队的PT，生成一系列新的PT
//     vector<PT> new_pts = priority.front().NewPTs();
//     for (PT pt : new_pts)
//     {
//         // 计算概率
//         CalProb(pt);
//         // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
//         for (auto iter = priority.begin(); iter != priority.end(); iter++)
//         {
//             // 对于非队首和队尾的特殊情况
//             if (iter != priority.end() - 1 && iter != priority.begin())
//             {
//                 // 判定概率
//                 if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
//                 {
//                     priority.emplace(iter + 1, pt);
//                     break;
//                 }
//             }
//             if (iter == priority.end() - 1)
//             {
//                 priority.emplace_back(pt);
//                 break;
//             }
//             if (iter == priority.begin() && iter->prob < pt.prob)
//             {
//                 priority.emplace(iter, pt);
//                 break;
//             }
//         }
//     }

//     // 现在队首的PT善后工作已经结束，将其出队（删除）
//     priority.erase(priority.begin());
// }
{
    // 使用批处理版本，一次处理多个PT
    BatchProcessPTs(4); // 可以调整批次大小
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
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
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
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}


// 简化的GPU核函数：处理单个PT的所有可能猜测
__global__ void gpu_batch_generate_guesses(
    char* d_all_segment_values,    // 所有segment值的连续存储
    int* d_pt_offsets,             // 每个PT在segment值数组中的起始偏移
    int* d_value_offsets,          // 每个值在segment值数组中的偏移
    int* d_value_lengths,          // 每个值的长度
    char* d_prefixes,              // 每个PT的前缀
    int* d_prefix_lengths,         // 前缀长度
    int* d_last_seg_counts,        // 每个PT最后一个segment的值数量
    char* d_output_guesses,        // 输出猜测
    int* d_output_counts,          // 每个PT实际生成的猜测数量
    int num_pts,                   // PT数量
    int max_guess_length           // 最大猜测长度
) {
    int pt_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    if (pt_id >= num_pts) return;
    
    int last_seg_count = d_last_seg_counts[pt_id];
    int prefix_len = d_prefix_lengths[pt_id];
    
    // 每个block处理一个PT，block内的线程并行处理该PT的所有可能值
    for (int value_idx = thread_id; value_idx < last_seg_count; value_idx += blockDim.x) {
        // 计算输出位置
        int output_offset = d_pt_offsets[pt_id] + value_idx;
        
        // 复制前缀
        for (int i = 0; i < prefix_len && i < max_guess_length; i++) {
            d_output_guesses[output_offset * max_guess_length + i] = 
                d_prefixes[pt_id * max_guess_length + i];
        }
        
        // 获取当前值的信息
        int value_data_offset = d_value_offsets[d_pt_offsets[pt_id] + value_idx];
        int value_len = d_value_lengths[d_pt_offsets[pt_id] + value_idx];
        
        // 添加最后一个segment的值
        for (int i = 0; i < value_len && (prefix_len + i) < max_guess_length; i++) {
            d_output_guesses[output_offset * max_guess_length + prefix_len + i] = 
                d_all_segment_values[value_data_offset + i];
        }
        
        // 记录实际长度（用于后续字符串构造）
        d_output_counts[output_offset] = min(prefix_len + value_len, max_guess_length);
    }
}

// 重新设计的批处理函数
void PriorityQueue::BatchProcessPTs(int batch_size)
{
    if (priority.empty()) return;
    
    vector<PT> batch_pts;
    vector<string> batch_prefixes;
    
    // 收集批次PT并为每个PT生成一次猜测
    for (int i = 0; i < batch_size && !priority.empty(); i++) {
        PT current_pt = priority.front();
        priority.erase(priority.begin());
        
        // 对于只有一个segment的PT，直接在CPU上处理
        if (current_pt.content.size() == 1) {
            segment *seg;
            if (current_pt.content[0].type == 1) {
                seg = &m.letters[m.FindLetter(current_pt.content[0])];
            } else if (current_pt.content[0].type == 2) {
                seg = &m.digits[m.FindDigit(current_pt.content[0])];
            } else {
                seg = &m.symbols[m.FindSymbol(current_pt.content[0])];
            }
            
            // 直接添加所有值到guesses
            for (const string& value : seg->ordered_values) {
                guesses.push_back(value);
            }
            
            // 不需要生成新PT（单segment PT不产生子PT）
            continue;
        }
        
        batch_pts.push_back(current_pt);
        
        // 构建前缀（除最后一个segment外的所有值）
        string prefix = "";
        int seg_idx = 0;
        for (int idx : current_pt.curr_indices) {
            if (seg_idx == current_pt.content.size() - 1) break;
            
            if (current_pt.content[seg_idx].type == 1) {
                prefix += m.letters[m.FindLetter(current_pt.content[seg_idx])].ordered_values[idx];
            } else if (current_pt.content[seg_idx].type == 2) {
                prefix += m.digits[m.FindDigit(current_pt.content[seg_idx])].ordered_values[idx];
            } else if (current_pt.content[seg_idx].type == 3) {
                prefix += m.symbols[m.FindSymbol(current_pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx++;
        }
        batch_prefixes.push_back(prefix);
    }
    
    if (batch_pts.empty()) {
        total_guesses = guesses.size();
        return;
    }
    
    const int MAX_GUESS_LENGTH = 64;
    
    // 准备GPU数据
    vector<char> all_segment_values;
    vector<int> pt_offsets;
    vector<int> value_offsets;
    vector<int> value_lengths;
    vector<int> last_seg_counts;
    vector<char> prefixes(batch_pts.size() * MAX_GUESS_LENGTH, '\0');
    vector<int> prefix_lengths;
    
    int total_values = 0;
    int data_offset = 0;
    
    for (int pt_idx = 0; pt_idx < batch_pts.size(); pt_idx++) {
        PT& pt = batch_pts[pt_idx];
        
        // 设置PT偏移
        pt_offsets.push_back(total_values);
        
        // 获取最后一个segment
        int last_idx = pt.content.size() - 1;
        segment* last_seg;
        if (pt.content[last_idx].type == 1) {
            last_seg = &m.letters[m.FindLetter(pt.content[last_idx])];
        } else if (pt.content[last_idx].type == 2) {
            last_seg = &m.digits[m.FindDigit(pt.content[last_idx])];
        } else {
            last_seg = &m.symbols[m.FindSymbol(pt.content[last_idx])];
        }
        
        last_seg_counts.push_back(last_seg->ordered_values.size());
        
        // 复制前缀
        string prefix = batch_prefixes[pt_idx];
        prefix_lengths.push_back(prefix.length());
        for (int i = 0; i < prefix.length() && i < MAX_GUESS_LENGTH; i++) {
            prefixes[pt_idx * MAX_GUESS_LENGTH + i] = prefix[i];
        }
        
        // 处理最后一个segment的所有值
        for (const string& value : last_seg->ordered_values) {
            value_offsets.push_back(data_offset);
            value_lengths.push_back(value.length());
            
            for (char c : value) {
                all_segment_values.push_back(c);
                data_offset++;
            }
            total_values++;
        }
    }
    
    // 分配GPU内存
    char *d_all_segment_values, *d_prefixes, *d_output_guesses;
    int *d_pt_offsets, *d_value_offsets, *d_value_lengths, *d_last_seg_counts, *d_prefix_lengths, *d_output_counts;
    
    cudaMalloc(&d_all_segment_values, all_segment_values.size() * sizeof(char));
    cudaMalloc(&d_pt_offsets, pt_offsets.size() * sizeof(int));
    cudaMalloc(&d_value_offsets, value_offsets.size() * sizeof(int));
    cudaMalloc(&d_value_lengths, value_lengths.size() * sizeof(int));
    cudaMalloc(&d_last_seg_counts, last_seg_counts.size() * sizeof(int));
    cudaMalloc(&d_prefixes, prefixes.size() * sizeof(char));
    cudaMalloc(&d_prefix_lengths, prefix_lengths.size() * sizeof(int));
    cudaMalloc(&d_output_guesses, total_values * MAX_GUESS_LENGTH * sizeof(char));
    cudaMalloc(&d_output_counts, total_values * sizeof(int));
    
    // 复制数据到GPU
    cudaMemcpy(d_all_segment_values, all_segment_values.data(), all_segment_values.size() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt_offsets, pt_offsets.data(), pt_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_offsets, value_offsets.data(), value_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_lengths, value_lengths.data(), value_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_last_seg_counts, last_seg_counts.data(), last_seg_counts.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefixes, prefixes.data(), prefixes.size() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_lengths, prefix_lengths.data(), prefix_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // 启动GPU核函数
    int num_blocks = batch_pts.size();
    int threads_per_block = 256;
    
    gpu_batch_generate_guesses<<<num_blocks, threads_per_block>>>(
        d_all_segment_values, d_pt_offsets, d_value_offsets, d_value_lengths,
        d_prefixes, d_prefix_lengths, d_last_seg_counts,
        d_output_guesses, d_output_counts,
        batch_pts.size(), MAX_GUESS_LENGTH
    );
    
    cudaDeviceSynchronize();
    
    // 复制结果回CPU
    vector<char> output_guesses(total_values * MAX_GUESS_LENGTH);
    vector<int> output_counts(total_values);
    
    cudaMemcpy(output_guesses.data(), d_output_guesses, total_values * MAX_GUESS_LENGTH * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_counts.data(), d_output_counts, total_values * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 构造字符串并添加到guesses
    for (int i = 0; i < total_values; i++) {
        if (output_counts[i] > 0) {
            string guess(output_guesses.begin() + i * MAX_GUESS_LENGTH,
                        output_guesses.begin() + i * MAX_GUESS_LENGTH + output_counts[i]);
            guesses.push_back(guess);
        }
    }
    
    // 生成新的PT并插入优先队列
    for (PT& pt : batch_pts) {
        vector<PT> new_pts = pt.NewPTs();
        for (PT& new_pt : new_pts) {
            CalProb(new_pt);
            // 按概率插入优先队列
            bool inserted = false;
            for (auto iter = priority.begin(); iter != priority.end(); iter++) {
                if (new_pt.prob > iter->prob) {
                    priority.insert(iter, new_pt);
                    inserted = true;
                    break;
                }
            }
            if (!inserted) {
                priority.push_back(new_pt);
            }
        }
    }
    
    // 释放GPU内存
    cudaFree(d_all_segment_values);
    cudaFree(d_pt_offsets);
    cudaFree(d_value_offsets);
    cudaFree(d_value_lengths);
    cudaFree(d_last_seg_counts);
    cudaFree(d_prefixes);
    cudaFree(d_prefix_lengths);
    cudaFree(d_output_guesses);
    cudaFree(d_output_counts);
    
    total_guesses = guesses.size();
}
