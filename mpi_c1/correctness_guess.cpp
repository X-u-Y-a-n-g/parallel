#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
using namespace std;
using namespace chrono;
#include <mpi.h>

int main(int argc, char **argv)
{
    // 初始化 MPI 环境
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 获取当前进程的 rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // 获取总进程数

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    if (rank == 0)
    {
        auto start_train = system_clock::now();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
        cout << "Training completed in " << time_train << " seconds." << endl;
        // 初始化优先队列
        q.init();
        cout << "Priority queue initialized." << endl;
    }

    // 加载测试数据（所有进程都需要加载）
    unordered_set<std::string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string pw;
    while (test_data >> pw)
    {
        test_count += 1;
        test_set.insert(pw);
        if (test_count >= 1000000)
        {
            break;
        }
    }
    // 确保所有进程都已加载测试数据
    MPI_Barrier(MPI_COMM_WORLD);

    int total_cracked = 0;
    int curr_num = 0;
    int total_guesses = 0;
    bool should_continue = true;

    // 只有 rank 0 的进程负责生成初始猜测
    if (rank == 0)
    {
        cout << "Starting guessing process..." << endl;
    }

    auto start_time = system_clock::now();

    // 设置每次处理的PT数量（可根据进程数调整）
    int pts_per_batch = max(1, min(size/2, 2)); // 一次最多取出2个PT或进程数个PT一半

    while (should_continue)
    {
        // 只有 rank 0 检查队列是否为空并生成猜测
        bool queue_empty = false;
        int batch_size = 0;

        if (rank == 0)
        {
            queue_empty = q.priority.empty();
            if (!queue_empty)
            {
                // 确定本次实际处理的PT数量（不超过队列大小）
                batch_size = min(pts_per_batch, (int)q.priority.size());
                
                // 清空之前的猜测列表
                q.guesses.clear();
                
                // 一次性处理多个PT，生成猜测
                q.ProcessMultiplePTs(batch_size);
                
                cout << "Processed " << batch_size << " PTs, generated " 
                     << q.guesses.size() << " guesses." << endl;
            }
        }
        
        // 广播队列状态给所有进程
        MPI_Bcast(&queue_empty, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        // 如果队列为空，所有进程退出循环
        if (queue_empty)
        {
            should_continue = false;
            continue;
        }

        // 广播猜测数量给所有进程
        int guesses_size = 0;
        if (rank == 0)
        {
            guesses_size = q.guesses.size();
        }
        MPI_Bcast(&guesses_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 如果没有猜测，继续下一轮
        if (guesses_size == 0)
        {
            continue;
        }

        // 非 rank 0 的进程需要为猜测分配空间
        if (rank != 0)
        {
            q.guesses.resize(guesses_size);
        }
        
        // 广播猜测给所有进程
        for (int i = 0; i < guesses_size; i++)
        {
            int str_len = 0;
            if (rank == 0)
            {
                str_len = q.guesses[i].length();
            }
            MPI_Bcast(&str_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            char *buffer = new char[str_len + 1];
            if (rank == 0)
            {
                strcpy(buffer, q.guesses[i].c_str());
            }
            MPI_Bcast(buffer, str_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
            
            if (rank != 0)
            {
                q.guesses[i] = string(buffer);
            }
            delete[] buffer;
        }

        // 并行处理猜测任务分配
        int chunk_size = (guesses_size + size - 1) / size;
        int start = rank * chunk_size;
        int end = min(start + chunk_size, guesses_size);
        
        // 每个进程处理自己的任务范围
        auto start_hash = system_clock::now();
        bit32 state[4];
        int local_cracked = 0;
        for (int i = start; i < end; i++)
        {
            string &pw = q.guesses[i];
            if (test_set.find(pw) != test_set.end())
            {
                local_cracked++;
            }
            MD5Hash(pw, state);
        }
        auto end_hash = system_clock::now();
        auto duration = duration_cast<microseconds>(end_hash - start_hash);
        time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

        // 汇总本次循环破解的密码数量
        int cracked_this_round = 0;
        MPI_Reduce(&local_cracked, &cracked_this_round, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // rank 0 负责统计结果
        if (rank == 0)
        {
            // 累加本轮破解的密码数量到总数
            total_cracked += cracked_this_round;
            
            // 更新总猜测数量
            total_guesses += guesses_size;
            curr_num += guesses_size;
            
            if (curr_num >= 1000000)
            {
                cout << "Guesses generated: " << total_guesses << endl;
                cout << "Cracked so far: " << total_cracked << endl;
                curr_num = 0;
            }

            if (total_guesses >= 10000000)
            {
                auto end_time = system_clock::now();
                auto total_duration = duration_cast<microseconds>(end_time - start_time);
                time_guess = double(total_duration.count()) * microseconds::period::num / microseconds::period::den;

                cout << "Guessing completed." << endl;
                cout << "Total guesses: " << total_guesses << endl;
                cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                cout << "Hash time: " << time_hash << " seconds" << endl;
                cout << "Train time: " << time_train << " seconds" << endl;
                cout << "Cracked passwords: " << total_cracked << endl;
                
                should_continue = false;
            }
        }
        
        // 广播是否继续循环
        MPI_Bcast(&should_continue, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    // 清理 MPI 环境
    MPI_Finalize();
    return 0;
}