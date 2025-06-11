#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
using namespace std;
using namespace chrono;
#include <mpi.h>
#include <unistd.h> // 添加此头文件以使用 usleep 函数


// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            cout << "需要至少2个进程来运行流水线并行算法" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // 进程0负责生成猜测
    if (rank == 0) {
        // 记录各阶段时间
        double time_train = 0;
        double time_guess = 0;
        PriorityQueue q;
        // 记录训练开始时间
        auto start_train = system_clock::now();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
        cout << "训练时间: " << time_train << " 秒" << endl;

        q.init();

        int total_guesses = 0;
        const int MAX_GUESSES = 1000000;
        auto start_guess = system_clock::now();

        while (!q.priority.empty()&& total_guesses < MAX_GUESSES) {
            // 生成新的猜测
            q.PopNext();
            
            total_guesses += q.guesses.size();

            // 等待上一轮哈希计算的结果
            int cracked_count;
            MPI_Status status;
            MPI_Recv(&cracked_count, 1, MPI_INT, 1, 3, MPI_COMM_WORLD, &status);
            
            if (total_guesses % 100000 == 0) {
                auto current = system_clock::now();
                auto duration = duration_cast<microseconds>(current - start_guess);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                
                cout << "当前进度:" << endl;
                cout << "- 已生成猜测数: " << total_guesses << "/" << MAX_GUESSES << endl;
                cout << "- 本轮破解数量: " << cracked_count << endl;
                cout << "- 已用时间: " << time_guess << " 秒" << endl;
                cout << "- 猜测生成速度: " << total_guesses/time_guess << " 个/秒" << endl;
                cout << "----------------------------------------" << endl;
            }
        }

        // 发送结束信号
        int end_signal = -1;
        MPI_Send(&end_signal, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // 计算总时间并输出最终统计信息
        auto end_guess = system_clock::now();
        auto duration_guess = duration_cast<microseconds>(end_guess - start_guess);
        time_guess = double(duration_guess.count()) * microseconds::period::num / microseconds::period::den;
        
        cout << "\n最终统计信息:" << endl;
        cout << "- 总训练时间: " << time_train << " 秒" << endl;
        cout << "- 总猜测时间: " << time_guess << " 秒" << endl;
        cout << "- 总猜测数量: " << total_guesses << endl;
        cout << "- 平均猜测速度: " << total_guesses/time_guess << " 个/秒" << endl;

    }
    // 进程1负责计算哈希
    else if (rank == 1) {
        // 记录哈希计算时间
        double time_hash = 0;
        int total_cracked = 0;

        // 加载测试数据
        unordered_set<string> test_set;
        ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
        int test_count = 0;
        string pw;
        while (test_data >> pw && test_count < 1000000) {
            test_set.insert(pw);
            test_count++;
        }
        test_data.close();

        while (true) {
            // 接收猜测数量
            int guess_count;
            MPI_Status status;
            MPI_Recv(&guess_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            // 检查是否收到结束信号
            if (guess_count == -1) break;

            // 接收并处理猜测
            int cracked_count = 0;
            auto start_hash = system_clock::now();

            for (int i = 0; i < guess_count; i++) {
                // 接收猜测长度
                int len;
                MPI_Recv(&len, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

                // 接收猜测内容
                vector<char> buffer(len + 1);
                MPI_Recv(buffer.data(), len, MPI_CHAR, 0, 2, MPI_COMM_WORLD, &status);
                buffer[len] = '\0';
                
                string guess(buffer.data());
                bit32 state[4];
                MD5Hash(guess, state);
                // 检查猜测是否正确
                if (test_set.find(guess) != test_set.end()) {
                    cracked_count++;
                    total_cracked++;
                }
            }

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 发送结果回进程0
            MPI_Send(&cracked_count, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        }
        cout << "\n哈希计算统计:" << endl;
        cout << "- 总哈希计算时间: " << time_hash << " 秒" << endl;
        cout << "- 总破解密码数: " << total_cracked << endl;
        cout << "- 破解成功率: " << (double)total_cracked/test_count * 100 << "%" << endl;
    }

    MPI_Finalize();
    return 0;
}