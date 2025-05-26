#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main()
{
    double time_hash_serial = 0;   // 串行哈希时间
    double time_hash_parallel = 0;  // 并行哈希时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    model pcfg_model;
    auto start_train = system_clock::now();
    pcfg_model.train("/guessdata/Rockyou-singleLined-full.txt");
    pcfg_model.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;


    
    // 加载一些测试数据
    unordered_set<std::string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count=0;
    string pw;
    while(test_data>>pw)
    {   
        test_count+=1;
        test_set.insert(pw);
        if (test_count>=1000000)
        {
            break;
        }
    }
    int cracked=0;

    // 创建多级队列系统
    MultiLevelQueue mlq(pcfg_model);
    cout << "Initializing multi-level queue system..." << endl;
    
    // 开始计时
    auto start = system_clock::now();
    
    // 初始化并开始并行处理
    mlq.init();
    mlq.process_parallel();
    
    const vector<string>& guesses = mlq.get_guesses();
    int batch_size = 1000000;
    size_t total_processed = 0;
    
    // 分批处理生成的猜测
    while(total_processed < guesses.size()) {
        size_t current_batch_size = min(static_cast<size_t>(batch_size), 
                                  guesses.size() - total_processed);
        vector<string> current_batch(guesses.begin() + total_processed,
                                   guesses.begin() + total_processed + current_batch_size);
        
        // 串行哈希处理
        auto start_serial = system_clock::now();
        bit32 state[4];
        for(const string& pw : current_batch) {
            if(test_set.find(pw) != test_set.end()) {
                cracked++;
            }
            MD5Hash(pw, state);
        }
        auto end_serial = system_clock::now();
        auto duration_serial = duration_cast<microseconds>(end_serial - start_serial);
        time_hash_serial += double(duration_serial.count()) * 
                          microseconds::period::num / microseconds::period::den;
        
        // 并行哈希处理
        auto start_parallel = system_clock::now();
        vector<bit32*> hash_states;
        MD5HashBatch(current_batch, hash_states);
        
        // 清理内存
        for(auto state_ptr : hash_states) {
            delete[] state_ptr;
        }
        
        auto end_parallel = system_clock::now();
        auto duration_parallel = duration_cast<microseconds>(end_parallel - start_parallel);
        time_hash_parallel += double(duration_parallel.count()) * 
                            microseconds::period::num / microseconds::period::den;
        
        // 输出当前批次统计
        cout << "\nBatch " << (total_processed / batch_size + 1) << " statistics:" << endl;
        cout << "Processed guesses: " << total_processed + current_batch_size << endl;
        cout << "Serial Hash time: " << double(duration_serial.count()) * 
                microseconds::period::num / microseconds::period::den << " seconds" << endl;
        cout << "Parallel Hash time: " << double(duration_parallel.count()) * 
                microseconds::period::num / microseconds::period::den << " seconds" << endl;
        cout << "Current speedup: " << double(duration_serial.count()) / 
                duration_parallel.count() << "x" << endl;
        cout << "----------------------------------------" << endl;
        
        total_processed += current_batch_size;
        
        // 检查是否达到生成上限
        if(total_processed >= 10000000) break;
    }
    
    // 计算总时间
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    
    // 输出最终统计结果
    cout << "\nFinal Statistics:" << endl;
    cout << "Total guesses generated: " << total_processed << endl;
    cout << "Guess time: " << time_guess - (time_hash_serial + time_hash_parallel) << " seconds" << endl;
    cout << "Serial Hash time: " << time_hash_serial << " seconds" << endl;
    cout << "Parallel Hash time: " << time_hash_parallel << " seconds" << endl;
    cout << "Overall parallel speedup: " << time_hash_serial / time_hash_parallel << "x" << endl;
    cout << "Train time: " << time_train << " seconds" << endl;
    cout << "Passwords cracked: " << cracked << endl;
    
    return 0;
}
