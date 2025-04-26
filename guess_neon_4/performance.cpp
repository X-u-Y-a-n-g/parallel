#include "md5.h"
#include <vector>
#include <chrono>
#include <iostream>

using namespace std;
using namespace chrono;

#define TEST_SIZE 10000  // 测试数据量
#define BATCH_SIZE 4     // 批次大小（模拟SIMD分组）
#define LOOP_TIMES 100   // 循环次数

int main() {
    // 生成随机测试数据
    vector<string> inputs;
    for(int i=0; i<TEST_SIZE; i++){
        inputs.push_back("sample_pwd_"+to_string(i)+"_"+string(50, 'a')); // 50字符填充
    }
    
    // 预热缓存
    for(int i=0; i<BATCH_SIZE; i++) {
        bit32* state = new bit32[4];
        MD5Hash(inputs[i], state);
        delete[] state; // 立即释放预热内存
    }
    
    auto start = high_resolution_clock::now();
    
    // 分批次处理（模拟批处理逻辑）
    for(int i=0; i<LOOP_TIMES; i++){
        for(size_t base=0; base<inputs.size(); base+=BATCH_SIZE){
            size_t remaining = min(static_cast<size_t>(BATCH_SIZE), inputs.size()-base);
            vector<bit32*> batch_states;
            
            // 逐个处理批次内的每个输入
            for(size_t j=0; j<remaining; j++){
                bit32* state = new bit32[4];
                MD5Hash(inputs[base+j], state);
                batch_states.push_back(state);
            }
            
            // 清理内存
            for(auto* s : batch_states) delete[] s;
            batch_states.clear();
        }
        if(i % 10 == 0) cerr << "Processing iteration " << i << endl;
    }
    
    auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    cout << "Total time: " << duration.count() << "ms" << endl;
    
    return 0;
}