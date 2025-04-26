#include "md5.h"
#include <vector>
#include <chrono>
#include<iostream>

using namespace std;
using namespace chrono;

#define TEST_SIZE 10000  // 测试数据量
#define BATCH_SIZE 4     // SIMD宽度
#define LOOP_TIMES 100   // 循环次数

int main() {
    // 生成随机测试数据
    vector<string> inputs;
    for(int i=0; i<TEST_SIZE; i++){
        inputs.push_back("sample_pwd_"+to_string(i)+"_"+string(50, 'a')); // 50字符填充
    }

    vector<bit32*> states;
    
    // 预热缓存
    MD5HashBatch(vector<string>(inputs.begin(), inputs.begin()+BATCH_SIZE), states);
    
    auto start = high_resolution_clock::now();
    
    // 分批次处理
    for(int i=0; i<LOOP_TIMES; i++){
        for(size_t base=0; base<inputs.size(); base+=BATCH_SIZE){
            size_t remaining = min(static_cast<size_t>(BATCH_SIZE), inputs.size()-base);
            vector<string> batch(inputs.begin()+base, inputs.begin()+base+remaining);
            
            MD5HashBatch(batch, states);
            
            // 清理内存
            for(auto* s : states) delete[] s;
            states.clear();
        }
        if(i % 10 == 0) cerr << "Processing iteration " << i << endl;
    }
    
    auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    cout << "Total time: " << duration.count() << "ms" << endl;
    
    return 0;
}