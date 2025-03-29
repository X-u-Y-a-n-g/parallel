#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <omp.h>  // OpenMP ���м���
#include <fstream>

using namespace std;
using namespace std::chrono;

// **��ͨ��ʽ�ۼ�**
int sum_chain(const vector<int>& a) {
    int sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += a[i];
    }
    return sum;
}

// **�ݹ����**

void recursion(std::vector<int>& a, size_t n) {
    if (n == 1) {
        return;
    } else {
        for (int i = 0; i < n / 2; i++) {
            a[i] += a[n - i - 1];  // Add the corresponding elements from the two halves
        }
        recursion(a, n / 2);  // Recurse on the first half of the array
    }
}

int sum_recursive(std::vector<int>& a) {
    if (a.empty()) {
        return 0;
    }
    recursion(a, a.size());
    return a[0];  // After recursion, the result is stored in a[0]
}

// **����ѭ�����**
int sum_iterative(vector<int>& a) {
    int n = a.size();
    while (n > 1) {
        for (int i = 0; i < n / 2; i++) {
            a[i] += a[n - i - 1];
        }
        n /= 2;
    }
    return a[0];
}

// **2·�ۼ�**
int sum_2_way(const vector<int>& a) {
    int sum1 = 0, sum2 = 0;
    for (int i=0; i  < a.size(); i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    return sum1+sum2;
}

// **4·�ۼ�**
int sum_4_way(const vector<int>& a) {
    int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    for (int i = 0; i< a.size(); i += 4) {
        sum1 += a[i];
        sum2 += a[i+1];
        sum3 += a[i+2];
        sum4 += a[i+3];
    }

    return sum1 + sum2 + sum3 + sum4;
}

// **8·�ۼ�**
int sum_8_way(const vector<int>& a) {
    int sum1=0,sum2=0,sum3=0,sum4=0,sum5=0,sum6=0,sum7=0,sum8=0;
    for (int i=0; i< a.size(); i += 8) {
         sum1 += a[i];
        sum2 += a[i+1];
        sum3 += a[i+2];
        sum4 += a[i+3];
        sum5 += a[i+4];
        sum6 += a[i+5];
        sum7 += a[i+6];
        sum8 += a[i+7];
    }
    return sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8;
}



int main() {

    int repeat = 1000;

    ofstream file("sum_benchmark_results.csv");
    file << "Array Size,Chain Time (us),Recursive Time (us),Iterative Time (us),2-way Time (us),4-way Time (us),8-way Time (us),"
            "Speedup (Chain/Recursive),Speedup (Chain/Iterative),Speedup (Chain/2-way),Speedup (Chain/4-way),Speedup (Chain/8-way)\n";

    for (int n=(1<<5);n<=(1<<22);n=(n<<1)) {
        cout << "==========================" << endl;
        cout << "�����С n = " << n << endl;

        vector<int> a(n, 1);  // ��ʼ������
        long int total_chain = 0, total_2_way = 0, total_4_way = 0, total_8_way = 0, total_16_way = 0;
        long int total_recursive = 0, total_iterative = 0;

        int res_chain = 0, res_2_way = 0, res_4_way = 0, res_8_way = 0, res_16_way = 0;
        int res_recursive = 0, res_iterative = 0;

        for (int k = 0; k < repeat; k++) {
            // ��������
            vector<int> a_copy(n,1);

            auto start = steady_clock::now();
            res_chain = sum_chain(a_copy);
            auto end = steady_clock::now();
            total_chain += duration_cast<nanoseconds>(end - start).count(); // ΢��
        }
        cout << "ƽ����ʽ�ۼӽ��: " << res_chain << " (ӦΪ " << n << ")"
             << "��ƽ��ʱ��: " << (total_chain / repeat) << " ΢��" << endl;


        for (int k = 0; k < repeat; k++) {
            // ��������
            vector<int> a_copy(n,1);

            auto start = steady_clock::now();
            res_recursive = sum_recursive(a_copy);
            auto end = steady_clock::now();
            total_recursive += duration_cast<nanoseconds>(end - start).count();
        }
        cout << "�ݹ���ͽ��: " << res_recursive << " (ӦΪ " << n << ")"
             << "��ƽ��ʱ��: " << (total_recursive / repeat) << " ΢��" << endl;

        for (int k = 0; k < repeat; k++) {
            // ��������
            vector<int> a_copy(n,1);

            auto start = steady_clock::now();
            res_iterative = sum_iterative(a_copy);
            auto end = steady_clock::now();
            total_iterative += duration_cast<nanoseconds>(end - start).count();
        }
        cout << "����ѭ����ͽ��: " << res_iterative << " (ӦΪ " << n << ")"
             << "��ƽ��ʱ��: " << (total_iterative / repeat) << " ΢��" << endl;


        for (int k = 0; k < repeat; k++) {
            // ��������
            vector<int> a_copy(n,1);

            auto start = steady_clock::now();
            res_2_way = sum_2_way(a_copy);
            auto end = steady_clock::now();
            total_2_way += duration_cast<nanoseconds>(end - start).count();
        }
        cout << "2·�ۼӽ��: " << res_2_way << " (ӦΪ " << n << ")"
             << "��ƽ��ʱ��: " << (total_2_way / repeat) << " ΢��" << endl;

        for (int k = 0; k < repeat; k++) {
            // ��������
            vector<int> a_copy(n,1);
            auto start = steady_clock::now();
            res_4_way = sum_4_way(a_copy);
            auto end = steady_clock::now();
            total_4_way += duration_cast<nanoseconds>(end - start).count();
        }
        cout << "4·�ۼӽ��: " << res_4_way << " (ӦΪ " << n << ")"
             << "��ƽ��ʱ��: " << (total_4_way / repeat) << " ΢��" << endl;


        for (int k = 0; k < repeat; k++) {
            // ��������
            vector<int> a_copy(n,1);

            auto start = steady_clock::now();
            res_8_way = sum_8_way(a_copy);
            auto end = steady_clock::now();
            total_8_way += duration_cast<nanoseconds>(end - start).count();
        }
        cout << "8·�ۼӽ��: " << res_8_way << " (ӦΪ " << n << ")"
             << "��ƽ��ʱ��: " << (total_8_way / repeat) << " ΢��" << endl;



        // ������ٱ�
        cout << "���ٱ�(ƽ��/�ݹ�): " << (double)total_chain/(double)total_recursive  << endl;
        cout << "���ٱ�(ƽ��/����ѭ��): " << (double)total_chain/(double)total_iterative   << endl;
        cout << "���ٱ�(ƽ��/2·): " << (double)total_chain/(double)total_2_way  << endl;
        cout << "���ٱ�(ƽ��/4·): " << (double)total_chain/(double)total_4_way  << endl;
        cout << "���ٱ�(ƽ��/8·): " << (double)total_chain/(double)total_8_way  << endl;


        double speedup_recursive = (double)total_chain / (double)total_recursive;
        double speedup_iterative = (double)total_chain / (int)total_iterative;
        double speedup_2_way = (double)total_chain / (double)total_2_way;
        double speedup_4_way = (double)total_chain / (double)total_4_way;
        double speedup_8_way = (double)total_chain / (double)total_8_way;

        file << n << "," << (total_chain / repeat) << "," << (total_recursive / repeat) << "," << (total_iterative / repeat) << ","
             << (total_2_way / repeat) << "," << (total_4_way / repeat) << "," << (total_8_way / repeat) << ","
             << speedup_recursive << "," << speedup_iterative << "," << speedup_2_way << "," << speedup_4_way << "," << speedup_8_way << "\n";
    }

    file.close();
    cout << "������д�� sum_benchmark_results.csv" << endl;
    return 0;
}
