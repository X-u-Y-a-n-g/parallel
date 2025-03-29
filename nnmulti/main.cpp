#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>

using namespace std;
using namespace std::chrono;

void naive_dot_product(const vector<vector<double>>& b, const vector<double>& a, vector<double>& sum, int n) {
    for (int i = 0; i < n; i++) {
        sum[i] = 0.0;
        for (int j = 0; j < n; j++) {
            sum[i] += b[j][i] * a[j];
        }
    }
}


void optimized_dot_product(const vector<vector<double>>& b, const vector<double>& a, vector<double>& sum, int n) {

    for (int i = 0; i < n; i++) {
        sum[i] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            sum[i] += b[j][i] * a[j];
        }
    }
}


void unrolled_dot_product(const vector<vector<double>>& b, const vector<double>& a, vector<double>& sum, int n) {

    for (int i = 0; i < n; i++) {
        sum[i] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        int i = 0;
        for (; i <= n - 4; i += 4) {
            sum[i]     += b[j][i]     * a[j];
            sum[i + 1] += b[j][i + 1] * a[j];
            sum[i + 2] += b[j][i + 2] * a[j];
            sum[i + 3] += b[j][i + 3] * a[j];
        }
        for (; i < n; i++) {
            sum[i] += b[j][i] * a[j];
        }
    }
}


void generate_test_data(int n, vector<vector<double>>& b, vector<double>& a) {
    b.assign(n, vector<double>(n, 0.0));
    a.assign(n, 0.0);
    for (int i = 0; i < n; i++) {
        a[i] = i;
        for (int j = 0; j < n; j++) {
            b[i][j] = i + j;
        }
    }
}

int main() {

    ofstream file("benchmark_results.csv");
    file << "Matrix Size,Naive Time (us),Optimized Time (us),Unrolled Time (us),Speedup (Naive/Optimized),Speedup (Naive/Unrolled)" << endl;



    int repeat = 1000;

    for (int n=50;n<=3000;n+=50) {
        cout << "==========================" << endl;
        cout << "�����С: " << n << " x " << n << endl;

        vector<vector<double>> b;
        vector<double> a;
        generate_test_data(n, b, a);

        vector<double> sum(n, 0.0);

        long long total_naive = 0, total_opt = 0, total_unroll = 0;

        for (int k = 0; k < repeat; k++) {
            fill(sum.begin(), sum.end(), 0.0);
            auto start = high_resolution_clock::now();
            naive_dot_product(b, a, sum, n);
            auto end = high_resolution_clock::now();
            total_naive += duration_cast<microseconds>(end - start).count();
        }
        fill(sum.begin(), sum.end(), 0.0);
        naive_dot_product(b, a, sum, n);
        double checksum_naive = accumulate(sum.begin(), sum.end(), 0.0);
        cout << "ƽ��ʱ��: " << (total_naive / repeat) << "΢��" << endl;
        cout << "ƽ�������ǰ5��Ԫ��: ";
        for (int i = 0; i < min(n, 5); i++) {
            cout << sum[i] << " ";
        }
        cout << "У���: " << checksum_naive << endl;


        for (int k = 0; k < repeat; k++) {
            fill(sum.begin(), sum.end(), 0.0);
            auto start = high_resolution_clock::now();
            optimized_dot_product(b, a, sum, n);
            auto end = high_resolution_clock::now();
            total_opt += duration_cast<microseconds>(end - start).count();
        }
        fill(sum.begin(), sum.end(), 0.0);
        optimized_dot_product(b, a, sum, n);
        double checksum_opt = accumulate(sum.begin(), sum.end(), 0.0);
        cout << "ƽ��ʱ��: " << (total_opt / repeat) << "΢��" << endl;
        cout << "ƽ�������ǰ5��Ԫ�أ�";
        for (int i = 0; i < min(n, 5); i++) {
            cout << sum[i] << " ";
        }
        cout << "У���: " << checksum_opt << endl;

        for (int k = 0; k < repeat; k++) {
            fill(sum.begin(), sum.end(), 0.0);
            auto start = high_resolution_clock::now();
            unrolled_dot_product(b, a, sum, n);
            auto end = high_resolution_clock::now();
            total_unroll += duration_cast<microseconds>(end - start).count();
        }
        fill(sum.begin(), sum.end(), 0.0);
        unrolled_dot_product(b, a, sum, n);
        double checksum_unroll = accumulate(sum.begin(), sum.end(), 0.0);
        cout << "ƽ��ʱ��: " << (total_unroll / repeat) << "΢��" << endl;
        cout << "ƽ�������ǰ5��Ԫ��: ";
        for (int i = 0; i < min(n, 5); i++) {
            cout << sum[i] << " ";
        }
        cout << "У���: " << checksum_unroll << endl;

        cout << "���ٱ�(ƽ��/�Ż�): " << (double)total_naive/(double)total_opt << endl;
        cout << "���ٱ�(ƽ��/ѭ��չ��): " <<  (double)total_naive/(double)total_unroll << endl;
        double speedup_opt = (double)total_naive/(double)total_opt;
        double speedup_unroll = (double)total_naive/(double)total_unroll;
        file << n << "," << (total_naive / repeat) << "," << (total_opt / repeat) << "," << (total_unroll / repeat) << "," << speedup_opt << "," << speedup_unroll << endl;
    }
    file.close();
    cout << "������д�� benchmark_results.csv" << endl;
    return 0;
}
