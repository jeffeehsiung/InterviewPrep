/**
 * FFT
 */


/** Heeaders */
#include <iostream>
#include <vector>
#include <memory> // for smart pointers
#include <cmath>
#include <complex>

using namespace std;

const double PI = 3.141592653589793238460;

void fft(unique_ptr<complex<double>[]>& a, int n){
    // base case
    if (n <= 1) return;

    // divide: n/2
    auto a_even = make_unique<complex<double>[]>(n/2);
    auto a_odd = make_unique<complex<double>[]>(n/2);
    // initialize
    for(int i = 0; i < n/2; i++){
        a_even[i] = a[i*2];
        a_odd[i] = a[i*2+1];
    }

    // conquer: revursive implementation
    fft(a_even, n/2);
    fft(a_odd, n/2);

    //combine
    for(int k = 0; k < n/2; k++){
        complex<double> t = polar(1.0, -2 * PI * k /n) * a_odd[k];
        // k and k + n/2
        a[k] = a_even[k] + t;
        a[k+n/2] = a_even[k] - t;
    }
}

int main(){
    // Example input: array of complex numbers
    int n = 8;
    /** data with unique pointers */
    auto data = make_unique<complex<double>[]>(n);

    // Fill the input array
    data[0] = 1; data[1] = 1; data[2] = 1; data[3] = 1;
    data[4] = 0; data[5] = 0; data[6] = 0; data[7] = 0;

    // perform FFT
    fft(data, n);

    //Output the results
    cout << "FFT results:\n";
    for (int i = 0; i < n; i++){
        cout << data [i] << "\n";
    }

    return 0;
}