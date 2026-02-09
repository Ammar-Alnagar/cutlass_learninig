#include "math_utils.h"

namespace utils {
    double power(double base, int exp) {
        if (exp == 0) return 1;
        
        double result = 1.0;
        int abs_exp = exp < 0 ? -exp : exp;
        
        for (int i = 0; i < abs_exp; i++) {
            result *= base;
        }
        
        return exp < 0 ? 1.0 / result : result;
    }
    
    bool is_prime(int num) {
        if (num <= 1) return false;
        if (num <= 3) return true;
        if (num % 2 == 0 || num % 3 == 0) return false;
        
        for (int i = 5; i * i <= num; i += 6) {
            if (num % i == 0 || num % (i + 2) == 0) {
                return false;
            }
        }
        
        return true;
    }
    
    double average(int arr[], int size) {
        if (size == 0) return 0.0;
        
        long long sum = 0;
        for (int i = 0; i < size; i++) {
            sum += arr[i];
        }
        
        return static_cast<double>(sum) / size;
    }
}