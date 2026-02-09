#include <iostream>
#include "string_utils.h"
#include "math_utils.h"

int main() {
    // String utilities demo
    std::string text = "Hello, World!";
    std::cout << "Original: " << text << std::endl;
    std::cout << "Upper: " << utils::to_upper(text) << std::endl;
    std::cout << "Reversed: " << utils::reverse_string(text) << std::endl;
    std::cout << "Word count: " << utils::count_words(text) << std::endl;
    
    // Math utilities demo
    std::cout << "\nMath utilities:" << std::endl;
    std::cout << "2^8 = " << utils::power(2, 8) << std::endl;
    std::cout << "Is 17 prime? " << (utils::is_prime(17) ? "Yes" : "No") << std::endl;
    
    int numbers[] = {1, 2, 3, 4, 5};
    std::cout << "Average of [1,2,3,4,5]: " << utils::average(numbers, 5) << std::endl;
    
    return 0;
}