#include <stdio.h>
#include "helper.h"

int main() {
    print_greeting("CMake Learner");
    
    int num = 5;
    int squared = calculate_square(num);
    printf("The square of %d is %d\n", num, squared);
    
    return 0;
}