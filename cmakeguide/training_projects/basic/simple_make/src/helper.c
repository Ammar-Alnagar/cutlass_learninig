#include <stdio.h>
#include "helper.h"

void print_greeting(const char* name) {
    printf("Hello, %s! Welcome to Makefiles.\n", name);
}

int calculate_square(int num) {
    return num * num;
}