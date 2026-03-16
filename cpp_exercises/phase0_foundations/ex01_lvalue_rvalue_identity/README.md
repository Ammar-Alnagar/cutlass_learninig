# ex01: lvalue vs rvalue Identity

Build and run this exercise to observe the fundamental difference between lvalues and rvalues.

## What You Build

A program that prints addresses of named variables vs temporaries, demonstrating that lvalues are addressable and rvalues are not.

## What You Observe

Named variables (`x`) have addresses. Literals (`42`) and function return values do not. `std::move(x)` changes the type to rvalue reference without changing the underlying object.

## CUTLASS/CUDA Mapping

Kernel argument passing follows the same rules. A named device pointer is an lvalue. A temporary from a device function is an rvalue. Generic CUDA wrappers must preserve value category when forwarding arguments.

## Build Command

```bash
g++ -std=c++20 -O2 -Wall -Wextra -o ex01 exercise.cpp && ./ex01
```
