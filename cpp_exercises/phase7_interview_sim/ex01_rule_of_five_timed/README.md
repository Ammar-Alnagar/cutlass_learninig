# ex01: Rule of Five — Timed (4 min)

Implement all five special member functions in 4 minutes.

## What You Build

A `Buffer` class with destructor, copy constructor, copy assignment, move constructor, and move assignment.

## Interview Rubric

- [ ] Destructor frees memory
- [ ] Copy constructor does deep copy
- [ ] Copy assignment frees old, deep copies, checks self-assignment
- [ ] Move constructor steals pointer, nullifies source
- [ ] Move assignment frees old, steals, nullifies, checks self-assignment

## Time Targets

- < 3 min: Excellent
- 3-4 min: Good
- 4-5 min: Acceptable
- > 5 min: Review ownership

## Build Command

```bash
g++ -std=c++20 -O2 -o ex01 exercise.cpp && ./ex01
```
