# ex02: unique_ptr from Memory — Timed (6 min)

Implement a `unique_ptr`-like class from memory in 6 minutes.

## What You Build

A `SimpleUniquePtr<T>` template with move-only semantics, `get()`, `release()`, `reset()`, and bool conversion.

## Interview Rubric

- [ ] Destructor deletes ptr
- [ ] Copy ops deleted (move-only)
- [ ] Move ctor steals and nullifies
- [ ] Move assign frees old, steals, nullifies
- [ ] `get()` returns raw pointer
- [ ] `release()` returns ptr and sets to null
- [ ] `reset()` deletes old, takes new

## Time Targets

- < 5 min: Excellent
- 5-6 min: Good
- 6-8 min: Acceptable
- > 8 min: Review unique_ptr

## Build Command

```bash
g++ -std=c++20 -O2 -o ex02 exercise.cpp && ./ex02
```
