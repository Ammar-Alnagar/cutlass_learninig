"""
Module 13 — Iterators & Generators
Exercise 03 — itertools for Benchmark Grids

WHAT YOU'RE BUILDING:
  itertools provides efficient iterator building blocks. For kernel
  benchmarking, you need product (grid search), permutations, combinations,
  and islice (limit iterations). This is cleaner than nested loops.

OBJECTIVE:
  - Use itertools.product for config grids
  - Use itertools.islice to limit infinite iterators
  - Use itertools.chain to combine iterators
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does itertools.product([1,2], ['a','b']) produce?
# Q2: How is itertools.chain different from list concatenation?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import itertools
from typing import List, Tuple, Iterator

BLOCK_SIZES = [32, 64, 128, 256, 512]
NUM_STAGES = [2, 3, 4]
NUM_WARPS = [4, 8]

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Generate all kernel configs using itertools.product.
#              This replaces nested for loops.
# HINT: itertools.product(BLOCK_SIZES, NUM_STAGES, NUM_WARPS)

def generate_config_grid() -> List[Tuple[int, int, int]]:
    """Generate all (block_size, num_stages, num_warps) combinations."""
    # TODO: use itertools.product
    pass

# TODO [MEDIUM]: Generate configs but limit to first N using islice.
#              Useful for quick testing without full grid.
# HINT: itertools.islice(iterator, n)

def generate_limited_configs(n: int) -> List[Tuple[int, int, int]]:
    """Generate first n configs from the grid."""
    # TODO: use itertools.islice with product
    pass

# TODO [EASY]: Chain multiple config sources together.
#              E.g., small configs + large configs in one iterator.
# HINT: itertools.chain(iter1, iter2)

def generate_all_scale_configs() -> Iterator[Tuple[int, int, int]]:
    """Chain small and large config ranges."""
    small_configs = itertools.product([32, 64], NUM_STAGES, NUM_WARPS)
    large_configs = itertools.product([1024, 2048], NUM_STAGES, NUM_WARPS)
    # TODO: chain them together
    pass

# TODO [MEDIUM]: Use itertools.permutations and combinations.
#              For ablation studies, you might test different orderings
#              or subsets of optimization techniques.
# HINT: itertools.permutations(items, r), itertools.combinations(items, r)

def generate_ablation_orders(techniques: List[str]) -> List[Tuple[str, ...]]:
    """Generate all orderings of techniques for ablation study."""
    # TODO: return all permutations of techniques
    pass

def generate_techniques_subsets(techniques: List[str], subset_size: int) -> List[Tuple[str, ...]]:
    """Generate all subsets of given size."""
    # TODO: return all combinations of techniques
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How many configs in the full grid? How does product compare to nested loops?
# C2: When would you use chain vs concatenating lists?

if __name__ == "__main__":
    print("Config grid with itertools.product...")
    grid = generate_config_grid()
    print(f"  Total configs: {len(grid)}")
    print(f"  First 3: {grid[:3]}\n")

    print("Limited configs with islice...")
    limited = generate_limited_configs(5)
    print(f"  First 5: {limited}\n")

    print("Chained configs...")
    all_scales = list(generate_all_scale_configs())
    print(f"  Total: {len(all_scales)}")
    print(f"  All: {all_scales}\n")

    print("Ablation study...")
    techniques = ["fusion", "tiling", "vectorization"]
    orders = generate_ablation_orders(techniques)
    print(f"  Orderings: {orders}")
    
    subsets = generate_techniques_subsets(techniques, 2)
    print(f"  Pairs: {subsets}\n")

    print("Done!")
