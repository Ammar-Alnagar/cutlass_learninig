# Kernel Implications: Non-Contiguous Access

## What This Is

PagedAttention requires non-contiguous memory access (gather pattern). This is less efficient than contiguous access but enables much larger batch sizes.

## The Kernel Implication

### CuTe ComposedLayout Connection

**PagedAttention as ComposedLayout:**
```cpp
// Logical layout: [batch, layer, logical_block, offset]
auto logical = make_layout(shape(B, L, num_blocks, block_size));

// Physical layout: block_table mapping
auto physical = make_layout(shape(num_physical_blocks, block_size));

// ComposedLayout: logical → physical via block_table
auto composed = make_composed_layout(logical, physical, block_table);
```

### Gather Kernel

```cuda
__device__ float gather_kv(float* kv_cache, int* block_table, 
                           int token_pos, int block_size) {
    int logical_block = token_pos / block_size;
    int offset = token_pos % block_size;
    int physical_block = block_table[logical_block];
    
    return kv_cache[physical_block * block_size + offset];
}
```

**Optimization:** Cache block_table in shared memory for reuse across threads.
