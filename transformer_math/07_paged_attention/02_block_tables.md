# Block Tables: Logical to Physical Mapping

## What This Is

Block tables map logical token positions to physical KV cache blocks, like OS page tables map virtual to physical addresses.

## The Math

### Block Table Structure

**Block table:**
```
block_table[batch_idx, layer, logical_block_idx] = physical_block_id
```

**Example (batch=0, layer=0):**
| Logical Block | Physical Block |
|---------------|----------------|
| 0 | 5 |
| 1 | 12 |
| 2 | 3 |
| 3 | 27 |

**Token position → KV cache access:**
```
token_pos = 35
block_size = 16
logical_block = token_pos // block_size = 2
offset_in_block = token_pos % block_size = 3
physical_block = block_table[logical_block] = 3
kv_offset = physical_block * block_size + offset_in_block = 3 * 16 + 3 = 51
```

## The Kernel Implication

**Block table lookup in kernel:**
```cuda
__global__ void paged_attention(Q, K_cache, V_cache, O, block_table) {
    int token_pos = ...;  // Position attending to
    int block_size = 16;
    
    int logical_block = token_pos / block_size;
    int offset = token_pos % block_size;
    int physical_block = block_table[batch, layer, logical_block];
    
    // Load KV from non-contiguous location
    float* k_ptr = K_cache + physical_block * block_size * stride;
    float k_val = k_ptr[offset];
}
```
