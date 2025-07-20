# Gemmini-Accelerated Llama2.c

This is a modified version of llama2.c that uses the Gemmini accelerator for improved performance on matrix operations.

## Key Modifications

### 1. Data Type Changes
- Changed from `float` to `elem_t` for weights and activations
- Added `acc_t` accumulator buffers for higher precision intermediate results
- Implemented conversion functions between float and elem_t/acc_t types

### 2. Accelerated Operations

#### Matrix Multiplication (`gemmini_matmul`)
- Replaced the original CPU matmul with Gemmini's `tiled_matmul_auto`
- Used for:
  - QKV projections in attention
  - Attention output projection
  - FFN layers (w1, w2, w3)
  - Final classifier

#### Attention Mechanism
- Accelerated dot products for attention scores
- Optimized weighted value aggregation
- Maintained RoPE positional encoding on CPU (due to trigonometric operations)

#### Feed-Forward Network
- Accelerated both linear projections (w1, w3)
- SwiGLU activation remains on CPU (due to exponential operations)
- Final projection (w2) uses Gemmini

#### RMSNorm
- Uses Gemmini's `tiled_norm_auto` with LAYERNORM activation
- Applied weight scaling after normalization

#### Residual Connections
- Uses Gemmini's `tiled_resadd_auto` for efficient residual additions

### 3. Memory Management
- Allocated additional accumulator buffers for Gemmini operations
- Converted float weights to elem_t format during model loading
- Maintained compatibility with original model file format

## Building

```bash
# Set the Gemmini root directory
export GEMMINI_ROOT=/path/to/gemmini

# Build the Gemmini version
make -f Makefile.gemmini

# Run a test
./gemmini_run stories15M.bin -i "Once upon a time"
```

## Performance Considerations

1. **Data Type Precision**: The conversion from float32 to elem_t (typically int8) may affect model accuracy. Consider using higher precision elem_t if supported.

2. **Memory Bandwidth**: Gemmini excels at compute-intensive operations but may be limited by memory bandwidth for small matrices.

3. **Kernel Fusion**: The current implementation could benefit from further optimization by fusing operations like attention score computation with softmax.

4. **Batch Processing**: For better utilization, consider processing multiple sequences in parallel.

## Limitations

1. Some operations remain on CPU:
   - RoPE positional encoding (trigonometric functions)
   - SwiGLU activation (exponential functions)
   - Softmax (currently CPU implementation)

2. The implementation assumes Gemmini is configured with appropriate tile sizes for the model dimensions.

3. Weight conversion happens at load time, which adds some overhead but ensures compatibility with existing model files.

## Future Optimizations

1. Implement Gemmini-accelerated softmax
2. Optimize memory layout for better cache utilization
3. Add support for batched inference
4. Implement custom kernels for activation functions
5. Add int8 quantization support throughout the pipeline