# LOG2FP32 - Hardware Logarithm Function Implementation

A high-performance hardware implementation of the base-2 logarithm function (log₂) for single-precision floating-point numbers (FP32), designed using Chisel HDL.

## Overview

This project implements **LOG2FP32**, which computes `log₂(x)` for FP32 inputs using a pipelined architecture that combines lookup tables (LUT) and polynomial approximation to achieve efficient hardware implementation with high accuracy.

This project reuses the **XiangShan Fudian** floating-point unit library for basic FP32 arithmetic operations (multiplication, addition, fused multiply-add).

## Algorithm

The computation is based on the mathematical transformation:

$$
\begin{equation}
\begin{aligned}
f                    &= \log_2(x) \\
\\
x                    &= 2^{E} \times (1 + M) \\
\\
\log_2(x)            &= E + \log_2(1 + M)
\\
\log_2(1 + M)        &= \log_2(1 + M_{high}) + \log_2(1 + \frac{M_{low}}{1 + M_{high}}) \\
\\
r                    &= \frac{M_{low}}{1 + M_{high}} \\
\\
\log_2(1 + M_{high}) &= \text{LUT}[M_{high}] \\
\\
\log_2(1 + r)        &= \text{poly}(r)
\\
\text{poly}          &= \log_2(e) \cdot r - \frac{\log_2(e)}{2} \cdot r^2 \\
                     &= r (\log_2(e) - \frac{\log_2(e)}{2} \cdot r)
\end{aligned}
\end{equation}
$$

### Key Components

- **$E$**: Exponent extraction serves as the integer part of the result
- **$M_{high}$**: High bits of mantissa (7 bits) used for LUT indexing to compute $\log_2(1+M_{high})$ and $\frac{1}{1+M_{high}}$
- **$M_{low}$**: Low bits of mantissa (16 bits), converted to FP32 for residual computation
- **$r$**: Normalized residual = $M_{low} \times \frac{1}{1+M_{high}}$, range $[0, \frac{1}{2^{m+1}}]$
- **Polynomial**: Second-order Taylor approximation for $\log_2(1+r)$ using Horner's method
- **Final result**: $\log_2(x) = E + \log_2(1+M_{high}) + \text{poly}(r)$

## Hardware Design

### Input Filtering

Special value handling and input validation:

- **Special values**: NaN, +Inf, -Inf, negative numbers → return NaN
- **Zero**: +0 or -0 → return -Inf
- **Subnormal numbers**: Handled separately or return -Inf

### Pipeline Structure

```
S0: Input Filtering and Special Value Handling
    - NaN, Inf, negative number detection → return NaN
    - Zero detection → return -Inf
    - Subnormal number handling

S1: Exponent Extraction and Mantissa Decomposition 
    - Extract exponent E from FP32 representation
    - Convert E to FP32 format for later addition
    - Split mantissa: M_high (top 7 bits) and M_low (bottom 16 bits)
    - Convert M_low to FP32 format

S2: LUT Access
    - LUT lookup using M_high as index:
      - log_base = log₂(1 + M_high)
      - inv = 1 / (1 + M_high)

S3: Normalized Residual Calculation (3 FP multiply stages)
    - r = M_low × inv
    - Uses FMUL pipeline from Fudian library

S4-S5: Polynomial Approximation log₂(1+r) (5 FCMA stages)
    - Second-order Taylor expansion using Horner form
    - S4: temp = C1 + C2 × r  (where C1 = log₂(e), C2 = -log₂(e)/2)
    - S5: poly = r × temp

S6: Result Composition (2 FADD stages)
    - result = E + log_base + poly
    - Final addition and assembly
```

## Performance Results

Verification on 1,000,000 test cases:

### CPU Reference (cmath log2f)

```
Total: 1,000,000 test cases
Pass:  1,000,000 (100.00%)
Fail:  0 (0.00%)

Average Error: 1.198306e-09
Maximum Error: 1.095356e-07

Average ULP: 0.02
Maximum ULP: 1

Total Cycles: 1,000,089
Throughput:   1 result/cycle
```

### GPU Reference (RTX 5060 with -use_fast_math)

```
Total: 1,000,000 test cases
Pass:  1,000,000 (100.00%)
Fail:  0 (0.00%)

Average Error: 2.961146e-08
Maximum Error: 1.176576e-07

Average ULP: 0.46
Maximum ULP: 1

Total Cycles: 1,000,089
Throughput:   1 result/cycle
```

## Dependencies

### Required

- **Chisel 6.6.0**: Hardware description language
- **Scala 2.13.15**: Programming language for Chisel
- **Mill**: Build tool for Scala/Chisel projects
- **Verilator**: For simulation and verification
- **XiangShan Fudian**: Floating-point arithmetic library (included as git submodule)

### Optional

- **CUDA/NVCC**: For GPU-accelerated reference implementation (NVIDIA GPU required)
- **Synopsys Design Compiler**: For ASIC synthesis (if targeting specific process technology)

## Building

### Initialize Dependencies

```bash
make init
```

This will initialize the XiangShan Fudian submodule.

### Generate SystemVerilog

```bash
# Generate LOG2FP32 RTL
./mill --no-server LOG2FP32.run
```

The generated SystemVerilog will be placed in `rtl/LOGFP32.sv`.

### Build and Run Simulation

```bash
make run
```

The build system automatically detects CUDA availability:

- **Without CUDA**: Uses CPU reference only (standard C library `log2f()`)
- **With CUDA**: Uses both CPU and GPU references simultaneously
  - CPU Reference: Standard C library `log2f()`
  - GPU Reference: NVIDIA CUDA math library with `-use_fast_math` flag
  - Both error statistics are computed and displayed for comparison

### Clean Build Artifacts

```bash
make clean
```

## Testing and Verification

### Simulation

Verilator-based testbench with:

- Comprehensive test vector generation (1M test cases)
- Random input generation across full FP32 range
- Special value testing (NaN, Inf, zero, negative numbers, subnormals)
- ULP (Unit in Last Place) error measurement
- Waveform generation (FST format) for debugging

### Reference Models

The testbench automatically uses available reference implementations:

- **CPU Reference**: Standard C library (`log2f`) - always available
- **GPU Reference**: NVIDIA CUDA math library with `-use_fast_math` - automatically enabled if CUDA is detected

When both references are available, error statistics are computed against both to provide comprehensive verification.

### Accuracy Metrics

- **ULP Error**: Measures floating-point accuracy in terms of "units in the last place"
- **Relative Error**: Standard floating-point error metrics
- **Pass/Fail**: Bit-exact comparison against reference implementation

## Future Improvements

- [ ] Add support for subnormal input handling
- [ ] Optimize polynomial coefficients using Remez algorithm
- [ ] Implement configurable rounding mode support

## Credits

- **XiangShan Fudian FPU Library**: Provides high-quality floating-point arithmetic components
  - Repository: <https://github.com/OpenXiangShan/fudian>
  - Used for: FMUL, FADD, FCMA (Fused Multiply-Add), RawFloat utilities

## References

- IEEE Standard for Floating-Point Arithmetic (IEEE 754-2008)
- XiangShan Fudian FPU: <https://github.com/OpenXiangShan/fudian>
- Chisel/FIRRTL Documentation: <https://www.chisel-lang.org/>
- Handbook of Floating-Point Arithmetic (Muller et al.)
- CUDA Math API: <https://docs.nvidia.com/cuda/cuda-math-api/>

## License

This project reuses the XiangShan Fudian library. Please refer to the respective license files in the `dependencies/fudian` directory for licensing terms.
