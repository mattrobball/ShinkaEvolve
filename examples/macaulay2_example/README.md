# Macaulay2 Example for ShinkaEvolve

This example demonstrates how to use ShinkaEvolve with Macaulay2 programs.

## Overview

Macaulay2 is a software system for algebraic geometry and commutative algebra research. This example shows how to evolve Macaulay2 programs using ShinkaEvolve's genetic programming framework.

## Files

- `initial.m2`: The initial Macaulay2 program with EVOLVE-BLOCK markers
- `evaluate.py`: Evaluation script that runs Macaulay2 programs and scores them
- `README.md`: This file

## Requirements

To use this example, you need:

1. Macaulay2 installed and available in your PATH as `M2`
2. ShinkaEvolve installed and configured

## Initial Program

The initial program (`initial.m2`) computes the Hilbert polynomial of an ideal:

```macaulay2
-- EVOLVE-BLOCK-START
-- Example Macaulay2 program to compute Hilbert polynomial
-- This is a simple initial program that will be evolved

R = QQ[x,y,z]
I = ideal(x^2, x*y, y^2)
hilbertPolynomial(I, Projective => false)
-- EVOLVE-BLOCK-END
```

## EVOLVE-BLOCK Markers

The code between `-- EVOLVE-BLOCK-START` and `-- EVOLVE-BLOCK-END` markers will be evolved by ShinkaEvolve. Note that Macaulay2 uses `--` for comments, so the markers use this comment style.

## Evaluation

The `evaluate.py` script:
1. Runs the Macaulay2 program using the `M2` interpreter
2. Captures stdout, stderr, and return code
3. Computes a score based on execution success
4. Saves metrics to JSON files

You can customize the evaluation logic to score programs based on:
- Correctness of mathematical results
- Computational efficiency
- Code quality metrics
- Problem-specific criteria

## Running Evolution

To run evolution with this example, you can create a run script similar to other examples, or use the Hydra configuration:

```bash
python -m shinka.run task=macaulay2_example
```

## Customization

To adapt this example for your own Macaulay2 problems:

1. Modify `initial.m2` with your starting code
2. Update `evaluate.py` to implement your scoring criteria
3. Adjust the task configuration in `configs/task/macaulay2_example.yaml`
4. Update the system message to provide domain-specific guidance

## Notes

- Macaulay2 programs use `.m2` file extension
- The syntax validator uses `M2 --script` to check for syntax errors
- Comment style: `--` (double dash)
- Complexity analysis uses regex-based pattern matching (same as C++/Rust)
