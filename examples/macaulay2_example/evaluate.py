"""
Evaluation script for Macaulay2 programs.

This script runs a Macaulay2 program and evaluates its output.
"""

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path


def run_macaulay2_program(program_path: str, timeout: int = 30) -> dict:
    """
    Run a Macaulay2 program and capture its output.

    Args:
        program_path: Path to the .m2 file to execute
        timeout: Timeout in seconds for program execution

    Returns:
        Dictionary containing execution results
    """
    try:
        # Run the Macaulay2 program using M2 interpreter
        result = subprocess.run(
            ["M2", "--script", program_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Program execution timed out after {timeout} seconds",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Error running program: {str(e)}",
            "returncode": -1,
        }


def main(program_path: str, results_dir: str) -> None:
    """
    Evaluate a Macaulay2 program and save results.

    Args:
        program_path: Path to the .m2 program file
        results_dir: Directory to save evaluation results
    """
    print(f"Evaluating Macaulay2 program: {program_path}")
    print(f"Saving results to: {results_dir}")

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    try:
        # Run the Macaulay2 program
        execution_result = run_macaulay2_program(program_path)

        # Calculate a simple score based on execution success
        # This is a placeholder - customize based on your specific evaluation criteria
        if execution_result["success"]:
            combined_score = 1.0
        else:
            combined_score = 0.0

        # Prepare metrics
        metrics = {
            "combined_score": combined_score,
            "public": {
                "success": execution_result["success"],
                "returncode": execution_result["returncode"],
            },
            "private": {
                "stdout": execution_result["stdout"],
                "stderr": execution_result["stderr"],
            },
        }

        # Save metrics to JSON
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save correctness result
        correct_path = os.path.join(results_dir, "correct.json")
        with open(correct_path, "w") as f:
            json.dump({"correct": execution_result["success"]}, f)

        print(f"Evaluation complete. Score: {combined_score}")
        print(f"Success: {execution_result['success']}")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        traceback.print_exc()

        # Save error metrics
        error_metrics = {
            "combined_score": 0.0,
            "public": {
                "success": False,
                "error": str(e),
            },
            "private": {
                "traceback": traceback.format_exc(),
            },
        }

        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(error_metrics, f, indent=2)

        correct_path = os.path.join(results_dir, "correct.json")
        with open(correct_path, "w") as f:
            json.dump({"correct": False}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Macaulay2 program")
    parser.add_argument("--program_path", type=str, required=True, help="Path to the .m2 program")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results")

    args = parser.parse_args()
    main(args.program_path, args.results_dir)
