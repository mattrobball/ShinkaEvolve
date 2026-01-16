"""
Evaluation script for Macaulay2 resurgence optimization.

This evaluates M2 programs that define ideals and scores them based on
their resurgence properties. Higher resurgence = better score.

The evolved M2 program should define an ideal I (typically an ideal of points).
This evaluation script:
1. Loads the ideal via resurgence_check.m2
2. Computes symbolic power containments I^(m) ⊆ I^r
3. Estimates resurgence as the score (higher is better)
"""

import argparse
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from m2_runner import ResurgenceResult, run_resurgence_check


def validate_resurgence_result(result: ResurgenceResult) -> Tuple[bool, Optional[str]]:
    """
    Validates resurgence computation results.

    Args:
        result: ResurgenceResult from run_resurgence_check

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    if not result.success:
        return False, f"Computation failed: {result.error_message}"

    if result.num_generators is None:
        return False, "Could not determine number of generators"

    if result.num_generators == 0:
        return False, "Ideal has no generators (zero ideal)"

    # Check that we got some containment results
    if not result.containments:
        return False, "No containment checks completed"

    return True, None


def construct_error_feedback(result: ResurgenceResult) -> str:
    """
    Construct detailed text feedback for failed M2 executions.
    This helps the LLM understand what went wrong and how to fix it.
    """
    feedback_parts = []

    feedback_parts.append("# M2 Execution Error\n")

    if result.error_message:
        feedback_parts.append(f"## Error Message:\n{result.error_message}\n")

    # Include relevant stderr (often contains M2 error details)
    if result.stderr and result.stderr.strip():
        # Truncate but keep useful info
        stderr_preview = result.stderr[:1500]
        if len(result.stderr) > 1500:
            stderr_preview += "\n... (truncated)"
        feedback_parts.append(f"## M2 stderr output:\n```\n{stderr_preview}\n```\n")

    # Include stdout if it might have useful info
    if result.stdout and result.stdout.strip():
        stdout_preview = result.stdout[:1000]
        if len(result.stdout) > 1000:
            stdout_preview += "\n... (truncated)"
        feedback_parts.append(f"## M2 stdout output:\n```\n{stdout_preview}\n```\n")

    # Common M2 issues and hints
    feedback_parts.append("## Common issues to check:\n")
    feedback_parts.append("- Ensure the ideal I is defined (not just pts or R)\n")
    feedback_parts.append("- Check M2 syntax: use 'ideal(...)' not 'Ideal(...)'\n")
    feedback_parts.append("- Point ideals should be: ideal(x-a, y-b) for point (a,b)\n")
    feedback_parts.append("- Use 'intersect' to combine ideals: I = intersect {i1, i2, ...}\n")
    feedback_parts.append("- Ensure all variables are defined in the ring R\n")
    feedback_parts.append("- AVOID underscores in variable names! M2 uses _ for subscripts.\n")
    feedback_parts.append("  Bad: lines_B3, pts_list  Good: linesB3, ptsList\n")
    feedback_parts.append("- 'set {}' creates empty set; add with: mySet = mySet + set{newElement}\n")

    return "\n".join(feedback_parts)


def aggregate_resurgence_metrics(
    results: List[ResurgenceResult], results_dir: str
) -> Dict[str, Any]:
    """
    Aggregates metrics from resurgence computation results.

    The score is based on the resurgence lower bound - higher resurgence means
    the ideal has more interesting containment failure properties.

    Args:
        results: List of ResurgenceResult from multiple runs
        results_dir: Directory to save extra data

    Returns:
        Metrics dictionary with combined_score based on resurgence
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    # Use the first (and typically only) result
    result = results[0]

    if not result.success:
        error_feedback = construct_error_feedback(result)
        return {
            "combined_score": 0.0,
            "error": result.error_message,
            "text_feedback": error_feedback,
            "public": {"success": False, "error": result.error_message},
            "private": {"stdout": result.stdout[:2000], "stderr": result.stderr[:2000]},
        }

    # The score is the resurgence lower bound
    # We add a small bonus for having more containment failures
    # and for the Waldschmidt constant if available
    resurgence_lb = result.resurgence_lower_bound

    # Count containment failures
    num_failures = sum(
        1 for c in result.containments
        if c.contained is False
    )

    # Base score is the resurgence lower bound
    # Multiply by 100 to make differences more visible
    combined_score = resurgence_lb * 100.0

    # Small bonus for each containment failure (encourages finding multiple)
    combined_score += num_failures * 5.0

    # Format containment results for display
    containment_summary = []
    for c in result.containments:
        status = "✓" if c.contained else "✗" if c.contained is False else "?"
        containment_summary.append(f"I^({c.m}) ⊆ I^{c.r}: {status} (ratio {c.ratio:.3f})")

    public_metrics = {
        "success": True,
        "resurgence_lower_bound": resurgence_lb,
        "num_containment_failures": num_failures,
        "num_generators": result.num_generators,
        "ideal_dimension": result.dimension,
        "ideal_codimension": result.codimension,
        "waldschmidt_constant": result.waldschmidt_constant,
        "containment_summary": containment_summary,
    }

    private_metrics = {
        "all_containments": [
            {"m": c.m, "r": c.r, "ratio": c.ratio, "contained": c.contained}
            for c in result.containments
        ],
        "stdout_preview": result.stdout[:500] if result.stdout else "",
    }

    return {
        "combined_score": combined_score,
        "public": public_metrics,
        "private": private_metrics,
    }


def run_resurgence_eval(
    program_path: str,
    results_dir: str,
    num_runs: int = 1,
    timeout: int = 120,
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """
    Runs resurgence evaluation on an M2 program.

    Args:
        program_path: Path to the .m2 file that defines ideal I
        results_dir: Directory to save metrics.json and correct.json
        num_runs: Number of times to run (typically 1 for deterministic computation)
        timeout: Timeout in seconds for each run

    Returns:
        Tuple of (metrics_dict, overall_correct, first_error_message)
    """
    os.makedirs(results_dir, exist_ok=True)

    overall_correct = True
    first_error_message: Optional[str] = None
    all_validation_errors: List[str] = []
    num_valid_runs = 0
    num_invalid_runs = 0

    all_results: List[ResurgenceResult] = []
    execution_times: List[float] = []

    try:
        for i in range(num_runs):
            start_time = time.perf_counter()
            result = run_resurgence_check(program_path, timeout)
            end_time = time.perf_counter()

            all_results.append(result)
            execution_times.append(end_time - start_time)

            is_valid, validation_err = validate_resurgence_result(result)
            if not is_valid:
                num_invalid_runs += 1
                overall_correct = False
                if validation_err:
                    if not first_error_message:
                        first_error_message = f"Validation failed: {validation_err}"
                    if validation_err not in all_validation_errors:
                        all_validation_errors.append(validation_err)
            else:
                num_valid_runs += 1

            print(f"Run {i + 1}/{num_runs} completed in {end_time - start_time:.2f}s")

        # Aggregate metrics
        metrics = aggregate_resurgence_metrics(all_results, results_dir)

        # Add timing stats
        metrics["execution_time_mean"] = (
            float(np.mean(execution_times)) if execution_times else 0.0
        )
        metrics["execution_time_std"] = (
            float(np.std(execution_times)) if execution_times else 0.0
        )

        metrics["num_valid_runs"] = num_valid_runs
        metrics["num_invalid_runs"] = num_invalid_runs
        metrics["all_validation_errors"] = all_validation_errors

    except Exception as e:
        print(f"Evaluation error: {e}")
        traceback.print_exc()
        metrics = {
            "combined_score": 0.0,
            "execution_time_mean": 0.0,
            "execution_time_std": 0.0,
            "num_valid_runs": 0,
            "num_invalid_runs": num_runs,
            "all_validation_errors": [str(e)],
        }
        first_error_message = str(e)
        overall_correct = False

    # Save results
    save_json_results(results_dir, metrics, overall_correct, first_error_message)

    return metrics, overall_correct, first_error_message


def save_json_results(
    results_dir: str,
    metrics: Dict[str, Any],
    correct: bool,
    error: Optional[str] = None,
) -> None:
    """Saves metrics and correctness status to JSON files."""
    os.makedirs(results_dir, exist_ok=True)

    correct_payload = {"correct": correct, "error": error}
    correct_file = os.path.join(results_dir, "correct.json")
    with open(correct_file, "w") as f:
        json.dump(correct_payload, f, indent=2)
    print(f"Correctness saved to {correct_file}")

    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")


def main(program_path: str, results_dir: str) -> None:
    """
    Evaluate a Macaulay2 program for resurgence properties.

    Args:
        program_path: Path to the .m2 program file (must define ideal I)
        results_dir: Directory to save evaluation results
    """
    print(f"Evaluating Macaulay2 program for resurgence: {program_path}")
    print(f"Saving results to: {results_dir}")
    print()

    metrics, correct, error_msg = run_resurgence_eval(
        program_path=program_path,
        results_dir=results_dir,
        num_runs=1,
        timeout=120,
    )

    print()
    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {error_msg}")

    print("\n=== Metrics ===")
    print(f"Combined Score: {metrics.get('combined_score', 0.0):.2f}")

    public = metrics.get("public", {})
    if public:
        print(f"\nResurgence Lower Bound: {public.get('resurgence_lower_bound', 'N/A')}")
        print(f"Containment Failures: {public.get('num_containment_failures', 0)}")
        print(f"Waldschmidt Constant: {public.get('waldschmidt_constant', 'N/A')}")
        print(f"Ideal Generators: {public.get('num_generators', 'N/A')}")
        print(f"Ideal Dimension: {public.get('ideal_dimension', 'N/A')}")

        containments = public.get("containment_summary", [])
        if containments:
            print("\nContainment Checks:")
            for c in containments:
                print(f"  {c}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Macaulay2 program for resurgence properties"
    )
    parser.add_argument(
        "--program_path", type=str, required=True,
        help="Path to the .m2 program (must define ideal I)"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Directory to save results"
    )

    args = parser.parse_args()
    main(args.program_path, args.results_dir)
