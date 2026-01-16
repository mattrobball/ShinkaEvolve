"""
M2 Runner - Python wrapper for executing Macaulay2 resurgence computations.

This module runs M2 programs and parses the output for resurgence-related
invariants (containment checks, Waldschmidt constant, etc.).
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ContainmentResult:
    """Result of checking I^(m) ⊆ I^r."""
    m: int
    r: int
    ratio: float
    contained: Optional[bool]  # None if error


@dataclass
class ResurgenceResult:
    """Result from resurgence computation."""
    success: bool
    error_message: Optional[str]

    # Ideal properties
    num_generators: Optional[int] = None
    dimension: Optional[int] = None
    codimension: Optional[int] = None

    # Containment results
    containments: List[ContainmentResult] = field(default_factory=list)

    # Resurgence bounds
    resurgence_lower_bound: float = 0.0
    waldschmidt_constant: Optional[float] = None

    # Raw output for debugging
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1


def get_resurgence_check_path() -> str:
    """Get the path to resurgence_check.m2 (in same directory as this file)."""
    return str(Path(__file__).parent / "resurgence_check.m2")


def parse_resurgence_output(stdout: str) -> ResurgenceResult:
    """
    Parse the output from resurgence_check.m2.

    Expected output format:
        IDEAL_NUMGENS <n>
        IDEAL_DIM <d>
        IDEAL_CODIM <c>
        CONTAINMENT_START
        CONTAINMENT <m> <r> <ratio> <true|false>
        ...
        CONTAINMENT_END
        RESURGENCE_LOWER_BOUND <value>
        WALDSCHMIDT <value>
        RESURGENCE_COMPLETE
    """
    result = ResurgenceResult(success=False, error_message=None, stdout=stdout)

    # Check for errors
    error_match = re.search(r'RESURGENCE_ERROR:\s*(.+)', stdout)
    if error_match:
        result.error_message = error_match.group(1)
        # Also capture the detailed M2 error if present
        m2_error_match = re.search(r'RESURGENCE_M2_ERROR:\s*(.+)', stdout)
        if m2_error_match:
            result.error_message += f"\nM2 Error: {m2_error_match.group(1)}"
        # Capture hints
        for hint_match in re.finditer(r'RESURGENCE_HINT:\s*(.+)', stdout):
            result.error_message += f"\nHint: {hint_match.group(1)}"
        # Capture defined symbols (for debugging)
        symbols_match = re.search(r'RESURGENCE_DEFINED_SYMBOLS:\s*(.+)', stdout)
        if symbols_match:
            result.error_message += f"\nDefined ideals: {symbols_match.group(1)}"
        return result

    # Parse ideal properties
    numgens_match = re.search(r'IDEAL_NUMGENS\s+(\d+)', stdout)
    if numgens_match:
        result.num_generators = int(numgens_match.group(1))

    dim_match = re.search(r'IDEAL_DIM\s+(\d+)', stdout)
    if dim_match:
        result.dimension = int(dim_match.group(1))

    codim_match = re.search(r'IDEAL_CODIM\s+(\d+)', stdout)
    if codim_match:
        result.codimension = int(codim_match.group(1))

    # Parse containment results
    containment_pattern = re.compile(
        r'CONTAINMENT\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(true|false|ERROR)'
    )
    for match in containment_pattern.finditer(stdout):
        m = int(match.group(1))
        r = int(match.group(2))
        ratio = float(match.group(3))
        contained_str = match.group(4)

        if contained_str == "ERROR":
            contained = None
        else:
            contained = contained_str == "true"

        result.containments.append(ContainmentResult(m, r, ratio, contained))

    # Parse resurgence lower bound
    resurgence_match = re.search(r'RESURGENCE_LOWER_BOUND\s+([\d.]+)', stdout)
    if resurgence_match:
        result.resurgence_lower_bound = float(resurgence_match.group(1))

    # Parse Waldschmidt constant
    waldschmidt_match = re.search(r'WALDSCHMIDT\s+([\d.]+)', stdout)
    if waldschmidt_match:
        result.waldschmidt_constant = float(waldschmidt_match.group(1))

    # Check for completion
    if "RESURGENCE_COMPLETE" in stdout:
        result.success = True
    else:
        result.error_message = "Computation did not complete"

    return result


def pre_check_m2_file(program_path: str, timeout: int = 30) -> Optional[str]:
    """
    Pre-check an M2 file for syntax/runtime errors before resurgence computation.
    Returns error message if there's a problem, None if OK.
    """
    try:
        proc = subprocess.run(
            ["M2", "--script", program_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            # Extract the error from stderr
            error_lines = proc.stderr.strip().split('\n') if proc.stderr else []
            # Find the actual error message (usually contains "error:")
            for line in error_lines:
                if 'error:' in line.lower():
                    return f"M2 syntax/runtime error: {line}"
            # If no specific error found, return the first few lines
            if error_lines:
                return f"M2 error: {' | '.join(error_lines[:3])}"
            return f"M2 failed with return code {proc.returncode}"
        return None
    except subprocess.TimeoutExpired:
        return f"M2 pre-check timed out after {timeout}s"
    except Exception as e:
        return f"Error running M2 pre-check: {e}"


def run_resurgence_check(
    program_path: str,
    timeout: int = 120
) -> ResurgenceResult:
    """
    Run resurgence computation on an M2 program that defines an ideal I.

    Args:
        program_path: Path to the .m2 file that defines ideal I
        timeout: Timeout in seconds for the computation

    Returns:
        ResurgenceResult with computation results
    """
    resurgence_check_path = get_resurgence_check_path()

    if not os.path.exists(resurgence_check_path):
        return ResurgenceResult(
            success=False,
            error_message=f"resurgence_check.m2 not found at {resurgence_check_path}"
        )

    if not os.path.exists(program_path):
        return ResurgenceResult(
            success=False,
            error_message=f"Program file not found: {program_path}"
        )

    # Convert to absolute path
    program_path = os.path.abspath(program_path)

    # Pre-check the M2 file for syntax/runtime errors
    pre_check_error = pre_check_m2_file(program_path, timeout=min(30, timeout))
    if pre_check_error:
        return ResurgenceResult(
            success=False,
            error_message=pre_check_error
        )

    try:
        proc = subprocess.run(
            ["M2", "--script", resurgence_check_path, "--args", program_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        result = parse_resurgence_output(proc.stdout)
        result.stdout = proc.stdout
        result.stderr = proc.stderr
        result.returncode = proc.returncode

        # If M2 returned non-zero but we didn't detect an error, note it
        if proc.returncode != 0 and result.success:
            result.success = False
            result.error_message = f"M2 returned exit code {proc.returncode}"

        return result

    except subprocess.TimeoutExpired:
        return ResurgenceResult(
            success=False,
            error_message=f"Computation timed out after {timeout} seconds"
        )
    except FileNotFoundError:
        return ResurgenceResult(
            success=False,
            error_message="M2 executable not found. Is Macaulay2 installed?"
        )
    except Exception as e:
        return ResurgenceResult(
            success=False,
            error_message=f"Error running M2: {str(e)}"
        )


# For backwards compatibility and simple syntax checking
@dataclass
class M2Result:
    """Simple result from running an M2 program (for syntax checking)."""
    success: bool
    score: Optional[int]
    stdout: str
    stderr: str
    returncode: int


def run_m2_subprocess(m2_path: str, timeout: int = 30) -> M2Result:
    """
    Run an M2 program via subprocess (simple execution, no resurgence).

    This is kept for backwards compatibility and simple syntax checking.
    """
    try:
        result = subprocess.run(
            ["M2", "--script", m2_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Try to extract a score if present (legacy format)
        score = None
        match = re.search(r"The score is (\d+)", result.stdout)
        if match:
            score = int(match.group(1))

        return M2Result(
            success=result.returncode == 0,
            score=score,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return M2Result(
            success=False,
            score=None,
            stdout="",
            stderr=f"Timed out after {timeout}s",
            returncode=-1,
        )
    except Exception as e:
        return M2Result(
            success=False,
            score=None,
            stdout="",
            stderr=str(e),
            returncode=-1,
        )
