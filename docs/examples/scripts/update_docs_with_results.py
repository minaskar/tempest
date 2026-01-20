#!/usr/bin/env python3
"""
Helper script to update the model_comparison.md documentation file
with actual computed values from running the example.

Usage:
    python update_docs_with_results.py
"""

import re
import sys
from pathlib import Path

# True parameter values (from the example)
TRUE_VALUES = {
    "A_true": 0.5,
    "B_true": 2.0,
    "omega_true": 6.283,
    "phi_true": 0.785,
    "sigma_true": 0.25,
}


def extract_results_from_script_output():
    """
    Extract results from model_comparison_standalone.py
    Assumes the script has already been run and produces stdout with specific patterns
    """
    # For now, we'll use hardcoded values from the actual run
    # These match the output we saw when running the script
    return {
        "logz_linear": -113.72,
        "logz_oscillatory": -26.36,
        "bayes_factor": 8.62e37,
        "log10_bayes_factor": 37.94,
        "a_fit": -0.396,
        "a_err": 0.331,
        "b_fit": 0.518,
        "b_err": 0.574,
        "sigma_lin_fit": 2.015,
        "sigma_lin_err": 0.208,
        "A_fit": 0.443,
        "A_err": 0.058,
        "B_fit": 2.106,
        "B_err": 0.099,
        "omega_fit": 6.308,
        "omega_err": 0.021,
        "phi_fit": 0.728,
        "phi_err": 0.042,
        "sigma_osc_fit": 0.247,
        "sigma_osc_err": 0.028,
    }


def update_model_comparison_md():
    """Update the model_comparison.md file with actual computed values."""

    results = extract_results_from_script_output()

    # Path to the documentation file
    doc_path = Path(__file__).parent.parent / "model_comparison.md"

    if not doc_path.exists():
        print(f"Error: {doc_path} not found")
        sys.exit(1)

    # Read the file content
    content = doc_path.read_text()

    # Define replacement patterns
    replacements = [
        # Evidence outputs
        (
            r'print\(f"Linear model: logZ = \{logz_lin:\.2f\}\"\)\s+# Expected:.*',
            f'print(f"Linear model: logZ = {{logz_lin:.2f}}")\n# Expected: {results["logz_linear"]:.2f}',
        ),
        (
            r'print\(f"Oscillatory model: logZ = \{logz_osc:\.2f\}\"\)\s+# Expected:.*',
            f'print(f"Oscillatory model: logZ = {{logz_osc:.2f}}")\n# Expected: {results["logz_oscillatory"]:.2f}',
        ),
        # Bayes factor outputs
        (
            r'print\(f"\\nBayes Factor \(Oscillatory/Linear\) = \{bayes_factor:\.2e\}\"\)\s+print\(f"log₁₀\(BF\) = \{log10_bayes_factor:\.2f\}\"\)\s+# Expected:.*',
            f'print(f"\\nBayes Factor (Oscillatory/Linear) = {{bayes_factor:.2e}}")\nprint(f"log₁₀(BF) = {{log10_bayes_factor:.2f}}")\n# Expected: {results["bayes_factor"]:.2e} and {results["log10_bayes_factor"]:.2f}',
        ),
        # Linear model parameters
        (
            r'print\(f"\s+a = \{a_fit:\.3f\} ± \{a_err:\.3f\}\"\)\s+# Expected.*',
            f'print(f"  a = {{a_fit:.3f}} ± {{a_err:.3f}}")\n# Expected: {results["a_fit"]:.3f} and {results["a_err"]:.3f}',
        ),
        (
            r'print\(f"\s+b = \{b_fit:\.3f\} ± \{b_err:\.3f\}\"\)\s+# Expected.*',
            f'print(f"  b = {{b_fit:.3f}} ± {{b_err:.3f}}")\n# Expected: {results["b_fit"]:.3f} and {results["b_err"]:.3f}',
        ),
        (
            r'print\(f"\s+σ = \{sigma_fit_lin:\.3f\} ± \{sigma_err_lin:\.3f\}\"\)\s+# Expected.*',
            f'print(f"  σ = {{sigma_fit_lin:.3f}} ± {{sigma_err_lin:.3f}}")\n# Expected: {results["sigma_lin_fit"]:.3f} and {results["sigma_lin_err"]:.3f}',
        ),
        # Oscillatory model parameters
        (
            r'print\(f"\s+A = \{A_fit:\.3f\} ± \{A_err:\.3f\}\s+\(true:.*',
            f'print(f"  A = {{A_fit:.3f}} ± {{A_err:.3f}}  (true: {A_true})")\n# Expected: {results["A_fit"]:.3f} and {results["A_err"]:.3f}',
        ),
        (
            r'print\(f"\s+B = \{B_fit:\.3f\} ± \{B_err:\.3f\}\s+\(true:.*',
            f'print(f"  B = {{B_fit:.3f}} ± {{B_err:.3f}}  (true: {B_true})")\n# Expected: {results["B_fit"]:.3f} and {results["B_err"]:.3f}',
        ),
        (
            r'print\(f"\s+ω = \{omega_fit:\.3f\} ± \{omega_err:\.3f\}\s+\(true:.*',
            f'print(f"  ω = {{omega_fit:.3f}} ± {{omega_err:.3f}}  (true: {omega_true:.3f})")\n# Expected: {results["omega_fit"]:.3f} and {results["omega_err"]:.3f}',
        ),
        (
            r'print\(f"\s+φ = \{phi_fit:\.3f\} ± \{phi_err:\.3f\}\s+\(true:.*',
            f'print(f"  φ = {{phi_fit:.3f}} ± {{phi_err:.3f}}  (true: {phi_true:.3f})")\n# Expected: {results["phi_fit"]:.3f} and {results["phi_err"]:.3f}',
        ),
        (
            r'print\(f"\s+σ = \{sigma_fit_osc:\.3f\} ± \{sigma_err_osc:\.3f\}\s+\(true:.*',
            f'print(f"  σ = {{sigma_fit_osc:.3f}} ± {{sigma_err_osc:.3f}}  (true: {sigma_true})")\n# Expected: {results["sigma_osc_fit"]:.3f} and {results["sigma_osc_err"]:.3f}',
        ),
    ]

    # Apply replacements
    for pattern, replacement in replacements:
        # Find and replace
        content = re.sub(pattern, replacement, content)

    # Write back to file
    doc_path.write_text(content)
    print(f"Updated {doc_path} with computed values")


if __name__ == "__main__":
    update_model_comparison_md()
