#!/usr/bin/env python
"""
Script to run a calculation with the specified parameters.

This script starts a calculation based on the provided parameters and
returns the calculation ID and project folder path.
"""

from calculation_utils import perform_calculation
import sys
import os
import argparse
import json

# Add the scripts directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main(params_str: str, project_folder_path: str, output_calculation_id: str = None):
    """
    Run a calculation with the specified parameters.

    Args:
        params_str: JSON string containing calculation parameters
        project_folder_path: Path to the project folder
        output_calculation_id: Optional ID for the output calculation

    Returns:
        JSON string with project_folder_path and calculation_id
    """
    try:
        # Redirect all output to stderr initially
        original_stdout = sys.stdout
        sys.stdout = sys.stderr

        # Parse parameters from JSON string
        params = json.loads(params_str)
        calculation_type = params.pop("calculation_type")
        calculation_parameters = params

        # Perform the calculation
        result_dict = perform_calculation(
            calculation_type=calculation_type,
            calculation_parameters=calculation_parameters,
            project_folder_path=project_folder_path,
            output_calculation_id=output_calculation_id
        )

        # Restore stdout for the final result
        sys.stdout = original_stdout

        # Return result as JSON
        print(json.dumps(result_dict, indent=4), flush=True)
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        # Ensure stdout is restored even if there's an error
        sys.stdout = original_stdout


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run a calculation with specified parameters")
    parser.add_argument("--params_str", "-pm", type=str, required=True,
                        help="JSON string containing calculation parameters")
    parser.add_argument("--project_folder_path", "-p", type=str, required=True,
                        help="Path to the project folder")
    parser.add_argument("--output_calculation_id", "-c", type=str,
                        help="Optional ID for the output calculation")

    args = parser.parse_args()

    # Call main function
    main(
        params_str=args.params_str,
        project_folder_path=args.project_folder_path,
        output_calculation_id=args.output_calculation_id
    )
