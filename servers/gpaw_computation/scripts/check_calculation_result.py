#!/usr/bin/env python
"""
Script to check the result of a calculation.

This script checks the status and results of a calculation with the specified ID
and returns the calculation status, log, and any available output data.
"""

import sys
import os
import argparse
import json

# Add the scripts directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from calculation_utils import check_calculation_result


def main(project_folder_path: str, calculation_id: str):
    """
    Check the result of a calculation.
    
    Args:
        project_folder_path: Path to the project folder
        calculation_id: ID of the calculation to check
        
    Returns:
        JSON string with calculation status, log, and any available output data
    """
    try:
        # Redirect all output to stderr initially
        original_stdout = sys.stdout
        sys.stdout = sys.stderr
        
        # Check the calculation result
        result = check_calculation_result(
            project_folder_path=project_folder_path,
            calculation_id=calculation_id
        )
        
        # Restore stdout for the final result
        sys.stdout = original_stdout
        
        # Return result as JSON
        result_str = json.dumps(result, indent=4)
        print(result_str, flush=True)
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        # Ensure stdout is restored even if there's an error
        sys.stdout = original_stdout


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check the result of a calculation")
    parser.add_argument('--project_folder_path', '-p', type=str, required=True,
                        help='Path to the project folder')
    parser.add_argument('--calculation_id', '-c', type=str, required=True,
                        help='ID of the calculation to check')
    
    args = parser.parse_args()
    
    # Call main function
    main(
        project_folder_path=args.project_folder_path,
        calculation_id=args.calculation_id
    )
