#!/usr/bin/env python
"""
Script to get a summary of a calculation project.

This script retrieves the calculation history for a project and returns it as a JSON string.
"""

import sys
import os
import argparse
import json

# Add the scripts directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_class import CalculationProject


def main(project_folder_path: str):
    """
    Get a summary of a calculation project.
    
    Args:
        project_folder_path: Path to the project folder
        
    Returns:
        JSON string with calculation history
    """
    try:
        # Redirect all output to stderr initially
        original_stdout = sys.stdout
        sys.stdout = sys.stderr
        
        # Load the calculation project
        calculation_project = CalculationProject(
            project_folder_path=project_folder_path
        )
        
        # Get calculation history
        calculation_history = calculation_project.get_calculation_history()
        
        # Restore stdout for the final result
        sys.stdout = original_stdout
        
        # Return result as JSON
        print(json.dumps({"calculation_history": calculation_history}, indent=4), flush=True)
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        # Ensure stdout is restored even if there's an error
        sys.stdout = original_stdout


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Get a summary of a calculation project")
    parser.add_argument("--project_folder_path", "-p", type=str, required=True,
                        help="Path to the project folder")
    
    args = parser.parse_args()
    
    # Call main function
    main(
        project_folder_path=args.project_folder_path
    )
