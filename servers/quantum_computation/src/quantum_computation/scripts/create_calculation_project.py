#!/usr/bin/env python
"""
Script to create a new calculation project with a given structure.

This script creates a new CalculationProject instance with the provided structure
and returns the calculation history as a JSON string.
"""

from data_class import CalculationProject
import sys
import os
import argparse
import json
from pymatgen.core.structure import Structure

# Add the scripts directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main(project_folder_path: str, structure_str: str):
    """
    Create a new project in the given project_folder_path.

    Args:
        project_folder_path: Path to the project folder
        structure_str: JSON string representing a pymatgen Structure

    Returns:
        JSON string with calculation history
    """
    try:
        # Redirect all output to stderr initially
        original_stdout = sys.stdout
        sys.stdout = sys.stderr

        # Parse structure from JSON string
        structure = Structure.from_dict(json.loads(structure_str))

        # Create new project
        new_project = CalculationProject(
            project_folder_path=project_folder_path,
            structure=structure
        )

        # Get calculation history
        calculation_history = new_project.get_calculation_history()

        # Restore stdout for the final result
        sys.stdout = original_stdout

        # Return result as JSON
        result = json.dumps({"calculation_history": calculation_history}, indent=4)
        print(result, flush=True)
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        # Ensure stdout is restored even if there's an error
        sys.stdout = original_stdout


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create a new calculation project")
    parser.add_argument("--project_folder_path", "-p", type=str, required=True,
                        help="Path to the project folder")
    parser.add_argument("--structure_str", "-s", type=str, required=True,
                        help="JSON string representing a pymatgen Structure")

    args = parser.parse_args()

    # Call main function
    main(
        project_folder_path=args.project_folder_path,
        structure_str=args.structure_str
    )
