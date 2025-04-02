#!/usr/bin/env python
"""
Utility functions for quantum mechanics calculations.

This module contains functions for performing various types of quantum mechanics calculations,
checking calculation results, and managing calculation data.
"""

import os
import sys
import time
import uuid
import shutil
from typing import Dict, Any, Optional

from data_class import CalculationProject, CalculationInfo, MAX_COMPUTATION_TIME
import plotly.io as pio


def remove_calculation_subfolder(calculation_id: str, project_folder_path: str):
    """Remove the calculation subfolder in project_folder_path."""
    try:
        shutil.rmtree(os.path.join(project_folder_path, calculation_id))
    except FileNotFoundError:
        print(
            f"Subfolder {calculation_id} does not exist in {project_folder_path}.")


def perform_calculation_by_id(input_calculation_id: str, output_calculation_id: str, project_folder_path: str):
    """
    Perform the calculation based on the input information specified by input_calculation_id
    and output the result to the folder specified by output_information_id.
    """
    input_info_file_path = os.path.join(
        project_folder_path, input_calculation_id, 'info.json')
    output_info_file_path = os.path.join(
        project_folder_path, output_calculation_id, 'info.json')

    input_info = CalculationInfo.from_file(input_info_file_path)
    output_info = CalculationInfo.from_file(output_info_file_path)

    if not input_info:
        raise ValueError(
            f"No input information found in path {input_info_file_path}.")
    if not output_info:
        raise ValueError(
            f"No output information found in path {output_info_file_path}.")

    # In a real implementation, this would call a function to perform the calculation
    # For now, we'll just update the status to simulate a completed calculation
    output_info.calculation_status = "finished"
    output_info.save_to_file()

    return output_info


def perform_calculation(calculation_type: str,
                        calculation_parameters: Dict[str, Any],
                        project_folder_path: str,
                        output_calculation_id: Optional[str] = None) -> Dict[str, str]:
    """
    Start a calculation subprocess based on the calculation type and parameters,
    and store the calculation info into a new calculation record.

    Args:
        calculation_type: Type of calculation to perform (e.g., 'relax', 'ground_state', 'band')
        calculation_parameters: Dictionary containing the calculation parameters
        project_folder_path: Path to the project folder
        output_calculation_id: Optional ID for the output calculation. If None, a new UUID is generated.

    Returns:
        A dictionary containing the project_folder_path and calculation_id.
    """
    calculation_record = CalculationProject(
        project_folder_path=project_folder_path)

    # Determine input calculation based on calculation type
    if calculation_type == 'relax' or calculation_type == 'ground_state':
        input_info = calculation_record.latest_finished_calc()
    else:
        input_info = calculation_record.latest_dos_calc()

    # If no suitable input calculation is found, use the initial structure
    if not input_info:
        input_info = calculation_record.info_list[0]

    # Generate a new calculation ID if not provided
    if not output_calculation_id:
        output_calculation_id = str(uuid.uuid4())

    # Add the new calculation to the project
    calculation_record.add_new(
        calculation_id=output_calculation_id,
        calculation_type=calculation_type,
        calculation_parameters=calculation_parameters
    )

    # Perform the calculation
    perform_calculation_by_id(
        input_calculation_id=input_info.calculation_id,
        output_calculation_id=output_calculation_id,
        project_folder_path=project_folder_path
    )

    return {
        "project_folder_path": project_folder_path,
        "calculation_id": output_calculation_id
    }


def check_calculation_result(calculation_id: str, project_folder_path: str) -> Dict[str, Any]:
    """
    Check the calculation result with the given calculation id and project folder path.

    Args:
        calculation_id: The ID of the calculation
        project_folder_path: The path to the project folder

    Returns:
        A dictionary containing the calculation status and log.
    """
    calculation_info_file_path = os.path.join(
        project_folder_path, calculation_id, 'info.json')
    calculation_info = None

    start_time = time.time()
    while True:
        try:
            calculation_info = CalculationInfo.from_file(
                calculation_info_file_path)
            calculation_status = calculation_info.get_calculation_status()
            status_file_content = calculation_info.status_file_content()
        except Exception as e:
            status_file_content = str(e)
            calculation_status = "status inaccessible"
            print(f"Warning: {str(e)}", file=sys.stderr, flush=True)

        if calculation_status == 'finished':
            result = {
                "calculation_status": calculation_status,
                "log": status_file_content
            }

            # Try to add figure data if available
            try:
                fig = calculation_info.plot_output()
                if fig:
                    fig_str = pio.to_json(fig)
                    result["figure_str"] = fig_str
            except Exception as e:
                print(
                    f"Warning: Error generating plot: {str(e)}", file=sys.stderr, flush=True)

            break

        # Check if calculation has timed out
        if time.time() - start_time > MAX_COMPUTATION_TIME:
            log = status_file_content + \
                f"\n\nCalculation exceeded maximum computation time {MAX_COMPUTATION_TIME} seconds"
            result = {
                "calculation_status": "timeout",
                "log": log
            }
            break

        # Wait before checking again
        time.sleep(5)

    return result


def stream_calculation_log(calculation_id: str, project_folder_path: str):
    """
    Stream the calculation log of a calculation subprocess, if the calculation is in progress.

    Args:
        calculation_id: The ID of the calculation
        project_folder_path: The path to the project folder

    Yields:
        The content of the status file.
    """
    calculation_info_file_path = os.path.join(
        project_folder_path, calculation_id, 'info.json')

    while True:
        try:
            calculation_info = CalculationInfo.from_file(
                calculation_info_file_path)
            calculation_status = calculation_info.get_calculation_status()
            status_file_content = calculation_info.status_file_content()
            yield status_file_content
        except Exception as e:
            status_file_content = str(e)
            calculation_status = "status inaccessible"

        if calculation_status == 'finished' or calculation_status == 'interrupted':
            break

        time.sleep(1)
