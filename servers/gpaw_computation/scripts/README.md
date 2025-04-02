# Quantum Mechanics Calculation Scripts

This directory contains standalone scripts for performing quantum mechanics calculations that can be run on a separate server.

## Overview

These scripts provide the following functionalities:

1. **Create Calculation Project**: Create a new calculation project with a given structure
2. **Run Calculation**: Run a calculation with specified parameters
3. **Check Calculation Result**: Check the status and results of a calculation
4. **Get Project Summary**: Get a summary of a calculation project's history

## File Structure

- `data_class.py`: Core data classes for quantum mechanics calculations
- `calculation_utils.py`: Utility functions for performing calculations
- `create_calculation_project.py`: Script to create a new calculation project
- `run_calculation.py`: Script to run a calculation
- `check_calculation_result.py`: Script to check calculation results
- `get_project_summary.py`: Script to get a project summary
- `requirements.txt`: Required Python packages

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Create Calculation Project

```bash
python create_calculation_project.py --project_folder_path /path/to/project --structure_str '{"@module": "pymatgen.core.structure", "@class": "Structure", "lattice": {"matrix": [[4.01, 0.0, 2.75], [1.45, 3.74, 2.75], [0.0, 0.0, 4.86]]}, "sites": [{"species": [{"element": "Bi", "occu": 1.0}], "abc": [0.23, 0.23, 0.23]}, {"species": [{"element": "Bi", "occu": 1.0}], "abc": [0.76, 0.76, 0.76]}]}'
```

### Run Calculation

```bash
python run_calculation.py --project_folder_path /path/to/project --params_str '{"calculation_type": "relax", "spinpol": true, "kpoints": [2, 2, 2], "is_bulk": true}'
```

### Check Calculation Result

```bash
python check_calculation_result.py --project_folder_path /path/to/project --calculation_id 12345678-1234-5678-1234-567812345678
```

### Get Project Summary

```bash
python get_project_summary.py --project_folder_path /path/to/project
```

## Notes

- All scripts return JSON output to stdout and error messages to stderr
- The scripts are designed to be modular and can be run independently
- Each script includes proper error handling and will exit with a non-zero status code on failure
