#!/usr/bin/env python
"""
Core data classes for quantum mechanics calculations.

This module contains the essential classes for handling quantum mechanics calculations:
- EnhancedAtoms: Extended ASE Atoms class with additional functionality
- CalculationInfo: Manages information about individual calculations
- CalculationProject: Manages a collection of calculations
"""

import os
import json
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from pymatgen.core.structure import Structure
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.dft.kpoints import BandPath, get_special_points

# Constants
FORCE_THRESHOLD = 0.001
MAX_COMPUTATION_TIME = 3600  # Default value, can be overridden by config


class EnhancedAtoms(Atoms):
    """Enhanced Atoms class with additional functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_relax_region(self, relax_intervals: Optional[List[List[float]]] = None):
        """Set the region to relax based on z-coordinate intervals."""
        if relax_intervals is None:
            return

        # Get the positions of all atoms
        positions = self.get_positions()

        # Create a constraint mask: True means constrained (frozen), False means free to move
        constraint_mask = np.ones(len(self), dtype=bool)

        # For each interval, mark atoms within that interval as free to move
        for interval in relax_intervals:
            z_min, z_max = interval
            for i, pos in enumerate(positions):
                if z_min <= pos[2] <= z_max:
                    constraint_mask[i] = False

        # Apply the constraints
        constraints = FixAtoms(mask=constraint_mask)
        self.set_constraint(constraints)

    def get_band_path(self, kpath: Optional[str] = None, npoints: int = 100):
        """Get the band path for the given atoms object."""
        if kpath is None:
            # Get suggested kpath
            kpath = self.get_suggested_kpath()

        # Create band path
        cell = self.get_cell()
        path = BandPath(cell, npoints=npoints, path=kpath)
        return path

    def get_suggested_kpath(self):
        """Get suggested k-path based on lattice type."""
        # Simplified implementation - in a real scenario, this would analyze the lattice
        return "GXMYG"

    def get_special_points(self):
        """Get special points in the Brillouin zone."""
        cell = self.get_cell()
        return get_special_points(cell)

    def check_convergence(self, force_threshold=FORCE_THRESHOLD):
        """Check if forces are below threshold."""
        if self.calc is None:
            return False

        try:
            forces = self.get_forces()
            max_force = np.max(np.abs(forces))
            return max_force < force_threshold
        except Exception:
            return False


class CalculationInfo:
    """A class to store GPAW calculation info."""

    def __init__(self,
                 project_folder_path: str,
                 calculation_id: str,
                 calculation_type: str,
                 calculation_parameters: Optional[Dict[str, Any]] = None):
        """Initialize a CalculationInfo object."""
        # Create folder for the calculation
        os.makedirs(os.path.join(project_folder_path,
                    calculation_id), exist_ok=True)

        # Set file paths
        self.info_file_path = os.path.join(
            project_folder_path, calculation_id, 'info.json')

        # If info file exists, load existing calculation
        if os.path.exists(self.info_file_path):
            instance = self.__class__.from_file(self.info_file_path)
            self.__dict__.update(instance.__dict__)
            print(
                f"Loading an existing calculation from {os.path.join(project_folder_path, calculation_id)}")
            return

        # Initialize new calculation
        self.project_folder_path = project_folder_path
        self.project_id = os.path.basename(self.project_folder_path)
        self.calculation_id = calculation_id
        self.calculation_type = calculation_type
        self.calculation_parameters = calculation_parameters
        self.calculation_status = 'initialized'

        # Define file paths
        self.define_file_paths()

        # Save to file
        self.save_to_file()

    def define_file_paths(self):
        """Define file paths for calculation outputs."""
        calculation_folder = os.path.join(
            self.project_folder_path, self.calculation_id)
        self.structure_file_path = os.path.join(
            calculation_folder, 'structure.json')
        self.status_file_path = os.path.join(calculation_folder, 'status.txt')
        self.trajectory_file_path = os.path.join(
            calculation_folder, 'trajectory.traj')
        self.log_file_path = os.path.join(calculation_folder, 'log.txt')
        self.ckpt_file_path = os.path.join(calculation_folder, 'ckpt.gpw')

    def save_to_file(self, file_path: Optional[str] = None):
        """Save calculation info to file."""
        if file_path is None:
            file_path = self.info_file_path

        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def to_dict(self):
        """Convert the CalculationInfo object to a dictionary."""
        return {k: self.convert_value(v) for k, v in self.__dict__.items() if not k.startswith('_')}

    @staticmethod
    def convert_value(value):
        """Convert numpy types to Python native types."""
        if isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: CalculationInfo.convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [CalculationInfo.convert_value(v) for v in value]
        else:
            return value

    @classmethod
    def from_file(cls, file_path: str):
        """Create a CalculationInfo object from a file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create a CalculationInfo object from a dictionary."""
        instance = cls.__new__(cls)
        for key, value in data.items():
            setattr(instance, key, value)
        return instance

    def update_structure(self, structure: Structure):
        """Update the structure by a pymatgen Structure object."""
        # Save structure to file
        with open(self.structure_file_path, 'w') as f:
            json.dump(structure.as_dict(), f)

    def get_structure(self):
        """Get the structure from file."""
        if not os.path.exists(self.structure_file_path):
            return None

        with open(self.structure_file_path, 'r') as f:
            structure_dict = json.load(f)

        return Structure.from_dict(structure_dict)

    def get_calculation_status(self):
        """Get the calculation status."""
        return self.calculation_status

    def get_summary(self):
        """Get a summary of the calculation."""
        summary = f"Type: {self.calculation_type}, Status: {self.calculation_status}"
        if self.calculation_parameters:
            summary += f", Parameters: {json.dumps(self.calculation_parameters, indent=4)}"
        return summary

    def status_file_content(self):
        """Get the content of the status file."""
        if not hasattr(self, 'status_file_path') or not os.path.exists(self.status_file_path):
            return ""
        try:
            with open(self.status_file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading status file: {str(e)}"

    def plot_output(self, **kwargs):
        """Plot output data (placeholder for actual implementation)."""
        # In a real implementation, this would generate plots based on calculation results
        return None


class CalculationProject:
    """A class to manage a collection of calculations for a project."""

    def __init__(self, project_folder_path: str, structure: Optional[Structure] = None):
        """Initialize a CalculationProject object."""
        if structure is None:
            self.load(project_folder_path)
        else:
            self.project_folder_path = project_folder_path
            # If there is a structure, create a new project
            # Check if project_folder_path is empty. If not empty, raise error
            if os.path.exists(project_folder_path) and os.listdir(project_folder_path):
                raise ValueError(
                    f"Initialize the project with a structure requires the folder {project_folder_path} to be empty")

            # Create project folder if it doesn't exist
            os.makedirs(project_folder_path, exist_ok=True)

            # Create initial structure calculation
            calculation_id = str(uuid.uuid4())
            info = CalculationInfo(
                project_folder_path=project_folder_path,
                calculation_id=calculation_id,
                calculation_type='initial_structure'
            )
            info.update_structure(structure)
            self.info_list = [info]
            self.update_calculation_id_list()

    def load(self, project_folder_path: str):
        """Load a project from a folder."""
        self.project_folder_path = project_folder_path
        calculation_id_path = os.path.join(
            project_folder_path, 'calculation_id_list.json')

        if os.path.exists(calculation_id_path):
            with open(calculation_id_path, 'r') as f:
                calculation_id_list = json.load(f)

            if not isinstance(calculation_id_list, list):
                raise ValueError(
                    f"Invalid calculation_id_list: {calculation_id_list}")

            self.info_list = []
            for calculation_id in calculation_id_list:
                info_path = os.path.join(
                    project_folder_path, calculation_id, 'info.json')
                if os.path.exists(info_path):
                    self.info_list.append(CalculationInfo.from_file(info_path))
        else:
            print(f"calculation_id_list not found at {calculation_id_path}")
            self.info_list = None

    def update_calculation_id_list(self):
        """Update the calculation_id_list.json file."""
        calculation_id_list = []
        if self.info_list is not None:
            calculation_id_list = [
                info.calculation_id for info in self.info_list]

        with open(os.path.join(self.project_folder_path, 'calculation_id_list.json'), 'w') as f:
            json.dump(calculation_id_list, f)

    def get_calculation_history(self):
        """Get the calculation history as a list of dictionaries."""
        history_list = []
        if self.info_list:
            for i, info in enumerate(self.info_list):
                history_list.append({
                    "calculation_number": i,
                    "calculation_type": info.calculation_type,
                    "calculation_parameters": json.dumps(info.calculation_parameters, indent=4) if info.calculation_parameters else None,
                    "calculation_id": info.calculation_id,
                    "calculation_status": info.get_calculation_status()
                })
        return history_list

    def add_new(self, calculation_id: str, calculation_type: str, calculation_parameters: Optional[Dict[str, Any]] = None):
        """Update the project with a new calculation info, and update the calculation_id_list."""
        if self.info_list is None:
            self.info_list = []
        new_info = CalculationInfo(
            calculation_id=calculation_id,
            calculation_type=calculation_type,
            calculation_parameters=calculation_parameters,
            project_folder_path=self.project_folder_path
        )
        self.info_list.append(new_info)
        new_info.save_to_file()
        self.update_calculation_id_list()
        return new_info

    def latest_finished_calc(self):
        """Get the latest finished calculation."""
        if self.info_list is None:
            return None
        for info in reversed(self.info_list):
            if info.calculation_status == 'finished':
                return info
        return None

    def latest_dos_calc(self):
        """Get the latest finished calculation with DOS, which means relax or ground_state."""
        if self.info_list is None:
            print("No calculation info found")
            return None
        for info in reversed(self.info_list):
            if (info.calculation_type == 'relax' or info.calculation_type == 'ground_state') and info.calculation_status == 'finished':
                return info
        print("No calculation with DOS has been finished")
        return None
