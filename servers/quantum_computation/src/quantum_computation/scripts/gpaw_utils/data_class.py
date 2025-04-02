# define the data class based on Atoms class, specialized for gpaw
import random
import time
import multiprocessing
import uuid
import copy
from ase.constraints import FixAtoms
from .document import extract_blocks
from ase.atoms import Atoms
from typing import TypedDict, Literal, Optional
from gpaw import GPAW
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
from . import plot_band_dos_utils as pbd_utils
import sys
import os
import shutil
import json
from dataclasses import dataclass


# Add the project root to the path
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(scripts_dir)
sys.path.append(project_root)


FORCE_THRESHOLD = 0.001
MAX_ATTEMPTS = 10


class RelaxCalculationParams(TypedDict):
    """Parameters for a gpaw relax calculation.

    Attributes:
        calculation_type: must be 'relax'
        spinpol: The boolean parameter determines whether the calculation is spin polarized.
        initial_magnetic_moments: An optional list that sets the initial magnetic moments for the atoms. This parameter is only needed if spinpol is True.
        kpoints: A list of 3 positive integers, representing the number of k-points in each direction of the reciprocal space. For example [1,1,1] or [4,4,2]
        is_bulk: The boolean parameter determines whether the calculation is for a bulk structure (periodic in all three directions) or a slab structure (periodic in two directions and open boundary condition in the third direction
        relax_intervals: A list of lists, such as [[z_1,z_2]] or [[z_1,z_2],[z_3,z_4]] which specifies intervals of z coordinates (in unit of angstrom) for relaxing atoms. If not selected, all atoms will be relaxed.
    """
    calculation_type: Literal["relax"]
    spinpol: bool
    initial_magnetic_moments: Optional[list[float]]
    kpoints: list[int]
    is_bulk: bool
    relax_intervals: Optional[list[list[float]]]


class GroundStateCalculationParams(TypedDict):
    """Parameters for a gpaw ground state calculation.

    Attributes:
        calculation_type: must be 'ground_state'
        spinpol: The boolean parameter determines whether the calculation is spin polarized.
        initial_magnetic_moments: A list that sets the initial magnetic moments for the atoms. This parameter is only needed if spinpol is True.
        kpoints: A list of 3 positive integers, representing the number of k-points in each direction of the reciprocal space. For example [1,1,1] or [4,4,2]
    """
    calculation_type: Literal["ground_state"]
    spinpol: bool
    initial_magnetic_moments: Optional[list[float]]
    kpoints: list[int]


class BandCalculationParams(TypedDict):
    """Parameters for a gpaw band structure calculation.

    Attributes:
        calculation_type: must be 'band'
        kpath: The string containing high-symmetry points that defines the k-path. For example 'GXMYG' or 'GXMYG,XG' if there are multiple segments. Use 'G' to represent the gamma point.
        npoints: The number of k-points along each segment of the k-path.
        include_soc: The parameter determines whether the calculation includes spin-orbit coupling.
    """
    calculation_type: Literal["band"]
    kpath: str
    npoints: int
    include_soc: Optional[bool]


class EnhancedAtoms(Atoms):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_relax_region(self, relax_intervals: list[list[float]] | None):
        if not relax_intervals:
            return
        positions = self.positions
        z_coords = positions[:, 2]
        relax_indices = []
        for atom_index, z in enumerate(z_coords):
            # print(f"Atom {atom_index}: z={z}")
            for interval in relax_intervals:
                if z >= interval[0] and z <= interval[1]:
                    relax_indices.append(atom_index)
                    break
        fixed_indices = [i for i in range(len(self)) if i not in relax_indices]

        constraint = FixAtoms(indices=fixed_indices)
        self.set_constraint(constraint)

    def get_band_path(self, kpath: str | None = None, npoints: int = 100):
        """
        Get the band path for the given atoms object.

        Args:
        - kpath (str): The k-path to use. Default is None.
        - npoints (int): The number of k-points along the path. Default is 100.

        Returns:
        - path (BandPath): The band path object.
        """
        rank = self.cell.rank
        if rank == 3:
            pbc = [1, 1, 1]
        elif rank == 2:
            pbc = [1, 1, 0]
        elif rank == 1:
            pbc = [1, 0, 0]
        else:
            raise ValueError(f"Invalid rank: {rank}")

        if not kpath:  # if kpath is not specified, get the suggested path
            suggested_path, special_points = self.get_suggested_kpath()
            kpath = suggested_path
        print(f"Kpath: {kpath}")
        path = self.cell.bandpath(kpath, npoints=npoints, pbc=pbc)
        return path

    def get_suggested_kpath(self):
        # Get the cell and lattice
        dimension = self.cell.rank
        lattice = self.cell.get_bravais_lattice()
        special_points = self.get_special_points()

        if dimension == 3:
            suggested_path = lattice.special_path
        else:
            suggested_path = ''.join(list(special_points.keys()))

        for label, point in special_points.items():
            print(f"{label} : [{point[0]:8.5f}, {point[1]:8.5f}, {point[2]:8.5f}]")

        return suggested_path, special_points

    def get_special_points(self):
        rank = self.cell.rank
        if rank == 3:
            pbc = [1, 1, 1]
        elif rank == 2:
            pbc = [1, 1, 0]
        elif rank == 1:
            pbc = [1, 0, 0]
        else:
            raise ValueError(f"Invalid rank: {rank}")
        special_points = self.cell.bandpath('G', npoints=10, pbc=pbc).special_points
        return special_points

    def check_convergence(self, force_threshold=FORCE_THRESHOLD):
        forces = self.get_forces()
        if forces is None:
            print("Warning: Forces are None. Cannot check force convergence.")
            return False

        max_force = max(np.linalg.norm(forces, axis=1))

        converged = max_force < force_threshold
        return converged


@dataclass
class CalculationInfo:
    """A class to store GPAW calculation info"""
    project_folder_path: str
    calculation_id: str
    calculation_type: str
    calculation_parameters: dict | None = None
    calculation_status: str | None = None
    ckpt_file_path: str | None = None
    structure_file_path: str | None = None
    status_file_path: str | None = None
    trajectory_file_path: str | None = None
    log_file_path: str | None = None
    info_file_path: str | None = None
    project_id: str | None = None
    _lock = multiprocessing.Lock()

    @classmethod
    def from_dict(cls, data: dict):
        """Override dataclass_json's from_dict to directly set attributes."""
        with cls._lock:
            instance = cls.__new__(cls)
            instance.__dict__.update(data)
        return instance

    def __init__(self,
                 project_folder_path: str,
                 calculation_id: str,
                 calculation_type: str,
                 calculation_parameters: dict | None = None):
        # create folder for the project_id/calculation id if not existing already
        os.makedirs(os.path.join(project_folder_path, calculation_id), exist_ok=True)
        # get the file paths
        self.info_file_path = os.path.join(project_folder_path, calculation_id, 'info.json')
        if os.path.exists(self.info_file_path):
            instance = self.__class__.from_file(self.info_file_path)
            self.__dict__.update(instance.__dict__)
            print(f"Loading an existing calculation from {os.path.join(project_folder_path, calculation_id)}")
            return
        self.project_folder_path = project_folder_path
        self.project_id = os.path.basename(self.project_folder_path)
        self.calculation_id = calculation_id
        self.calculation_type = calculation_type
        self.calculation_parameters = calculation_parameters
        self.calculation_status = 'initialized'
        self.define_file_paths()
        self.save_to_file()

    def define_file_paths(self):
        project_folder_path = self.project_folder_path
        calculation_id = self.calculation_id
        self.status_file_path = os.path.join(project_folder_path, calculation_id, 'status.txt')
        self.trajectory_file_path = os.path.join(project_folder_path, calculation_id, 'trajectory.traj')
        self.log_file_path = os.path.join(project_folder_path, calculation_id, 'log.log')
        self.ckpt_file_path = os.path.join(project_folder_path, calculation_id, 'checkpoint.ckpt')
        self.interrupt_status_path = os.path.join(project_folder_path, calculation_id, 'interrupt.json')
        self.structure_file_path = os.path.join(project_folder_path, calculation_id, 'structure.cif')
        self.info_file_path = os.path.join(project_folder_path, calculation_id, 'info.json')

    @property
    def status_file_content(self):
        return self.get_status_file_content()

    def save_to_file(self, file_path: str | None = None) -> None:
        """Dump the JSON representation of the instance to a file."""
        if not file_path:
            file_path = self.info_file_path
        if not file_path:
            raise ValueError("No info_file_path provided.")
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # update project_folder_path and project_id it is different from the one in the instance
        self.project_folder_path = os.path.dirname(os.path.dirname(file_path))
        self.project_id = os.path.basename(self.project_folder_path)
        self.calculation_id = os.path.basename(os.path.dirname(file_path))
        self.define_file_paths()
        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

    def update_status(self, calculation_status: str):
        self.calculation_status = calculation_status
        self.save_to_file()

    def refresh(self):
        """load from file to keep updated with the disk version"""
        instance = self.__class__.from_file(self.info_file_path)
        self.__dict__.update(instance.__dict__)

    def get_interrupt_status(self):
        if not os.path.exists(self.interrupt_status_path):
            return False
        with open(self.interrupt_status_path, 'r') as f:
            interrupt_dict = json.load(f)
        return interrupt_dict['interrupt']

    def update_interrupt_status(self, interrupt: bool):
        interrupt_dict = {'interrupt': interrupt}
        with open(self.interrupt_status_path, 'w') as f:
            json.dump(interrupt_dict, f)

    @classmethod
    def from_file(cls, file_path: str):
        jj = 0
        while True:
            try:
                with open(file_path, 'r') as file:
                    data_dict = json.load(file)
                # print(f"Loaded data_dict:\n\n {data_dict}")
                return cls.from_dict(data_dict)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                sleep_time = random.uniform(0.01, 0.1)
                time.sleep(sleep_time)
            if jj > MAX_ATTEMPTS:
                raise ValueError(f"Failed to load {file_path} after {MAX_ATTEMPTS} attempts.")
            jj += 1
        return None

    def copy(self):
        new_info = copy.deepcopy(self)
        return new_info

    def to_dict(self):
        """Convert the CalculationInfo object to a dictionary, handling numpy.bool_ types."""
        def convert_value(value):
            if isinstance(value, np.bool_):
                return bool(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(convert_value(v) for v in value)
            return value

        return {key: convert_value(value) for key, value in self.__dict__.items()}

    def get_structure(self) -> Structure | None:
        if self.structure_file_path and os.path.exists(self.structure_file_path):
            return Structure.from_file(self.structure_file_path)
        else:
            return None

    def get_atom_number(self):
        structure = self.get_structure()
        if structure is None:
            return 0
        else:
            return structure.num_sites

    def update_structure(self, structure: Structure):
        """Update the structure by a pymatgen Structure object"""
        # save the structure to the structure_file_path
        structure.to_file(self.structure_file_path)
        if self.calculation_type == "initial_structure":
            # for initial structure, update the status to "finished". For other types
            # of calculation, no update to status.
            self.update_status("finished")

    def update_structure_from_atoms(self, atoms: Atoms):
        """Update the structure by an ASE Atoms object"""
        structure = AseAtomsAdaptor.get_structure(atoms)
        self.update_structure(structure)

    def get_atoms(self) -> Atoms | None:
        if self.ckpt_file_path and os.path.exists(self.ckpt_file_path):
            calc = GPAW(self.ckpt_file_path, txt=None)
            atoms = calc.get_atoms()
            atoms.calc = calc
        else:
            structure = self.get_structure()
            if structure:
                atoms = AseAtomsAdaptor.get_atoms(structure)
            else:
                atoms = None
        if atoms is not None:
            atoms = EnhancedAtoms(atoms)
        return atoms

    def get_calc(self):
        print(f"Getting calculator for {self.calculation_id}")
        try:
            if self.ckpt_file_path and os.path.exists(self.ckpt_file_path):
                print(f"Loading calculator from checkpoint: {self.ckpt_file_path}")
                calc = GPAW(self.ckpt_file_path, txt=None)
                print("Successfully loaded calculator")
                return calc
            else:
                print(f"No checkpoint file found at: {self.ckpt_file_path}")
                return None
        except Exception as e:
            print(f"Error loading calculator: {str(e)}")
            return None

    def get_calculation_status(self):
        instance = self.__class__.from_file(self.info_file_path)
        self.calculation_status = instance.calculation_status
        return self.calculation_status

    def get_status_file_content(self) -> str | None:
        if self.status_file_path and os.path.exists(self.status_file_path):
            with open(self.status_file_path, 'r') as file:
                return file.read()
        else:
            return None

    def update_status_file_content(self, content: str | None = None):
        if not content:
            return
        status_file_content = self.get_status_file_content()
        if status_file_content:
            status_file_content += "\n\n" + content
        else:
            status_file_content = content
        with open(self.status_file_path, 'w') as file:
            file.write(status_file_content)

    def plot_band(self,
                  locfun=None,
                  emin=-6,
                  emax=3,
                  scale_factor=100,
                  xlabel='K-path',
                  ylabel='Energy (eV)',
                  colors=['blue', 'red'],
                  labels=['Bands', 'Projected Bands']):
        calc = self.get_calc()
        if not calc:
            return None

        spinpol = (calc.get_number_of_spins() == 2)
        include_soc = self.calculation_parameters.get('include_soc', True)
        if not spinpol:
            if locfun:
                return pbd_utils.plot_band_projection(calc=calc,
                                                      locfun=locfun,
                                                      scale_factor=scale_factor,
                                                      emin=emin,
                                                      emax=emax,
                                                      ylabel=ylabel,
                                                      xlabel=xlabel,
                                                      colors=colors,
                                                      labels=labels)
            else:
                return pbd_utils.plot_band(calc=calc,
                                           emin=emin,
                                           emax=emax,
                                           ylabel=ylabel,
                                           colors=colors,
                                           labels=labels,
                                           soc=include_soc,
                                           only_show_soc=False)
        else:
            if not locfun:
                return pbd_utils.plot_band_spinpol(calc=calc,
                                                   emin=emin,
                                                   emax=emax,
                                                   soc=include_soc,
                                                   only_show_soc=True,
                                                   show_spinprojection=True,
                                                   ylabel=ylabel,
                                                   xlabel=xlabel,
                                                   marker_size=5,
                                                   marker_style='circle')
            else:
                return pbd_utils.plot_band_spinpol_projection(calc=calc,
                                                              locfun=locfun,
                                                              emin=emin,
                                                              emax=emax,
                                                              colors=colors,
                                                              labels=['Spin up', 'Spin down'],
                                                              scale_factor=scale_factor,
                                                              xlabel=xlabel,
                                                              ylabel=ylabel,
                                                              only_show_projection=False)

    def plot_dos(
            self,
            locfun=None,
            emin=-6,
            emax=3,
            xlabel='Energy (eV)',
            ylabel='DOS (states/eV)',
            npts=1000,
            color='black'):
        calc = self.get_calc()
        if calc is None:
            return None

        spinpol = (calc.get_number_of_spins() == 2)
        if not spinpol:
            if locfun:
                return pbd_utils.plot_dos_projection(calc=calc,
                                                     locfun=locfun,
                                                     emin=emin,
                                                     emax=emax,
                                                     xlabel=xlabel,
                                                     ylabel=ylabel,
                                                     npts=npts,
                                                     color=color)
            else:
                return pbd_utils.plot_dos(calc=calc,
                                          emin=emin,
                                          emax=emax,
                                          npts=npts,
                                          xlabel=xlabel,
                                          ylabel=ylabel,
                                          width=0.1,
                                          color='black',
                                          label='Total DOS')
        else:
            if locfun:
                return pbd_utils.plot_dos_spinpol_projection(calc=calc,
                                                             locfun=locfun,
                                                             emin=emin,
                                                             emax=emax,
                                                             npts=npts,
                                                             xlabel=xlabel,
                                                             ylabel=ylabel,
                                                             colors=['red', 'blue'],
                                                             labels=['Spin up', 'Spin down'])
            return pbd_utils.plot_dos_spinpol(calc=calc,
                                              emin=emin,
                                              emax=emax,
                                              npts=npts,
                                              xlabel=xlabel,
                                              ylabel=ylabel,
                                              width=0.1,
                                              colors=['red', 'blue'],
                                              labels=['Spin up', 'Spin down'])

    def plot_output(
        self,
        atom_labels: list | None = None,
        orbitals: list | None = None,
        **kwargs
    ):
        print(f"Plotting output for {self.calculation_type} calculation")
        calc = self.get_calc()
        if calc is None:
            print(f"No calculator found for {self.calculation_id}")
            print(f"Checkpoint file: {self.ckpt_file_path}")
            print(f"Checkpoint file exists: {os.path.exists(self.ckpt_file_path)}")
            return None
        print(f"Got calculator for {self.calculation_id}")
        locfun = self.get_locfun(atom_labels=atom_labels, orbitals=orbitals)
        if self.calculation_type == 'band':
            print("Generating band plot")
            return self.plot_band(locfun=locfun, **kwargs)
        elif self.calculation_type == 'relax' or self.calculation_type == 'ground_state':
            print("Generating DOS plot")
            return self.plot_dos(locfun=locfun, **kwargs)
        print(f"No plot for calculation type: {self.calculation_type}")
        return None

    def get_chemical_symbols(self):
        atoms = self.get_atoms()
        if not atoms:
            return None
        return atoms.get_chemical_symbols()

    def get_chemical_symbols_no_duplicate(self):
        """Get a list of chemical symbols without duplicate"""
        chemical_symbols = list(set(self.get_chemical_symbols()))
        # sort the chemical symbols alphabetically
        chemical_symbols.sort()
        return chemical_symbols

    def get_atom_index_list(self):
        """Get a list of atom indices including the numbers of atoms, and atom symbols, such as [0,1,2,3,4,'Bi','Se']"""
        atoms = self.get_atoms()
        atom_index_list = list(range(len(atoms)))
        atom_index_list.extend(self.get_chemical_symbols_no_duplicate())
        return atom_index_list

    def get_orbital_list(self):
        """Get a list of orbitals such as ['s','p','d','f']"""
        return ['s', 'p', 'd', 'f']

    def get_locfun(
        self,
        atom_labels: list | None = None,
        orbitals: list | None = None,
        sigma: float = 0.1
    ):
        """
        Get the locfun for the given atom symbol (such as 'Bi') or atom index such as 0,1,2, and orbitals (such as 's,p,d,f')
        For example, if atom_labels=[0,1] and orbitals=['s','p'], this will select the s and p orbitals of the first two atoms
        if atom_labels=['Bi','Se'], orbitals=['s','p'], this will select the s and p orbitals of all atoms with Bi and Se
        """
        if not atom_labels and not orbitals:
            return None
        atoms = self.get_atoms()
        if not atom_labels or len(atom_labels) == 0:
            atom_labels = list(range(len(atoms)))
        if not orbitals or len(orbitals) == 0:
            orbitals = ['s', 'p', 'd', 'f']
        atom_symbols_list = atoms.get_chemical_symbols()
        # translate orbital label 's','p','d','f' to 0,1,2,3
        orbital_dict = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        l_list = [orbital_dict[orbital] for orbital in orbitals]
        # convert strings like '2' to integers like 2
        updated_atom_labels = []
        for atom_label in atom_labels:
            try:
                updated_atom_labels.append(int(atom_label))
            except BaseException:
                updated_atom_labels.append(atom_label)
        atom_index_list = []
        for atom_label in updated_atom_labels:
            if isinstance(atom_label, int) and atom_label < len(atoms):
                atom_index_list.append(atom_label)
            elif isinstance(atom_label, str) and atom_label in atom_symbols_list:
                atom_index_list.extend([j for j, atom_symbol in enumerate(
                    atom_symbols_list) if atom_symbol == atom_label])
            else:
                raise ValueError(f"Invalid atom label: {atom_label}")
        # remove duplicate atom indices
        atom_index_list = list(set(atom_index_list))

        locfun = []
        for atom_index in atom_index_list:
            for l in l_list:
                locfun.append([atom_index, l, sigma])

        return locfun

    def has_charge_density(self):
        """Check if the calculation has charge density"""
        calc = self.get_calc()
        if calc is None:
            return False
        if hasattr(calc, 'density') and calc.density is not None:
            return True
        else:
            return False

    def get_special_points(self):
        atoms = self.get_atoms()
        rank = atoms.cell.rank
        if rank == 3:
            pbc = [1, 1, 1]
        elif rank == 2:
            pbc = [1, 1, 0]
        elif rank == 1:
            pbc = [1, 0, 0]
        else:
            raise ValueError(f"Invalid rank: {rank}")
        special_points = atoms.cell.bandpath('G', npoints=10, pbc=pbc).special_points
        return special_points

    def get_suggested_kpath(self):
        atoms = self.get_atoms()
        # Get the cell and lattice
        dimension = atoms.cell.rank
        lattice = atoms.cell.get_bravais_lattice()
        special_points = self.get_special_points()

        if dimension == 3:
            suggested_path = lattice.special_path
        else:
            suggested_path = ''.join(list(special_points.keys()))

        for label, point in special_points.items():
            print(f"{label} : [{point[0]:8.5f}, {point[1]:8.5f}, {point[2]:8.5f}]")

        return suggested_path, special_points

    def get_band_path(self, kpath: str | None = None, npoints: int = 100):
        """
        Get the band path for the given atoms object.

        Args:
        - kpath (str): The k-path to use. Default is None.
        - npoints (int): The number of k-points along the path. Default is 100.

        Returns:
        - path (BandPath): The band path object.
        """
        atoms = self.get_atoms()
        rank = atoms.cell.rank
        if rank == 3:
            pbc = [1, 1, 1]
        elif rank == 2:
            pbc = [1, 1, 0]
        elif rank == 1:
            pbc = [1, 0, 0]
        else:
            raise ValueError(f"Invalid rank: {rank}")

        if not kpath:  # if kpath is not specified, get the suggested path
            suggested_path, special_points = self.get_suggested_kpath()
            kpath = suggested_path
        print(f"Kpath: {kpath}")
        path = atoms.cell.bandpath(kpath, npoints=npoints, pbc=pbc)
        return path

    def set_relax_region(self, relax_intervals: list[list[float]]):
        """
        Mark certain region (intervals in z direction) as relax region, which means they will be relaxed in the relax structure calculation.
        All other regions will be frozen.

        Args:
        - structure: the bulk structure
        - intervals: a list of intervals in the z direction that will be marked as bulk region. Each item
        is a list of two numbers [z1,z2], defining the region in Angstrum.
        """
        if not relax_intervals:
            return

        atoms = self.get_atoms()
        positions = atoms.positions
        z_coords = positions[:, 2]
        relax_indices = []
        for atom_index, z in enumerate(z_coords):
            # print(f"Atom {atom_index}: z={z}")
            for interval in relax_intervals:
                if z >= interval[0] and z <= interval[1]:
                    relax_indices.append(atom_index)
                    break
        fixed_indices = [i for i in range(len(self.atoms)) if i not in relax_indices]

        constraint = FixAtoms(indices=fixed_indices)
        atoms.set_constraint(constraint)
        self.update_structure_from_atoms(atoms)

    def set_initial_magnetic_moments(self, magnetic_moments: list[float]):
        """
        Set the initial magnetic moments for the atoms.
        """
        if len(magnetic_moments) != len(self.atoms):
            raise ValueError("The number of magnetic moments must match the number of atoms.")
        atoms = self.get_atoms()
        atoms.set_initial_magnetic_moments(magnetic_moments)
        self.update_structure_from_atoms(atoms)

    def set_initial_magnetic_moments_by_atom(self, magnetic_moments_dict: dict):
        """
        Set the initial magnetic moments for the atoms.
        """
        chemical_symbols_list = self.get_chemical_symbols()
        magnetic_moments_list = [0] * len(chemical_symbols_list)
        for j, chemical_symbol in enumerate(chemical_symbols_list):
            if chemical_symbol in magnetic_moments_dict:
                magnetic_moments_list[j] = magnetic_moments_dict[chemical_symbol]
        self.set_initial_magnetic_moments(magnetic_moments_list)

    def ckpt_status(self):
        """Check the status of the ckpt file. Note that this might be different from self.calculation_status"""
        calc = self.get_calc()
        if calc is not None:
            if calc.scf.converged:
                return 'converged'
            else:
                return 'not converged'
        else:
            return 'no calculator'

    def update_calculation_status_by_ckpt_status(self):
        """Update the calculation status by checking the ckpt file status"""
        ckpt_status = self.ckpt_status()
        if ckpt_status == 'converged':
            self.calculation_status = 'finished'
        # if the ckpt file is not converged, but the calculation status is finished, it is interrupted
        # if the calculation status is "processing" or "initialized" or "error", keep it unchanged
        elif ckpt_status == 'not converged' and self.calculation_status == 'finished':
            self.calculation_status = 'interrupted'
        self.save_to_file()

    def get_summary(self):
        # Keywords to search for
        keywords = [
            'Date:',
            'cores:',
            'Input parameters:',
            'Spin',
            'Non-collinear calculation',
            "Total magnetic moment",
            "Local magnetic moments",
            "Magnetic moment",
            'Symmetries',
            'Positions:',
            'Unit cell:',
            'Free energy:',
            'Fermi level:',
            'Gap:',
            'No gap',
            'Forces in eV/Ang:',
            # 'Memory usage:'
        ]

        keywords_band = [
            'Date:',
            'cores:',
            # 'Input parameters:',
            'Spin',
            'Non-collinear calculation',
            "Total magnetic moment",
            "Local magnetic moments",
            "Magnetic moment",
            # 'Symmetries',
            'Positions:',
            'Unit cell:',
            'Free energy:',
            'Fermi level:',
            'Gap:',
            'No gap',
            # 'Forces in eV/Ang:',
            # 'Memory usage:'
        ]
        content = self.status_file_content
        calculation_type = self.calculation_type
        summary = ""
        if calculation_type == 'relax':
            initial_blocks, final_blocks = extract_blocks(content, keywords)
            summary += "Optimization calculation:\n\n"
            summary += "Initial structure:\n\n"
            for keyword, block in initial_blocks.items():
                summary += f"{block.rstrip()}\n"
            summary += f"\nOptimized structure:\n\n"
            for keyword, block in final_blocks.items():
                summary += f"{block.rstrip()}\n"
        elif calculation_type == 'ground_state':
            initial_blocks, final_blocks = extract_blocks(content, keywords)
            summary += "Ground state calculation:\n\n"
            for keyword, block in initial_blocks.items():
                summary += f"{block.rstrip()}\n"
        elif calculation_type == 'band':
            initial_blocks, final_blocks = extract_blocks(content, keywords_band)
            summary += "Band structure calculation:\n\n"
            for keyword, block in initial_blocks.items():
                summary += f"{block.rstrip()}\n"
        else:
            raise ValueError(
                f"Unsupported calculation type: {calculation_type}. Use 'relax', 'ground_state', or 'band'.")

        return summary


class CalculationProject:
    info_list: list[CalculationInfo] | None = None
    project_folder_path: str | None = None

    def __init__(self, project_folder_path: str, structure: Structure | None = None):
        if structure is None:
            self.load(project_folder_path)
        else:
            self.project_folder_path = project_folder_path
            # if there is a structure, create a new project
            # check if project_folder_path is empty. If not empty, raise error
            if os.path.exists(project_folder_path) and os.listdir(project_folder_path):
                raise ValueError(
                    f"Initialize the project with a structure requires the folder {project_folder_path} to be empty")

            calculation_id = str(uuid.uuid4())
            info = CalculationInfo(project_folder_path=project_folder_path,
                                   calculation_id=calculation_id,
                                   calculation_type='initial_structure')
            info.update_structure(structure)
            self.info_list = [info]
            self.update_calculation_id_list()

    def load(self, project_folder_path: str):
        self.project_folder_path = project_folder_path
        calculation_id_path = os.path.join(project_folder_path, 'calculation_id_list.json')
        if os.path.exists(calculation_id_path):
            with open(calculation_id_path, 'r') as f:
                calculation_id_list = json.load(f)
            if not isinstance(calculation_id_list, list):
                raise ValueError(f"Invalid calculation_id_list: {calculation_id_list}")
            self.info_list = []
            for calculation_id in calculation_id_list:
                info_path = os.path.join(project_folder_path, calculation_id, 'info.json')
                if os.path.exists(info_path):
                    self.info_list.append(CalculationInfo.from_file(info_path))
        else:
            print(f"calculation_id_list not found at {calculation_id_path}")
            self.info_list = None

    def update_calculation_id_list(self):
        calculation_id_list = []
        if self.info_list is not None:
            calculation_id_list = [info.calculation_id for info in self.info_list]
        with open(os.path.join(self.project_folder_path, 'calculation_id_list.json'), 'w') as f:
            json.dump(calculation_id_list, f)

    def latest_calc(self):
        if self.info_list is None:
            return None
        latest_info = self.info_list[-1]
        return latest_info

    def latest_finished_calc(self):
        if self.info_list is None:
            return None
        for info in reversed(self.info_list):
            if info.calculation_status == 'finished':
                return info
        return None

    def latest_dos_calc(self):
        """Get the latest finished calculation with DOS, which means relax or ground_state"""
        if self.info_list is None:
            print("No calculation info found")
            return None
        for info in reversed(self.info_list):
            if (info.calculation_type == 'relax' or info.calculation_type ==
                    'ground_state') and info.calculation_status == 'finished':
                return info
        print("No calculation with DOS has been finished")
        return None

    def latest_band_calc(self):
        """Get the latest finished band calculation"""
        if self.info_list is None:
            print("No calculation info found")
            return None
        for info in reversed(self.info_list):
            if info.calculation_type == 'band' and info.calculation_status == 'finished':
                return info
        print("No band calculation has been finished")
        return None

    def latest_calc_status(self):
        if self.latest_calc():
            return self.latest_calc().calculation_status
        else:
            return None

    def plot_band(self,
                  atom_labels: list | None = None,
                  orbitals: list | None = None,
                  **kwargs):
        if self.latest_band_calc():
            locfun = self.latest_band_calc().get_locfun(atom_labels=atom_labels, orbitals=orbitals)
            return self.latest_band_calc().plot_band(locfun=locfun, **kwargs)
        else:
            return None

    def plot_dos(self,
                 atom_labels: list | None = None,
                 orbitals: list | None = None,
                 **kwargs):
        if self.latest_dos_calc():
            locfun = self.latest_dos_calc().get_locfun(atom_labels=atom_labels, orbitals=orbitals)
            return self.latest_dos_calc().plot_dos(locfun=locfun, **kwargs)
        else:
            return None

    def add_new(self,
                calculation_id: str,
                calculation_type: str,
                calculation_parameters: dict | None = None):
        """Update the project with a new calculation info, and update the calculation_id_list"""
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

    def get_calculation_summary(self):
        summary = ""
        for i, calc in enumerate(self.info_list):
            # output the summary only for finished calculations
            if calc.calculation_type != 'initial_structure':
                summary += f"{i}. " + calc.get_summary()
                summary += "\n\n"
        return summary

    def get_calculation_history(self):
        """Get the calculation history as a list of dictionaries"""
        history_list = []
        for i, info in enumerate(self.info_list):
            history_list.append({
                "calculation_number": i,
                "calculation_type": info.calculation_type,
                "calculation_parameters": json.dumps(info.calculation_parameters, indent=4),
                "calculation_id": info.calculation_id,
                "calculation_status": info.get_calculation_status()
            })
        return history_list

    def discard_calculation(self, calculation_id: str | None = None):
        """discard a calculation by calculation_id. If calculation_id is None, discard the latest calculation"""
        if calculation_id is None:
            calculation_id = self.latest_calc().calculation_id
        for i, info in enumerate(self.info_list):
            if info.calculation_id == calculation_id:
                del self.info_list[i]
                # remove the calculation folder
                shutil.rmtree(os.path.join(self.project_folder_path, calculation_id))
                self.update_calculation_id_list()
                return
        raise ValueError(f"Calculation with id {calculation_id} not found")

    def get_atom_and_orbital_options(self):
        initial_data = self.info_list[0]
        atom_options = initial_data.get_atom_index_list()
        orbital_options = initial_data.get_orbital_list()
        return atom_options, orbital_options

    def get_calculation_info(self, calculation_id: str):
        for info in self.info_list:
            if info.calculation_id == calculation_id:
                return info
        return None

    def refresh(self):
        """refresh the calculation project by reloading the project folder"""
        if self.project_folder_path:
            self.load(self.project_folder_path)
