from pymatgen.io.ase import AseAtomsAdaptor
from ase.atoms import Atoms
from pymatgen.core import Structure, Lattice
from gpaw import GPAW, restart
from ase.optimize import BFGS
import numpy as np
import shutil
import copy
import json
import time
import uuid
import os
import sys
from .data_class import CalculationInfo, EnhancedAtoms, CalculationProject
import plotly.io as pio


GPAW_TEMP_FOLDER_PATH = "working/temp"

FORCE_THRESHOLD = 0.001

_MAX_TRY = 10

MAX_COMPUTATION_TIME = 3600


def resume_relax_calculation(ckpt_file_path: str,
                             trajectory_file_path: str,
                             steps=100,
                             force_threshold=FORCE_THRESHOLD):
    """Resume relax calculation from a checkpoint file."""
    if os.path.exists(ckpt_file_path):
        # Restart from checkpoint
        atoms, calc = restart(ckpt_file_path)
        atoms = EnhancedAtoms(atoms)
        is_converged = atoms.check_convergence(force_threshold=force_threshold)
        if is_converged:
            # The checkpoint is a converged calculation. Output results
            atoms.calc = calc
            return atoms
        else:
            print("The previous calculation is not converged. Resume the calculation now.")
            optimizer = BFGS(atoms, trajectory=trajectory_file_path)
            # Enable checkpointing in the optimizer
            optimizer.attach(lambda: calc.write(ckpt_file_path), interval=1)
            optimizer.run(fmax=force_threshold, steps=steps)
            return atoms


def relax_calculation(atoms: Atoms,
                      is_bulk: bool,
                      kpoints: list[int],
                      spinpol: bool,
                      status_file_path: str,
                      log_file_path: str,
                      trajectory_file_path: str,
                      ckpt_file_path: str | None = None,
                      initial_magnetic_moments: list[float] | None = None,
                      relax_intervals: list[list[float]] | None = None,
                      ecut: float = 350,
                      xc: str = 'PBE',
                      nbands: int = -10,
                      mode_name: str = 'pw',
                      steps: int = 100,
                      force_threshold: float = FORCE_THRESHOLD):

    logfile = open(log_file_path, 'w')

    # check if the file at ckpt_file_path exists. If exists, resume the calculation from the ckpt_file_path
    if ckpt_file_path and os.path.exists(ckpt_file_path):
        atoms = resume_relax_calculation(ckpt_file_path=ckpt_file_path,
                                         trajectory_file_path=trajectory_file_path,
                                         force_threshold=force_threshold,
                                         steps=steps)
        return atoms

    if not is_bulk:
        kpoints[2] = 1
    # print(atoms.get_chemical_symbols())
    chemical_symbols_list = list(set(atoms.get_chemical_symbols()))
    chemical_symbols_list.sort()
    if isinstance(initial_magnetic_moments, list) and len(initial_magnetic_moments) == len(atoms):
        atoms.set_initial_magnetic_moments(initial_magnetic_moments)
    # set the initial magnetic moments by atom
    elif isinstance(initial_magnetic_moments, list) and len(initial_magnetic_moments) == len(chemical_symbols_list):
        initial_magnetic_moments_dict = {
            symbol: moment for symbol, moment in zip(chemical_symbols_list, initial_magnetic_moments)}
        initial_magnetic_moments_full_list = []
        for chemical_symbol in atoms.get_chemical_symbols():
            initial_magnetic_moments_full_list.append(initial_magnetic_moments_dict[chemical_symbol])
        atoms.set_initial_magnetic_moments(initial_magnetic_moments_full_list)
    else:
        atoms.set_initial_magnetic_moments([0] * len(atoms))
    # set the calculator
    calc = GPAW(mode={'name': mode_name, 'ecut': ecut},  # plane wave basis set cutoff
                xc=xc,  # EX-Crl functional (e.g., 'LDA', 'PBE', 'RPBE')
                nbands=nbands,  # total number of bands
                kpts={'size': kpoints, 'gamma': True},  # k-points
                spinpol=spinpol,  # non-spin-polarized calculation
                txt=status_file_path)  # output file

    atoms.calc = calc
    atoms = EnhancedAtoms(atoms)
    atoms.set_relax_region(relax_intervals)

    optimizer = BFGS(atoms=atoms, trajectory=trajectory_file_path, logfile=logfile)
    if ckpt_file_path:
        optimizer.attach(lambda: calc.write(ckpt_file_path), interval=1)
    optimizer.run(fmax=force_threshold, steps=steps)  # fmax is the convergence threshold for the force,  eV/A.
    return atoms


def ground_state_calculation(atoms: Atoms,
                             kpoints: list[int],
                             spinpol: bool,
                             status_file_path: str,
                             ckpt_file_path: str | None = None,
                             initial_magnetic_moments: list[float] | None = None,
                             ecut: float = 350,
                             xc: str = 'PBE',
                             nbands: int = -10,
                             mode_name: str = 'pw',
                             n_write: int = 3):
    if ckpt_file_path and os.path.exists(ckpt_file_path):
        atoms = resume_ground_state_calculation(ckpt_file_path=ckpt_file_path, n_write=n_write)
        return atoms
    chemical_symbols_list = list(set(atoms.get_chemical_symbols()))
    chemical_symbols_list.sort()
    if isinstance(initial_magnetic_moments, list) and len(initial_magnetic_moments) == len(atoms):
        atoms.set_initial_magnetic_moments(initial_magnetic_moments)
    elif isinstance(initial_magnetic_moments, list) and len(initial_magnetic_moments) == len(chemical_symbols_list):
        initial_magnetic_moments_dict = {
            symbol: moment for symbol, moment in zip(
                chemical_symbols_list, initial_magnetic_moments)}
        atoms.set_initial_magnetic_moments_by_atom(initial_magnetic_moments_dict)
    else:
        atoms.set_initial_magnetic_moments([0] * len(atoms))
    calc = GPAW(mode={'name': mode_name, 'ecut': ecut},
                xc=xc,  # EX-Crl functional (e.g., 'LDA', 'PBE', 'RPBE')
                nbands=nbands,  # total number of bands
                kpts={'size': kpoints, 'gamma': True},
                spinpol=spinpol,
                txt=status_file_path)
    if ckpt_file_path:
        calc.attach(calc.write, n_write, ckpt_file_path, mode='all')
    atoms.calc = calc
    # carry the relaxation calculation
    e = atoms.get_potential_energy()

    print(f'Ground state (SCF) calculations finished. The energy is {e} eV')
    return atoms


def resume_ground_state_calculation(ckpt_file_path: str, n_write: int = 3):
    """Resume calculation from a checkpoint file."""
    if os.path.exists(ckpt_file_path):
        # Restart from checkpoint
        atoms, calc = restart(ckpt_file_path)
        if calc.scf.converged:
            atoms.calc = calc
            # The checkpoint is a converged calculation. Output results
            return atoms
        else:
            # The checkpoint is NOT converged. Resume calculation now.
            print("The previous calculation is not converged. Resume the calculation now.")
            calc.attach(calc.write, n_write, ckpt_file_path, mode='all')
            atoms.get_potential_energy()
            return atoms


def band_calculation(atoms: Atoms,
                     status_file_path: str,
                     kpath: str | None = None,
                     npoints: int = 60,
                     symmetry: str = 'off',
                     include_soc: bool = True,
                     ckpt_file_path: str | None = None):
    # if status_file_path is not .txt, add .txt to the end of status_file_path
    # get the band path
    atoms = EnhancedAtoms(atoms)
    path = atoms.get_band_path(kpath=kpath, npoints=npoints)
    # to do: check if atoms.calc is defined and scf calculation is done
    calc = atoms.calc

    calc = calc.fixed_density(
        kpts=path,
        symmetry=symmetry,
        txt=status_file_path)

    atoms.calc = calc
    if ckpt_file_path:
        calc.write(ckpt_file_path, mode='all')
    return atoms


def perform_calculation_service(input_info: CalculationInfo,
                                output_info: CalculationInfo):
    """
    Perform the calculation for the given calculation type and parameters, based on the input information.
    This function assumes the input_info and output_info folders have been created.

    Args:
    - input_info: a CalculationInfo object containing the input data.
    - output_info: a CalculationInfo object containing the output data.

    Returns:
    - output_info: The output data after the calculation is performed.
    """
    atoms = input_info.get_atoms()
    calculation_type = output_info.calculation_type
    calculation_parameters = output_info.calculation_parameters
    # print(f"Calculation type: {calculation_type}")
    # print(f"Calculation parameters: {json.dumps(calculation_parameters, indent=4)}")
    output_info.update_status('processing')
    if calculation_type == 'relax':
        atoms = relax_calculation(atoms=atoms,
                                  status_file_path=output_info.status_file_path,
                                  ckpt_file_path=output_info.ckpt_file_path,
                                  trajectory_file_path=output_info.trajectory_file_path,
                                  log_file_path=output_info.log_file_path,
                                  **calculation_parameters)
    elif calculation_type == 'ground_state':
        atoms = ground_state_calculation(atoms=atoms,
                                         status_file_path=output_info.status_file_path,
                                         ckpt_file_path=output_info.ckpt_file_path,
                                         **calculation_parameters)
    elif calculation_type == 'band':
        # for band calculation, need to specify a ckpt file path to provide the initial charge density
        input_ckpt_file_path = getattr(input_info, 'ckpt_file_path', None)
        if not input_ckpt_file_path or not os.path.exists(input_ckpt_file_path):
            raise ValueError(
                "Input ckpt_file_path is empty. A ckpt file with charge density isrequired for band calculation.")
        else:
            calc = GPAW(input_ckpt_file_path, txt=None)
            if not hasattr(calc, 'density') or calc.density is None:
                raise ValueError("Input ckpt_file_path is not a valid input because it does not have charge density.")
            atoms.calc = calc
            atoms = band_calculation(atoms=atoms,
                                     status_file_path=output_info.status_file_path,
                                     ckpt_file_path=output_info.ckpt_file_path,
                                     **calculation_parameters)
    # save the structure information in .cif file in the output info folder
    output_info.update_structure_from_atoms(atoms)
    output_info.update_status('finished')
    return output_info


def remove_calculation_subfolder(calculation_id: str, project_folder_path: str):
    """Remove the calculation subfolder in project_folder_path"""
    try:
        shutil.rmtree(os.path.join(project_folder_path, calculation_id))
    except FileNotFoundError:
        print(f"Subfolder {calculation_id} does not exist in {project_folder_path}.")


def perform_calculation_by_id(input_calculation_id: str, output_calculation_id: str, project_folder_path: str):
    """
    Perform the calculation based on the input information secified by input_calculation_id and output the result to the folder specified by output_information_id
    """
    input_info_file_path = os.path.join(project_folder_path, input_calculation_id, 'info.json')
    output_info_file_path = os.path.join(project_folder_path, output_calculation_id, 'info.json')
    input_info = CalculationInfo.from_file(input_info_file_path)
    output_info = CalculationInfo.from_file(output_info_file_path)
    # print(output_info)
    if not input_info:
        raise ValueError(f"No input information found in path {input_info_file_path}.")
    if not output_info:
        raise ValueError(f"No output information found in path {output_info_file_path}.")
    output_info = perform_calculation_service(input_info, output_info)
    return output_info


def perform_calculation(calculation_type: str,
                        calculation_parameters: dict,
                        project_folder_path: str,
                        output_calculation_id: str | None = None) -> str:
    """
    Start a relax calculation subprocessbased on the calculation type and parameters, and store the calculation info into a new calculation record.
    The input_calculation_id is chosen automatically. The output_calculation_id is a new uuid generated. If the calculation cannot be calculated,
    the function will return an error message.

    Args:
        relax_calculation_parameters: a dictionary containing the calculation parameters
        project_folder_path: the path to the project folder

    Returns:
        A dictionary containing the calculation id.
    """
    calculation_record = CalculationProject(project_folder_path=project_folder_path)
    if calculation_type == 'relax' or calculation_type == 'ground_state':
        input_info = calculation_record.latest_finished_calc()
    else:
        input_info = calculation_record.latest_dos_calc()
    if not input_info:
        input_info = calculation_record.info_list[0]
    if not output_calculation_id:
        output_calculation_id = str(uuid.uuid4())
    calculation_record.add_new(
        calculation_id=output_calculation_id,
        calculation_type=calculation_type,
        calculation_parameters=calculation_parameters)
    perform_calculation_by_id(input_calculation_id=input_info.calculation_id,
                              output_calculation_id=output_calculation_id,
                              project_folder_path=project_folder_path)
    return {
        "project_folder_path": project_folder_path,
        "calculation_id": output_calculation_id
    }


def check_calculation_result(calculation_id: str, project_folder_path: str):
    """
    Check the calculation result with the given calculation id and project folder path.

    Args:
        calculation_id: the id of the calculation
        project_folder_path: the path to the project folder

    Returns:
        A dictionary containing the calculation log.
    """
    calculation_info_file_path = os.path.join(project_folder_path, calculation_id, 'info.json')
    calculation_info = None

    start_time = time.time()
    while True:
        try:
            calculation_info = CalculationInfo.from_file(calculation_info_file_path)
            calculation_status = calculation_info.calculation_status
            status_file_content = calculation_info.status_file_content
        except Exception as e:
            status_file_content = str(e)
            calculation_status = "status inaccessible"
            print(f"Warning: {str(e)}", file=sys.stderr, flush=True)

        # result = {
        #         "calculation_status": calculation_status,
        #         "log": status_file_content
        #     }

        if calculation_status == 'finished':
            result = {
                "calculation_status": calculation_status,
                "log": status_file_content
            }
            # return the figure as a dictionary for JSON serialization
            try:
                fig = calculation_info.plot_output()
                if fig:
                    fig_str = pio.to_json(fig)
                    result["figure_str"] = fig_str
            except Exception as e:
                print(f"Warning: Error generating plot: {str(e)}", file=sys.stderr, flush=True)
            break

        # if time exceeds max_computation_time, exit
        if time.time() - start_time > MAX_COMPUTATION_TIME:
            log = status_file_content + \
                f"\n\nCalculation exceeded maximum computation time {MAX_COMPUTATION_TIME} seconds"
            result = {
                "calculation_status": "timeout",
                "log": log
            }
            break

        time.sleep(5)

    return result


def stream_calculation_log(calculation_id: str, project_folder_path: str):
    """
    Stream the calculation log of a calculation subprocess, if the calculation is in progress.

    Args:
        calculation_id: the id of the calculation
        project_folder_path: the path to the project folder

    Returns:
        A dictionary containing the calculation log.
    """
    calculation_info_file_path = os.path.join(project_folder_path, calculation_id, 'info.json')

    while True:
        try:
            calculation_info = CalculationInfo.from_file(calculation_info_file_path)
            calculation_status = calculation_info.calculation_status
            status_file_content = calculation_info.status_file_content
            yield status_file_content
        except Exception as e:
            status_file_content = e
            calculation_status = "status inaccessible"

        if calculation_status == 'finished' or calculation_status == 'interrupted':
            break
        time.sleep(1)
