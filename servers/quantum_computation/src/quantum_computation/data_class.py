from __future__ import annotations

import json
import os
import uuid
from typing import TypedDict

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .rester import mp_rester


class SupercellParameters(TypedDict):
    """Parameters for creating and modifying a supercell structure.

    Attributes:
        supercell_size: Size multipliers for each dimension a, b, and c of the unit cell
        removed_region: Region to remove from the structure, specified as [zmin, zmax] where zmin and zmax are relative coordinates between 0 and 1.
    """

    supercell_size: list[int]  # Size multipliers for each dimension [a, b, c]
    removed_region: list[float]  # Region to remove [zmin, zmax]


class MoireParameters(TypedDict):
    pass


class StructureData:
    structure: Structure  # the structure in Pymatgen format
    structure_id: str  # a unique identifier
    material_id: str | None = None  # the material id from MPR in case of bulk structure
    parameters: SupercellParameters | MoireParameters | None = (
        None  # parameters that are used to build the supercell or moire structure
    )
    # parent structures which are usually bulk structures
    parent_ids: list[str] | None = None

    def __init__(
        self,
        structure: Structure
        | str
        | None = None,  # if the structure is a string, it is treated as poscar and converted to Structure
        structure_id: str | None = None,
        material_id: str | None = None,
        parameters: SupercellParameters | MoireParameters | None = None,
        parent_ids: list[str] | None = None,
    ):
        if structure and isinstance(structure, Structure):
            self.structure = structure
        elif structure and isinstance(structure, str):
            self.structure = Structure.from_str(structure, fmt="poscar")
        elif material_id:
            self.material_id = material_id
            result = mp_rester.summary.get_data_by_id(material_id)
            self.structure = result.structure
        else:  # raise error if no structure or material_id
            raise ValueError(
                "Either structure or material_id must be provided")
        if structure_id:
            self.structure_id = structure_id
        else:  # if no structure id is provided, generate a new one
            self.structure_id = str(uuid.uuid4())
        if parameters:
            self.parameters = parameters
        if parent_ids:
            self.parent_ids = parent_ids

    def to_folder(self, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data = {k: v for k, v in self.__dict__.items() if k != "structure"}
        dict_path = os.path.join(folder_path, "structure_info.json")
        with open(dict_path, "w") as f:
            json.dump(data, f, indent=2)
        file_path = os.path.join(folder_path, "structure.cif")
        self.structure.to(fmt="cif", filename=file_path)

    @classmethod
    def from_folder(cls, folder_path: str):
        dict_path = os.path.join(folder_path, "structure_info.json")
        if not os.path.exists(dict_path):
            raise FileNotFoundError(
                f"structure_info.json not found in {folder_path}")

        with open(dict_path, "r") as f:
            data = json.load(f)

        instance = cls.__new__(cls)
        instance.__dict__.update(data)

        file_path = os.path.join(folder_path, "structure.cif")
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"structure.cif not found in {folder_path}")
        instance.structure = Structure.from_file(file_path)

        return instance

    @property
    def poscar_str(self) -> str:
        # Create a Poscar object from the structure
        poscar: Poscar = Poscar(structure=self.structure)
        # Convert to a string
        poscar_string: str = poscar.get_str()
        return poscar_string

    @property
    def description(self):
        return self.get_description()

    def get_description(self) -> str:
        description = "Bulk Structure Information\n\n"

        description += f"""
Material id: {self.material_id if 'material_id' in self.__dict__ else 'N/A'}

Species:
{self.structure.species}

Lattice Parameters:
a={self.structure.lattice.a:.4f}
b={self.structure.lattice.b:.4f}
c={self.structure.lattice.c:.4f}
Angles:
alpha={self.structure.lattice.alpha:.4f}
beta={self.structure.lattice.beta:.4f}
gamma={self.structure.lattice.gamma:.4f}

Number of atoms: {len(self.structure)}
"""
        return json.loads(json.dumps(description))

    @classmethod
    def build_supercell(cls, bulk_structure_data: StructureData, supercell_parameters: SupercellParameters):
        bulk_structure = bulk_structure_data.structure
        # Convert to conventional cell
        spacegroup = SpacegroupAnalyzer(bulk_structure)
        conventional_structure = spacegroup.get_conventional_standard_structure()
        # logger.info(conventional_structure)
        supercell_size = [int(n)
                          for n in supercell_parameters["supercell_size"]]

        supercell_structure = conventional_structure * supercell_size

        zs = [float(z) for z in supercell_parameters["removed_region"]]
        # Remove atoms with z > 1-zc or z < zc
        sites_to_remove = []
        for site in supercell_structure:
            if site.frac_coords[2] > zs[0] and site.frac_coords[2] < zs[1]:
                sites_to_remove.append(site)

        if sites_to_remove:
            for site in sites_to_remove:
                supercell_structure.remove(site)

        instance = cls.__new__(cls)
        instance.structure = supercell_structure
        instance.structure_id = str(uuid.uuid4())
        instance.parameters = supercell_parameters
        instance.parent_ids = [bulk_structure_data.structure_id]
        instance.material_id = None
        return instance
