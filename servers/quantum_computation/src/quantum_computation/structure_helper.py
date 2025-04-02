import os
import json
from loguru import logger

from .data_class import StructureData

structure_temp_folder_path = os.getenv(
    "STRUCTURE_TEMP_FOLDER_PATH", "working/structures/temp")
root_path = os.getenv("ROOT_PATH", "")
structure_temp_folder_abs_path = os.path.join(
    root_path, structure_temp_folder_path)
material_id_dict_path = os.path.join(
    root_path, structure_temp_folder_path, "material_id_dict.json")


def get_structure_folder_path(structure_uri: str) -> str:
    if not structure_uri.startswith("structure://"):
        raise ValueError(
            "Error: structure uri does not start with structure://")
    structure_id = structure_uri.split("structure://")[-1]
    structure_folder_path = os.path.join(
        structure_temp_folder_abs_path, f"{structure_id}")
    return structure_folder_path


def get_structure_info(structure_uri: str) -> StructureData | None:
    structure_folder_path = get_structure_folder_path(structure_uri)
    if structure_folder_path and os.path.exists(structure_folder_path):
        structure_data = StructureData.from_folder(structure_folder_path)
        return structure_data
    else:
        logger.warning("Error: structure folder not found")
        return None


def get_structure_from_material_id(material_id: str) -> StructureData | None:
    """
    Get a structure from a material_id. First check material_id_dict for the structure. If material_id_dict.keys() does not contain the material_id, retrieve it from MPR and save it to the temp folder.

    Args:
        material_id: the material_id of the structure

    Returns:
        the structure data, an instance of StructureData class
    """
    # load material_id_dict
    if os.path.exists(material_id_dict_path):
        with open(material_id_dict_path, "r") as f:
            material_id_dict = json.load(f)
    else:
        material_id_dict = {}
    # If the list already contains the material id, load it from folder
    if material_id_dict.get(material_id) is not None:
        structure_id = material_id_dict[material_id]
        structure_uri = f"structure://{structure_id}"
        structure_data = get_structure_info(structure_uri)
        if structure_data is not None:
            return structure_data
    else:
        # retrieve from MPR
        structure_data = StructureData(material_id=material_id)
        structure_id = structure_data.structure_id
        structure_uri = f"structure://{structure_id}"
        structure_folder_path = get_structure_folder_path(structure_uri)
        structure_data.to_folder(structure_folder_path)
        material_id_dict[material_id] = structure_id
        with open(material_id_dict_path, "w") as f:
            json.dump(material_id_dict, f)
        return structure_data
    return None
