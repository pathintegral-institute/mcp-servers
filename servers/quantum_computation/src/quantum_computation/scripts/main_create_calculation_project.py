import sys
import argparse
import json
from pymatgen.core.structure import Structure
from gpaw_utils.data_class import CalculationProject


def main(project_folder_path: str, structure_str: str):
    """
    Create a new project in the given project_folder_path
    """
    try:
        structure = Structure.from_dict(json.loads(structure_str))

        new_project = CalculationProject(
            project_folder_path=project_folder_path,
            structure=structure)

        calculation_history = new_project.get_calculation_history()

        result = json.dumps({"calculation_history": calculation_history}, indent=4)
        print(result)
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_folder_path", "-p", type=str)
    parser.add_argument("--structure_str", "-s", type=str)

    args = parser.parse_args()
    main(
        project_folder_path=args.project_folder_path,
        structure_str=args.structure_str
    )
