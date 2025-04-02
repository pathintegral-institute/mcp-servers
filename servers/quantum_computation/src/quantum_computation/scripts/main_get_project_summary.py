import argparse
import json
from gpaw_utils.data_class import CalculationProject


def main(project_folder_path: str):
    """
    Create a new project in the given project_folder_path
    """
    calculation_project = CalculationProject(
        project_folder_path=project_folder_path)

    calculation_history = calculation_project.get_calculation_history()
    print(json.dumps({"calculation_history": calculation_history}, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_folder_path", "-p", type=str)

    args = parser.parse_args()
    main(
        project_folder_path=args.project_folder_path
    )
