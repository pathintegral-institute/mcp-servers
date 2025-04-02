from gpaw_utils.lib import perform_calculation
import argparse
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main(params_str, project_folder_path, output_calculation_id):
    from gpaw_utils.lib import perform_calculation

    params = json.loads(params_str)
    calculation_type = params["calculation_type"]
    calculation_parameters = params
    calculation_parameters.pop("calculation_type")

    result_dict = perform_calculation(
        calculation_type=calculation_type,
        calculation_parameters=calculation_parameters,
        project_folder_path=project_folder_path,
        output_calculation_id=output_calculation_id
    )

    print(json.dumps(result_dict, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_str", "-pm", type=str)
    parser.add_argument("--project_folder_path", "-p", type=str)
    parser.add_argument("--output_calculation_id", "-c", type=str)

    args = parser.parse_args()
    main(
        params_str=args.params_str,
        project_folder_path=args.project_folder_path,
        output_calculation_id=args.output_calculation_id
    )
