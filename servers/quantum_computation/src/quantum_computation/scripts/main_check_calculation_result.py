

import sys
import argparse
import json
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main(project_folder_path, calculation_id):
    from gpaw_utils.lib import check_calculation_result

    # Redirect all output to stderr initially
    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        result = check_calculation_result(
            project_folder_path=project_folder_path,
            calculation_id=calculation_id)

        # Restore stdout only for the final result
        sys.stdout = original_stdout
        result_str = json.dumps(result, indent=4)
        print(result_str, flush=True)
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        # Ensure stdout is restored even if there's an error
        sys.stdout = original_stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder_path', '-p',
                        required=True,
                        help='Path to the project folder')
    parser.add_argument('--calculation_id', '-c',
                        required=True,
                        help='ID of the calculation to check')

    args = parser.parse_args()

    main(
        project_folder_path=args.project_folder_path,
        calculation_id=args.calculation_id
    )
