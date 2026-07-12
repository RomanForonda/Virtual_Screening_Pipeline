"""
protein_to_pdbqt.py

Convert the repaired protein from PDB to PDBQT using ADFRsuite
(prepare_receptor), instead of Meeko.

ADFRsuite must be installed separately and its "bin" folder must be
available on the system PATH, so the "prepare_receptor" command works
directly from the terminal.
"""

import argparse
import shutil
import subprocess


def prepare_receptor(
    input_file: str,
    output_file: str,
    prepare_receptor_command: str = "prepare_receptor"
):

    print("\n==============================")
    print("Protein -> PDBQT (ADFRsuite)")
    print("==============================")

    if shutil.which(prepare_receptor_command) is None:
        raise FileNotFoundError(
            f"Could not find the '{prepare_receptor_command}' command on PATH.\n"
            "Check that ADFRsuite is installed and that its 'bin' folder "
            "has been added to your system PATH. If you just edited PATH, "
            "open a completely new terminal before trying again."
        )

    command = [
        prepare_receptor_command,
        "-r", input_file,
        "-o", output_file
    ]

    print("\nRunning protein conversion to PDBQT...")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        shell=True
    )

    if result.returncode != 0:
        print("\nProtein conversion failed:")
        print(result.stderr)
        raise RuntimeError(
            f"prepare_receptor failed with exit code {result.returncode}. "
            "Review the error message above."
        )

    print(f"\nReceptor saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--command", default="prepare_receptor")
    args = parser.parse_args()

    prepare_receptor(args.input, args.output, args.command)
