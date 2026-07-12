from pathlib import Path
import argparse
import ast
import subprocess
import sys
import traceback

import pandas as pd

from config import load_config
from data_curation import curate_data
from mol_filters import apply_filters
from clustering import cluster_molecules
from ligand_prep import prepare_ligands
from sdf_to_pdbqt import sdf_to_pdbqt
from protein_to_pdbqt import prepare_receptor
from docking import run_docking


BASE = Path(__file__).parent.resolve()
RESULTS = BASE / "results"

# -------------------------------------------------------------
# CONFIGURATION
# Name of the conda environment that contains Biopython,
# PDBFixer and OpenMM. This is the environment used by
# protein_clean.py and protein_fix.py.
# -------------------------------------------------------------
DEFAULT_CONFIG_FILE = "configs/config.yaml"


def result_exists(filename):
    """
    Check whether a step output already exists in results/.
    This lets the pipeline resume without repeating completed steps.
    """
    return (RESULTS / filename).exists()


def results_exist(*filenames):
    """
    Check whether all requested step outputs already exist in results/.
    """
    return all(
        result_exists(filename)
        for filename in filenames
    )


def docking_env_python(env_name):
    # In a standard conda install, environments live as sibling folders.
    env_dir = Path(sys.executable).parent.parent / env_name
    candidates = [
        env_dir / "python.exe",
        env_dir / "bin" / "python",
        env_dir / "Scripts" / "python.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def run_in_docking_env(config, script, *args):
    """
    Run a Python script inside docking_env instead of the current
    rdkit_env process. Steps 6 and 7 need packages that are only
    installed in docking_env.
    """
    env_name = config["environments"]["docking_env_name"]
    env_python = docking_env_python(env_name)

    if not env_python.exists():
        raise FileNotFoundError(
            f"Could not find Python for '{env_name}' at:\n"
            f"{env_python}\n"
            "Check that you are running main.py with 'rdkit_env' activated "
            f"(not 'base') and that the '{env_name}' environment exists."
        )

    subprocess.run(
        [str(env_python), script, *args],
        cwd=str(BASE),
        check=True
    )


def read_python_constant(script_name, constant_name, default=None):
    """
    Read a simple constant assignment from a Python file without importing it.
    This avoids importing protein-processing modules from the RDKit environment.
    """
    script_path = BASE / script_name

    try:
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
    except OSError:
        return default

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == constant_name:
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    return default

    return default


def count_excel_rows(path):
    if not path.exists():
        return None

    return len(pd.read_excel(path))


def count_sdf_records(path):
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.count("$$$$")


def count_files(path, pattern):
    if not path.exists():
        return None

    return len(list(path.glob(pattern)))


def read_binding_site(path):
    if not path.exists():
        return {}

    values = {}

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        try:
            values[key.strip()] = float(value.strip())
        except ValueError:
            values[key.strip()] = value.strip()

    return values


def format_count(value):
    return "not generated" if value is None else str(value)


def generate_pipeline_summary(config):
    """
    Create a human-readable text summary of the full virtual-screening run.
    Values are read from the generated files so the summary also works when
    the pipeline resumes and skips already completed steps.
    """
    RESULTS.mkdir(exist_ok=True)

    paths = config["paths"]
    outputs = config["outputs"]
    protein = config["protein"]
    clustering = config["clustering"]
    tools = config["tools"]
    docking = config["docking"]

    dataset_rows = count_excel_rows(BASE / "data" / paths["dataset_file"])
    curated_rows = count_excel_rows(RESULTS / outputs["curated_dataset"])
    filtered_rows = count_excel_rows(RESULTS / outputs["filtered_dataset"])
    clustered_rows = count_excel_rows(RESULTS / outputs["clustered_dataset"])
    selected_rows = count_excel_rows(RESULTS / outputs["selected_dataset"])
    prepared_ligands = count_sdf_records(RESULTS / outputs["prepared_ligands"])
    pdbqt_ligands = count_files(RESULTS / outputs["pdbqt_ligands_dir"], "*.pdbqt")
    docking_pose_files = count_files(RESULTS / outputs["docking_poses_dir"], "*_out.pdbqt")

    binding_site = read_binding_site(RESULTS / outputs["binding_site"])

    ligand_resname = protein["ligand_resname"]
    padding = protein["padding_angstroms"]
    summary_lines = [
        "Virtual Screening Pipeline Summary",
        "=" * 35,
        "",
        "Input and molecule counts",
        "-" * 25,
        f"Input dataset molecules: {format_count(dataset_rows)}",
        f"After curation: {format_count(curated_rows)}",
        f"After drug-likeness filters: {format_count(filtered_rows)}",
        f"After clustering table generation: {format_count(clustered_rows)}",
        f"Selected for docking: {format_count(selected_rows)}",
        f"Prepared 3D ligands in SDF: {format_count(prepared_ligands)}",
        f"PDBQT ligand files: {format_count(pdbqt_ligands)}",
        "",
        "Adjustable parameters used",
        "-" * 26,
        f"Docking conda environment: {config['environments']['docking_env_name']}",
        f"Binding-site ligand residue name: {ligand_resname}",
        f"Binding-site padding Angstroms: {padding}",
        f"Tanimoto similarity threshold for Butina clustering: {clustering['similarity_threshold']}",
        f"Butina distance cutoff used internally: {1 - clustering['similarity_threshold']:.3f}",
        f"Maximum molecules kept per small cluster: {clustering['max_cluster_size']}",
        "Morgan fingerprint radius: 2",
        "Morgan fingerprint bits: 2048",
        "2D diversity map method: PCA",
        "2D diversity map random_state: 42",
        "Ligand conformer method: RDKit ETKDGv3",
        "Ligand conformer random seed: 42",
        "Ligand conformers generated per molecule: 1",
        "Ligand geometry optimization force field: MMFF",
        f"Protein hydrogenation pH: {protein['hydrogenation_ph']}",
        f"Ligand PDBQT conversion tool: OpenBabel {tools['obabel']}",
        f"Receptor PDBQT conversion tool: ADFRsuite {tools['prepare_receptor']}",
        f"Vina executable: {tools['vina']}",
        f"Vina docking precision/search thoroughness, exhaustiveness: {docking['exhaustiveness']}",
        f"Vina CPU threads: {docking['cpu']}",
        f"Vina max ligands per run: {docking['max_ligands']}",
        "",
        "Docking box",
        "-" * 11,
    ]

    if binding_site:
        for key in [
            "center_x",
            "center_y",
            "center_z",
            "size_x",
            "size_y",
            "size_z",
        ]:
            summary_lines.append(f"{key}: {binding_site.get(key, 'not generated')}")
    else:
        summary_lines.append("Docking box file not generated.")

    summary_lines.extend([
        "",
        "Generated output files",
        "-" * 22,
        f"Curated dataset: {RESULTS / outputs['curated_dataset']}",
        f"Filtered dataset: {RESULTS / outputs['filtered_dataset']}",
        f"Clustered dataset: {RESULTS / outputs['clustered_dataset']}",
        f"Selected dataset: {RESULTS / outputs['selected_dataset']}",
        f"Cluster plot: {RESULTS / outputs['cluster_plot']}",
        f"Prepared ligands SDF: {RESULTS / outputs['prepared_ligands']}",
        f"Ligand PDBQT directory: {RESULTS / outputs['pdbqt_ligands_dir']}",
        f"Clean protein PDB: {RESULTS / outputs['clean_protein']}",
        f"Fixed protein PDB: {RESULTS / outputs['fixed_protein']}",
        f"Receptor PDBQT: {RESULTS / outputs['receptor_pdbqt']}",
        f"Docking pose files: {format_count(docking_pose_files)}",
        f"Docking results table: {RESULTS / outputs['docking_results']}",
        "",
        "Docking results",
        "-" * 15,
    ])

    docking_results_path = RESULTS / outputs["docking_results"]

    if docking_results_path.exists():
        docking_df = pd.read_excel(docking_results_path)
        total_docking_rows = len(docking_df)

        if "affinity_kcal_mol" in docking_df.columns:
            successful_df = docking_df[docking_df["affinity_kcal_mol"].notna()]
            failed_count = total_docking_rows - len(successful_df)
        else:
            successful_df = pd.DataFrame()
            failed_count = "unknown"

        summary_lines.append(f"Docking result rows: {total_docking_rows}")
        summary_lines.append(f"Successful docking affinities: {len(successful_df)}")
        summary_lines.append(f"Failed or missing affinities: {failed_count}")

        if len(successful_df) > 0:
            best_row = successful_df.sort_values("affinity_kcal_mol").iloc[0]
            summary_lines.append(
                "Best predicted binder: "
                f"{best_row['ligand']} "
                f"({best_row['affinity_kcal_mol']} kcal/mol)"
            )

        summary_lines.extend([
            "",
            "Full ranked docking table",
            "-" * 25,
            docking_df.to_string(index=False),
        ])
    else:
        summary_lines.append("Docking results were not generated.")

    output_path = RESULTS / outputs["summary"]
    output_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8"
    )

    print(f"\nPipeline summary saved to: {output_path}")
    return output_path


def run_pipeline(config):

    paths = config["paths"]
    outputs = config["outputs"]
    protein = config["protein"]
    clustering = config["clustering"]
    tools = config["tools"]

    print("\n=== 1) Dataset curation ===")
    if result_exists(outputs["curated_dataset"]):
        print(f"{outputs['curated_dataset']} already exists; skipping this step.")
    else:
        curate_data(paths["dataset_file"], outputs["curated_dataset"])

    print("\n=== 2) Drug-likeness filters ===")
    if result_exists(outputs["filtered_dataset"]):
        print(f"{outputs['filtered_dataset']} already exists; skipping this step.")
    else:
        apply_filters(outputs["curated_dataset"], outputs["filtered_dataset"])

    print("\n=== 3) Diversity clustering ===")
    if results_exist(
        outputs["clustered_dataset"],
        outputs["selected_dataset"],
        outputs["cluster_plot"]
    ):
        print("Clustering outputs already exist; skipping this step.")
    else:
        cluster_molecules(
            outputs["filtered_dataset"],
            outputs["clustered_dataset"],
            selected_output_file=outputs["selected_dataset"],
            plot_file=outputs["cluster_plot"],
            similarity_threshold=clustering["similarity_threshold"],
            max_cluster_size=clustering["max_cluster_size"]
        )

    print("\n=== 4) Ligand preparation (3D) ===")
    if result_exists(outputs["prepared_ligands"]):
        print(f"{outputs['prepared_ligands']} already exists; skipping this step.")
    else:
        prepare_ligands(outputs["selected_dataset"], outputs["prepared_ligands"])

    print("\n=== 5) Ligand conversion to PDBQT ===")
    if result_exists(outputs["pdbqt_ligands_dir"]):
        print(f"{outputs['pdbqt_ligands_dir']} already exists; skipping this step.")
    else:
        sdf_to_pdbqt(
            input_sdf=outputs["prepared_ligands"],
            output_dir=str(RESULTS / outputs["pdbqt_ligands_dir"]),
            obabel_path=tools["obabel"]
        )

    # -----------------------------------------------------------
    # Steps 6 and 7 require Biopython, PDBFixer and OpenMM, which
    # are installed only in docking_env. They are run as separate
    # processes instead of being imported into this environment.
    # -----------------------------------------------------------

    print("\n=== 6) Protein cleaning ===")
    if result_exists(outputs["clean_protein"]):
        print(f"{outputs['clean_protein']} already exists; skipping this step.")
    else:
        run_in_docking_env(
            config,
            "protein_clean.py",
            "--input", paths["protein_file"],
            "--output", outputs["clean_protein"],
            "--ligand-resname", protein["ligand_resname"],
            "--padding", str(protein["padding_angstroms"]),
            "--binding-site-output", outputs["binding_site"]
        )

    print("\n=== 7) Protein repair ===")
    if result_exists(outputs["fixed_protein"]):
        print(f"{outputs['fixed_protein']} already exists; skipping this step.")
    else:
        run_in_docking_env(
            config,
            "protein_fix.py",
            "--input", outputs["clean_protein"],
            "--output", outputs["fixed_protein"],
            "--ph", str(protein["hydrogenation_ph"])
        )

    # -----------------------------------------------------------
    # Step 8 uses ADFRsuite (prepare_receptor), an external
    # program rather than a Python library. It does not need
    # docking_env.
    # -----------------------------------------------------------

    print("\n=== 8) Protein conversion to PDBQT ===")
    if result_exists(outputs["receptor_pdbqt"]):
        print(f"{outputs['receptor_pdbqt']} already exists; skipping this step.")
    else:
        prepare_receptor(
            str(RESULTS / outputs["fixed_protein"]),
            str(RESULTS / outputs["receptor_pdbqt"]),
            prepare_receptor_command=tools["prepare_receptor"]
        )

    print("\n=== 9) Docking with Vina ===")
    if result_exists(outputs["docking_results"]):
        print(f"{outputs['docking_results']} already exists; skipping this step.")
    else:
        run_docking(config)

    print("\n=== 10) Run summary ===")
    generate_pipeline_summary(config)

    print("\nPipeline complete. Protein and ligands are already docked.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help="YAML configuration file to use."
    )
    args = parser.parse_args()

    try:
        run_pipeline(load_config(args.config))
    except Exception as error:
        print(f"\nThe pipeline stopped because this step failed:\n{error}")
        print("\nFull technical traceback:")
        traceback.print_exc()
        sys.exit(1)
