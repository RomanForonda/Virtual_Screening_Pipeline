from Bio.PDB import PDBParser, PDBIO, Select
import os


# Three-letter residue name of the original co-crystallized ligand.
# It is used to calculate the docking box center and size.
# Change this value when using a different protein-ligand complex.
LIGAND_RESNAME = "AQ4"

# Extra margin, in Angstroms, added around the original ligand to define
# the Vina search box size. Increase it for a larger box; decrease it for
# a tighter and more focused search region.
PADDING_ANGSTROMS = 8.0


class ProteinSelect(Select):
    """
    Keep only standard protein residues.
    Remove waters and heteroatoms.
    """

    def accept_residue(self, residue):
        return residue.id[0] == " "


def save_docking_box(
    structure,
    output_path,
    ligand_resname=LIGAND_RESNAME,
    padding_angstroms=PADDING_ANGSTROMS
):
    """
    Find the original co-crystallized ligand, calculate
    the docking box center and size around it, and save the values to a
    text file for Vina.
    """

    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == ligand_resname:
                    for atom in residue:
                        coords.append(atom.get_coord())

    if not coords:
        print(
            f"Warning: ligand '{ligand_resname}' was not found in the "
            "structure; the docking box file was not generated."
        )
        return

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    center_z = sum(zs) / len(zs)

    size_x = (max(xs) - min(xs)) + 2 * padding_angstroms
    size_y = (max(ys) - min(ys)) + 2 * padding_angstroms
    size_z = (max(zs) - min(zs)) + 2 * padding_angstroms

    with open(output_path, "w") as f:
        f.write(f"center_x = {center_x:.3f}\n")
        f.write(f"center_y = {center_y:.3f}\n")
        f.write(f"center_z = {center_z:.3f}\n")
        f.write(f"size_x = {size_x:.3f}\n")
        f.write(f"size_y = {size_y:.3f}\n")
        f.write(f"size_z = {size_z:.3f}\n")

    print(f"Binding site saved to: {output_path}")


def clean_protein(
    input_file,
    output_file,
    ligand_resname=LIGAND_RESNAME,
    padding_angstroms=PADDING_ANGSTROMS,
    binding_site_file="binding_site.txt"
):

    input_path = os.path.join("data", input_file)
    output_path = os.path.join("results", output_file)

    os.makedirs("results", exist_ok=True)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_path)

    binding_site_path = os.path.join("results", binding_site_file)
    save_docking_box(
        structure,
        binding_site_path,
        ligand_resname=ligand_resname,
        padding_angstroms=padding_angstroms
    )

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, ProteinSelect())

    print("Clean protein saved to:", output_path)

    return output_path


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser()
    cli.add_argument("--input", required=True, help="Input PDB filename inside data/")
    cli.add_argument("--output", required=True, help="Output PDB filename inside results/")
    cli.add_argument("--ligand-resname", default=LIGAND_RESNAME)
    cli.add_argument("--padding", type=float, default=PADDING_ANGSTROMS)
    cli.add_argument("--binding-site-output", default="binding_site.txt")
    args = cli.parse_args()

    clean_protein(
        args.input,
        args.output,
        ligand_resname=args.ligand_resname,
        padding_angstroms=args.padding,
        binding_site_file=args.binding_site_output
    )
