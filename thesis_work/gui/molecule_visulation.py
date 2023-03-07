"""Molecule visualization using RDKit and Streamlit."""
import py3Dmol
import streamlit as st

# from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from stmol import showmol


def create_static_visualization(
    compound_smiles: str = "c1cc(C(=O)O)c(OC(=O)C)cc1",
) -> None:
    """Create static image of molecule from SMILES string."""
    m = Chem.MolFromSmiles(compound_smiles)
    im = Draw.MolToImage(m)
    st.image(im)

    # Draw.MolToFile(m, "mol.png")
    # st.image("mol.png")


def makeblock(smi: str):
    """Create molecule block from SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)

    return mblock


def render_mol(xyz):
    """Render molecule from xyz string."""
    xyzview = py3Dmol.view()  # (width=400,height=400)
    xyzview.addModel(xyz, "mol")  # pdb, sdf, xyz
    xyzview.setStyle({"stick": {}})
    xyzview.setBackgroundColor("white")
    xyzview.zoomTo()
    showmol(xyzview, height=500, width=500)


def create_dynamic_visualization(
    compound_smiles: str = "c1cc(C(=O)O)c(OC(=O)C)cc1",
) -> None:
    """Create interactive molecule visualization from SMILES string."""
    blk = makeblock(compound_smiles)
    render_mol(blk)


if __name__ == "__main__":
    default_smiles = "CC"
    default_smiles: str = "c1cc(C(=O)O)c(OC(=O)C)cc1"
    compound_smiles = st.text_input("SMILES please", default_smiles)
    # create_static_visualization(compound_smiles=compound_smiles)
    create_dynamic_visualization(compound_smiles=compound_smiles)
