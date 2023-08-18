"""Molecule visualization with RDKit"""
import py3Dmol
import streamlit as st

# from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from stmol import showmol

from thesis_work.chemberta.model_descriptors import (
    get_model_descriptor,
    initialize_model_tokenizer,
)
from thesis_work.utils.utils import get_ecfp_descriptor, is_valid_smiles

CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def create_static_visualization(
    compound_smiles: str = CAFFEINE_SMILES,
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


def render_mol(xyz, height: int = 600, width: int = 600):
    """Render molecule from xyz string."""
    xyzview = py3Dmol.view()  # (width=600,height=600)
    xyzview.addModel(xyz, "mol")  # pdb, sdf, xyz
    xyzview.setStyle({"stick": {}})
    xyzview.setBackgroundColor("white")
    xyzview.zoomTo()
    showmol(xyzview, height=height, width=width)


def create_dynamic_visualization(
    compound_smiles: str = CAFFEINE_SMILES,
) -> None:
    """Create interactive molecule visualization from SMILES string."""
    blk = makeblock(compound_smiles)
    render_mol(blk)


def create_model_attention_visualization(
    compound_smiles: str = CAFFEINE_SMILES,
) -> None:
    pass


if __name__ == "__main__":
    st.set_page_config(page_title="Model Visualization", layout="wide")

    default_smiles = CAFFEINE_SMILES
    compound_smiles = st.text_input("SMILES string", default_smiles)

    if not is_valid_smiles(compound_smiles):
        st.error("Invalid SMILES string")
        st.stop()

    # visualization_choice = st.radio("Visualization type", ("Static", "Dynamic"))
    visualization_choice = st.selectbox(
        "Visualization type", (None, "Static", "Dynamic", "Model.Attention")
    )

    if not visualization_choice:
        st.stop()

    if visualization_choice == "Static":
        create_static_visualization(compound_smiles=compound_smiles)
    elif visualization_choice == "Dynamic":
        create_dynamic_visualization(compound_smiles=compound_smiles)
    elif visualization_choice == "Model.Attention":
        create_model_attention_visualization(compound_smiles=compound_smiles)

    if visualization_choice != "Model.Attention":
        st.stop()

    descriptor_choice = st.selectbox("Descriptor type", (None, "ECFP", "ChemBERTa"))

    if not descriptor_choice:
        st.stop()

    if descriptor_choice == "ECFP":
        ecfp_descriptor = get_ecfp_descriptor(
            smiles_str=compound_smiles, return_type="numpy"
        )
        st.write(ecfp_descriptor)
    elif descriptor_choice == "ChemBERTa":
        model_name = "DeepChem/ChemBERTa-77M-MLM"

        # TODO: Cache
        model, tokenizer = initialize_model_tokenizer(model_name=model_name)

        model_descriptor = get_model_descriptor(
            model=model, tokenizer=tokenizer, smiles_str=compound_smiles
        )
        st.write(model_descriptor)
