import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser


def is_valid_smiles(smiles: str) -> bool:
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            return True
        else:
            return False
    except Exception:
        return False


def get_ecfp_descriptor(smiles_str: str, nBits: int = 1024):
    if not is_valid_smiles(smiles_str):
        raise ValueError("Invalid SMILES string")

    mol = Chem.MolFromSmiles(smiles_str)
    fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
    # fp = np.array(fp)

    return fp


def get_largest_fragment_from_smiles(s: str):
    """Returns the largest fragment of a SMILES string

    NOTE:
        - Not used right now!
        - From https://github.com/seyonechithrananda/bert-loves-chemistry/blob/master/chemberta/visualization/viz_utils.py
        - Can be used to remove extra fragments in SMILES (typically salts, which are irrelevant to BBB permeability)
    """
    mol = Chem.MolFromSmiles(s)
    if mol:
        clean_mol = LargestFragmentChooser().choose(mol)
        return Chem.MolToSmiles(clean_mol)
    return None


def plot_global_embeddings_with_clusters(  # noqa: PLR0913
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str,
    title: str = "",
    x_lim=None,
    y_lim=None,
):
    """Plots data colored by soft HDBSCAN clusters

    If the cluster of a compund is not specified (cluster < 0), it will be
    plotted gray, otherwise it will be colored by the cluster value.

    NOTE:
        - Not used right now!
        - From https://github.com/seyonechithrananda/bert-loves-chemistry/blob/master/chemberta/visualization/viz_utils.py
    """
    clustered = df[cluster_col].values >= 0

    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(
        data=df.iloc[~clustered],
        x=x_col,
        y=y_col,
        color=(0.5, 0.5, 0.5),
        s=10,
        alpha=0.1,
    )
    sns.scatterplot(
        data=df.iloc[clustered],
        x=x_col,
        y=y_col,
        hue=cluster_col,
        alpha=0.5,
        palette="nipy_spectral",
        ax=ax,
    )
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)

    sm = plt.cm.ScalarMappable(cmap="nipy_spectral")
    sm.set_array([])
    ax.get_legend().remove()
    ax.figure.colorbar(sm, label="Global Cluster")

    plt.title(title)
    plt.show()
