"""Molecule clustering"""
import streamlit as st

if __name__ == "__main__":
    st.set_page_config(page_title="Compound Clustering", layout="wide")

    visualization_choice = st.selectbox(
        "CLustering type", (None, "K.Means", "UMAP", "T.SNE")
    )

    if not visualization_choice:
        st.stop()
