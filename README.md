# MolecuScope — Molecular Explorer

**Explore Molecules. Visualize Chemistry.**

MolecuScope is a Streamlit cheminformatics app for exploring common molecules, looking up compound names through PubChem, entering SMILES, and importing a CSV molecule list.

## Included files

- `app.py` — complete Streamlit app
- `requirements.txt` — Python dependencies
- `runtime.txt` — Python runtime setting for Streamlit Community Cloud
- `molecules.csv` — sample molecule library for CSV import

## Features

- Searchable starter molecule library
- PubChem compound-name lookup
- Direct SMILES input with RDKit validation
- CSV upload with required `smiles` and optional `name` and `category` columns
- Browser-based 2D structure rendering with SmilesDrawer
- Interactive 3D molecular viewer with 3Dmol.js
- Molecular formula, molecular weight, exact mass, LogP estimate, TPSA, H-bond donors/acceptors, rotatable bonds, ring count, atom count, formal charge, and fraction Csp3
- Atom table
- Downloadable 3D MOL file, canonical SMILES, and property report

## Design notes

This version deliberately does **not** import `rdkit.Chem.Draw`, `stmol`, `py3Dmol`, or `ipython_genutils`. RDKit is used to parse SMILES, calculate descriptors, and generate 3D coordinates. The 2D and 3D visualizers are JavaScript libraries loaded in the browser, so an internet connection is required for those visual elements.

## Deploy on Streamlit Community Cloud

1. Download and extract the project ZIP.
2. In your GitHub repository, upload all five project files to the repository root and commit them to the `main` branch. If you are reusing an old repository, replace its `app.py`, `requirements.txt`, and `runtime.txt` with these files. Upload `molecules.csv` and `README.md` too.
3. Remove the old `packages.txt` from the GitHub repository if it exists; this version does not require it.
4. In Streamlit Community Cloud, set the main file to `app.py` and use Python 3.12 if the setting is available.
5. Reboot/redeploy the app so the dependencies are rebuilt from the new `requirements.txt`.

The suggested GitHub repository name is `molecuscope`. The public Streamlit URL is assigned by Streamlit and must be checked when creating the deployment; availability is not guaranteed.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## CSV format

```csv
name,smiles,category
Ethanol,CCO,Alcohol
Benzene,c1ccccc1,Aromatic hydrocarbon
```

For educational use. Calculated properties and conformer coordinates are estimates, not experimental measurements.
