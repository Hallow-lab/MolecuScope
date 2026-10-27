from __future__ import annotations

import json
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

st.set_page_config(page_title="MolecuScope | Molecular Explorer", page_icon="🧪", layout="wide")

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

STARTER_MOLECULES = [
    {"name": "Ethanol", "smiles": "CCO", "category": "Alcohol"},
    {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "category": "Drug / analgesic"},
    {"name": "Benzene", "smiles": "c1ccccc1", "category": "Aromatic hydrocarbon"},
    {"name": "Acetic acid", "smiles": "CC(=O)O", "category": "Carboxylic acid"},
    {"name": "Phenol", "smiles": "Oc1ccccc1", "category": "Phenol"},
    {"name": "Caffeine", "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "category": "Alkaloid"},
    {"name": "Glucose", "smiles": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "category": "Carbohydrate"},
    {"name": "Paracetamol", "smiles": "CC(=O)NC1=CC=C(C=C1)O", "category": "Drug / analgesic"},
    {"name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "category": "Drug / NSAID"},
    {"name": "Nicotine", "smiles": "CN1CCC[C@H]1c2cccnc2", "category": "Alkaloid"},
    {"name": "Lactic acid", "smiles": "C[C@@H](O)C(=O)O", "category": "Organic acid"},
    {"name": "Urea", "smiles": "NC(=O)N", "category": "Organic compound"},
    {"name": "Glycine", "smiles": "NCC(=O)O", "category": "Amino acid"},
    {"name": "Acetone", "smiles": "CC(=O)C", "category": "Ketone"},
    {"name": "Citric acid", "smiles": "O=C(O)CC(O)(CC(=O)O)C(=O)O", "category": "Organic acid"},
    {"name": "Dopamine", "smiles": "NCCc1ccc(O)c(O)c1", "category": "Biogenic amine"},
]

STYLE_OPTIONS = ["Ball and Stick", "Space Filling", "Wedge and Dash", "Stick", "Wireframe", "Cartoon"]
STYLE_CONFIG: dict[str, dict[str, Any]] = {
    "Ball and Stick": {"stick": {"radius": 0.16}, "sphere": {"scale": 0.25}},
    "Space Filling": {"sphere": {"scale": 1.0}},
    "Wedge and Dash": {"stick": {"radius": 0.20}},
    "Stick": {"stick": {"radius": 0.22}},
    "Wireframe": {"line": {"linewidth": 2}},
    "Cartoon": {"stick": {"radius": 0.16}, "sphere": {"scale": 0.25}},
}


def safe_json(value: Any) -> str:
    """JSON-encode data for inline JavaScript without closing its script tag."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


@st.cache_data(show_spinner=False, ttl=3600)
def pubchem_name_to_smiles(name: str) -> tuple[str, str]:
    clean_name = str(name).strip()
    if not clean_name:
        raise ValueError("Enter a molecule name to search PubChem.")
    encoded_name = requests.utils.quote(clean_name, safe="")
    # Try current PubChem SMILES property names, with a fallback for older responses.
    property_sets = [
        "CanonicalSMILES,IsomericSMILES",
        "ConnectivitySMILES,SMILES",
    ]
    last_message = "PubChem did not return a usable SMILES string."
    for props in property_sets:
        url = f"{PUBCHEM}/compound/name/{encoded_name}/property/{props}/JSON"
        try:
            response = requests.get(url, timeout=20)
        except requests.RequestException as exc:
            raise ValueError(f"Could not connect to PubChem: {exc}") from exc
        if response.status_code != 200:
            last_message = f"PubChem could not find '{clean_name}'. Try another name or enter SMILES directly."
            continue
        try:
            records = response.json().get("PropertyTable", {}).get("Properties", [])
        except (ValueError, AttributeError):
            records = []
        if records:
            record = records[0]
            smiles = next((record.get(key) for key in ("IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES", "SMILES") if record.get(key)), None)
            if smiles and isinstance(smiles, str):
                title = str(record.get("Title") or clean_name)
                return title, smiles
    raise ValueError(last_message)


@st.cache_data(show_spinner=False, max_entries=256)
def analyze_smiles(smiles_input: str) -> dict[str, Any]:
    """Parse a SMILES string, calculate descriptors, and create a 3D MOL block.

    Intentionally does not import rdkit.Chem.Draw, py3Dmol, or stmol. This avoids
    the native drawing dependency and Jupyter-widget import failures seen in the
    previous deployment.
    """
    smiles_input = str(smiles_input).strip()
    if not smiles_input:
        raise ValueError("Enter a SMILES string.")
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        raise ValueError("RDKit could not parse this SMILES. Check the spelling and bonding notation.")

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    # Generate a 3D conformer. Hydrogen atoms improve the visual representation.
    mol_3d = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    embed_status = AllChem.EmbedMolecule(mol_3d, params)
    if embed_status != 0:
        # One deterministic retry with random coordinates for difficult structures.
        params.useRandomCoords = True
        embed_status = AllChem.EmbedMolecule(mol_3d, params)
    if embed_status != 0:
        raise ValueError("A 3D conformer could not be generated for this molecule. Try a different SMILES.")

    optimization = "Generated coordinates; not optimized"
    if AllChem.MMFFHasAllMoleculeParams(mol_3d):
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
        optimization = "MMFF force-field optimization attempted"
    elif AllChem.UFFHasAllMoleculeParams(mol_3d):
        AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
        optimization = "UFF force-field optimization attempted"

    properties = {
        "Molecular formula": formula,
        "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 4),
        "Exact mass (Da)": round(Descriptors.ExactMolWt(mol), 5),
        "LogP estimate": round(Descriptors.MolLogP(mol), 4),
        "Topological polar surface area (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "Hydrogen-bond donors": int(rdMolDescriptors.CalcNumHBD(mol)),
        "Hydrogen-bond acceptors": int(rdMolDescriptors.CalcNumHBA(mol)),
        "Rotatable bonds": int(Lipinski.NumRotatableBonds(mol)),
        "Ring count": int(rdMolDescriptors.CalcNumRings(mol)),
        "Heavy atoms": int(mol.GetNumHeavyAtoms()),
        "Total atoms (without added 3D hydrogens)": int(mol.GetNumAtoms()),
        "Formal charge": int(Chem.GetFormalCharge(mol)),
        "Fraction Csp³": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
    }
    atoms = [
        {
            "Atom": atom.GetIdx(),
            "Element": atom.GetSymbol(),
            "Formal charge": atom.GetFormalCharge(),
            "Degree": atom.GetDegree(),
            "Aromatic": atom.GetIsAromatic(),
            "In ring": atom.IsInRing(),
            "Chiral tag": str(atom.GetChiralTag()),
        }
        for atom in mol.GetAtoms()
    ]
    return {
        "canonical_smiles": canonical,
        "formula": formula,
        "properties": properties,
        "atoms": atoms,
        "mol_block": Chem.MolToMolBlock(mol_3d),
        "optimization": optimization,
        "input_smiles": smiles_input,
    }


def render_2d(smiles: str, element_id: str = "molecule-svg") -> str:
    smiles_js = safe_json(str(smiles))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{margin:0;background:#fff;font-family:Arial,sans-serif}} .card{{width:100%;min-height:440px;display:flex;align-items:center;justify-content:center}} svg{{max-width:100%;height:auto}} .msg{{color:#a33;padding:16px;text-align:center}}</style></head>
<body><div class="card"><svg id="{element_id}" width="680" height="440" viewBox="0 0 680 440" aria-label="2D molecular structure"></svg></div>
<div id="error" class="msg"></div>
<script src="https://unpkg.com/smiles-drawer@2.1.7/dist/smiles-drawer.min.js"></script>
<script>
(function(){{
 const smiles = {smiles_js};
 const err = document.getElementById('error');
 if (!window.SmilesDrawer) {{ err.textContent='The 2D renderer could not load. Check your internet connection, then reload.'; return; }}
 const drawer = new SmilesDrawer.SmiDrawer({{width:680,height:440, bondThickness:1.4, fontSize:16, padding:24, compactDrawing:false}});
 drawer.draw(smiles, '#{element_id}', 'light', function(){{}}, function(e){{err.textContent='Could not draw this SMILES in the 2D viewer.';}});
}})();
</script></body></html>"""


def render_3d(mol_block: str, style: str) -> str:
    mol_js = safe_json(str(mol_block))
    if style == "Wedge and Dash":
        effective_style = "Stick"
        notice = "Wedge-and-dash is a 2D stereochemical convention; this 3D view uses Stick. Use the 2D depiction to inspect stereochemistry."
    elif style == "Cartoon":
        effective_style = "Ball and Stick"
        notice = "Cartoon is intended for protein backbones, not small molecules; this view uses Ball and Stick."
    else:
        effective_style = style
        notice = ""
    style_js = safe_json(STYLE_CONFIG[effective_style])
    notice_js = safe_json(notice)
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body{{margin:0;padding:0;background:#fff;font-family:Arial,sans-serif}} #viewer{{height:520px;width:100%;position:relative;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden}} #notice{{font-size:13px;color:#555;padding:6px 2px}} #error{{color:#a33;padding:18px}}</style></head>
<body><div id="viewer"></div><div id="notice"></div><div id="error"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script><script>
(function(){{
 const molData = {mol_js};
 document.getElementById('notice').textContent = {notice_js};
 const errorBox=document.getElementById('error');
 if (!window.$3Dmol) {{ errorBox.textContent='The 3D viewer could not load. Check your internet connection, then reload.'; return; }}
 try {{
   const el=document.getElementById('viewer');
   const viewer=$3Dmol.createViewer(el, {{backgroundColor:'white', antialias:true}});
   viewer.addModel(molData, 'mol');
   viewer.setStyle({{}}, {style_js});
   viewer.zoomTo(); viewer.render();
   window.addEventListener('resize', function(){{viewer.resize();}});
 }} catch(e) {{ errorBox.textContent='The 3D viewer could not display this molecule.'; }}
}})();
</script></body></html>"""


def set_selected(name: str, smiles: str, category: str = "User input") -> None:
    st.session_state["current_molecule"] = {"name": str(name), "smiles": str(smiles), "category": str(category)}


if "current_molecule" not in st.session_state:
    st.session_state.current_molecule = {
        "name": "Ethanol", "smiles": "CCO", "category": "Alcohol"
    }

st.title("🧪 MolecuScope")
st.caption("Explore molecules. Visualize chemistry. Browse common compounds or inspect a molecule by name or SMILES, with 2D and interactive 3D views, calculated properties, and downloads.")

with st.sidebar:
    st.header("Find a molecule")
    input_mode = st.radio("Input method", ["Molecule library", "Search PubChem by name", "Enter SMILES", "Upload CSV"])
    if input_mode == "Molecule library":
        filter_text = st.text_input("Filter library", placeholder="e.g. caffeine, acid, drug")
        filtered = [row for row in STARTER_MOLECULES if filter_text.lower() in f"{row['name']} {row['category']} {row['smiles']}".lower()]
        if not filtered:
            st.warning("No library molecules match that filter.")
        else:
            choice = st.selectbox("Choose molecule", filtered, format_func=lambda row: row["name"])
            if st.button("Open molecule", type="primary", use_container_width=True):
                set_selected(choice["name"], choice["smiles"], choice["category"])
    elif input_mode == "Search PubChem by name":
        pubchem_query = st.text_input("Molecule name", placeholder="e.g. acetaminophen")
        if st.button("Search and open", type="primary", use_container_width=True):
            try:
                with st.spinner("Searching PubChem..."):
                    found_name, found_smiles = pubchem_name_to_smiles(pubchem_query)
                    # Parse before storing the result to avoid saving a non-molecule response.
                    if Chem.MolFromSmiles(found_smiles) is None:
                        raise ValueError("PubChem returned a SMILES string that RDKit could not parse.")
                    set_selected(found_name or pubchem_query, found_smiles, "PubChem result")
            except Exception as exc:
                st.error(str(exc))
    elif input_mode == "Enter SMILES":
        smiles_query = st.text_area("SMILES", value="CCO", height=90, help="Example: CCO for ethanol, c1ccccc1 for benzene.")
        molecule_name = st.text_input("Display name (optional)", value="Custom molecule")
        if st.button("Analyze SMILES", type="primary", use_container_width=True):
            try:
                if Chem.MolFromSmiles(smiles_query.strip()) is None:
                    raise ValueError("RDKit could not parse this SMILES. Check the string and try again.")
                set_selected(molecule_name.strip() or "Custom molecule", smiles_query.strip(), "User SMILES")
            except Exception as exc:
                st.error(str(exc))
    else:
        uploaded = st.file_uploader("Upload a CSV", type=["csv"], help="Required column: smiles. Optional columns: name, category.")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                normalized = {str(c).strip().lower(): c for c in df.columns}
                if "smiles" not in normalized:
                    st.error("Your CSV needs a column named 'smiles'.")
                else:
                    smiles_col = normalized["smiles"]
                    name_col = normalized.get("name")
                    category_col = normalized.get("category")
                    valid_rows = []
                    invalid_count = 0
                    for i, row in df.iterrows():
                        raw = row.get(smiles_col, "")
                        if pd.isna(raw) or not str(raw).strip() or Chem.MolFromSmiles(str(raw).strip()) is None:
                            invalid_count += 1
                            continue
                        valid_rows.append({
                            "name": str(row.get(name_col, f"Molecule {i+1}")) if name_col else f"Molecule {i+1}",
                            "smiles": str(raw).strip(),
                            "category": str(row.get(category_col, "Uploaded CSV")) if category_col else "Uploaded CSV",
                        })
                    if invalid_count:
                        st.warning(f"Skipped {invalid_count} row(s) with missing or invalid SMILES.")
                    if valid_rows:
                        csv_choice = st.selectbox("Choose uploaded molecule", valid_rows, format_func=lambda row: row["name"])
                        if st.button("Open uploaded molecule", type="primary", use_container_width=True):
                            set_selected(csv_choice["name"], csv_choice["smiles"], csv_choice["category"])
                    else:
                        st.info("No valid molecules were found in this CSV.")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

    st.divider()
    st.caption("2D structures use SmilesDrawer in the browser. The 3D viewer uses 3Dmol.js. No Jupyter widgets or native RDKit drawing module are required.")

current = st.session_state.current_molecule
try:
    result = analyze_smiles(current["smiles"])
except Exception as exc:
    st.error(f"Could not analyze {current['name']}: {exc}")
    st.stop()

st.subheader(current["name"])
st.caption(f"Category: {current['category']}")
st.code(result["canonical_smiles"], language=None)

props = result["properties"]
metric_cols = st.columns(4)
metric_cols[0].metric("Molecular formula", result["formula"])
metric_cols[1].metric("Molecular weight", f"{props['Molecular weight (g/mol)']:.3f} g/mol")
metric_cols[2].metric("Exact mass", f"{props['Exact mass (Da)']:.4f} Da")
metric_cols[3].metric("Heavy atoms", str(props["Heavy atoms"]))

tab_2d, tab_3d, tab_props, tab_atoms, tab_download = st.tabs(["🧬 2D Structure", "🧊 3D Viewer", "📊 Properties", "⚛️ Atoms", "⬇️ Export"])
with tab_2d:
    st.markdown("**2D molecular structure**")
    components.html(render_2d(result["canonical_smiles"]), height=475, scrolling=False)
    st.markdown("**Canonical isomeric SMILES**")
    st.code(result["canonical_smiles"], language=None)
    st.caption("The 2D drawing is generated from the SMILES string in your browser. Stereochemistry is determined from the structure provided.")
with tab_3d:
    style = st.selectbox("3D visualization style", STYLE_OPTIONS, index=0, key="3d_style")
    if style == "Wedge and Dash":
        st.info("Wedge-and-dash is a 2D stereochemical convention, not a native 3Dmol.js molecular style. The 3D viewer uses Stick; inspect the 2D structure for stereochemical notation.")
    elif style == "Cartoon":
        st.info("Cartoon is designed for protein backbones and is not appropriate for a small molecule. The 3D viewer uses Ball and Stick.")
    st.caption("Drag to rotate and use the mouse wheel or trackpad gesture to zoom. Atom-index labels are intentionally disabled. Coordinates are computationally generated, not experimental measurements.")
    components.html(render_3d(result["mol_block"], style), height=575, scrolling=False)
    st.caption(f"Conformer generation: {result['optimization']}.")
with tab_props:
    st.dataframe(pd.DataFrame([{"Property": key, "Value": value} for key, value in props.items()]), use_container_width=True, hide_index=True)
    st.info("LogP and the other calculated descriptors are estimates derived from the structure. They are not experimental measurements.")
with tab_atoms:
    st.dataframe(pd.DataFrame(result["atoms"]), use_container_width=True, hide_index=True)
with tab_download:
    st.download_button("Download 3D structure (.mol)", data=result["mol_block"], file_name=f"{current['name'].replace(' ', '_')}.mol", mime="chemical/x-mdl-molfile")
    st.download_button("Download canonical SMILES (.smi)", data=f"{result['canonical_smiles']}\t{current['name']}\n", file_name=f"{current['name'].replace(' ', '_')}.smi", mime="text/plain")
    report_lines = [f"Name: {current['name']}", f"Category: {current['category']}", f"Input SMILES: {current['smiles']}", f"Canonical SMILES: {result['canonical_smiles']}"] + [f"{key}: {value}" for key, value in props.items()]
    st.download_button("Download properties report (.txt)", data="\n".join(report_lines), file_name=f"{current['name'].replace(' ', '_')}_report.txt", mime="text/plain")

with st.expander("Browse the starter library"):
    st.dataframe(pd.DataFrame(STARTER_MOLECULES), use_container_width=True, hide_index=True)

st.divider()
st.caption("MolecuScope • Explore Molecules. Visualize Chemistry. For educational use; validate structures and calculated properties before research or clinical use.")
