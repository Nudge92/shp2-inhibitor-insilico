import os, glob
from plip.structure.preparation import PDBComplex

POCKET = ["32","42","55","35","34","36","53"]

def count_bonds(work):
    prot = f"{work}/protein_fixed.pdb"
    lig  = f"{work}/ligand_pose.pdb"
    if not (os.path.exists(prot) and os.path.exists(lig)):
        return None
    complex_pdb = f"{work}/_complex_tmp.pdb"
    with open(complex_pdb, "w") as out:
        for fn in [prot, lig]:
            for line in open(fn):
                if line.startswith(("ATOM", "HETATM")):
                    out.write(line)
        out.write("END\n")
    try:
        mol = PDBComplex()
        mol.load_pdb(complex_pdb)
        mol.analyze()
    except Exception as e:
        return f"ERROR: {e}"
    res = {}
    for key, site in mol.interaction_sets.items():
        for it in site.hbonds_pdon + site.hbonds_ldon:
            rn = f"{it.restype}{it.resnr}"; res[rn] = res.get(rn,0)+1
        for it in site.saltbridge_lneg + site.saltbridge_pneg:
            rn = f"{it.restype}{it.resnr}"; res[rn] = res.get(rn,0)+1
        for it in site.hydrophobic_contacts:
            rn = f"{it.restype}{it.resnr}"; res[rn] = res.get(rn,0)+1
    return res

for tag in ["md_runs_4class", "md_runs_3pt"]:
    print(f"\n===== {tag} =====")
    for work in sorted(glob.glob(f"{tag}/*/")):
        name = os.path.basename(work.rstrip("/"))
        b = count_bonds(work.rstrip("/"))
        if b is None:
            continue
        if isinstance(b, str):
            print(f"{name:22} {b}")
            continue
        tot = sum(b.values())
        pocket = {k:v for k,v in b.items() if any(r in k for r in POCKET)}
        print(f"{name:22} total={tot:>2}  pocket={pocket}")
