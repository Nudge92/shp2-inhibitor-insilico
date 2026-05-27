#!/usr/bin/env python3
"""
md_stability_v2.py  --  설폰아미드/카복실산 4후보 x 50ns MD (restraint equil 개선판)

개선점 (v1 대비):
  - position restraint 를 단백질+리간드 heavy atom 에 걸고 단계적으로 풀며 equilibration
    (리간드가 docked pose 에서 초기에 튕겨나가는 것 방지 -> 공정한 안정성 평가)
  - equilibration 궤적도 저장 (초반 거동 확인)
  - 시작 거리를 docked pose 에 최대한 가깝게 유지

run (md env):
  conda run -n md python3 md_stability_v2.py > md_v2.log 2>&1 &
"""
import time, json
from pathlib import Path
T0=time.time()
def stamp(m): print(f"[{time.time()-T0:8.1f}s] {m}", flush=True)

# ---------------- CONFIG (확인 후 채움) ----------------
CANDIDATES = [
    # (name, complex_pdb, smiles)
    ("sulfon_1", "./runs/gen100k/dock/strict_work/plip/complex_1412.pdb", "Cn1nc(NCc2ccc(S(N)(=O)=O)cc2)c2ccccc21"),
    ("sulfon_2", "./runs/gen100k/dock/strict_work/plip/complex_1062.pdb", "CC(C)n1c(=O)n(Cc2ccc(S(N)(=O)=O)cc2)c2ccccc21"),
    ("sulfon_3", "./runs/gen100k/dock/strict_work/plip/complex_1055.pdb", "Cc1cccc(Nc2nccnc2-c2ccc(S(N)(=O)=O)cc2)c1"),
    ("hydrox_1", "./runs/gen100k/dock/strict_work/plip/complex_3156.pdb", "O=C(NO)C1CC(n2cnc3ccccc32)CCN1S(=O)(=O)c1ccccc1"),
    ("hydrox_2", "./runs/gen100k/dock/strict_work/plip/complex_3844.pdb", "CN(C)c1ccc(-c2ccc(N=C3SCC(=O)N3CC(=O)NO)s2)cc1"),
    ("hydrox_3", "./runs/gen100k/dock/strict_work/plip/complex_3624.pdb", "Cc1cccc(C=Cc2nnc(C3(C(=O)NO)CCN(C)CC3)o2)c1"),
    ("tetraz_1", "./runs/gen100k/dock/strict_work/plip/complex_4818.pdb", "CCCCCc1ccnc(NC(=O)c2nnn[nH]2)n1"),
    ("tetraz_2", "./runs/gen100k/dock/strict_work/plip/complex_4653.pdb", "CC(=O)c1cc(C(C)c2nnn[nH]2)ccc1OC(C)(C)C"),
    ("tetraz_3", "./runs/gen100k/dock/strict_work/plip/complex_4602.pdb", "O=C(O)c1cccnc1Oc1cc(Br)ccc1-c1nnn[nH]1"),
    ("carbox_1", "./runs/gen100k/dock/strict_work/plip/complex_2712.pdb", "Cc1cc(OCCCc2ccncc2)ccc1OC(=O)N1CCCC1C(=O)O"),
    ("carbox_2", "./runs/gen100k/dock/strict_work/plip/complex_2439.pdb", "CNCn1c(Cn2cc(C(F)(F)F)nn2)c(C(=O)O)c2c(F)cccc21"),
    ("carbox_3", "./runs/gen100k/dock/strict_work/plip/complex_1626.pdb", "CSc1ccc(CO)c(-c2nc3c(CC(=O)O)ccnc3cc2Cl)c1"),
]
LIG_RESNAME="LIG"
OUT_ROOT=Path("./md_runs_4class")
PROD_NS=30.0
TIMESTEP_FS=2.0
TEMP_K=310.0
REPORT_PS=10.0
SALT_CUT_A=4.5
R32_RESSEQ=32
RESTRAINT_K=10.0   # kcal/mol/A^2 초기 구속 강도
# -------------------------------------------------------

OUT_ROOT.mkdir(exist_ok=True)
stamp("import")
from rdkit import Chem
from rdkit.Chem import AllChem as _AC
import numpy as np
import openmm
from openmm import app, unit, Platform, LangevinMiddleIntegrator, Vec3, CustomExternalForce, MonteCarloBarostat
from openff.toolkit import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from pdbfixer import PDBFixer
import mdtraj as md
stamp("import done")


def prepare_ligand(complex_pdb, smiles, work):
    lig_lines=[l for l in open(complex_pdb)
               if l.startswith("HETATM") and l[17:20].strip()==LIG_RESNAME]
    lig_pdb=work/"ligand_pose.pdb"
    with open(lig_pdb,"w") as f: f.writelines(lig_lines); f.write("END\n")
    pose=Chem.MolFromPDBFile(str(lig_pdb), sanitize=False, removeHs=False,
                             proximityBonding=True)
    tmpl=Chem.MolFromSmiles(smiles)
    pose_bo=_AC.AssignBondOrdersFromTemplate(tmpl, pose)
    pose_h=Chem.AddHs(pose_bo, addCoords=True)
    off=Molecule.from_rdkit(pose_h, allow_undefined_stereo=True)
    off.assign_partial_charges(partial_charge_method="am1bcc")
    return off, pose_h


def add_restraint(system, positions, restrained_idx, k):
    """heavy atom 들을 초기 위치에 구속하는 force 반환 (k 조절 가능)"""
    force=CustomExternalForce("k_rest*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k_rest", k*4.184*100.0)  # kcal/mol/A^2 -> kj/mol/nm^2
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for i in restrained_idx:
        p=positions[i]
        force.addParticle(i, [p.x, p.y, p.z])
    system.addForce(force)
    return force


def run_one(name, complex_pdb, smiles):
    work=OUT_ROOT/name; work.mkdir(exist_ok=True)
    stamp(f"=== {name} ===")
    off, pose_h = prepare_ligand(complex_pdb, smiles, work)

    prot_lines=[l for l in open(complex_pdb) if l.startswith("ATOM")]
    prot_raw=work/"protein_raw.pdb"
    with open(prot_raw,"w") as f: f.writelines(prot_lines); f.write("END\n")
    fixer=PDBFixer(filename=str(prot_raw))
    fixer.findMissingResidues(); fixer.findMissingAtoms()
    fixer.addMissingAtoms(); fixer.addMissingHydrogens(7.4)
    prot_fixed=work/"protein_fixed.pdb"
    with open(prot_fixed,"w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    prot=app.PDBFile(str(prot_fixed))
    n_prot=prot.topology.getNumAtoms()

    gaff=GAFFTemplateGenerator(molecules=off, forcefield="gaff-2.11")
    modeller=app.Modeller(prot.topology, prot.positions)
    lig_omm=off.to_topology().to_openmm()
    lig_pos=pose_h.GetConformer().GetPositions()*0.1
    modeller.add(lig_omm, [Vec3(*xyz) for xyz in lig_pos]*unit.nanometer)
    ff=app.ForceField("amber14-all.xml","amber14/tip3pfb.xml")
    ff.registerTemplateGenerator(gaff.generator)
    modeller.addSolvent(ff, model="tip3p", padding=1.0*unit.nanometer,
                        ionicStrength=0.15*unit.molar)
    n_atoms=modeller.topology.getNumAtoms()
    stamp(f"  시스템 {n_atoms} atoms (단백질 {n_prot})")
    with open(work/"system.pdb","w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    system=ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                           nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds)

    # (restraint 제거 - v1 방식)

    integrator=LangevinMiddleIntegrator(TEMP_K*unit.kelvin, 1.0/unit.picosecond,
                                        TIMESTEP_FS*unit.femtoseconds)
    sim=app.Simulation(modeller.topology, system, integrator,
                       Platform.getPlatformByName("CUDA"))
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy()
    stamp("  최소화 완료")

    sim.context.setVelocitiesToTemperature(TEMP_K*unit.kelvin)
    sim.step(int(100*unit.picoseconds/(TIMESTEP_FS*unit.femtoseconds)))
    stamp("  equilibration 100ps 완료 (restraint 없음)")

    # production
    n_steps=int(PROD_NS*unit.nanoseconds/(TIMESTEP_FS*unit.femtoseconds))
    save_int=int(REPORT_PS*unit.picoseconds/(TIMESTEP_FS*unit.femtoseconds))
    sim.reporters.append(app.DCDReporter(str(work/"prod.dcd"), save_int))
    stamp(f"  production {PROD_NS}ns 시작")
    t=time.time()
    sim.step(n_steps)
    stamp(f"  production 완료 ({time.time()-t:.0f}s)")

    res=analyze(work, n_prot, pose_h.GetNumAtoms())
    res.update(name=name, smiles=smiles, n_atoms=n_atoms)
    json.dump(res, open(work/"stability.json","w"), indent=2)
    stamp(f"  [{name}] occ={res['occupancy_pct']:.0f}% "
          f"start={res['r32_dist_start']:.2f} end={res['r32_dist_end']:.2f} "
          f"min={res['r32_dist_min']:.2f} RMSD={res['lig_rmsd_mean']:.2f}")
    return res


def analyze(work, n_prot, n_lig):
    top=md.load(str(work/"system.pdb")).topology
    traj=md.load(str(work/"prod.dcd"), top=str(work/"system.pdb"))
    r32=[a.index for a in top.atoms if a.residue.resSeq==R32_RESSEQ
         and a.residue.name=="ARG" and a.name in ("NH1","NH2","NE")]
    lig_on=[a.index for a in top.atoms if a.residue.name=="LIG"
            and a.element.symbol in ("O","N")]
    lig_all=[a.index for a in top.atoms if a.residue.name=="LIG"]
    prot_ca=[a.index for a in top.atoms if a.name=="CA"]
    # PBC minimum image 거리: 모든 (lig_on, r32) 원자쌍을 mdtraj 로
    import itertools
    pairs=np.array([[i,j] for i in lig_on for j in r32])
    dd=md.compute_distances(traj, pairs, periodic=True)  # (frames, npairs) nm
    dists=dd.min(axis=1)*10.0  # 각 프레임 최소거리 -> A
    traj.superpose(traj, frame=0, atom_indices=prot_ca)
    lig_rmsd=md.rmsd(traj, traj, frame=0, atom_indices=lig_all)*10.0
    np.savez(work/"timeseries.npz", r32_dist=dists, lig_rmsd=lig_rmsd)
    return dict(n_frames=int(traj.n_frames),
        occupancy_pct=float(np.mean(dists<=SALT_CUT_A)*100),
        r32_dist_start=float(np.mean(dists[:10])),
        r32_dist_end=float(np.mean(dists[-10:])),
        r32_dist_min=float(np.min(dists)),
        r32_dist_mean=float(np.mean(dists)),
        lig_rmsd_mean=float(np.mean(lig_rmsd)))


def main():
    rows=[]
    for name,cpdb,smi in CANDIDATES:
        if "__FILL__" in (cpdb,smi):
            stamp(f"SKIP {name}: CONFIG 미완성"); continue
        if not Path(cpdb).exists():
            stamp(f"SKIP {name}: complex 없음 {cpdb}"); continue
        try:
            rows.append(run_one(name,cpdb,smi))
        except Exception as e:
            import traceback; stamp(f"ERROR {name}: {e}"); traceback.print_exc()
    print("\n"+"="*72)
    print(f"{'name':14s} {'occ%':>6s} {'start':>7s} {'end':>7s} {'min':>6s} {'mean':>7s} {'RMSD':>6s}")
    print("-"*72)
    for r in rows:
        print(f"{r['name']:14s} {r['occupancy_pct']:6.0f} {r['r32_dist_start']:7.2f} "
              f"{r['r32_dist_end']:7.2f} {r['r32_dist_min']:6.2f} {r['r32_dist_mean']:7.2f} "
              f"{r['lig_rmsd_mean']:6.2f}")
    print("="*72)
    json.dump(rows, open(OUT_ROOT/"summary.json","w"), indent=2)
    stamp(f"전체 완료 ({time.time()-T0:.0f}s)")


if __name__=="__main__":
    main()
