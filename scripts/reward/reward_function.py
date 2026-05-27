"""
SHP2 N-SH2 PPI inhibitor reward function

Target: SHP2 N-SH2 domain pY binding pocket
Hotspots: R32, S34, K55 (blocks GAB1/IRS1 pY peptide binding)

6 components:
    1. Docking          (Uni-Dock, SHP2 N-SH2)
    2. Pharmacophore    (PLIP, R32/S34/K55 contact)
    3. Stability        (ADMET-AI Clearance_Microsome_AZ)
    4. Permeability     (ADMET-AI Caco2_Wang)
    5. AD score         (Morgan FP + Tanimoto MTS, hard gate)
    6. SAScore          (RDKit)
"""

from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path

from components.docking import run_unidock
from components.pharmacophore import run_plip
from components.admet import run_admet_ai
from components.ad_score import compute_ad_score
from components.sascore import compute_sascore

from transformations import (
    reverse_sigmoid,
    sigmoid,
    normalize_sascore,
    hard_gate,
)


# ============================================================
# SHP2-tuned hyperparameters
# ============================================================
DEFAULT_PARAMS = {
    # Docking — SHP2 N-SH2 pY pocket
    # Our 968-mol baseline: mean -5.9, best -7.8
    # d0 adjusted to -7.0 to give meaningful gradient in this range
    'dock_k': 1.0,      # was 0.5 — steeper for sharper discrimination near d0
    'dock_d0': -7.0,    # was -9.0 (STING-TRIM29)
    # Pharmacophore — R32/S34/K55 contacts
    'pharma_k': 1.0,
    'pharma_p0': 1.0,   # R32/S34/T42 기준 분포 중심(median 1)에 맞춤
    # Stability — same
    'stab_k': 0.05,
    'stab_x0': 30.0,
    # Permeability — same
    'perm_k': 2.0,
    'perm_x0': -5.0,
    # AD hard gate
    'ad_threshold': 0.30,
}


def compute_reward(
    smiles_list: List[str],
    setup: str = 'C',
    params: Dict = None,
    work_dir: Path = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute reward for a batch of SMILES against SHP2 N-SH2.

    Returns:
        rewards: numpy array, shape (N,), values in [0, 1]
        details: dict with per-component raw + normalized values
    """
    if params is None:
        params = DEFAULT_PARAMS
    N = len(smiles_list)

    # 1.1 Docking
    docking_scores = run_unidock(smiles_list, work_dir=work_dir)

    # 1.2 Pharmacophore — SHP2 hotspots
    pharma_contacts = run_plip(
        smiles_list,
        docking_poses=docking_scores,
        target_residues=[(32, 'A'), (34, 'A'), (42, 'A')],  # R32, S34, T42
        work_dir=work_dir,
    )

    # 1.3-1.4 ADMET
    admet_results = run_admet_ai(
        smiles_list,
        properties=['Clearance_Microsome_AZ', 'Caco2_Wang'],
        work_dir=work_dir,
    )
    stability = admet_results['Clearance_Microsome_AZ']
    permeability = admet_results['Caco2_Wang']

    # 1.5 AD score
    ad_results = compute_ad_score(
        smiles_list,
        training_sets=['Caco2_Wang', 'Clearance_Microsome_AZ'],
    )
    mts_caco2 = ad_results['Caco2_Wang']
    mts_hlm = ad_results['Clearance_Microsome_AZ']

    # 1.6 SAScore
    sa_scores = compute_sascore(smiles_list)

    # ============================================================
    # 2. Normalization
    # ============================================================
    d_norm = reverse_sigmoid(docking_scores, k=params['dock_k'], x0=params['dock_d0'])
    p_norm = sigmoid(pharma_contacts, k=params['pharma_k'], x0=params['pharma_p0'])
    s_norm = reverse_sigmoid(stability, k=params['stab_k'], x0=params['stab_x0'])
    r_norm = sigmoid(permeability, k=params['perm_k'], x0=params['perm_x0'])
    sa_norm = normalize_sascore(sa_scores)

    G_caco2 = hard_gate(mts_caco2, threshold=params['ad_threshold'])
    G_hlm = hard_gate(mts_hlm, threshold=params['ad_threshold'])
    G = G_caco2 * G_hlm

    # ============================================================
    # 3. Combination
    # ============================================================
    if setup == 'A':
        reward = G * (d_norm + p_norm + s_norm + r_norm + sa_norm) / 5
    elif setup == 'B':
        reward = G * (d_norm * p_norm * s_norm * r_norm * sa_norm) ** (1/5)
    elif setup == 'C':
        # Hybrid ⭐ MAIN
        reward = G * d_norm * p_norm * (s_norm + r_norm + sa_norm) / 3
    elif setup == 'D':
        reward = G * d_norm * (p_norm + s_norm + r_norm + sa_norm) / 4
    elif setup == 'E':
        reward = _pareto_reward(
            np.stack([d_norm, p_norm, s_norm, r_norm, sa_norm], axis=1), G,
        )
    elif setup == 'F':
        # Minimal multiplicative: docking × pharma only
        reward = d_norm * p_norm
    elif setup == 'G':
        # Minimal additive: (docking + pharma) / 2
        reward = (d_norm + p_norm) / 2
    else:
        raise ValueError(f"Unknown setup: {setup}")

    details = {
        'docking_raw': docking_scores,
        'pharma_raw': pharma_contacts,
        'stability_raw': stability,
        'permeability_raw': permeability,
        'mts_caco2': mts_caco2,
        'mts_hlm': mts_hlm,
        'sascore_raw': sa_scores,
        'docking_norm': d_norm,
        'pharma_norm': p_norm,
        'stability_norm': s_norm,
        'permeability_norm': r_norm,
        'sascore_norm': sa_norm,
        'AD_gate': G,
        'reward': reward,
        'setup': setup,
    }

    if verbose:
        _print_summary(smiles_list, details)

    return reward, details


def _pareto_reward(score_matrix: np.ndarray, G: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Pareto setup E not yet implemented")


def _print_summary(smiles_list, details):
    print(f"\n{'SMILES':<40} {'dock':>7} {'pharma':>7} {'AD':>5} {'reward':>7}")
    print('-' * 70)
    for i, smi in enumerate(smiles_list):
        smi_short = smi[:38] + '..' if len(smi) > 40 else smi
        print(
            f"{smi_short:<40} "
            f"{details['docking_raw'][i]:>7.2f} "
            f"{details['pharma_raw'][i]:>7d} "
            f"{int(details['AD_gate'][i]):>5d} "
            f"{details['reward'][i]:>7.3f}"
        )


# ============================================================
# SHP2 pY-anchor reward (data-driven 2025, "setup D/H")
#   signal: pharma = 2*R32_salt_bridge + 1*T42_hydrogen_bond  (Anselmi 2020)
#   gates : QEPPI>0.4 (PPI drug-likeness), SA<5, docking<=-4.5
#   docking is NOT a score component (orthogonal to pY anchor), gate only.
# ============================================================
_QEPPI_CALC = None

def compute_qeppi(smiles_list):
    """QEPPI (PPI-targeting drug-likeness) per SMILES. 0 if invalid."""
    global _QEPPI_CALC
    import QEPPI as _Q
    from rdkit import Chem
    if _QEPPI_CALC is None:
        _QEPPI_CALC = _Q.QEPPI_Calculator()
        _QEPPI_CALC.read()
    out = np.zeros(len(smiles_list), dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            out[i] = _QEPPI_CALC.qeppi(m)
        except Exception:
            out[i] = 0.0
    return out


def compute_reward_pY(
    smiles_list,
    work_dir=None,
    pharma_weights=None,
    pharma_k=2.0, pharma_x0=0.5,
    qeppi_gate=0.4, sa_gate=5.0, dock_gate=-4.5,
    verbose=False,
):
    """
    SHP2 N-SH2 pY-anchor reward.
    reward = sigmoid(2*R32_sb + 1*T42_hb) * GATE
    GATE = (QEPPI > qeppi_gate) AND (SA < sa_gate) AND (docking <= dock_gate)
    """
    from components.pharmacophore import run_plip_typed, score_pY_third_path, run_plip_bioisostere
    from components.sascore import compute_sascore
    N = len(smiles_list)

    # 1. docking (pose 생성 + score; score는 게이트로만)
    docking = np.asarray(run_unidock(smiles_list, work_dir=work_dir), dtype=np.float64)

    # 2. typed pharma: 2*R32_salt_bridge + 1*T42_hydrogen_bond
    pharma_raw = run_plip_bioisostere(smiles_list, work_dir=work_dir)

    # 3. QEPPI, 4. SA
    qeppi = compute_qeppi(smiles_list)
    sa = np.asarray(compute_sascore(smiles_list), dtype=np.float64)

    # 5. normalize + gate
    p_norm = sigmoid(pharma_raw, k=pharma_k, x0=pharma_x0)
    gate = (qeppi > qeppi_gate) & (sa < sa_gate) & (docking <= dock_gate)
    reward = p_norm * gate.astype(np.float64)

    details = {
        'docking_raw': docking,
        'pharma_raw': pharma_raw,
        'qeppi_raw': qeppi,
        'sascore_raw': sa,
        'pharma_norm': p_norm,
        'gate': gate.astype(np.float64),
        'reward': reward,
        'setup': 'pY',
    }
    if verbose:
        print(f"  N={N}  pharma_raw>0: {(pharma_raw>0).sum()}  gate_pass: {int(gate.sum())}  mean_reward: {reward.mean():.4f}")
    return reward, details


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('smiles_file', type=str)
    parser.add_argument('--setup', type=str, default='C', choices=list('ABCDEFG'))
    parser.add_argument('--work_dir', type=str, default='./reward_work')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    with open(args.smiles_file) as f:
        smiles = [line.strip() for line in f if line.strip()]

    rewards, details = compute_reward(
        smiles, setup=args.setup, work_dir=work_dir, verbose=args.verbose
    )

    print(f"\nMean reward: {rewards.mean():.3f}")
    print(f"AD pass rate: {details['AD_gate'].mean():.1%}")