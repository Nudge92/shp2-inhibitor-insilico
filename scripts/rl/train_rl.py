"""
RL fine-tuning for Hot2Mol — Augmented Likelihood (REINVENT 4 DAP).

Run in `Hot2Mol` conda environment.

Usage:
    cd ~/Project/sh2/tools/Hot2Mol
    conda activate Hot2Mol
    python ./scripts/phase4_rl/train_rl.py \
        --posp ./data/posp/sh2_chainA.posp \
        --out_dir ./results/phase4_rl \
        --n_steps 100 --batch_size 32 --sigma 96 --lr 1e-5
"""

import sys
import argparse
import pickle
import time
from pathlib import Path
import numpy as np
import torch
import dgl

# Hot2Mol imports (must run from Hot2Mol directory or have it in PYTHONPATH)
HOT2MOL_DIR = Path('./tools/Hot2Mol')
sys.path.insert(0, str(HOT2MOL_DIR))

from model.Hot2Mol import Hot2Mol
from utils.file_utils import load_phar_file

# Reward bridge (same dir as this script)
sys.path.insert(0, str(Path(__file__).parent))
from reward_bridge import get_rewards


# ============================================================
# Config
# ============================================================
MODEL_PARAMS = {
    'max_len': 128, 'pp_v_dim': 8, 'pp_e_dim': 1, 'pp_encoder_n_layer': 4,
    'hidden_dim': 384, 'n_layers': 8, 'ff_dim': 1024, 'n_head': 8,
    'non_vae': False, 'remove_pp_dis': False,
}
PRETRAINED_PTH = HOT2MOL_DIR / 'pretrained_model' / 'epoch32.pth'
TOKENIZER_PATH = HOT2MOL_DIR / 'pretrained_model' / 'tokenizer_r_iso.pkl'
TOKENIZER_PROP_PATH = HOT2MOL_DIR / 'pretrained_model' / 'tokenizer_delta_qeppi.pkl'
TRAINABLE_MODULES = ['decoder', 'word_pred']


# ============================================================
# Model loading
# ============================================================
def load_model(device, tokenizer, tokenizer_prop):
    params = dict(MODEL_PARAMS)
    params['device'] = device
    m = Hot2Mol(params, tokenizer, tokenizer_prop)
    states = torch.load(PRETRAINED_PTH, map_location=device)
    states['model'].update({
        k: m.state_dict()[k] for k in m.state_dict().keys() if k not in states['model']
    })
    m.load_state_dict(states['model'], strict=False)
    m.to(device)
    return m


def freeze_for_prior(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def setup_agent(model):
    """Only decoder + word_pred trainable."""
    for name, p in model.named_parameters():
        p.requires_grad = any(t in name for t in TRAINABLE_MODULES)


# ============================================================
# Token / SMILES utilities
# ============================================================
def decode_smiles(tokens_2d, tokenizer, eos_value, pad_value):
    """
    tokens_2d: (B, max_len-1) tensor of token ids
    Returns list of SMILES strings (may include invalid SMILES from raw decode)
    """
    smiles_list = []
    for i in range(tokens_2d.shape[0]):
        toks = tokens_2d[i].cpu().tolist()
        # Truncate at EOS / PAD
        clean = []
        for t in toks:
            if t == eos_value or t == pad_value:
                break
            clean.append(t)
        if not clean:
            smiles_list.append('')
            continue
        smi = tokenizer.get_text([clean])[0]
        smiles_list.append(smi)
    return smiles_list


def compute_log_p_seq(scores, tokens, eos_value):
    """
    Compute sequence-level log probability.

    Args:
        scores: (B, gen_len, V) logits
        tokens: (B, gen_len) sampled token ids
        eos_value: int

    Returns:
        log_p_seq: (B,) sum of log p over actual sequence length
        seq_lens: list of int, actual length per sample (incl. EOS if present)
    """
    log_probs_all = torch.log_softmax(scores, dim=-1)
    log_p_per_token = log_probs_all.gather(2, tokens.unsqueeze(-1)).squeeze(-1)  # (B, gen_len)

    B, gen_len = tokens.shape
    mask = torch.ones_like(tokens, dtype=torch.float)
    seq_lens = []
    for i in range(B):
        toks = tokens[i].tolist()
        if eos_value in toks:
            end = toks.index(eos_value) + 1
            seq_lens.append(end)
            if end < gen_len:
                mask[i, end:] = 0.0
        else:
            seq_lens.append(gen_len)

    log_p_seq = (log_p_per_token * mask).sum(dim=1)
    return log_p_seq, seq_lens


# ============================================================
# Main training loop
# ============================================================
def train(args):
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizers
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(TOKENIZER_PROP_PATH, 'rb') as f:
        tokenizer_prop = pickle.load(f)

    # Pharmacophore graph
    g = load_phar_file(Path(args.posp))

    # Models
    print('Loading prior...')
    prior = load_model(device, tokenizer, tokenizer_prop)
    freeze_for_prior(prior)

    print('Loading agent...')
    agent = load_model(device, tokenizer, tokenizer_prop)
    setup_agent(agent)
    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable_params)
    print(f'Trainable params: {n_train:,}')

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Logging
    log_lines = []
    log_lines.append('step,time,mean_reward,max_reward,valid_rate,mean_log_p_agent,mean_log_p_prior,loss,frac_saltbridge,frac_hbond_only,mean_pharma')
    log_path = out_dir / 'rl_log.csv'

    # === Training loop ===
    print(f'\n=== Training: {args.n_steps} steps, batch={args.batch_size}, sigma={args.sigma} ===\n')
    
    t0 = time.time()
    for step in range(1, args.n_steps + 1):
        # Build batch graph (same posp repeated)
        g_batch = dgl.batch([g] * args.batch_size).to(device)

        # (1) Agent samples
        agent.train()
        predict, scores_agent, z = agent.generate_rl(g_batch)
        B, gen_len, V = scores_agent.shape
        tokens = predict[:, :gen_len]

        # (2) Prior evaluates with same z (no grad)
        with torch.no_grad():
            logits_prior = prior.evaluate_rl(g_batch, tokens, z)

        # (3) Log probabilities (sequence-level)
        log_p_agent, _ = compute_log_p_seq(scores_agent, tokens, agent.eos_value)
        log_p_prior, _ = compute_log_p_seq(logits_prior, tokens, agent.eos_value)

        # (4) Decode → SMILES → reward
        smiles_list = decode_smiles(predict, tokenizer, agent.eos_value, agent.pad_value)
        t_reward = time.time()
        rewards_np, pharma_np = get_rewards(smiles_list, setup=args.setup, return_pharma=True)
        reward_time = time.time() - t_reward

        # NaN -> 0 (invalid SMILES gets no reward signal)
        rewards_np = np.nan_to_num(rewards_np, nan=0.0)
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)

        # (5) Augmented likelihood loss
        log_p_aug = log_p_prior.detach() + args.sigma * rewards
        loss = ((log_p_aug - log_p_agent) ** 2).mean()
        print(f"  [DBG] prior={log_p_prior.mean().item():.1f} aug={log_p_aug.mean().item():.1f} agent={log_p_agent.mean().item():.1f} | reward_term={(args.sigma*rewards).mean().item():.1f} | r_max={rewards.max().item():.3f} r_min={rewards.min().item():.3f}")

        # (6) Backward + step
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
        optimizer.step()

        # Logging
        valid_rate = float(np.mean([s != '' for s in smiles_list]))
        elapsed = time.time() - t0

        # Typed pharma breakdown (setup 'pY'). pharma_np may be None (other setups
        # or failure) -> report zeros. Value -1 marks invalid/unscored molecules.
        if pharma_np is not None:
            pv = np.asarray(pharma_np, dtype=np.float32)
            pv = pv[pv >= 0]  # drop -1 (invalid / not scored)
            if pv.size:
                frac_sb = float(np.mean(pv >= 2.0))            # R32 salt bridge formed
                frac_hb = float(np.mean((pv > 0) & (pv < 2.0)))  # H-bond only (no salt)
                mean_ph = float(pv.mean())
            else:
                frac_sb = frac_hb = mean_ph = 0.0
        else:
            frac_sb = frac_hb = mean_ph = 0.0

        line = (f'{step},{elapsed:.1f},{rewards_np.mean():.4f},{rewards_np.max():.4f},'
                f'{valid_rate:.3f},{log_p_agent.mean().item():.2f},'
                f'{log_p_prior.mean().item():.2f},{loss.item():.2f},'
                f'{frac_sb:.3f},{frac_hb:.3f},{mean_ph:.3f}')
        log_lines.append(line)
        log_path.write_text('\n'.join(log_lines) + '\n')

        print(f'[step {step:>3}/{args.n_steps}] '
              f'reward={rewards_np.mean():.3f} (max={rewards_np.max():.3f}) | '
              f'salt_br={frac_sb:.0%} hbond={frac_hb:.0%} | '
              f'log_p_agent={log_p_agent.mean().item():.1f} | '
              f'loss={loss.item():.1f} | '
              f'grad={grad_norm:.1f} | '
              f'reward_t={reward_time:.0f}s | '
              f'valid={valid_rate:.0%}')

        # Save checkpoint
        if step % args.save_every == 0 or step == args.n_steps:
            ckpt_path = out_dir / f'agent_step{step}.pth'
            torch.save({
                'step': step,
                'model': agent.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': vars(args),
            }, ckpt_path)
            print(f'  -> saved {ckpt_path}')

    print(f'\nTraining complete. Logs: {log_path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--posp', required=True, help='.posp file (pharmacophore hypothesis)')
    p.add_argument('--out_dir', required=True, help='checkpoints + log destination')
    p.add_argument('--n_steps', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--sigma', type=float, default=96.0)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--setup', default='C', choices=list('ABCDEFG')+['pY'])
    p.add_argument('--save_every', type=int, default=20)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())