"""
amber_style_sasa.py

Usage:
    python amber_style_sasa.py fold_dimer_hts_model_0.pdb
or (trajectory):
    python amber_style_sasa.py traj.nc topology.pdb

Output:
    - Shows a matplotlib figure of per-residue SASA (Å^2)
    - Saves "sasa_per_residue.png" and "sasa_per_residue.csv" 
"""

import sys
import os
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt

def load_input(args):
    # If one argument -> single PDB file (or multi-frame PDB)
    if len(args) == 2:
        return md.load_pdb(args[1])
    # If two args -> trajectory + topology
    if len(args) == 3:
        traj = md.load(args[1], top=args[2])
        return traj
    raise ValueError("Usage: python amber_style_sasa.py <pdb_or_traj> [topology.pdb]")

def compute_residue_sasa(traj, probe_radius=0.14, n_sphere_points=960):
    """
    Returns:
      - res_sasa: numpy array (n_frames, n_residues) of per-residue SASA (Å^2)
      - residue_list: list of (chain_id, resname, resid) tuples
    """
    # mdtraj.shrake_rupley returns per-atom SASA per frame
    sasa_atoms = md.shrake_rupley(traj,
                                 probe_radius=probe_radius,
                                 n_sphere_points=n_sphere_points,
                                 mode='atom')  # shape (n_frames, n_atoms)
    # group by residue
    topology = traj.topology
    residues = list(topology.residues)
    # create mapping atom index -> residue index
    atom_to_res = [atom.residue.index for atom in topology.atoms]  # length = n_atoms
    n_frames = sasa_atoms.shape[0]
    n_res = len(residues)
    res_sasa = np.zeros((n_frames, n_res))
    for atom_idx, res_idx in enumerate(atom_to_res):
        res_sasa[:, res_idx] += sasa_atoms[:, atom_idx]
    # build residue list (chain, resname, resid)
    residue_list = [(residue.chain.index if residue.chain is not None else 0,
                     residue.name,
                     residue.resSeq if hasattr(residue, "resSeq") else residue.index)
                    for residue in residues]
    return res_sasa, residues

def make_amber_style_plot(residue_objs, sasa_avg, sasa_std=None, out_png="sasa_per_residue.png"):
    # x: residue indices (1..N) or actual residue numbers if available
    # We'll use integer positions (1..N) but label ticks with resname#chain optionally.
    N = len(residue_objs)
    x = np.arange(1, N+1)

    # get labels for x ticks: e.g. A:LYS12 or LYS12:A
    labels = []
    chain_breaks = []
    last_chain = None
    for i,res in enumerate(residue_objs):
        chain = res.chain.index if res.chain is not None else 0
        resname = res.name
        resid = res.resSeq if hasattr(res, "resSeq") else res.index
        labels.append(f"{resname}{resid}")
        if last_chain is None:
            last_chain = chain
        elif chain != last_chain:
            chain_breaks.append(i + 0.5)  # vertical line between residues
            last_chain = chain

    fig, ax = plt.subplots(figsize=(12,5))
    # line (Amber often shows per-residue line)
    ax.plot(x, sasa_avg, linewidth=1.5)
    # optional ribbon of std deviation
    if sasa_std is not None:
        ax.fill_between(x, sasa_avg - sasa_std, sasa_avg + sasa_std, alpha=0.2)

    # vertical lines to denote chain breaks
    for cb in chain_breaks:
        ax.axvline(cb, linestyle='--', linewidth=0.7)

    # improve ticks: show a tick every ~10 residues for readability
    if N <= 40:
        xticks = x
    else:
        step = int(max(1, round(N/40)))
        xticks = x[::step]
    ax.set_xticks(xticks)
    ax.set_xticklabels([labels[i-1] for i in xticks], rotation=90, fontsize=8)

    ax.set_xlabel("Residue (name+number)")
    ax.set_ylabel("SASA per residue (Å²)")
    ax.set_title("Per-residue SASA (Amber-style)")
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.8)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")
    plt.show()

def save_csv(residue_objs, sasa_mean, sasa_std, out_csv="sasa_per_residue.csv"):
    rows = []
    for i,res in enumerate(residue_objs):
        chain = res.chain.index if res.chain is not None else 0
        resname = res.name
        resid = res.resSeq if hasattr(res, "resSeq") else res.index
        rows.append({
            "chain": chain,
            "resname": resname,
            "resid": resid,
            "sasa_mean": float(sasa_mean[i]),
            "sasa_std": float(sasa_std[i]) if sasa_std is not None else None
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV to {out_csv}")

def main():
    traj = load_input(sys.argv)
    print(f"Loaded trajectory with {traj.n_frames} frames, {traj.n_atoms} atoms, {traj.n_residues} residues")

    res_sasa, residues = compute_residue_sasa(traj)
    # average across frames (if single frame, it's just the frame)
    sasa_mean = res_sasa.mean(axis=0)
    sasa_std = res_sasa.std(axis=0) if traj.n_frames > 1 else None

    # Plot and save
    make_amber_style_plot(residues, sasa_mean, sasa_std)
    save_csv(residues, sasa_mean, sasa_std)

if __name__ == "__main__":
    main()
