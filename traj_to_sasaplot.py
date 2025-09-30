# amber_style_sasa_time.py
"""
Usage:
    python amber_style_sasa_time.py traj.dcd topology.pdb
    or
    python amber_style_sasa_time.py output.pdb   # if multi-model PDB

Outputs:
    - sasa_vs_time.png
    - sasa_vs_time.csv
"""

import sys
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt

def load_input(args):
    if len(args) == 2:
        return md.load_pdb(args[1])   # can handle multi-model PDBs
    if len(args) == 3:
        return md.load(args[1], top=args[2])  # e.g. DCD/NC + topology
    raise ValueError("Usage: python amber_style_sasa_time.py <pdb_or_traj> [topology.pdb]")

def compute_total_sasa(traj, probe_radius=0.14, n_sphere_points=960):
    # Per-atom SASA (nm²)
    sasa_atoms = md.shrake_rupley(traj,
                                  probe_radius=probe_radius,
                                  n_sphere_points=n_sphere_points,
                                  mode='atom')
    # Per-frame total SASA (nm² → Å²)
    sasa_total = sasa_atoms.sum(axis=1) * 100.0
    return sasa_total

def make_time_plot(sasa_total, dt=1.0, out_png="sasa_vs_time.png"):
    """
    sasa_total: np.array of total SASA per frame
    dt: timestep between frames (ps or ns — user sets)
    """
    frames = np.arange(len(sasa_total))
    time = frames * dt
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(time, sasa_total, color='blue', lw=1.2)
    ax.set_xlabel("Time (arb. units)")  # replace with ns if known
    ax.set_ylabel("Total SASA (Å²)")
    ax.set_title("Total SASA vs Time (Amber-style)")
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")
    plt.show()

def save_csv(sasa_total, out_csv="sasa_vs_time.csv"):
    df = pd.DataFrame({"frame": np.arange(len(sasa_total)),
                       "sasa_total_A2": sasa_total})
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV to {out_csv}")

def main():
    traj = load_input(sys.argv)
    print(f"Loaded trajectory with {traj.n_frames} frames, {traj.n_atoms} atoms, {traj.n_residues} residues")
    sasa_total = compute_total_sasa(traj)
    make_time_plot(sasa_total)
    save_csv(sasa_total)

if __name__ == "__main__":
    main()
