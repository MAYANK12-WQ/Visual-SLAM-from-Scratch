"""
KITTI Trajectory Evaluation — ATE & RPE Metrics
================================================
Evaluates Visual SLAM against KITTI ground truth using:
  - ATE  (Absolute Trajectory Error) — global consistency
  - RPE  (Relative Pose Error)       — local accuracy

Usage:
    python benchmark/evaluate_kitti.py \\
        --gt    data/kitti/00/ground_truth.txt \\
        --est   results/estimated_trajectory.txt \\
        --seq   00

Reference: Geiger et al., "Are we ready for Autonomous Driving?
The KITTI Vision Benchmark Suite", CVPR 2012.
"""

import argparse
import json
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# I/O Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_trajectory_kitti(path: str) -> np.ndarray:
    """
    Load KITTI-format trajectory file.

    Each line: 12 values -> 3x4 pose matrix (row-major)
    Returns: (N, 4, 4) homogeneous transformation matrices
    """
    poses = []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return np.array(poses)


def align_umeyama(P: np.ndarray, Q: np.ndarray):
    """
    Umeyama alignment — finds s, R, t minimising ||s*R*P + t - Q||^2

    Args:
        P: (N,3) estimated positions
        Q: (N,3) ground-truth positions

    Returns:
        scale (float), R (3,3), t (3,)
    """
    n = P.shape[0]
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)

    P_c = P - mu_P
    Q_c = Q - mu_Q

    var_P = (P_c ** 2).sum() / n
    cov = Q_c.T @ P_c / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    scale = (D * S.diagonal()).sum() / max(var_P, 1e-9)
    t = mu_Q - scale * R @ mu_P
    return scale, R, t


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_ate(gt_poses: np.ndarray, est_poses: np.ndarray, align: bool = True) -> dict:
    """
    Absolute Trajectory Error (ATE).

    Measures global consistency by aligning trajectories and computing
    per-frame translational error.
    """
    n = min(len(gt_poses), len(est_poses))
    gt_pos = gt_poses[:n, :3, 3]
    est_pos = est_poses[:n, :3, 3]

    if align:
        scale, R, t = align_umeyama(est_pos, gt_pos)
        aligned = (scale * (R @ est_pos.T)).T + t
    else:
        aligned = est_pos

    errors = np.linalg.norm(gt_pos - aligned, axis=1)

    return {
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "std": float(errors.std()),
        "max": float(errors.max()),
        "min": float(errors.min()),
        "errors": errors,
        "aligned_est": aligned,
        "gt_pos": gt_pos,
    }


def compute_rpe(gt_poses: np.ndarray, est_poses: np.ndarray,
                delta: int = 1) -> dict:
    """
    Relative Pose Error (RPE).

    Measures local accuracy by comparing relative transformations
    over fixed frame intervals.
    """
    n = min(len(gt_poses), len(est_poses))
    trans_errors = []
    rot_errors = []

    for i in range(0, n - delta, delta):
        j = min(i + delta, n - 1)

        Q_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[j]
        P_rel = np.linalg.inv(est_poses[i]) @ est_poses[j]
        E_rel = np.linalg.inv(Q_rel) @ P_rel

        trans_errors.append(np.linalg.norm(E_rel[:3, 3]))

        R_err = E_rel[:3, :3]
        cos_a = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        rot_errors.append(np.degrees(np.arccos(cos_a)))

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)

    return {
        "trans_rmse": float(np.sqrt((trans_errors ** 2).mean())),
        "trans_mean": float(trans_errors.mean()),
        "rot_rmse": float(np.sqrt((rot_errors ** 2).mean())),
        "rot_mean": float(rot_errors.mean()),
        "trans_errors": trans_errors,
        "rot_errors": rot_errors,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

KITTI_BASELINES = {
    "00": {"ORB-SLAM2": 0.0091, "DSO": 0.0111, "LSD-SLAM": 0.0211, "SVO": 0.0193},
    "05": {"ORB-SLAM2": 0.0099, "DSO": 0.0130, "LSD-SLAM": 0.0181, "SVO": 0.0178},
    "07": {"ORB-SLAM2": 0.0076, "DSO": 0.0088, "LSD-SLAM": 0.0145, "SVO": 0.0162},
    "08": {"ORB-SLAM2": 0.0115, "DSO": 0.0245, "LSD-SLAM": 0.0329, "SVO": 0.0271},
}


def print_report(ate: dict, rpe: dict, seq: str, our_name: str = "Ours"):
    print("\n" + "=" * 60)
    print(f"  KITTI Sequence {seq} — Evaluation Report")
    print("=" * 60)

    print(f"\n  ATE (m) — Absolute Trajectory Error")
    print(f"    RMSE:   {ate['rmse']:.4f} m")
    print(f"    Mean:   {ate['mean']:.4f} m")
    print(f"    Median: {ate['median']:.4f} m")
    print(f"    Std:    {ate['std']:.4f} m")

    print(f"\n  RPE — Relative Pose Error (delta=1 frame)")
    print(f"    Trans RMSE: {rpe['trans_rmse']:.4f} m")
    print(f"    Rot   RMSE: {rpe['rot_rmse']:.4f} deg")

    if seq in KITTI_BASELINES:
        print(f"\n  Comparison vs Published Methods — ATE RMSE (m)")
        print(f"  {'Method':<20} {'ATE RMSE':>10}  {'We Beat It?':>12}")
        print("  " + "-" * 46)
        for method, val in KITTI_BASELINES[seq].items():
            better = "YES" if ate["rmse"] < val else "No"
            print(f"  {method:<20} {val:>10.4f}  {better:>12}")
        print(f"  {our_name:<20} {ate['rmse']:>10.4f}")
    print("\n" + "=" * 60)


def save_results(ate: dict, rpe: dict, output_path: str):
    results = {
        "ate": {k: float(v) for k, v in ate.items()
                if k not in ("errors", "aligned_est", "gt_pos")},
        "rpe": {k: float(v) for k, v in rpe.items()
                if k not in ("trans_errors", "rot_errors")},
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectory_comparison(ate: dict, output_path: str = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    gt_pos = ate["gt_pos"]
    aligned = ate["aligned_est"]
    errors = ate["errors"]

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(gt_pos[:, 0], gt_pos[:, 2], "k-", lw=2, label="Ground Truth", alpha=0.9)
    ax1.plot(aligned[:, 0], aligned[:, 2], "r--", lw=1.5, label="Ours (aligned)", alpha=0.85)
    ax1.scatter(gt_pos[0, 0], gt_pos[0, 2], s=80, c="green", zorder=5, label="Start")
    ax1.scatter(gt_pos[-1, 0], gt_pos[-1, 2], s=80, c="red", zorder=5, label="End")
    ax1.set_xlabel("X (m)", fontsize=11)
    ax1.set_ylabel("Z (m)", fontsize=11)
    ax1.set_title("Trajectory Comparison (Top-Down View)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    frames = np.arange(len(errors))
    ax2.fill_between(frames, 0, errors, alpha=0.3, color="steelblue")
    ax2.plot(frames, errors, color="steelblue", lw=1.2)
    ax2.axhline(errors.mean(), color="red", lw=1.5, ls="--",
                label=f"Mean={errors.mean():.3f}m")
    ax2.axhline(errors.mean() + errors.std(), color="orange", lw=1, ls=":",
                label=f"mu+sigma={errors.mean() + errors.std():.3f}m")
    ax2.set_xlabel("Frame", fontsize=11)
    ax2.set_ylabel("ATE (m)", fontsize=11)
    ax2.set_title(f"Absolute Trajectory Error\nRMSE = {ate['rmse']:.4f} m",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved -> {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SLAM trajectory against KITTI ground truth")
    parser.add_argument("--gt", required=True, help="Ground truth trajectory (.txt)")
    parser.add_argument("--est", required=True, help="Estimated trajectory (.txt)")
    parser.add_argument("--seq", default="00", help="KITTI sequence ID")
    parser.add_argument("--delta", type=int, default=1, help="RPE interval in frames")
    parser.add_argument("--align", action="store_true", default=True,
                        help="Align trajectories with Umeyama (default: True)")
    parser.add_argument("--plot", type=str, default=None,
                        help="Save trajectory plot (e.g. docs/images/trajectory.png)")
    parser.add_argument("--out", type=str, default=None,
                        help="Save JSON results (e.g. results/seq00.json)")
    args = parser.parse_args()

    print(f"Loading ground truth:  {args.gt}")
    gt_poses = load_trajectory_kitti(args.gt)
    print(f"Loading estimated:     {args.est}")
    est_poses = load_trajectory_kitti(args.est)
    print(f"GT frames: {len(gt_poses)}  |  Est frames: {len(est_poses)}")

    ate = compute_ate(gt_poses, est_poses, align=args.align)
    rpe = compute_rpe(gt_poses, est_poses, delta=args.delta)

    print_report(ate, rpe, seq=args.seq)

    if args.out:
        save_results(ate, rpe, args.out)

    if args.plot:
        plot_trajectory_comparison(ate, output_path=args.plot)


if __name__ == "__main__":
    main()
