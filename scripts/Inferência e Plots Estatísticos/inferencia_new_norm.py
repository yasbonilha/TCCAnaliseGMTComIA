#!/usr/bin/env python3
"""
inference_and_plots.py — Inferência + plots da PINN treinada (z-score)

Standalone: NÃO depende do script de treino. Recria a arquitetura, carrega
pesos de um checkpoint, e gera todos os plots de pós-processamento.

ATUALIZADO PARA NOVA NORMALIZAÇÃO:
  - Coordenadas espaciais: z-score isotrópico (σ único para x, y, z)
  - Tempo: z-score
  - Saídas u, w: min-max simétrico (V_REF = 5 m/s)
  - Saídas v, P: z-score
  - HardConstraint com ansatz em coords físicas para v e P (offset != 0)

Uso no cluster:
    python inference_and_plots.py
    python inference_and_plots.py --checkpoint pinn_segregated_ep000300.weights.h5
    python inference_and_plots.py --list
    python inference_and_plots.py --no-slices    # mais rápido
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow.keras import layers, Model


# ================================================================================
# Configuração de caminhos (cluster)
# ================================================================================
CHECKPOINT_DIR = "/home/tmoraes/treino_normalizado/pinn_checkpoints_segregated"
SNAPSHOT_DIR   = "/home/tmoraes/CSVs"
OUTPUT_DIR     = "/home/tmoraes/treino_normalizado/plots_results"
CSV_LOG_PATH   = "/home/tmoraes/treino_normalizado/training_log_segregated.csv"

SLICES_DIR   = os.path.join(OUTPUT_DIR, "slices")
SCATTER_DIR  = os.path.join(OUTPUT_DIR, "scatter")
HIST_DIR     = os.path.join(OUTPUT_DIR, "histograms")
SUMMARY_DIR  = os.path.join(OUTPUT_DIR, "summary")
LOSSHIST_DIR = os.path.join(OUTPUT_DIR, "loss_history")


# ================================================================================
# CONSTANTES FÍSICAS (idênticas à Cell 2 do script principal)
# ================================================================================
X_MIN, X_MAX = 0.0, 280.0
Y_MIN, Y_MAX = 0.0, 700.0
Z_MIN, Z_MAX = 0.0, 190.0

D_CYL = 54.0
H_CYL = 65.0
R_CYL = D_CYL / 2.0
XC_CYL = 140.0
YC_CYL = 200.0

V_INF   = 17.0
P_OP    = 101325.0
RHO_INF = 1.225
MU      = 1.7894e-5


# ================================================================================
# ESCALAS DE NORMALIZAÇÃO Z-SCORE (idênticas à Cell 4)
# ================================================================================
# Espacial: z-score isotrópico (translação por eixo, mesmo σ)
X_MEAN = 140.01174139636194
Y_MEAN = 290.9150766181903
Z_MEAN = 66.39128138069277
DP_ISO = 101.234187329139

# Temporal: z-score
T_MEAN = 50.05
T_DP   = 28.867499025720953

# Saídas — esquema HÍBRIDO:
#   u, w: min-max simétrico (V_REF = 5 m/s; calibrado para Atacama)
#   v, P: z-score
V_REF  = 5.0
V_MEAN = 14.321999262773115
V_DP   = 7.399817765728233
P_MEAN = -32.78900000449526
P_DP   = 98.41228792498488

# Cilindro em coords normalizadas
XC_N = (XC_CYL - X_MEAN) / DP_ISO
YC_N = (YC_CYL - Y_MEAN) / DP_ISO
R_N  = R_CYL   / DP_ISO
H_N  = (H_CYL  - Z_MEAN) / DP_ISO

# Limites do domínio em coords normalizadas
XN_MIN = (X_MIN - X_MEAN) / DP_ISO
XN_MAX = (X_MAX - X_MEAN) / DP_ISO
YN_MIN = (Y_MIN - Y_MEAN) / DP_ISO
YN_MAX = (Y_MAX - Y_MEAN) / DP_ISO
ZN_MIN = (Z_MIN - Z_MEAN) / DP_ISO
ZN_MAX = (Z_MAX - Z_MEAN) / DP_ISO


# ================================================================================
# HARDCONSTRAINTLAYER (cópia idêntica à Cell 9 com ansatz híbrido)
# ================================================================================
class HardConstraintLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def smooth_max(a, b, k=50.0):
        m = tf.maximum(a, b)
        return m + (1.0 / k) * tf.math.log(
            tf.exp(k * (a - m)) + tf.exp(k * (b - m))
        )

    def call(self, inputs):
        xyzt, raw_pred = inputs

        xn = xyzt[:, 0:1]
        yn = xyzt[:, 1:2]
        zn = xyzt[:, 2:3]

        un = raw_pred[:, 0:1]
        vn = raw_pred[:, 1:2]
        wn = raw_pred[:, 2:3]
        Pn = raw_pred[:, 3:4]

        # Constantes recalibradas para z-score isotrópico (DP_ISO ≈ 101.23)
        mt       = 135.0
        rate     = 72.0
        k_smooth = 75.0
        k_r2     = 238.0

        # Funções de distância (zero sobre cada fronteira)
        D_inlet  = tf.math.tanh(mt * (yn - YN_MIN))
        D_outlet = tf.math.tanh(mt * (YN_MAX - yn))
        D_ground = tf.math.tanh(mt * (zn - ZN_MIN))
        D_xmin   = tf.math.tanh(mt * (xn - XN_MIN))
        D_xmax   = tf.math.tanh(mt * (XN_MAX - xn))
        D_ztop   = tf.math.tanh(mt * (ZN_MAX - zn))

        r2_normalized = tf.square(xn - XC_N) + tf.square(yn - YC_N)
        d_radial_sq   = r2_normalized - R_N * R_N
        d_top         = zn - H_N
        d_cyl_3d      = self.smooth_max(d_radial_sq, d_top, k=k_smooth)
        D_cyl         = tf.math.tanh(k_r2 * tf.maximum(d_cyl_3d, 0.0))

        D_vel = D_inlet * D_ground * D_cyl * D_xmin * D_xmax * D_ztop

        decay_inlet = tf.exp(-rate * (yn - YN_MIN))
        decay_xmin  = tf.exp(-rate * (xn - XN_MIN))
        decay_xmax  = tf.exp(-rate * (XN_MAX - xn))
        decay_ztop  = tf.exp(-rate * (ZN_MAX - zn))
        decay_freestream = 1.0 - (1.0 - decay_inlet) \
                               * (1.0 - decay_xmin)  \
                               * (1.0 - decay_xmax)  \
                               * (1.0 - decay_ztop)

        # ANSATZ HÍBRIDO
        # u, w: min-max simétrico (V_REF=5), v_norm=0 corresponde a v_phys=0
        u_hard = un * D_vel
        w_hard = wn * D_vel

        # v: z-score (offset V_MEAN). Pensa em coords físicas e renormaliza.
        v_phys_raw  = vn * V_DP + V_MEAN
        v_phys_hard = v_phys_raw * D_vel + V_INF * decay_freestream * D_ground * D_cyl
        v_hard = (v_phys_hard - V_MEAN) / V_DP

        # P: z-score, P_phys = 0 no outlet
        P_phys_raw  = Pn * P_DP + P_MEAN
        P_phys_hard = P_phys_raw * D_outlet
        P_hard = (P_phys_hard - P_MEAN) / P_DP

        return tf.concat([u_hard, v_hard, w_hard, P_hard], axis=1)


# ================================================================================
# Arquitetura MLP (idêntica à Cell 9)
# ================================================================================
def build_pinn(n_layers=6, n_neurons=256):
    inp = layers.Input(shape=(4,), name="xyzt")
    h = inp
    for i in range(n_layers):
        h = layers.Dense(
            n_neurons,
            activation="tanh",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name=f"dense_{i+1}",
        )(h)
    raw_out = layers.Dense(4, name="raw_fields")(h)
    final_out = HardConstraintLayer(name="hard_constraints")([inp, raw_out])
    return Model(inp, final_out, name="PINN_MLP")


# ================================================================================
# Helpers de normalização e desnormalização
# ================================================================================
def normalize_inputs(x_phys, y_phys, z_phys, t_phys):
    """Converte arrays físicos para tensor (N, 4) em z-score."""
    x_phys = np.atleast_1d(x_phys).astype(np.float32).ravel()
    y_phys = np.atleast_1d(y_phys).astype(np.float32).ravel()
    z_phys = np.atleast_1d(z_phys).astype(np.float32).ravel()
    n = len(x_phys)

    if np.isscalar(t_phys):
        t_arr = np.full(n, float(t_phys), dtype=np.float32)
    else:
        t_arr = np.atleast_1d(t_phys).astype(np.float32).ravel()
        assert len(t_arr) == n, f"len(t)={len(t_arr)} != len(x)={n}"

    xn = (x_phys - X_MEAN) / DP_ISO
    yn = (y_phys - Y_MEAN) / DP_ISO
    zn = (z_phys - Z_MEAN) / DP_ISO
    tn = (t_arr  - T_MEAN) / T_DP

    return np.column_stack([xn, yn, zn, tn]).astype(np.float32)


def denormalize_outputs(pred):
    """Saídas (N, 4) normalizadas → unidades físicas (esquema híbrido):
      u, w: min-max simétrico
      v, P: z-score
    """
    u = pred[:, 0] * V_REF
    v = pred[:, 1] * V_DP + V_MEAN
    w = pred[:, 2] * V_REF
    P = pred[:, 3] * P_DP + P_MEAN
    return u, v, w, P


# ================================================================================
# Loader de CSV (idêntico à Cell 5)
# ================================================================================
NEEDED_COLS_PATTERNS = {
    "x": [r"x.coordinate", r"^x$"],
    "y": [r"y.coordinate", r"^y$"],
    "z": [r"z.coordinate", r"^z$"],
    "P": [r"^pressure$", r"static.pressure", r"^p$"],
    "u": [r"x.velocity", r"^u$"],
    "v": [r"y.velocity", r"^v$"],
    "w": [r"z.velocity", r"^w$"],
}

def _norm_col(s):
    return re.sub(r'["\'\ufeff]', '', s).strip().lower()

def resolve_column_names(df_cols):
    norm_map = {_norm_col(c): c for c in df_cols}
    mapping = {}
    for key, patterns in NEEDED_COLS_PATTERNS.items():
        found = None
        for pat in patterns:
            for norm_name, orig_name in norm_map.items():
                if re.fullmatch(pat, norm_name):
                    found = orig_name
                    break
            if found:
                break
        if not found:
            raise KeyError(f"No match for '{key}'. Patterns: {patterns}. Available: {list(norm_map.keys())}")
        mapping[key] = found
    return mapping

def load_snapshot(path, col_map=None):
    df = pd.read_csv(path, sep=r'\s+', engine='python', dtype=np.float32)
    df = df.drop(columns=["cellnumber"], errors="ignore")
    df.columns = [_norm_col(c) for c in df.columns]
    if col_map is None:
        col_map = resolve_column_names(df.columns)
    out = {k: df[col_map[k]].to_numpy(dtype=np.float32, copy=False)
            for k in NEEDED_COLS_PATTERNS}
    return out, col_map


# ================================================================================
# Helpers de checkpoint
# ================================================================================
def list_checkpoints(checkpoint_dir):
    """Lista checkpoints .h5 ordenados por época."""
    patterns = [
        os.path.join(checkpoint_dir, "pinn_segregated_ep*.weights.h5"),
        os.path.join(checkpoint_dir, "pinn_ep*.weights.h5"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = list(set(files))

    def get_epoch(path):
        m = re.search(r'ep(\d+)\.weights\.h5', os.path.basename(path))
        return int(m.group(1)) if m else 0

    files.sort(key=get_epoch)
    return files


def find_latest_checkpoint(checkpoint_dir):
    """Mais recente por época, com fallback para 'best'/'final'."""
    files = list_checkpoints(checkpoint_dir)
    if files:
        return files[-1]
    for name in ["pinn_best_segregated.weights.h5",
                 "pinn_final_segregated.weights.h5",
                 "pinn_best.weights.h5",
                 "pinn_final.weights.h5"]:
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            return p
    return None


def discover_snapshots(snapshot_dir):
    """timestep-XXXX.csv ordenados por t_phys = idx * 0.1s."""
    pattern = os.path.join(snapshot_dir, "timestep-*.csv")
    files = sorted(glob.glob(pattern))
    snaps = []
    for f in files:
        m = re.search(r'timestep-(\d+)\.csv', os.path.basename(f))
        if m:
            idx = int(m.group(1))
            t_phys = idx * 0.1
            snaps.append((t_phys, f))
    return snaps


# ================================================================================
# PLOTS — Loss history
# ================================================================================
def plot_loss_history(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"[skip] CSV não encontrado: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    print(f"Loss history: {len(df)} épocas no CSV")

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    ax = axes[0]
    ax.semilogy(df["epoch"], df["loss_total"], lw=2.0, label="total", color='black')
    ax.semilogy(df["epoch"], df["loss_data"], lw=1.2, label="data", color='tab:blue')
    ax.semilogy(df["epoch"], df["loss_phys_sum"], lw=1.2, label="phys (sum)", color='tab:red')

    if "loss_val" in df.columns:
        val_changed = np.concatenate([[True], np.diff(df["loss_val"].values) != 0])
        ax.semilogy(df["epoch"][val_changed], df["loss_val"][val_changed],
                     'o-', ms=4, lw=1.0, label="val", color='tab:green', alpha=0.8)

    ax.set_ylabel("Loss (log)")
    ax.set_title(f"PINN Training — {len(df)} epochs total")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogy(df["epoch"], df["loss_cont"], lw=1.5, label="continuidade", color='tab:purple')
    ax.semilogy(df["epoch"], df["loss_mom_u"], lw=1.0, label="momento u", color='tab:orange')
    ax.semilogy(df["epoch"], df["loss_mom_v"], lw=1.0, label="momento v", color='tab:brown')
    ax.semilogy(df["epoch"], df["loss_mom_w"], lw=1.0, label="momento w", color='tab:pink')
    ax.set_ylabel("Physics residual (log)")
    ax.set_title("Resíduos físicos por componente")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    if "gnorm" in df.columns:
        ax.plot(df["epoch"], df["gnorm"], lw=1.0, color='tab:gray')
        ax.set_ylabel("Gradient norm (pre-clip)")
        ax.set_title("Norma do gradiente — estabilidade")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='red', ls='--', lw=0.8, alpha=0.5, label='clip threshold (1.0)')
        ax.legend(fontsize=9)

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    out = os.path.join(output_dir, "loss_history.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {out}")
    return out


# ================================================================================
# PLOTS — Slices XZ/XY
# ================================================================================
def add_cylinder_xz(ax, y_slice):
    if abs(y_slice - YC_CYL) > R_CYL:
        return
    rect = mpatches.Rectangle((XC_CYL - R_CYL, 0), 2 * R_CYL, H_CYL,
                                linewidth=1.5, edgecolor='black', facecolor='white', zorder=10)
    ax.add_patch(rect)

def add_cylinder_xy(ax, z_slice):
    if z_slice > H_CYL:
        return
    circle = mpatches.Circle((XC_CYL, YC_CYL), R_CYL,
                               linewidth=1.5, edgecolor='black', facecolor='white', zorder=10)
    ax.add_patch(circle)

def scatter_compare(ax_cfd, ax_pinn, h_coord, v_coord, val_cfd, val_pinn,
                      title_var, unit, cmap, plane, slice_value):
    vmin = min(val_cfd.min(), val_pinn.min())
    vmax = max(val_cfd.max(), val_pinn.max())

    sc_cfd = ax_cfd.scatter(h_coord, v_coord, c=val_cfd, cmap=cmap,
                              s=3, vmin=vmin, vmax=vmax, zorder=2)
    sc_pinn = ax_pinn.scatter(h_coord, v_coord, c=val_pinn, cmap=cmap,
                                s=3, vmin=vmin, vmax=vmax, zorder=2)

    if plane == "xz":
        add_cylinder_xz(ax_cfd, slice_value)
        add_cylinder_xz(ax_pinn, slice_value)
        xlabel, ylabel = "x [m]", "z [m]"
        xlim, ylim = (X_MIN, X_MAX), (Z_MIN, Z_MAX)
    else:
        add_cylinder_xy(ax_cfd, slice_value)
        add_cylinder_xy(ax_pinn, slice_value)
        xlabel, ylabel = "x [m]", "y [m]"
        xlim, ylim = (X_MIN, X_MAX), (Y_MIN, Y_MAX)

    for ax, sc, label in [(ax_cfd, sc_cfd, "CFD"), (ax_pinn, sc_pinn, "PINN")]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{title_var} {label} [{unit}]", fontsize=10)
        ax.set_aspect('equal' if plane == "xy" else 'auto')
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)


def plot_slices(model, snaps, t_targets, plane="xz", slab_tol=8.0):
    if plane == "xz":
        slice_values = [0, 100, 200, 250, 300, 400, 500, 600, 700]
        slice_label = "y"
    else:
        slice_values = [10, 32, 65, 100, 150]
        slice_label = "z"

    times_all = np.array([t for t, _ in snaps])
    errors_by_slice = []
    col_map_cached = None

    for t_target in t_targets:
        idx_t = int(np.argmin(np.abs(times_all - t_target)))
        actual_t = float(times_all[idx_t])
        csv_path = snaps[idx_t][1]

        print(f"\n  t={t_target}s (real {actual_t:.1f}s) — carregando {os.path.basename(csv_path)}")
        _d, col_map_cached = load_snapshot(csv_path, col_map=col_map_cached)
        x_cfd = _d['x']; y_cfd = _d['y']; z_cfd = _d['z']
        u_cfd = _d['u']; v_cfd = _d['v']; w_cfd = _d['w']
        P_cfd = _d['P']

        if P_cfd.mean() > 1e4:
            P_cfd = P_cfd - P_OP

        for slice_value in slice_values:
            if plane == "xz":
                slab = np.abs(y_cfd - slice_value) < slab_tol
            else:
                slab = np.abs(z_cfd - slice_value) < slab_tol

            inside_cyl = (((x_cfd - XC_CYL)**2 + (y_cfd - YC_CYL)**2 <= R_CYL**2) &
                          (z_cfd <= H_CYL))
            valid = slab & ~inside_cyl
            n_valid = int(valid.sum())

            if n_valid < 10:
                continue

            x_s = x_cfd[valid]; y_s = y_cfd[valid]; z_s = z_cfd[valid]
            u_s = u_cfd[valid]; v_vel_s = v_cfd[valid]; w_s = w_cfd[valid]; P_s = P_cfd[valid]

            if plane == "xz":
                h_s, vc_s = x_s, z_s
            else:
                h_s, vc_s = x_s, y_s

            Vmag_cfd_s = np.sqrt(u_s**2 + v_vel_s**2 + w_s**2)

            # === PINN inference: nova normalização z-score ===
            xyzt = normalize_inputs(x_s, y_s, z_s, t_phys=actual_t)
            pred = model(tf.constant(xyzt), training=False).numpy()
            u_p, v_p, w_p, P_p = denormalize_outputs(pred)
            Vmag_p_s = np.sqrt(u_p**2 + v_p**2 + w_p**2)

            fig, axes = plt.subplots(2, 2, figsize=(13, 10))
            scatter_compare(axes[0, 0], axes[0, 1], h_s, vc_s,
                              Vmag_cfd_s, Vmag_p_s, "|V|", "m/s", "viridis", plane, slice_value)
            scatter_compare(axes[1, 0], axes[1, 1], h_s, vc_s,
                              P_s, P_p, "P", "Pa", "coolwarm", plane, slice_value)

            plt.suptitle(f"CFD vs PINN — {plane.upper()} | {slice_label}={slice_value}m | "
                          f"t={actual_t:.1f}s | n={n_valid}", fontsize=11)
            plt.tight_layout()

            fname = f"pinn_vs_cfd_{plane}_{slice_label}{slice_value:04d}_t{int(actual_t):03d}s.png"
            fpath = os.path.join(SLICES_DIR, fname)
            plt.savefig(fpath, dpi=120, bbox_inches='tight')
            plt.close(fig)

            err_V = np.abs(Vmag_p_s - Vmag_cfd_s)
            err_P = np.abs(P_p - P_s)

            errors_by_slice.append({
                't': actual_t, 'slice_label': slice_label, 'slice_value': slice_value,
                'n_pts': n_valid,
                'V_mae': float(err_V.mean()), 'V_max': float(err_V.max()),
                'P_mae': float(err_P.mean()), 'P_max': float(err_P.max()),
            })

            print(f"    {slice_label}={slice_value:4d}m | n={n_valid:>5d} | "
                  f"|V| MAE={err_V.mean():5.2f}  max={err_V.max():6.2f}  | "
                  f"P MAE={err_P.mean():5.0f}  max={err_P.max():6.0f}")

    df_errors = pd.DataFrame(errors_by_slice)
    df_errors.to_csv(os.path.join(SUMMARY_DIR, "errors_by_slice.csv"), index=False)
    print(f"\n  Resumo salvo: {SUMMARY_DIR}/errors_by_slice.csv")
    if len(df_errors) > 0:
        print(f"  Médias agregadas:")
        print(f"    |V| MAE médio: {df_errors['V_mae'].mean():.3f} m/s")
        print(f"    P MAE médio:   {df_errors['P_mae'].mean():.1f} Pa")
    return df_errors


# ================================================================================
# PLOTS — Scatter quantitativo + histogramas
# ================================================================================
def plot_scatter_and_histograms(model, snaps, t_target=None):
    if t_target is None:
        actual_t, csv_path = snaps[-1]
    else:
        times_all = np.array([t for t, _ in snaps])
        idx = int(np.argmin(np.abs(times_all - t_target)))
        actual_t, csv_path = snaps[idx]

    print(f"\n  Scatter/Hist em t={actual_t:.1f}s")
    _d, _ = load_snapshot(csv_path)
    x_cfd = _d['x']; y_cfd = _d['y']; z_cfd = _d['z']
    u_cfd = _d['u']; v_cfd = _d['v']; w_cfd = _d['w']
    P_cfd = _d['P']

    if P_cfd.mean() > 1e4:
        P_cfd = P_cfd - P_OP

    Vmag_cfd = np.sqrt(u_cfd**2 + v_cfd**2 + w_cfd**2)
    n = len(x_cfd)

    # === PINN inference com nova normalização ===
    xyzt = normalize_inputs(x_cfd, y_cfd, z_cfd, t_phys=actual_t)
    pred = model(tf.constant(xyzt), training=False).numpy()
    u_p, v_p, w_p, P_p = denormalize_outputs(pred)
    Vmag_p = np.sqrt(u_p**2 + v_p**2 + w_p**2)

    # === Scatter ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    datasets = [
        (u_cfd, u_p, 'u', 'm/s', axes[0, 0]),
        (v_cfd, v_p, 'v', 'm/s', axes[0, 1]),
        (w_cfd, w_p, 'w', 'm/s', axes[0, 2]),
        (Vmag_cfd, Vmag_p, '|V|', 'm/s', axes[1, 0]),
        (P_cfd, P_p, 'P', 'Pa', axes[1, 1]),
    ]
    for cfd_vals, pinn_vals, name, unit, ax in datasets:
        ax.scatter(cfd_vals, pinn_vals, s=1, alpha=0.3, c='steelblue')
        vmin = min(cfd_vals.min(), pinn_vals.min())
        vmax = max(cfd_vals.max(), pinn_vals.max())
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1.5, label='y=x (ideal)')
        mae = np.mean(np.abs(pinn_vals - cfd_vals))
        rmse = np.sqrt(np.mean((pinn_vals - cfd_vals)**2))
        bias = np.mean(pinn_vals - cfd_vals)
        corr = np.corrcoef(cfd_vals, pinn_vals)[0, 1]
        ax.set_xlabel(f'{name} CFD [{unit}]')
        ax.set_ylabel(f'{name} PINN [{unit}]')
        ax.set_title(f'{name}: MAE={mae:.2f} | RMSE={rmse:.2f} | bias={bias:+.2f} | r={corr:.3f}', fontsize=9)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')
    axes[1, 2].axis('off')
    plt.suptitle(f'PINN vs CFD scatter — t={actual_t:.1f}s | n={n} pts', fontsize=12)
    plt.tight_layout()
    out_scatter = os.path.join(SCATTER_DIR, 'pinn_vs_cfd_scatter.png')
    plt.savefig(out_scatter, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Salvo: {out_scatter}")

    # === Histogramas ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    error_datasets = [
        (u_p - u_cfd, 'u', 'm/s', axes[0, 0]),
        (v_p - v_cfd, 'v', 'm/s', axes[0, 1]),
        (w_p - w_cfd, 'w', 'm/s', axes[0, 2]),
        (Vmag_p - Vmag_cfd, '|V|', 'm/s', axes[1, 0]),
        (P_p - P_cfd, 'P', 'Pa', axes[1, 1]),
    ]
    for errors, name, unit, ax in error_datasets:
        ax.hist(errors, bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='red', ls='--', lw=1.5, label='zero error')
        median = np.median(errors)
        ax.axvline(median, color='green', ls=':', lw=1.5, label=f'median={median:.2f}')
        pct25 = np.percentile(errors, 25)
        pct75 = np.percentile(errors, 75)
        iqr = pct75 - pct25
        ax.set_xlabel(f'Erro {name} (PINN - CFD) [{unit}]')
        ax.set_ylabel('Frequência')
        ax.set_title(f'{name}: Q1={pct25:.2f}, Q3={pct75:.2f}, IQR={iqr:.2f}', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[1, 2].axis('off')
    plt.suptitle(f'Distribuição de erros PINN - CFD — t={actual_t:.1f}s', fontsize=12)
    plt.tight_layout()
    out_hist = os.path.join(HIST_DIR, 'pinn_vs_cfd_errors_hist.png')
    plt.savefig(out_hist, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Salvo: {out_hist}")

    # === Tabela quantitativa ===
    # Escalas físicas do problema (NÃO usar V_REF=5 que é só de normalização da rede)
    # Para "acurácia", usa V_INF (velocidade do escoamento) e P_PHYS (pressão dinâmica)
    P_PHYS_SCALE = 0.5 * RHO_INF * V_INF**2   # ≈ 177 Pa
    scales = {'u': V_INF, 'v': V_INF, 'w': V_INF, '|V|': V_INF, 'P': P_PHYS_SCALE}
    summary_rows = []

    print(f"\n  {'='*90}")
    print(f"  TABELA QUANTITATIVA — t={actual_t:.1f}s | n={n} pontos")
    print(f"  {'='*90}")
    print(f"  {'Var':<5} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'Median|err|':>11} "
          f"{'R²':>7} {'r':>7} {'NRMSE%':>7} {'Acc10%':>7} {'Acc20%':>7}")
    print(f"  {'-'*90}")

    for cfd_vals, pinn_vals, name, unit, _ in datasets:
        err = pinn_vals - cfd_vals
        abs_err = np.abs(err)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err**2)))
        bias = float(np.mean(err))
        median_abs_err = float(np.median(abs_err))
        corr = float(np.corrcoef(cfd_vals, pinn_vals)[0, 1])

        ss_res = np.sum(err**2)
        ss_tot = np.sum((cfd_vals - cfd_vals.mean())**2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        cfd_range = float(cfd_vals.max() - cfd_vals.min())
        nrmse_pct = 100.0 * rmse / cfd_range if cfd_range > 0 else 0.0

        scale = scales.get(name, cfd_range)
        tol_10pct = 0.10 * scale
        tol_20pct = 0.20 * scale
        acc_10pct = float(100.0 * np.mean(abs_err < tol_10pct))
        acc_20pct = float(100.0 * np.mean(abs_err < tol_20pct))

        rel_mae = 100.0 * mae / scale

        print(f"  {name:<5} {mae:>8.3f} {rmse:>8.3f} {bias:>+8.3f} "
              f"{median_abs_err:>11.3f} {r_squared:>+7.3f} {corr:>+7.3f} "
              f"{nrmse_pct:>6.1f}% {acc_10pct:>6.1f}% {acc_20pct:>6.1f}%")

        summary_rows.append({
            'var': name, 'unit': unit,
            'MAE': mae, 'RMSE': rmse, 'bias': bias,
            'median_abs_err': median_abs_err,
            'R2': r_squared, 'corr': corr,
            'NRMSE_pct': nrmse_pct,
            'accuracy_10pct': acc_10pct,
            'accuracy_20pct': acc_20pct,
            'rel_mae_pct': rel_mae,
        })

    print(f"  {'='*90}")
    print(f"")
    print(f"  Legenda:")
    print(f"    MAE/RMSE/Bias/Median: em unidades da variável (m/s ou Pa)")
    print(f"    R²:        coef. determinação (1=perfeito, 0=trivial, <0=pior que média)")
    print(f"    r:         correlação Pearson (-1 a +1)")
    print(f"    NRMSE%:    RMSE / range dos dados CFD")
    print(f"    Acc10/20%: %% de pontos com |erro| < 10%/20% da escala física")
    print(f"               (V_INF={V_INF} m/s, P_phys_scale={P_PHYS_SCALE:.0f} Pa)")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(SUMMARY_DIR, "final_metrics_lastsnap.csv"), index=False)
    print(f"\n  Resumo salvo: {SUMMARY_DIR}/final_metrics_lastsnap.csv")


# ================================================================================
# Diagnóstico de configuração
# ================================================================================
def print_config():
    print("="*72)
    print("  PINN INFERENCE — CONFIGURAÇÃO Z-SCORE")
    print("="*72)
    print(f"  Espacial (z-score isotrópico):")
    print(f"    X_MEAN={X_MEAN:.3f}, Y_MEAN={Y_MEAN:.3f}, Z_MEAN={Z_MEAN:.3f}")
    print(f"    DP_ISO={DP_ISO:.3f} m  (σ isotrópico)")
    print(f"  Temporal (z-score):")
    print(f"    T_MEAN={T_MEAN:.3f}, T_DP={T_DP:.3f}")
    print(f"  Saídas (esquema híbrido):")
    print(f"    u, w: min-max simétrico, V_REF={V_REF}")
    print(f"    v:    z-score, V_MEAN={V_MEAN:.3f}, V_DP={V_DP:.3f}")
    print(f"    P:    z-score, P_MEAN={P_MEAN:.3f}, P_DP={P_DP:.3f}")
    print(f"  Cilindro normalizado:")
    print(f"    centro=({XC_N:+.4f}, {YC_N:+.4f}), R_N={R_N:.4f}, topo z_n={H_N:+.4f}")
    print(f"  Domínio normalizado:")
    print(f"    xn=[{XN_MIN:+.3f}, {XN_MAX:+.3f}]")
    print(f"    yn=[{YN_MIN:+.3f}, {YN_MAX:+.3f}]")
    print(f"    zn=[{ZN_MIN:+.3f}, {ZN_MAX:+.3f}]")
    print("="*72)


# ================================================================================
# MAIN
# ================================================================================
def main():
    parser = argparse.ArgumentParser(description="Inferência standalone + plots PINN (z-score)")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Nome do arquivo .h5 ou caminho completo. Default: mais recente")
    parser.add_argument("--list", action="store_true",
                         help="Lista checkpoints disponíveis e sai")
    parser.add_argument("--no-slices", action="store_true",
                         help="Pula geração dos plots de slice")
    parser.add_argument("--no-scatter", action="store_true",
                         help="Pula scatter e histogramas")
    parser.add_argument("--no-loss", action="store_true",
                         help="Pula loss history")
    parser.add_argument("--times", type=str, default="10,30,58,100",
                         help="Tempos físicos (s) para slices, separados por vírgula")
    args = parser.parse_args()

    if args.list:
        files = list_checkpoints(CHECKPOINT_DIR)
        if not files:
            print(f"Nenhum checkpoint encontrado em {CHECKPOINT_DIR}")
            return
        print(f"Checkpoints disponíveis em {CHECKPOINT_DIR}:")
        for f in files:
            size_kb = os.path.getsize(f) / 1024
            print(f"  {os.path.basename(f)}  ({size_kb:.0f} KB)")
        return

    for d in [SLICES_DIR, SCATTER_DIR, HIST_DIR, SUMMARY_DIR, LOSSHIST_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"Output em: {OUTPUT_DIR}/")

    print_config()

    if not args.no_loss:
        print(f"\n[1/3] Loss history a partir do CSV...")
        plot_loss_history(CSV_LOG_PATH, LOSSHIST_DIR)

    if args.no_slices and args.no_scatter:
        print("\nNada mais a fazer (--no-slices e --no-scatter).")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\nGPU(s): {[g.name for g in gpus]}")
        except RuntimeError as e:
            print(e)
    else:
        print("\nWARNING: sem GPU — inferência será em CPU (mais lenta)")
    print(f"TensorFlow: {tf.__version__}")

    # === Resolver checkpoint ===
    if args.checkpoint is None:
        ckpt_path = find_latest_checkpoint(CHECKPOINT_DIR)
        if ckpt_path is None:
            print(f"ERRO: nenhum checkpoint encontrado em {CHECKPOINT_DIR}")
            print(f"Rode com --list para ver o que está disponível")
            sys.exit(1)
    elif os.path.isabs(args.checkpoint):
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(CHECKPOINT_DIR, args.checkpoint)

    if not os.path.exists(ckpt_path):
        print(f"ERRO: checkpoint não existe: {ckpt_path}")
        sys.exit(1)

    print(f"\n[2/3] Construindo modelo e carregando pesos...")
    print(f"  Checkpoint: {ckpt_path}")
    model = build_pinn(n_layers=6, n_neurons=256)
    model.load_weights(ckpt_path)
    print(f"  Parâmetros: {model.count_params():,}")

    # === Sanity check ===
    # Inputs em coords normalizadas com range realista do z-score
    test_xn = np.random.uniform(-1.0, 1.0, 5).astype(np.float32)
    test_yn = np.random.uniform(-2.0, 3.0, 5).astype(np.float32)
    test_zn = np.random.uniform(-0.5, 1.0, 5).astype(np.float32)
    test_tn = np.zeros(5, dtype=np.float32)
    test_in = tf.constant(np.column_stack([test_xn, test_yn, test_zn, test_tn]))
    test_out = model(test_in, training=False).numpy()
    print(f"  Sanity output (norm) range: [{test_out.min():.3f}, {test_out.max():.3f}]")
    print(f"  All finite: {np.all(np.isfinite(test_out))}")

    u_t, v_t, w_t, P_t = denormalize_outputs(test_out)
    print(f"  Sanity desnormalizado: u={u_t.mean():+.1f} m/s, v={v_t.mean():+.1f} m/s, "
          f"w={w_t.mean():+.1f} m/s, P={P_t.mean():+.1f} Pa")

    snaps = discover_snapshots(SNAPSHOT_DIR)
    print(f"  Snapshots disponíveis: {len(snaps)} (t={snaps[0][0]:.1f}s … {snaps[-1][0]:.1f}s)")

    if not args.no_slices:
        print(f"\n[3/3] Gerando slices CFD vs PINN...")
        t_targets = [float(t) for t in args.times.split(",")]
        plot_slices(model, snaps, t_targets, plane="xz", slab_tol=8.0)

    if not args.no_scatter:
        print(f"\nGerando scatter e histogramas (último snapshot)...")
        plot_scatter_and_histograms(model, snaps)

    print(f"\n{'='*60}")
    print(f"  CONCLUÍDO")
    print(f"{'='*60}")
    print(f"Plots em: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()