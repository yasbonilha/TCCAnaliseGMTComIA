#!/usr/bin/env python3
"""
generate_movie.py — Pipeline PINN vs CFD comparativo (4 vídeos)
==================================================================

Gera 4 vídeos comparando inferência da PINN com dados CFD em dois cortes:
  - Vista superior: corte XY em z=32.5m (meio do cilindro)
  - Vista lateral:  corte YZ em x=140m (centro do cilindro)

Cada vista é renderizada em duas escalas de cor:
  - Escala CFD: V∈[14,20] m/s, P∈[-50,+50] Pa (replicando vídeo CFD existente)
  - Escala global: percentil 1/99 dos dados reais (visualização completa)

Cada frame é um layout 2×3:
  | CFD V  | PINN V | |erro V| |
  | CFD P  | PINN P | |erro P| |

USO:
    # Teste com 5 frames antes de rodar tudo:
    python generate_movie.py --test-mode
    
    # Execução completa:
    python generate_movie.py
    
    # Pula a Fase 1 (cache CFD) se já existe:
    python generate_movie.py --skip-cache
    
    # Só REPLOTAR (ajuste estético) sem reinferir nem reinterpolar:
    python generate_movie.py --skip-cache --skip-inference
    
    # Pula geração de PNGs (só recompila vídeos):
    python generate_movie.py --skip-cache --skip-inference --skip-frames

PIPELINE:
    Fase 1:  Cache CFD interpolado para grades 2D (~12 min, uma vez)
    Fase 2:  Cálculo de escalas globais (~1 min)
    Fase 3a: Inferência PINN -> cache .npy (uma vez; rápida, gargalo é o plot)
    Fase 3b: Geração de frames a partir dos caches (replotável em minutos)
    Fase 4:  Compilação dos 4 vídeos com ffmpeg (~1-2 min)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import tensorflow as tf
from tensorflow.keras import layers, Model

from scipy.interpolate import LinearNDInterpolator


# ============================================================================
# CONFIGURAÇÃO — AJUSTE SE NECESSÁRIO
# ============================================================================
# Diretórios
HOME_DIR    = Path("/home/tmoraes/treino_normalizado")
CSV_DIR     = Path("/home/tmoraes/CSVs")
WORK_DIR    = HOME_DIR / "inference_movie"
CACHE_DIR   = WORK_DIR / "cfd_interp_cache"
FRAMES_DIR  = WORK_DIR / "frames"
VIDEO_DIR   = WORK_DIR
CKPT_PATH   = HOME_DIR / "pinn_checkpoints_segregated" / "pinn_best_segregated.weights.h5"

# Padrão dos arquivos CSV
CSV_PATTERN = "timestep-{:04d}.csv"  # timestep-0001.csv, etc

# Domínio físico
X_MIN, X_MAX = 0.0, 280.0
Y_MIN, Y_MAX = 0.0, 700.0
Z_MIN, Z_MAX = 0.0, 190.0

# Cilindro
XC, YC = 140.0, 200.0
R_CYL  = 27.0    # raio físico = D/2 = 54/2
H_CYL  = 65.0    # altura

# Cortes
Z_SLICE = 32.5   # vista superior (XY)
X_SLICE = 140.0  # vista lateral  (YZ)

# Tolerâncias para filtrar pontos próximos ao plano de corte
TOL_Z = 5.0      # m, para corte XY
TOL_X = 10.0     # m, para corte YZ

# Resolução das grades 2D
DX = 1.0
DY = 1.0
DZ = 1.0

# Parâmetros do vídeo
T_START   = 15.0   # s
T_END     = 100.0  # s
N_FRAMES  = 192
FPS       = 24
DT_CFD    = 0.1    # s, intervalo entre snapshots no CFD

# Escalas da colorbar
V_CFD_RANGE = (14.0, 20.0)     # m/s
P_CFD_RANGE = (-50.0, 50.0)    # Pa

# Escala fixa para o plot de erro: fração de uma referência física.
# Em vez de deixar o vmax do erro flutuar com o percentil 95 de cada frame
# (que faz QUALQUER campo de erro parecer "todo vermelho"), fixamos o teto
# em uma fração conhecida da grandeza, e dizemos isso no título da coluna.
# Assim a banca lê a magnitude direto: vermelho-escuro = ERR_FRAC do campo.
ERR_FRAC      = 0.05               # 5%
ERR_V_VMAX    = ERR_FRAC * 17.0                        # 0.85 m/s  (V_INF=17)
ERR_P_VMAX    = ERR_FRAC * (P_CFD_RANGE[1] - P_CFD_RANGE[0])  # 5.0 Pa (5% da faixa CFD de 100 Pa)

# Inferência: chunk size para PINN
PINN_CHUNK = 50_000

# Arquitetura PINN
N_NEURONS = 512
N_BLOCKS  = 6

# ============================================================================
# CONSTANTES FÍSICAS E NORMALIZAÇÃO (mesmas do treino)
# ============================================================================
V_INF   = 17.0
RHO_INF = 1.225
MU      = 1.7894e-5

X_MEAN = 140.01174139636194
Y_MEAN = 290.9150766181903
Z_MEAN = 66.39128138069277
DP_ISO = 101.234187329139
T_MEAN = 50.05
T_DP   = 28.867499025720953

V_REF  = 5.0
V_MEAN_OUT = 14.321999262773115
V_DP   = 7.399817765728233
P_MEAN_OUT = -32.78900000449526
P_DP   = 98.41228792498488

# Geometria normalizada (para a HardConstraintLayer)
XC_N = (XC - X_MEAN) / DP_ISO
YC_N = (YC - Y_MEAN) / DP_ISO
R_N  = R_CYL / DP_ISO
H_N  = (H_CYL - Z_MEAN) / DP_ISO
XN_MIN = (X_MIN - X_MEAN) / DP_ISO
XN_MAX = (X_MAX - X_MEAN) / DP_ISO
YN_MIN = (Y_MIN - Y_MEAN) / DP_ISO
YN_MAX = (Y_MAX - Y_MEAN) / DP_ISO
ZN_MIN = (Z_MIN - Z_MEAN) / DP_ISO
ZN_MAX = (Z_MAX - Z_MEAN) / DP_ISO


# ============================================================================
# HARDCONSTRAINTLAYER (idêntica à do treino)
# ============================================================================
class HardConstraintLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def smooth_max(a, b, k=50.0):
        m = tf.maximum(a, b)
        return m + (1.0 / k) * tf.math.log(
            tf.exp(k * (a - m)) + tf.exp(k * (b - m)))

    def call(self, inputs):
        xyzt, raw_pred = inputs
        xn = xyzt[:, 0:1]; yn = xyzt[:, 1:2]; zn = xyzt[:, 2:3]
        un = raw_pred[:, 0:1]; vn = raw_pred[:, 1:2]
        wn = raw_pred[:, 2:3]; Pn = raw_pred[:, 3:4]

        mt = 135.0; rate = 72.0; k_smooth = 75.0; k_r2 = 238.0

        D_inlet  = tf.math.tanh(mt * (yn - YN_MIN))
        D_outlet = tf.math.tanh(mt * (YN_MAX - yn))
        D_ground = tf.math.tanh(mt * (zn - ZN_MIN))
        D_xmin   = tf.math.tanh(mt * (xn - XN_MIN))
        D_xmax   = tf.math.tanh(mt * (XN_MAX - xn))
        D_ztop   = tf.math.tanh(mt * (ZN_MAX - zn))

        r2 = tf.square(xn - XC_N) + tf.square(yn - YC_N)
        d_rad_sq = r2 - R_N * R_N
        d_top = zn - H_N
        d_cyl_3d = self.smooth_max(d_rad_sq, d_top, k=k_smooth)
        D_cyl = tf.math.tanh(k_r2 * tf.maximum(d_cyl_3d, 0.0))

        D_vel = D_inlet * D_ground * D_cyl * D_xmin * D_xmax * D_ztop

        decay_inlet = tf.exp(-rate * (yn - YN_MIN))
        decay_xmin  = tf.exp(-rate * (xn - XN_MIN))
        decay_xmax  = tf.exp(-rate * (XN_MAX - xn))
        decay_ztop  = tf.exp(-rate * (ZN_MAX - zn))
        decay_freestream = 1.0 - (1.0 - decay_inlet) * (1.0 - decay_xmin) \
                              * (1.0 - decay_xmax) * (1.0 - decay_ztop)

        u_hard = un * D_vel
        w_hard = wn * D_vel
        v_phys_raw = vn * V_DP + V_MEAN_OUT
        v_phys_hard = v_phys_raw * D_vel + V_INF * decay_freestream * D_ground * D_cyl
        v_hard = (v_phys_hard - V_MEAN_OUT) / V_DP
        P_phys_raw = Pn * P_DP + P_MEAN_OUT
        P_phys_hard = P_phys_raw * D_outlet
        P_hard = (P_phys_hard - P_MEAN_OUT) / P_DP

        return tf.concat([u_hard, v_hard, w_hard, P_hard], axis=1)


def build_pinn(n_layers=N_BLOCKS, n_neurons=N_NEURONS):
    inp = layers.Input(shape=(4,), name="xyzt")
    h = inp
    for i in range(n_layers):
        h = layers.Dense(n_neurons, activation="tanh",
                         kernel_initializer="glorot_uniform",
                         bias_initializer="zeros",
                         name=f"dense_{i+1}")(h)
    raw_out = layers.Dense(4, name="raw_fields")(h)
    final_out = HardConstraintLayer(name="hard_constraints")([inp, raw_out])
    return Model(inp, final_out, name="PINN_MLP")


# ============================================================================
# CRIA GRADES 2D REGULARES
# ============================================================================
def create_grids():
    """Cria grades 2D fixas e máscaras do cilindro. Retorna dict com tudo."""
    # Grade XY em z=Z_SLICE
    x_xy_1d = np.arange(X_MIN, X_MAX + DX, DX, dtype=np.float64)
    y_xy_1d = np.arange(Y_MIN, Y_MAX + DY, DY, dtype=np.float64)
    X_xy, Y_xy = np.meshgrid(x_xy_1d, y_xy_1d, indexing='xy')
    Z_xy = np.full_like(X_xy, Z_SLICE)
    # Máscara: True onde o ponto está dentro do cilindro
    mask_xy_cyl = ((X_xy - XC)**2 + (Y_xy - YC)**2 <= R_CYL**2)

    # Grade YZ em x=X_SLICE
    y_yz_1d = np.arange(Y_MIN, Y_MAX + DY, DY, dtype=np.float64)
    z_yz_1d = np.arange(Z_MIN, Z_MAX + DZ, DZ, dtype=np.float64)
    Y_yz, Z_yz = np.meshgrid(y_yz_1d, z_yz_1d, indexing='xy')
    X_yz = np.full_like(Y_yz, X_SLICE)
    # Máscara: True onde o ponto está dentro do cilindro
    # No corte YZ em x=140, o cilindro é um retângulo y∈[173,227], z∈[0,65]
    mask_yz_cyl = ((np.abs(Y_yz - YC) <= R_CYL) & (Z_yz <= H_CYL))

    print(f"  Grade XY (z={Z_SLICE}): {X_xy.shape} = {X_xy.size:,} pontos")
    print(f"    Pontos no cilindro: {mask_xy_cyl.sum():,}")
    print(f"  Grade YZ (x={X_SLICE}): {Y_yz.shape} = {Y_yz.size:,} pontos")
    print(f"    Pontos no cilindro: {mask_yz_cyl.sum():,}")

    return {
        'xy': {
            'X': X_xy, 'Y': Y_xy, 'Z': Z_xy,
            'x_1d': x_xy_1d, 'y_1d': y_xy_1d,
            'mask_cyl': mask_xy_cyl,
            # --- Exibição DEITADA: horizontal = Y (fluxo p/ direita),
            #     vertical = X crescendo p/ baixo (origin='upper').
            #     Os dados (701,281) são transpostos só no plot (data.T).
            'extent': [Y_MIN, Y_MAX, X_MAX, X_MIN],  # h: Y 0->700 ; v: X invertido (baixo cresce)
            'xlabel': 'y [m]', 'ylabel': 'x [m]',
            'transpose': True,
            'origin': 'upper',
        },
        'yz': {
            'X': X_yz, 'Y': Y_yz, 'Z': Z_yz,
            'y_1d': y_yz_1d, 'z_1d': z_yz_1d,
            'mask_cyl': mask_yz_cyl,
            'extent': [Y_MIN, Y_MAX, Z_MIN, Z_MAX],
            'xlabel': 'y [m]', 'ylabel': 'z [m]',
            'transpose': False,
            'origin': 'lower',
        }
    }


# ============================================================================
# FASE 1 — CACHE DE CFD INTERPOLADO
# ============================================================================
def cache_paths(ts_idx):
    """Retorna paths dos 4 arquivos .npy do cache de um timestep."""
    xy_dir = CACHE_DIR / "xy_z32.5"
    yz_dir = CACHE_DIR / "yz_x140"
    return {
        'xy_V': xy_dir / f"V_t{ts_idx:04d}.npy",
        'xy_P': xy_dir / f"P_t{ts_idx:04d}.npy",
        'yz_V': yz_dir / f"V_t{ts_idx:04d}.npy",
        'yz_P': yz_dir / f"P_t{ts_idx:04d}.npy",
    }


def cache_exists(ts_idx):
    """Verifica se TODOS os 4 .npy do cache existem."""
    paths = cache_paths(ts_idx)
    return all(p.exists() for p in paths.values())


def load_from_cache(ts_idx, grids):
    """Carrega .npy do cache. Reaplica máscara do cilindro (NaN no interior)."""
    paths = cache_paths(ts_idx)
    V_xy = np.load(paths['xy_V'])
    P_xy = np.load(paths['xy_P'])
    V_yz = np.load(paths['yz_V'])
    P_yz = np.load(paths['yz_P'])
    # Máscara já foi aplicada quando salvou, mas reaplicamos por garantia
    V_xy[grids['xy']['mask_cyl']] = np.nan
    P_xy[grids['xy']['mask_cyl']] = np.nan
    V_yz[grids['yz']['mask_cyl']] = np.nan
    P_yz[grids['yz']['mask_cyl']] = np.nan
    return {'V_xy': V_xy, 'P_xy': P_xy, 'V_yz': V_yz, 'P_yz': P_yz}


def interpolate_snapshot(ts_idx, grids):
    """
    Interpola UM snapshot do CFD para as duas grades 2D e salva em cache.
    
    Para acelerar, filtra pontos próximos ao plano de corte antes de interpolar.
    """
    csv_path = CSV_DIR / CSV_PATTERN.format(ts_idx)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    # Carrega CSV
    df = pd.read_csv(csv_path, sep=r'\s+', engine='python')
    # Detecta nomes das colunas (Fluent usa 'x-coordinate', 'pressure', etc)
    cols = {c.lower().strip(): c for c in df.columns}
    
    def find_col(candidates):
        for cand in candidates:
            for k, v in cols.items():
                if cand in k:
                    return v
        return None

    col_x = find_col(['x-coordinate', 'x-coord', 'x_coord'])
    col_y = find_col(['y-coordinate', 'y-coord', 'y_coord'])
    col_z = find_col(['z-coordinate', 'z-coord', 'z_coord'])
    col_u = find_col(['x-velocity', 'x-vel', 'u'])
    col_v = find_col(['y-velocity', 'y-vel'])
    col_w = find_col(['z-velocity', 'z-vel'])
    col_P = find_col(['pressure'])

    if not all([col_x, col_y, col_z, col_u, col_v, col_w, col_P]):
        raise ValueError(f"Colunas não encontradas no CSV {csv_path}\n"
                         f"Encontradas: {list(df.columns)}")

    x = df[col_x].values.astype(np.float64)
    y = df[col_y].values.astype(np.float64)
    z = df[col_z].values.astype(np.float64)
    u = df[col_u].values.astype(np.float64)
    v = df[col_v].values.astype(np.float64)
    w = df[col_w].values.astype(np.float64)
    P = df[col_P].values.astype(np.float64)
    V_mag = np.sqrt(u*u + v*v + w*w)

    # ---- Corte XY (z=Z_SLICE) ----
    mask_z = np.abs(z - Z_SLICE) < TOL_Z
    pts_xy = np.column_stack([x[mask_z], y[mask_z], z[mask_z]])
    
    # Interpola V e P
    interp_V = LinearNDInterpolator(pts_xy, V_mag[mask_z], fill_value=np.nan)
    interp_P = LinearNDInterpolator(pts_xy, P[mask_z], fill_value=np.nan)
    
    grid_xy = grids['xy']
    query_xy = np.column_stack([
        grid_xy['X'].flatten(),
        grid_xy['Y'].flatten(),
        grid_xy['Z'].flatten()
    ])
    V_xy = interp_V(query_xy).reshape(grid_xy['X'].shape)
    P_xy = interp_P(query_xy).reshape(grid_xy['X'].shape)
    
    # Aplica máscara do cilindro
    V_xy[grid_xy['mask_cyl']] = np.nan
    P_xy[grid_xy['mask_cyl']] = np.nan

    # ---- Corte YZ (x=X_SLICE) ----
    mask_x = np.abs(x - X_SLICE) < TOL_X
    pts_yz = np.column_stack([x[mask_x], y[mask_x], z[mask_x]])
    interp_V = LinearNDInterpolator(pts_yz, V_mag[mask_x], fill_value=np.nan)
    interp_P = LinearNDInterpolator(pts_yz, P[mask_x], fill_value=np.nan)
    
    grid_yz = grids['yz']
    query_yz = np.column_stack([
        grid_yz['X'].flatten(),
        grid_yz['Y'].flatten(),
        grid_yz['Z'].flatten()
    ])
    V_yz = interp_V(query_yz).reshape(grid_yz['Y'].shape)
    P_yz = interp_P(query_yz).reshape(grid_yz['Y'].shape)
    V_yz[grid_yz['mask_cyl']] = np.nan
    P_yz[grid_yz['mask_cyl']] = np.nan

    # Salva como float32 (metade do tamanho, precisão suficiente)
    paths = cache_paths(ts_idx)
    np.save(paths['xy_V'], V_xy.astype(np.float32))
    np.save(paths['xy_P'], P_xy.astype(np.float32))
    np.save(paths['yz_V'], V_yz.astype(np.float32))
    np.save(paths['yz_P'], P_yz.astype(np.float32))

    return {'V_xy': V_xy, 'P_xy': P_xy, 'V_yz': V_yz, 'P_yz': P_yz}


def run_phase1_cache(timesteps_idx, grids, force=False):
    """Fase 1: pré-computa interpolação do CFD para todos os timesteps."""
    (CACHE_DIR / "xy_z32.5").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "yz_x140").mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*72}")
    print(f"FASE 1: Interpolando CFD para grades 2D")
    print(f"{'='*72}")
    print(f"  Cache em: {CACHE_DIR}")
    print(f"  Timesteps: {len(timesteps_idx)}")
    
    t0 = time.time()
    for i, ts_idx in enumerate(timesteps_idx):
        if cache_exists(ts_idx) and not force:
            continue
        try:
            interpolate_snapshot(ts_idx, grids)
        except Exception as e:
            print(f"  [ERRO] ts={ts_idx}: {e}")
            continue
        if (i+1) % 10 == 0 or i == len(timesteps_idx) - 1:
            elapsed = time.time() - t0
            eta = elapsed * (len(timesteps_idx) - i - 1) / (i + 1)
            print(f"  [{i+1:>3}/{len(timesteps_idx)}] ts={ts_idx} "
                  f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min")
    
    print(f"  Fase 1 concluída em {(time.time()-t0)/60:.1f} min")


# ============================================================================
# FASE 2 — ESCALAS GLOBAIS
# ============================================================================
def run_phase2_scales(timesteps_idx, grids):
    """Calcula percentis 1/99 dos campos para cada corte."""
    print(f"\n{'='*72}")
    print(f"FASE 2: Calculando escalas globais")
    print(f"{'='*72}")
    
    all_V_xy, all_P_xy, all_V_yz, all_P_yz = [], [], [], []
    
    for i, ts_idx in enumerate(timesteps_idx):
        if not cache_exists(ts_idx):
            continue
        data = load_from_cache(ts_idx, grids)
        all_V_xy.append(data['V_xy'].flatten())
        all_P_xy.append(data['P_xy'].flatten())
        all_V_yz.append(data['V_yz'].flatten())
        all_P_yz.append(data['P_yz'].flatten())
    
    all_V_xy = np.concatenate(all_V_xy)
    all_P_xy = np.concatenate(all_P_xy)
    all_V_yz = np.concatenate(all_V_yz)
    all_P_yz = np.concatenate(all_P_yz)
    
    scales = {
        'V_xy_global': (float(np.nanpercentile(all_V_xy, 1)),
                        float(np.nanpercentile(all_V_xy, 99))),
        'P_xy_global': (float(np.nanpercentile(all_P_xy, 1)),
                        float(np.nanpercentile(all_P_xy, 99))),
        'V_yz_global': (float(np.nanpercentile(all_V_yz, 1)),
                        float(np.nanpercentile(all_V_yz, 99))),
        'P_yz_global': (float(np.nanpercentile(all_P_yz, 1)),
                        float(np.nanpercentile(all_P_yz, 99))),
        'V_cfd_range': list(V_CFD_RANGE),
        'P_cfd_range': list(P_CFD_RANGE),
    }
    
    for k, v in scales.items():
        print(f"  {k}: [{v[0]:.2f}, {v[1]:.2f}]")
    
    out_path = WORK_DIR / "scales.json"
    with open(out_path, 'w') as f:
        json.dump(scales, f, indent=2)
    print(f"  Salvo em: {out_path}")
    
    return scales


# ============================================================================
# FASE 3 — INFERÊNCIA PINN E GERAÇÃO DE FRAMES
# ============================================================================
def predict_pinn(model, X_flat, Y_flat, Z_flat, t_phys):
    """
    Avalia a PINN em pontos físicos arbitrários.
    Retorna |V| e P em unidades físicas.
    """
    n_pts = len(X_flat)
    
    # Normaliza
    xn = ((X_flat - X_MEAN) / DP_ISO).astype(np.float32)
    yn = ((Y_flat - Y_MEAN) / DP_ISO).astype(np.float32)
    zn = ((Z_flat - Z_MEAN) / DP_ISO).astype(np.float32)
    tn = (np.full(n_pts, (t_phys - T_MEAN) / T_DP)).astype(np.float32)
    
    xyzt = np.column_stack([xn, yn, zn, tn])
    
    # Inferência em chunks (evita OOM em grades grandes)
    V_all = np.zeros(n_pts, dtype=np.float32)
    P_all = np.zeros(n_pts, dtype=np.float32)
    
    for start in range(0, n_pts, PINN_CHUNK):
        end = min(start + PINN_CHUNK, n_pts)
        chunk = tf.constant(xyzt[start:end])
        pred = model(chunk, training=False).numpy()
        
        # Desnormaliza
        u = pred[:, 0] * V_REF
        v = pred[:, 1] * V_DP + V_MEAN_OUT
        w = pred[:, 2] * V_REF
        P = pred[:, 3] * P_DP + P_MEAN_OUT
        
        V_all[start:end] = np.sqrt(u*u + v*v + w*w)
        P_all[start:end] = P
    
    return V_all, P_all


def predict_pinn_on_grids(model, t_phys, grids):
    """Avalia PINN em ambas as grades 2D."""
    # Corte XY
    grid_xy = grids['xy']
    V_xy_flat, P_xy_flat = predict_pinn(
        model,
        grid_xy['X'].flatten(),
        grid_xy['Y'].flatten(),
        grid_xy['Z'].flatten(),
        t_phys
    )
    V_xy = V_xy_flat.reshape(grid_xy['X'].shape)
    P_xy = P_xy_flat.reshape(grid_xy['X'].shape)
    V_xy[grid_xy['mask_cyl']] = np.nan
    P_xy[grid_xy['mask_cyl']] = np.nan
    
    # Corte YZ
    grid_yz = grids['yz']
    V_yz_flat, P_yz_flat = predict_pinn(
        model,
        grid_yz['X'].flatten(),
        grid_yz['Y'].flatten(),
        grid_yz['Z'].flatten(),
        t_phys
    )
    V_yz = V_yz_flat.reshape(grid_yz['Y'].shape)
    P_yz = P_yz_flat.reshape(grid_yz['Y'].shape)
    V_yz[grid_yz['mask_cyl']] = np.nan
    P_yz[grid_yz['mask_cyl']] = np.nan
    
    return {'V_xy': V_xy, 'P_xy': P_xy, 'V_yz': V_yz, 'P_yz': P_yz}


# ---- Cache da inferência PINN (separa inferência de plotagem) --------------
# Motivo: a plotagem (matplotlib) é o gargalo real (~150 min). Cacheando a
# inferência em .npy, qualquer ajuste ESTÉTICO futuro (escala, aspect, cores)
# replota em minutos sem reinferir. A inferência roda uma vez e fica salva.
PINN_CACHE_DIR = WORK_DIR / "pinn_interp_cache"


def pinn_cache_paths(ts_idx):
    xy_dir = PINN_CACHE_DIR / "xy_z32.5"
    yz_dir = PINN_CACHE_DIR / "yz_x140"
    return {
        'xy_V': xy_dir / f"V_t{ts_idx:04d}.npy",
        'xy_P': xy_dir / f"P_t{ts_idx:04d}.npy",
        'yz_V': yz_dir / f"V_t{ts_idx:04d}.npy",
        'yz_P': yz_dir / f"P_t{ts_idx:04d}.npy",
    }


def pinn_cache_exists(ts_idx):
    return all(p.exists() for p in pinn_cache_paths(ts_idx).values())


def load_pinn_from_cache(ts_idx, grids):
    paths = pinn_cache_paths(ts_idx)
    V_xy = np.load(paths['xy_V']); P_xy = np.load(paths['xy_P'])
    V_yz = np.load(paths['yz_V']); P_yz = np.load(paths['yz_P'])
    V_xy[grids['xy']['mask_cyl']] = np.nan
    P_xy[grids['xy']['mask_cyl']] = np.nan
    V_yz[grids['yz']['mask_cyl']] = np.nan
    P_yz[grids['yz']['mask_cyl']] = np.nan
    return {'V_xy': V_xy, 'P_xy': P_xy, 'V_yz': V_yz, 'P_yz': P_yz}


def save_pinn_to_cache(ts_idx, pinn_data):
    paths = pinn_cache_paths(ts_idx)
    np.save(paths['xy_V'], pinn_data['V_xy'].astype(np.float32))
    np.save(paths['xy_P'], pinn_data['P_xy'].astype(np.float32))
    np.save(paths['yz_V'], pinn_data['V_yz'].astype(np.float32))
    np.save(paths['yz_P'], pinn_data['P_yz'].astype(np.float32))


def run_phase3a_inference(timesteps_idx, model, grids, force=False):
    """Fase 3a: infere a PINN em todas as grades e SALVA em cache .npy."""
    (PINN_CACHE_DIR / "xy_z32.5").mkdir(parents=True, exist_ok=True)
    (PINN_CACHE_DIR / "yz_x140").mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"FASE 3a: Inferência PINN (com cache em disco)")
    print(f"{'='*72}")
    print(f"  Cache em: {PINN_CACHE_DIR}")

    t0 = time.time()
    for i, ts_idx in enumerate(timesteps_idx):
        if pinn_cache_exists(ts_idx) and not force:
            continue
        t_phys = ts_idx * DT_CFD
        pinn_data = predict_pinn_on_grids(model, t_phys, grids)
        save_pinn_to_cache(ts_idx, pinn_data)
        if (i+1) % 20 == 0 or i == len(timesteps_idx) - 1:
            elapsed = time.time() - t0
            eta = elapsed * (len(timesteps_idx) - i - 1) / (i + 1)
            print(f"  [{i+1:>3}/{len(timesteps_idx)}] ts={ts_idx} "
                  f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min")
    print(f"  Fase 3a concluída em {(time.time()-t0)/60:.1f} min")


def get_cylinder_patch(view):
    """Cria patch matplotlib para mostrar contorno do cilindro no plot."""
    if view == 'xy':
        # Vista DEITADA: horizontal=Y, vertical=X. Centro (YC, XC)=(200,140).
        # Círculo de raio 27 (continua círculo, só os eixos foram trocados).
        return Circle((YC, XC), R_CYL, fill=False, edgecolor='black',
                      linewidth=1.5, linestyle='--')
    else:  # yz
        # Retângulo: y∈[173, 227], z∈[0, 65]
        return Rectangle((YC - R_CYL, 0), 2*R_CYL, H_CYL,
                         fill=False, edgecolor='black',
                         linewidth=1.5, linestyle='--')


def plot_frame(t_phys, idx, cfd_data, pinn_data, grids, scales,
               view, scale_type, frames_dir):
    """
    Gera UM frame PNG para uma combinação (view, scale_type).
    
    Layout 2×2:
      | CFD V  | PINN V |
      | CFD P  | PINN P |
    Vista XY exibida deitada (horizontal=y, vertical=x p/ baixo) para casar
    com a orientação da vista YZ no slide.
    """
    grid = grids[view]
    
    # Seleciona campos
    V_cfd = cfd_data[f'V_{view}']
    P_cfd = cfd_data[f'P_{view}']
    V_pin = pinn_data[f'V_{view}']
    P_pin = pinn_data[f'P_{view}']
    
    # --- Orientação de exibição ---
    # A vista XY é exibida DEITADA (horizontal=Y, vertical=X p/ baixo). Para isso
    # os arrays (Y nas linhas, X nas colunas) são transpostos só aqui no plot,
    # e usamos origin='upper' + extent com X invertido. A vista YZ não muda.
    do_T = grid.get('transpose', False)
    origin = grid.get('origin', 'lower')
    if do_T:
        V_cfd, P_cfd = V_cfd.T, P_cfd.T
        V_pin, P_pin = V_pin.T, P_pin.T
    
    # Define escala
    if scale_type == 'cfd':
        v_range = V_CFD_RANGE
        p_range = P_CFD_RANGE
    else:
        v_range = scales[f'V_{view}_global']
        p_range = scales[f'P_{view}_global']
    
    # --- Aspect físico: 1 m == 1 m em ambos os eixos (sem distorção) ---
    # extent = [h0, h1, v0, v1]; v pode estar invertido (X p/ baixo), por isso abs.
    extent = grid['extent']
    phys_w = abs(extent[1] - extent[0])
    phys_h = abs(extent[3] - extent[2])
    aspect_ratio = phys_w / phys_h  # >1 = subplot largo e baixo; <1 = alto e estreito
    
    # figsize calculado a partir do aspect real de cada vista, para não sobrar
    # espaço branco. 2 colunas lado a lado, 2 linhas. Folga p/ colorbars/títulos.
    PANEL_H = 3.0  # altura base de cada subplot em polegadas
    panel_w = PANEL_H * aspect_ratio
    fig_w = 2 * panel_w + 2.0   # 2 colunas + margem p/ colorbars
    fig_h = 2 * PANEL_H + 1.2   # 2 linhas + margem p/ títulos
    # Limita extremos para não gerar figuras absurdas
    fig_w = float(np.clip(fig_w, 7.0, 22.0))
    fig_h = float(np.clip(fig_h, 5.0, 22.0))
    
    # Cria figura (2×2: colunas CFD e PINN; linhas |V| e P)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), dpi=120,
                             constrained_layout=True)
    
    titles = ['CFD', 'PINN']
    
    # Configura colormap com NaN cinza
    cmap_jet = plt.cm.jet.copy()
    cmap_jet.set_bad('gray', alpha=1.0)
    
    # ---- Linha 0: |V| ----
    for col, (data, title) in enumerate(zip([V_cfd, V_pin], titles)):
        ax = axes[0, col]
        im = ax.imshow(data, extent=extent, origin=origin,
                       vmin=v_range[0], vmax=v_range[1],
                       cmap=cmap_jet, interpolation='bilinear', aspect='equal')
        ax.set_title(f'{title}  |V| [m/s]', fontsize=11)
        ax.add_patch(get_cylinder_patch(view))
        if col == 0:
            ax.set_ylabel(grid['ylabel'])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # ---- Linha 1: P ----
    for col, (data, title) in enumerate(zip([P_cfd, P_pin], titles)):
        ax = axes[1, col]
        im = ax.imshow(data, extent=extent, origin=origin,
                       vmin=p_range[0], vmax=p_range[1],
                       cmap=cmap_jet, interpolation='bilinear', aspect='equal')
        ax.set_title(f'{title}  P [Pa]', fontsize=11)
        ax.add_patch(get_cylinder_patch(view))
        ax.set_xlabel(grid['xlabel'])
        if col == 0:
            ax.set_ylabel(grid['ylabel'])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Título geral
    view_name = "Vista Superior (z=32.5m)" if view == 'xy' else "Vista Lateral (x=140m)"
    scale_name = "escala CFD" if scale_type == 'cfd' else "escala global"
    fig.suptitle(f'{view_name}  |  {scale_name}  |  t = {t_phys:.2f} s',
                 fontsize=13, fontweight='bold')
    
    # Salva — SEM bbox_inches='tight' (era a causa das dimensões ímpares no
    # libx264). Com figsize proporcional + constrained_layout não sobra branco.
    out_path = frames_dir / f'frame_{idx:04d}.png'
    fig.savefig(out_path, pil_kwargs={'optimize': True})
    plt.close(fig)


def run_phase3_frames(timesteps_idx, grids, scales):
    """Fase 3b: gera todos os PNGs lendo a inferência PINN do cache (Fase 3a)."""
    print(f"\n{'='*72}")
    print(f"FASE 3b: Geração de frames (lendo inferência do cache)")
    print(f"{'='*72}")
    
    # Cria diretórios
    dirs = {}
    for view in ['xy', 'yz']:
        for st in ['cfd', 'global']:
            d = FRAMES_DIR / f'{view}_{st}_scale'
            d.mkdir(parents=True, exist_ok=True)
            dirs[(view, st)] = d
    
    t0 = time.time()
    for i, ts_idx in enumerate(timesteps_idx):
        t_phys = ts_idx * DT_CFD
        
        if not cache_exists(ts_idx):
            print(f"  [SKIP] ts={ts_idx}: cache CFD ausente")
            continue
        if not pinn_cache_exists(ts_idx):
            print(f"  [SKIP] ts={ts_idx}: cache PINN ausente (rode a Fase 3a)")
            continue
        
        cfd_data = load_from_cache(ts_idx, grids)
        pinn_data = load_pinn_from_cache(ts_idx, grids)
        
        # Gera 4 frames
        for view in ['xy', 'yz']:
            for st in ['cfd', 'global']:
                plot_frame(t_phys, i, cfd_data, pinn_data, grids, scales,
                           view, st, dirs[(view, st)])
        
        if (i+1) % 10 == 0 or i == len(timesteps_idx) - 1:
            elapsed = time.time() - t0
            eta = elapsed * (len(timesteps_idx) - i - 1) / (i + 1)
            print(f"  [{i+1:>3}/{len(timesteps_idx)}] ts={ts_idx} "
                  f"t={t_phys:.1f}s  elapsed={elapsed/60:.1f}min "
                  f"eta={eta/60:.1f}min")
    
    print(f"  Fase 3b concluída em {(time.time()-t0)/60:.1f} min")


# ============================================================================
# FASE 4 — COMPILAÇÃO DOS VÍDEOS
# ============================================================================
def run_phase4_videos():
    """Fase 4: chama ffmpeg para compilar os 4 vídeos."""
    print(f"\n{'='*72}")
    print(f"FASE 4: Compilando vídeos com ffmpeg")
    print(f"{'='*72}")
    
    for view in ['xy', 'yz']:
        for st in ['cfd', 'global']:
            frames_pattern = FRAMES_DIR / f'{view}_{st}_scale' / 'frame_%04d.png'
            video_path = VIDEO_DIR / f'video_{view}_{st}_scale.mp4'
            
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(FPS),
                '-i', str(frames_pattern),
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '18',
                str(video_path)
            ]
            
            print(f"  Compilando: {video_path.name}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    size_mb = video_path.stat().st_size / 1024 / 1024
                    print(f"    OK ({size_mb:.1f} MB)")
                else:
                    print(f"    ERRO no ffmpeg:")
                    print(result.stderr[-500:])
            except FileNotFoundError:
                print(f"    ERRO: ffmpeg não encontrado. Instale com:")
                print(f"      conda install -c conda-forge ffmpeg")
                return


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', action='store_true',
                        help='Roda apenas 5 frames (validação rápida)')
    parser.add_argument('--skip-cache', action='store_true',
                        help='Pula Fase 1 (assume cache CFD já existe)')
    parser.add_argument('--skip-frames', action='store_true',
                        help='Pula Fase 3b (só recompila vídeos)')
    parser.add_argument('--skip-inference', action='store_true',
                        help='Pula Fase 3a (assume cache PINN já existe). '
                             'Use junto com --skip-cache para só replotar.')
    parser.add_argument('--force-cache', action='store_true',
                        help='Recomputa cache mesmo se já existir')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path para checkpoint .h5 da PINN')
    args = parser.parse_args()
    
    # Setup
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Diretório de trabalho: {WORK_DIR}")
    
    # GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print(f"GPUs disponíveis: {len(gpus)}")
    
    # Cria grades
    print(f"\n{'='*72}")
    print("Criando grades 2D regulares")
    print(f"{'='*72}")
    grids = create_grids()
    
    # Define timesteps
    t_starts = np.linspace(T_START, T_END, N_FRAMES)
    timesteps_idx = np.round(t_starts / DT_CFD).astype(int)
    
    if args.test_mode:
        timesteps_idx = timesteps_idx[::N_FRAMES // 5][:5]
        print(f"  [TEST MODE] usando apenas {len(timesteps_idx)} frames")
    
    print(f"  Timesteps selecionados: {len(timesteps_idx)} "
          f"(t={timesteps_idx[0]*DT_CFD:.1f}s a {timesteps_idx[-1]*DT_CFD:.1f}s)")
    
    # FASE 1: Cache
    if not args.skip_cache:
        run_phase1_cache(timesteps_idx, grids, force=args.force_cache)
    else:
        print("\n[SKIP] Fase 1 (cache CFD)")
    
    # FASE 2: Escalas globais
    scales = run_phase2_scales(timesteps_idx, grids)
    
    # FASE 3a: Inferência PINN (com cache). Só carrega o modelo se necessário.
    if not args.skip_inference:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else CKPT_PATH
        print(f"\n{'='*72}")
        print(f"Carregando modelo PINN")
        print(f"{'='*72}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Arquitetura: {N_BLOCKS} blocos × {N_NEURONS} neurônios")
        
        model = build_pinn()
        model.load_weights(str(ckpt_path))
        print(f"  Parâmetros: {model.count_params():,}")
        
        run_phase3a_inference(timesteps_idx, model, grids, force=args.force_cache)
    else:
        print("\n[SKIP] Fase 3a (inferência PINN — usando cache existente)")
    
    # FASE 3b: Geração dos frames (lê inferência do cache, sem modelo)
    if not args.skip_frames:
        run_phase3_frames(timesteps_idx, grids, scales)
    else:
        print("\n[SKIP] Fase 3b (frames)")
    
    # FASE 4: Vídeos
    run_phase4_videos()
    
    print(f"\n{'='*72}")
    print("CONCLUÍDO")
    print(f"{'='*72}")
    print(f"Vídeos em: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
