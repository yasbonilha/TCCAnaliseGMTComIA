# ============================================================
# Análise de Escoamento do Ar no Telescópio Gigante de Magalhães Usando IA:
# Trabalho de Conclusão de Curso - IMT
# Código usado no supercomputador 
# ============================================================

# Importações

import os
import re
import glob
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

# Reprodutibilidade

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
rng = np.random.default_rng(SEED)

# Device/mixed precision - acelera o treinamento ao reduzir o consumo de RAM 

USE_MIXED_PRECISION = False

if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("WARNING: no GPU detected — training will be very slow on CPU.")

print(f"TensorFlow version: {tf.__version__}")



# ============================================================
# Definição dos parâmetros físicos - configurados de acordo com a simulação CFD

# --- Domínio ---
X_MIN, X_MAX = 0.0, 280.0
Y_MIN, Y_MAX = 0.0, 700.0
Z_MIN, Z_MAX = 0.0, 190.0

# --- Cilindro ---
D_CYL = 54.0
H_CYL = 65.0
R_CYL = D_CYL / 2.0
XC_CYL = 140.0
YC_CYL = 200.0
ZC_BASE = 0.0

# --- Escoamento ---
V_INF = 17.0               # Velocidade de entrada (inlet) [m/s] - direção +Y
V_MAX = 28.8               # Velocidade máxima registrada no domínio
P_SCALE = 15

# --- BOI - Caixa de Refinamento ao redor do cilindro ---
BOI_C_X = (100.0, 180.0)
BOI_C_Y = (160.0, 240.0)
BOI_C_Z = (0.0,   100.0)

# --- BOI - Prisma Triangular da esteira ---
BOI_T_V1 = (100.0, 240.0)
BOI_T_V2 = (180.0, 240.0)
BOI_T_V3 = (140.0, 600.0)
BOI_T_Z  = (0.0,   100.0)

# --- Tolerâncias para classificação de cell centers ---
BC_TOL  = 3.0     # Distância às paredes externas [m]
CYL_TOL = 2.0     # Distância à parede/topo do cilindro [m]

# --- Constantes Termodinâmicas e do Ar ---
P_OP = 101325.0                # Pressão de operação
R_SP = 287.058                 # Constante específica do ar [J/(kg·K)]
MU = 1.7894e-5                 # Viscosidade dinâmica [Pa·s]
G_ACC = 9.81                   # Gravidade em -Z
RHO_INF = 1.225

print(f"Domain:    X[{X_MIN},{X_MAX}] Y[{Y_MIN},{Y_MAX}] Z[{Z_MIN},{Z_MAX}]  (m)")
print(f"Cylinder:  D={D_CYL}m H={H_CYL}m centered at ({XC_CYL},{YC_CYL},{ZC_BASE})")
print(f"BOI_C:     X{BOI_C_X} Y{BOI_C_Y} Z{BOI_C_Z}")
print(f"BOI_T:     V1={BOI_T_V1} V2={BOI_T_V2} V3={BOI_T_V3} Z{BOI_T_Z}")
print(f"Tolerâncias: BC_TOL={BC_TOL}m  CYL_TOL={CYL_TOL}m")
print(f"Flow:      V_inf={V_INF} m/s (+Y)")
print(f"Air:       rho_inf={RHO_INF:.4f} kg/m³, mu={MU:.2e}")



# ============================================================
# Análise dos nomes dos arquivos de Snapshots

# --- Configurações iniciais ---
SNAPSHOT_DIR      = "/home/tmoraes/CSVs/"
SNAPSHOT_EXT      = ".csv"
DT_PHYSICAL       = 0.1          # Passo de tempo
TIMESTEP_IS_INDEX = True

# --- Stride temporal ---
# (pega 1 arquivo a cada SNAPSHOT_STRIDE. Dessa forma, um STRIDE=1 lê todos os arquivos)
SNAPSHOT_STRIDE   = 1            
SNAPSHOT_START    = 0            # Índice do primeiro arquivo a usar (default: 0)
SNAPSHOT_MAX      = None         # Limite de arquivos a usar (default: None → todos após o start)

# Pre-compiled pattern: o último número não-sinalizado no basename
_TIME_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)(?=\D*$)")

def parse_timestamp(filename: str,
                    dt_physical: float = DT_PHYSICAL,
                    timestep_is_index: bool = TIMESTEP_IS_INDEX):

    stem = os.path.splitext(os.path.basename(filename))[0]
    matches = _TIME_TOKEN_RE.findall(stem)
    if not matches:
        return None, None
    raw = float(matches[-1])
    t_phys = raw * dt_physical if timestep_is_index else raw

    return float(t_phys), raw


def discover_snapshots(snapshot_dir: str,
                       ext: str = SNAPSHOT_EXT,
                       dt_physical: float = DT_PHYSICAL,
                       timestep_is_index: bool = TIMESTEP_IS_INDEX,
                       stride: int = 1,
                       start_idx: int = 0,
                       max_snapshots: int = None,
                       verbose: bool = True):

    if not os.path.isdir(snapshot_dir):
        raise NotADirectoryError(f"Snapshot directory not found: {snapshot_dir}")

    pattern = os.path.join(snapshot_dir, f"*{ext}")
    candidates = sorted(glob.glob(pattern))
    if verbose:
        print(f"Scanning '{snapshot_dir}' for *{ext}: "
              f"{len(candidates)} candidate(s)")

    parsed = []
    skipped = []
    for path in candidates:
        t_phys, raw = parse_timestamp(path, dt_physical, timestep_is_index)
        if t_phys is None:
            skipped.append(os.path.basename(path))
            continue
        parsed.append((t_phys, path))

    parsed.sort(key=lambda s: s[0])

    n_total = len(parsed)
    selected = parsed[start_idx::stride]
    if max_snapshots is not None:
        selected = selected[:max_snapshots]

    if verbose:
        if skipped:
            print(f"  [skip] {len(skipped)} unparseable file(s), "
                  f"e.g. {skipped[:3]}")
        print(f"  [filter] {n_total} parseable → {len(selected)} after "
              f"stride={stride}, start={start_idx}, max={max_snapshots}")

    return selected


# --- Leitura arquivos na pasta ---
snaps = discover_snapshots(
    SNAPSHOT_DIR,
    stride=SNAPSHOT_STRIDE,
    start_idx=SNAPSHOT_START,
    max_snapshots=SNAPSHOT_MAX,
)

if len(snaps) == 0:
    raise RuntimeError(
        f"No usable '*{SNAPSHOT_EXT}' files in '{SNAPSHOT_DIR}' "
        f"(stride={SNAPSHOT_STRIDE}, start={SNAPSHOT_START})."
    )

times_all = np.array([t for t, _ in snaps], dtype=np.float32)

print(f"\nFinal snapshot selection: {len(snaps)} frames")
print(f"  Time range:    {times_all[0]:.4f} s → {times_all[-1]:.4f} s")

if len(times_all) > 1:
    dts = np.diff(times_all)

    print(f"  dt (between selected snaps): "
          f"mean={dts.mean():.4f}s  min={dts.min():.4f}s  max={dts.max():.4f}s")
    
    expected_dt = DT_PHYSICAL * SNAPSHOT_STRIDE

    print(f"  Expected dt (DT_PHYSICAL * stride): {expected_dt:.4f}s")

print(f"  First 3 selected files: {[os.path.basename(p) for _, p in snaps[:3]]}")
print(f"  Last  3 selected files: {[os.path.basename(p) for _, p in snaps[-3:]]}")



# ============================================================
# Constantes usadas para normalização dos dados

L_REF   = max(X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN)   # [m]
V_REF   = V_MAX                                              # [m/s]
P_REF   = 0.5 * RHO_INF * V_INF**2                           # pressão dinâmica [Pa]
TIME_REF = L_REF / V_REF                                     # [s]

# Tempo normalizado
TN_MIN = float(times_all[0])  / TIME_REF
TN_MAX = float(times_all[-1]) / TIME_REF

# Geometria Normalizada
XC_N = XC_CYL / L_REF
YC_N = YC_CYL / L_REF
R_N  = R_CYL  / L_REF
H_N  = H_CYL  / L_REF
XN_MIN, XN_MAX = X_MIN / L_REF, X_MAX / L_REF
YN_MIN, YN_MAX = Y_MIN / L_REF, Y_MAX / L_REF
ZN_MIN, ZN_MAX = Z_MIN / L_REF, Z_MAX / L_REF

VN_INLET   = V_INF / V_REF                # = 1.0

print(f"Normalization scales:")
print(f"  L_ref  = {L_REF:.2f} m")
print(f"  V_ref  = {V_REF:.2f} m/s")
print(f"  P_ref  = {P_REF:.3f} Pa")
print(f"  t_ref  = {TIME_REF:.4f} s")
print(f"Normalized domain: "
      f"X[{XN_MIN:.3f},{XN_MAX:.3f}] "
      f"Y[{YN_MIN:.3f},{YN_MAX:.3f}] "
      f"Z[{ZN_MIN:.3f},{ZN_MAX:.3f}] "
      f"t[{TN_MIN:.4f},{TN_MAX:.4f}]")


# ================================================================================
# Carregar os Snapshots já extratificados por categoria (condições de contorno, 
# detalhamento e interior do domínio)
#
#   PONTOS POR CATEGORIA PARA CADA ARQUIVO
#  -------------------------------------------------------------------------
#   Subtotal das BCs (16k):
#     1. cyl_lateral  (z ≤ H_CYL, |r - R_CYL| < CYL_TOL)         3500 pts
#     2. cyl_top      (z > H_CYL, r ≤ R_CYL, z - H_CYL < CYL_TOL) 500 pts
#     3. bc_inlet     (dist ao plano y=Y_MIN < BC_TOL)           2000 pts
#     4. bc_outlet    (dist ao plano y=Y_MAX < BC_TOL)           2000 pts
#     5. bc_xmin      (dist ao plano x=X_MIN < BC_TOL)           2000 pts
#     6. bc_xmax      (dist ao plano x=X_MAX < BC_TOL)           2000 pts
#     7. bc_ground    (dist ao plano z=Z_MIN < BC_TOL)           2000 pts
#     8. bc_top       (dist ao plano z=Z_MAX < BC_TOL)           2000 pts
#
#   Subtotal da zona de detalhamento (16k):
#     9. boi_c        (interior do bbox BOI_Cylinder)           10000 pts
#    10. boi_t        (interior do prisma BOI_Triangle)          6000 pts
#
#   Subtotal do interior do domínio (16k):
#    11. freestream   (resto do domínio)                        16000 pts
#                                                          -----------------
#                                                  Total:    48000 pts/snap

# --- Distribuição de pontos por categoria ---
QUOTAS = {
    # Parede do cilindro (no-slip)
    "cyl_lateral": 3500,
    "cyl_top":      500,
    # BCs externas
    "bc_inlet":    2000,
    "bc_outlet":   2000,
    "bc_xmin":     2000,
    "bc_xmax":     2000,
    "bc_ground":   2000,
    "bc_top":      2000,
    # Detalhamento volumétrico (BOIs)
    "boi_c":      10_000,
    "boi_t":       6_000,
    # Interior livre
    "freestream": 16_000,
}

N_POINTS_PER_SNAP = sum(QUOTAS.values())   # 48000
MAX_SNAPSHOTS     = None
# ----------------------------------

CATEGORY_NAMES = list(QUOTAS.keys())

import re

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
    """Normaliza: strip, lower, remove aspas e caracteres invisíveis."""
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
            raise KeyError(f"No match for '{key}'. Patterns: {patterns}. "
                            f"Available normalized: {list(norm_map.keys())}")
        mapping[key] = found
    return mapping

def load_snapshot(path, col_map=None):
    df = pd.read_csv(
        path,
        sep=r'\s+',   
        engine='python', 
        dtype=np.float32,
    )

    df = df.drop(columns=["cellnumber"], errors="ignore")
    df.columns = [_norm_col(c) for c in df.columns]
    if col_map is None:
        col_map = resolve_column_names(df.columns)
    out = {k: df[col_map[k]].to_numpy(dtype=np.float32, copy=False)
            for k in NEEDED_COLS_PATTERNS}
    return out, col_map



# ================================================================================
# Funções modulares de classificação geométrica
# (todas operam em arrays Numpy)

def is_inside_cylinder(x, y, z):
    """Dentro do cilindro sólido (deve ser excluído antes de classificar)."""
    r2 = (x - XC_CYL)**2 + (y - YC_CYL)**2
    return (r2 <= R_CYL**2) & (z <= H_CYL)


def is_adj_cyl_lateral(x, y, z):
    """Camada lateral do cilindro: z ≤ H_CYL e |r - R_CYL| < CYL_TOL."""
    r = np.sqrt((x - XC_CYL)**2 + (y - YC_CYL)**2)
    return (z <= H_CYL) & (np.abs(r - R_CYL) < CYL_TOL)


def is_adj_cyl_top(x, y, z):
    """Camada do topo: z > H_CYL, r ≤ R_CYL, (z - H_CYL) < CYL_TOL."""
    r = np.sqrt((x - XC_CYL)**2 + (y - YC_CYL)**2)
    return (z > H_CYL) & (r <= R_CYL) & ((z - H_CYL) < CYL_TOL)


def is_inside_boi_cyl(x, y, z):
    """Dentro da caixa BOI_Cylinder."""
    return ((x >= BOI_C_X[0]) & (x <= BOI_C_X[1]) &
            (y >= BOI_C_Y[0]) & (y <= BOI_C_Y[1]) &
            (z >= BOI_C_Z[0]) & (z <= BOI_C_Z[1]))


def is_inside_boi_tri(x, y, z):
    """Dentro do prisma triangular BOI_Triangle (teste baricêntrico em XY)."""
    v1, v2, v3 = BOI_T_V1, BOI_T_V2, BOI_T_V3
    denom = (v2[1]-v3[1])*(v1[0]-v3[0]) + (v3[0]-v2[0])*(v1[1]-v3[1])
    a = ((v2[1]-v3[1])*(x-v3[0]) + (v3[0]-v2[0])*(y-v3[1])) / denom
    b = ((v3[1]-v1[1])*(x-v3[0]) + (v1[0]-v3[0])*(y-v3[1])) / denom
    c = 1.0 - a - b
    in_xy = (a >= 0) & (b >= 0) & (c >= 0)
    in_z  = (z >= BOI_T_Z[0]) & (z <= BOI_T_Z[1])
    return in_xy & in_z


def classify_external_bc(x, y, z):
    """
    Para cada ponto, retorna:
      is_adj_bc : True onde dist < BC_TOL a alguma BC externa
      bc_idx    : índice 0..5 da BC mais próxima (atribuição exclusiva)

    Convenção:
      0 = bc_inlet (y=Y_MIN)
      1 = bc_outlet (y=Y_MAX)
      2 = bc_xmin (x=X_MIN)
      3 = bc_xmax (x=X_MAX)
      4 = bc_ground (z=Z_MIN)
      5 = bc_top (z=Z_MAX)
    """
    d_inlet  = np.abs(y - Y_MIN)
    d_outlet = np.abs(y - Y_MAX)
    d_xmin   = np.abs(x - X_MIN)
    d_xmax   = np.abs(x - X_MAX)
    d_ground = np.abs(z - Z_MIN)
    d_top    = np.abs(z - Z_MAX)
    dists = np.stack([d_inlet, d_outlet, d_xmin, d_xmax, d_ground, d_top], axis=1)
    d_min = dists.min(axis=1)
    bc_idx = dists.argmin(axis=1).astype(np.int8)
    is_adj_bc = d_min < BC_TOL
    return is_adj_bc, bc_idx


BC_IDX_TO_NAME = {
    0: "bc_inlet",
    1: "bc_outlet",
    2: "bc_xmin",
    3: "bc_xmax",
    4: "bc_ground",
    5: "bc_top",
}


def classify_points(x, y, z):
    """
    Classifica pontos em CATEGORY_NAMES seguindo hierarquia (mais específico vence):
      freestream → boi_t → boi_c → bc_* → cyl_top → cyl_lateral

    Pontos no interior sólido do cilindro são marcados como None (devem ter
    sido removidos antes; mantido por segurança).
    """
    n = len(x)
    category = np.full(n, None, dtype=object)

    # Step 0: pontos no interior do cilindro ficam None
    inside_cyl = is_inside_cylinder(x, y, z)

    # Step 1: freestream cobre tudo que não é interior do cilindro
    category[~inside_cyl] = "freestream"

    # Step 2: BOI_T sobrescreve freestream
    mask = is_inside_boi_tri(x, y, z) & ~inside_cyl
    category[mask] = "boi_t"

    # Step 3: BOI_C sobrescreve BOI_T (e freestream) — onde há overlap
    mask = is_inside_boi_cyl(x, y, z) & ~inside_cyl
    category[mask] = "boi_c"

    # Step 4: BCs externas sobrescrevem BOIs e freestream
    is_bc, bc_idx = classify_external_bc(x, y, z)
    for k, name in BC_IDX_TO_NAME.items():
        mask = is_bc & (bc_idx == k) & ~inside_cyl
        category[mask] = name

    # Step 5: cilindro topo sobrescreve BCs
    category[is_adj_cyl_top(x, y, z)] = "cyl_top"

    # Step 6: cilindro lateral (prioridade máxima)
    category[is_adj_cyl_lateral(x, y, z)] = "cyl_lateral"

    return category


def _stratified_sample(category, quotas, rng):
    """
    Sampling sem reposição respeitando as quotas. Se uma categoria tiver menos
    pontos disponíveis que a quota, usa todos os disponíveis.

    Returns: picked_idx, avail_counts (dict), taken_counts (dict)
    """
    avail_counts = {}
    taken_counts = {}
    picked = []
    for cat_name, quota in quotas.items():
        idx_cat = np.where(category == cat_name)[0]
        n_avail = len(idx_cat)
        avail_counts[cat_name] = n_avail
        if n_avail == 0:
            taken_counts[cat_name] = 0
            continue
        taken = idx_cat if n_avail <= quota else rng.choice(idx_cat, size=quota, replace=False)
        taken_counts[cat_name] = len(taken)
        picked.append(taken)
    picked_idx = np.concatenate(picked) if picked else np.array([], dtype=np.int64)
    return picked_idx, avail_counts, taken_counts


# ============================================================
# Mapeamento de categorias para grupos 

GROUP_TO_ID = {
    "cyl":   0,
    "bc":    1,
    "boi_c": 2,
    "boi_t": 3,
    "free":  4,
}

CAT_TO_GROUP_ID = {
    "cyl_lateral": GROUP_TO_ID["cyl"],
    "cyl_top":     GROUP_TO_ID["cyl"],
    "bc_inlet":    GROUP_TO_ID["bc"],
    "bc_outlet":   GROUP_TO_ID["bc"],
    "bc_xmin":     GROUP_TO_ID["bc"],
    "bc_xmax":     GROUP_TO_ID["bc"],
    "bc_ground":   GROUP_TO_ID["bc"],
    "bc_top":      GROUP_TO_ID["bc"],
    "boi_c":       GROUP_TO_ID["boi_c"],
    "boi_t":       GROUP_TO_ID["boi_t"],
    "freestream":  GROUP_TO_ID["free"],
}

# IDs como float32 pra serem usados em tensor
GID_CYL   = float(GROUP_TO_ID["cyl"])
GID_BC    = float(GROUP_TO_ID["bc"])
GID_BOI_C = float(GROUP_TO_ID["boi_c"])
GID_BOI_T = float(GROUP_TO_ID["boi_t"])
GID_FREE  = float(GROUP_TO_ID["free"])

print(f"Group IDs: cyl={GID_CYL}, bc={GID_BC}, boi_c={GID_BOI_C}, "
      f"boi_t={GID_BOI_T}, free={GID_FREE}")


def load_all_snapshots(snaps,
                       quotas=QUOTAS,
                       max_snapshots=MAX_SNAPSHOTS,
                       seed=SEED,
                       keep_first_raw=True,
                       verbose_every=20):
    """
    Stream-load com sampling estratificado por categoria.

    Returns:
      data_np        : (N_total, 8) float32 — colunas normalizadas [xn,yn,zn,tn,un,vn,wn,Pn]
      snap_offsets   : list[(t_phys, start, end)]
      first_snap_raw : dict com arrays brutos do primeiro snapshot, ou None
      stats_df       : DataFrame com (snapshot, t_phys, category, available, taken, quota)
                       — usado para a tabela agregada (Cell 5b).
    """
    if max_snapshots is not None:
        snaps = snaps[:max_snapshots]
    n_snaps = len(snaps)
    if n_snaps == 0:
        raise RuntimeError("No snapshots to load.")

    pts_per_snap = sum(quotas.values())
    cap = n_snaps * pts_per_snap
    data_np = np.empty((cap, 9), dtype=np.float32)
    snap_offsets = []
    write_ptr = 0

    col_map = None
    P_offset = None
    first_snap_raw = None
    stats_records = []
    rng_local = np.random.default_rng(seed)

    print(f"Loading {n_snaps} snapshots with stratified sampling.")
    print(f"Quotas per snap (total = {pts_per_snap:,}):")
    for k, v in quotas.items():
        print(f"  {k:<14} {v:>6,}")
    ram_mb = cap * 9 * 4 / 1e6
    print(f"Upper-bound RAM ~{ram_mb:.0f} MB ({ram_mb/1024:.2f} GB)")
    t0 = time.time()

    for i, (t_phys, path) in enumerate(snaps):
        try:
            data, col_map = load_snapshot(path, col_map=col_map)
        except Exception as exc:
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue

        x, y, z = data["x"], data["y"], data["z"]
        u, v, w = data["u"], data["v"], data["w"]
        P       = data["P"]

        # Auto-detect pressure convention (absolute vs gauge) uma vez
        if P_offset is None:
            if P.mean() > 1e4:
                P_offset = P_OP
                print(f"  Pressure mean={P.mean():.1f} Pa -> absolute "
                      f"(subtract P_op={P_OP})")
            else:
                P_offset = 0.0
                print(f"  Pressure mean={P.mean():.3f} Pa -> gauge")
        P = P - P_offset

        # Excluir interior sólido do cilindro
        inside = is_inside_cylinder(x, y, z)
        if inside.any():
            keep = ~inside
            x, y, z = x[keep], y[keep], z[keep]
            u, v, w = u[keep], v[keep], w[keep]
            P = P[keep]

        # Classificar e amostrar
        category = classify_points(x, y, z)
        picked, avail_counts, taken_counts = _stratified_sample(category, quotas, rng_local)

        for cat_name in quotas:
            stats_records.append({
                "snapshot": i,
                "t_phys": t_phys,
                "category": cat_name,
                "available": avail_counts.get(cat_name, 0),
                "taken": taken_counts.get(cat_name, 0),
                "quota": quotas[cat_name],
            })

        n_take = len(picked)
        end = write_ptr + n_take
        sl = slice(write_ptr, end)
        data_np[sl, 0] = x[picked] / L_REF
        data_np[sl, 1] = y[picked] / L_REF
        data_np[sl, 2] = z[picked] / L_REF
        data_np[sl, 3] = t_phys / TIME_REF
        data_np[sl, 4] = u[picked] / V_REF
        data_np[sl, 5] = v[picked] / V_REF
        data_np[sl, 6] = w[picked] / V_REF
        data_np[sl, 7] = P[picked] / P_REF

        # Coluna 8: group_id (RUN 6)
        cat_picked = category[picked]
        group_ids = np.array(
            [CAT_TO_GROUP_ID[c] for c in cat_picked],
            dtype=np.float32
        )
        data_np[sl, 8] = group_ids

        snap_offsets.append((t_phys, write_ptr, end))
        write_ptr = end

        if keep_first_raw and first_snap_raw is None:
            first_snap_raw = dict(
                x=x.copy(), y=y.copy(), z=z.copy(),
                u=u.copy(), v=v.copy(), w=w.copy(),
                P=P.copy(), t=float(t_phys),
            )

        del data, x, y, z, u, v, w, P, category, picked

        if (i + 1) % verbose_every == 0 or i == 0 or i == n_snaps - 1:
            print(f"  [{i+1:4d}/{n_snaps}] t={t_phys:.4f}s  "
                  f"+{n_take} pts (total={write_ptr:,})")

    data_np = data_np[:write_ptr]
    stats_df = pd.DataFrame(stats_records)

    print(f"\nDone. Supervised points: {len(data_np):,} "
          f"across {len(snap_offsets)} snapshots")
    print(f"Wall time: {time.time() - t0:.1f}s    "
          f"Memory: {data_np.nbytes / 1e6:.0f} MB ({data_np.nbytes/1e9:.2f} GB)")
    return data_np, snap_offsets, first_snap_raw, stats_df

# ----- Run loader -----
data_np, snap_offsets, first_snap_raw, stats_df = load_all_snapshots(snaps)
n_snaps = len(snap_offsets)



# ================================================================================
# Confirmação de que os pontos extraídos de forma randômica da simulação correspondem
# à distribuição de pontos esperada por categoria


print(f"\n{'='*82}")
print(f"  TABELA AGREGADA por categoria  (n_snapshots = {n_snaps})")
print(f"  Tolerâncias: BC_TOL={BC_TOL}m  CYL_TOL={CYL_TOL}m")
print(f"{'='*82}")
print(f"{'Categoria':<14}{'Quota':>8}{'Avail (μ±σ)':>20}"
      f"{'Taken (μ±σ)':>20}{'% Quota':>10}{'Status':>10}")
print(f"{'-'*82}")

summary_records = []
for cat_name in CATEGORY_NAMES:
    sub = stats_df[stats_df["category"] == cat_name]
    avail_mean = sub["available"].mean()
    avail_std  = sub["available"].std()
    taken_mean = sub["taken"].mean()
    taken_std  = sub["taken"].std()
    quota      = QUOTAS[cat_name]
    pct_quota  = 100.0 * taken_mean / quota if quota > 0 else 0.0
    status = "OK" if pct_quota >= 99.9 else ("DÉFICIT" if pct_quota < 95.0 else "near")
    summary_records.append({
        "category": cat_name, "quota": quota,
        "avail_mean": avail_mean, "avail_std": avail_std,
        "taken_mean": taken_mean, "taken_std": taken_std,
        "pct_quota": pct_quota, "status": status,
    })
    print(f"{cat_name:<14}{quota:>8d}"
          f"{avail_mean:>11.1f} ± {avail_std:<5.1f}"
          f"{taken_mean:>11.1f} ± {taken_std:<5.1f}"
          f"{pct_quota:>9.1f}%{status:>10}")

print(f"{'-'*82}")
total_quota = sum(QUOTAS.values())
total_taken_mean = sum(r["taken_mean"] for r in summary_records)
print(f"{'TOTAL':<14}{total_quota:>8d}{'—':>20}"
      f"{total_taken_mean:>11.1f}{'':<8}{100*total_taken_mean/total_quota:>9.1f}%")
print(f"{'='*82}\n")

# Save per-snapshot stats e summary
stats_df.to_csv("snapshot_sampling_stats.csv", index=False)
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv("snapshot_sampling_summary.csv", index=False)
print(f"Per-snapshot stats salvos em: snapshot_sampling_stats.csv")
print(f"Summary agregado salvos em:  snapshot_sampling_summary.csv\n")
print("Summary:")
print(summary_df.to_string(index=False, float_format='%.1f'))


# ============================================================
# Divisão entre treino e validação 
# (usando estratégia de segregação temporal por snapshot)

TRAIN_FILES = 18
VAL_FILES   = 2
CYCLE_SIZE  = TRAIN_FILES + VAL_FILES  # Ciclo de 20 arquivos

SKIP_FIRST  = False  # mantém o primeiro arquivo fora (apenas inicialização do dom.)

if n_snaps < CYCLE_SIZE:
    VAL_FILES = max(1, n_snaps // 10)
    TRAIN_FILES = n_snaps - VAL_FILES
    CYCLE_SIZE = n_snaps
    print(f"[info] apenas {n_snaps} snapshots -> reduzindo para {TRAIN_FILES} treino e {VAL_FILES} val")

val_snap_indices = set()
for i in range(n_snaps):
    if (i % CYCLE_SIZE) >= TRAIN_FILES:
        val_snap_indices.add(i)

if SKIP_FIRST:
    val_snap_indices.discard(0)

val_mask = np.zeros(len(data_np), dtype=bool)
for i, (_t_phys, start, end) in enumerate(snap_offsets):
    if i in val_snap_indices:
        val_mask[start:end] = True

train_np = data_np[~val_mask]
val_np   = data_np[val_mask]

val_times_phys = [snap_offsets[i][0] for i in sorted(val_snap_indices)]
print(f"Train points: {len(train_np):,}    Val points: {len(val_np):,}")
print(f"Validation snapshots: {len(val_snap_indices)} held out")
print(f"  Physical times (s): "
      f"{[f'{t:.3f}' for t in val_times_phys[:5]]}"
      f"{' ...' if len(val_times_phys) > 5 else ''}")



# ================================================================================
# Determinação dos pontos de colocação em 11 subregiõe (collocation pool)

#   adj_inlet     [0,      6000 ]       quota: 2000 por step × 3
#   adj_outlet    [6000,   12000]
#   adj_xmin      [12000,  18000]
#   adj_xmax      [18000,  24000]
#   adj_ground    [24000,  30000]
#   adj_top       [30000,  36000]
#   adj_cyl_lat   [36000,  46500]      quota: 3500 × 3 = 10500
#   adj_cyl_top   [46500,  48000]      quota: 500 × 3 = 1500
#   boi_c         [48000,  81000]      quota: 11000 × 3 = 33000
#   boi_t         [81000,  96000]      quota: 5000 × 3 = 15000
#   free          [96000,  144000]     quota: 16000 × 3 = 48000
#
# A hierarquia de exclusão evita duplicação geométrica entre regiões:
#   - adj_cyl_lat/top: cascas mais específicas (prioridade máxima)
#   - adj_BC (6 faces): cascas das paredes externas
#   - BOI_C/T: caixas de refinamento (excluem cilindro sólido + cascas adj_cyl)
#   - FREE: domínio menos todas as outras regiões
#
# A função is_adj_cyl_lateral (Cell 5) usa |r - R| < CYL_TOL (anel de espessura
# 2×CYL_TOL). Aqui amostramos apenas o exterior r ∈ [R, R+CYL_TOL] pois é onde
# há fluido — interior é cilindro sólido.
 
POOL_MULT = 3   # Multiplicador do pool sobre per-step (maior = mais variedade)
 
_REGION_DEFS = [
    # (nome,         per-step)
    ("adj_inlet",    2_000),
    ("adj_outlet",   2_000),
    ("adj_xmin",     2_000),
    ("adj_xmax",     2_000),
    ("adj_ground",   2_000),
    ("adj_top",      2_000),
    ("adj_cyl_lat",  3_500),
    ("adj_cyl_top",    500),
    ("boi_c",       11_000),
    ("boi_t",        5_000),
    ("free",        16_000),
]
# ----------------------------------
 
 
# Compute layout do pool
COLLOCATION_REGIONS = []
_cursor = 0
for _name, _per_step in _REGION_DEFS:
    _pool_size = _per_step * POOL_MULT
    COLLOCATION_REGIONS.append({
        "name": _name,
        "per_step": _per_step,
        "pool_size": _pool_size,
        "start": _cursor,
        "end": _cursor + _pool_size,
    })
    _cursor += _pool_size
 
N_COLL = _cursor    # 144.000
N_COLL_PER_STEP = sum(r["per_step"] for r in COLLOCATION_REGIONS)   # 48.000
 
 
# ================================================================================
# Funções de sampling 
 
def _make_slab_sampler(axis, side):
    """
    Factory: retorna função que amostra n pts uniformemente em uma camada
    de espessura BC_TOL adjacente a uma face do domínio.
 
    axis: 'x' | 'y' | 'z'    side: 'min' | 'max'
    """
    AXIS_BOUNDS = {'x': (X_MIN, X_MAX), 'y': (Y_MIN, Y_MAX), 'z': (Z_MIN, Z_MAX)}
    a_min, a_max = AXIS_BOUNDS[axis]
    if side == 'min':
        slab_lo, slab_hi = a_min, a_min + BC_TOL
    else:
        slab_lo, slab_hi = a_max - BC_TOL, a_max
 
    bounds = dict(AXIS_BOUNDS)   # copia
    bounds[axis] = (slab_lo, slab_hi)
 
    def sampler(n, rng):
        out = np.empty((n, 3), dtype=np.float32)
        written = 0
        while written < n:
            n_try = (n - written) + 256
            x = rng.uniform(*bounds['x'], n_try).astype(np.float32)
            y = rng.uniform(*bounds['y'], n_try).astype(np.float32)
            z = rng.uniform(*bounds['z'], n_try).astype(np.float32)

            keep = ~is_inside_cylinder(x, y, z)
            n_keep = int(keep.sum())
            take = min(n_keep, n - written)
            out[written:written+take, 0] = x[keep][:take]
            out[written:written+take, 1] = y[keep][:take]
            out[written:written+take, 2] = z[keep][:take]
            written += take
        return out
    return sampler
 
sample_adj_inlet_pts  = _make_slab_sampler('y', 'min')
sample_adj_outlet_pts = _make_slab_sampler('y', 'max')
sample_adj_xmin_pts   = _make_slab_sampler('x', 'min')
sample_adj_xmax_pts   = _make_slab_sampler('x', 'max')
sample_adj_ground_pts = _make_slab_sampler('z', 'min')
sample_adj_top_pts    = _make_slab_sampler('z', 'max')
 
 
def sample_adj_cyl_lateral_pts(n, rng):
    """
    Anel cilíndrico exterior à parede do cilindro: r ∈ [R, R+CYL_TOL], z ∈ [0, H].
    Usa r²-uniforme para garantir densidade uniforme em área (XY).
    """
    out = np.empty((n, 3), dtype=np.float32)
    written = 0
    while written < n:
        n_try = (n - written) + 256
        # r² uniforme → área uniforme
        r = np.sqrt(rng.uniform(R_CYL**2, (R_CYL + CYL_TOL)**2, n_try)).astype(np.float32)
        theta = rng.uniform(0.0, 2*np.pi, n_try).astype(np.float32)
        x = XC_CYL + r * np.cos(theta)
        y = YC_CYL + r * np.sin(theta)
        z = rng.uniform(Z_MIN, H_CYL, n_try).astype(np.float32)
        take = min(n_try, n - written)
        out[written:written+take, 0] = x[:take]
        out[written:written+take, 1] = y[:take]
        out[written:written+take, 2] = z[:take]
        written += take
    return out
 
 
def sample_adj_cyl_top_pts(n, rng):
    """
    Disco no topo do cilindro: r ∈ [0, R], z ∈ [H, H+CYL_TOL].
    r²-uniforme p/ densidade uniforme em área.
    """
    out = np.empty((n, 3), dtype=np.float32)
    written = 0
    while written < n:
        n_try = (n - written) + 256
        r = np.sqrt(rng.uniform(0.0, R_CYL**2, n_try)).astype(np.float32)
        theta = rng.uniform(0.0, 2*np.pi, n_try).astype(np.float32)
        x = XC_CYL + r * np.cos(theta)
        y = YC_CYL + r * np.sin(theta)
        z = rng.uniform(H_CYL, H_CYL + CYL_TOL, n_try).astype(np.float32)
        take = min(n_try, n - written)
        out[written:written+take, 0] = x[:take]
        out[written:written+take, 1] = y[:take]
        out[written:written+take, 2] = z[:take]
        written += take
    return out
 
 
def sample_boi_cyl_pts(n, rng):
    """
    BOI_C, EXCLUINDO cilindro sólido + cascas adj_cyl (estão em outras regiões).
    Rejection: cyl sólido ~23% + adj_cyl ~3% = ~26% rejeição → mult 1.6×.
    """
    out = np.empty((n, 3), dtype=np.float32)
    written = 0
    while written < n:
        n_try = int((n - written) * 1.6) + 256
        x = rng.uniform(BOI_C_X[0], BOI_C_X[1], n_try).astype(np.float32)
        y = rng.uniform(BOI_C_Y[0], BOI_C_Y[1], n_try).astype(np.float32)
        z = rng.uniform(BOI_C_Z[0], BOI_C_Z[1], n_try).astype(np.float32)
        keep = ~(is_inside_cylinder(x, y, z)
                 | is_adj_cyl_lateral(x, y, z)
                 | is_adj_cyl_top(x, y, z))
        n_keep = int(keep.sum())
        take = min(n_keep, n - written)
        out[written:written+take, 0] = x[keep][:take]
        out[written:written+take, 1] = y[keep][:take]
        out[written:written+take, 2] = z[keep][:take]
        written += take
    return out
 
 
def sample_boi_tri_pts(n, rng):
    """
    Prisma triangular BOI_T. Bounding box 80 x 360 m²; triângulo ocupa ~50%.
    Rejection rate ~50% → mult 2.5 x.
    BOI_T não overlap com cilindro nem com adj_cyl (longe), sem exclusões extra.
    """
    v1, v2, v3 = BOI_T_V1, BOI_T_V2, BOI_T_V3
    x_min = min(v1[0], v2[0], v3[0]); x_max = max(v1[0], v2[0], v3[0])
    y_min = min(v1[1], v2[1], v3[1]); y_max = max(v1[1], v2[1], v3[1])
 
    out = np.empty((n, 3), dtype=np.float32)
    written = 0
    while written < n:
        n_try = int((n - written) * 2.5) + 256
        x = rng.uniform(x_min, x_max, n_try).astype(np.float32)
        y = rng.uniform(y_min, y_max, n_try).astype(np.float32)
        z = rng.uniform(BOI_T_Z[0], BOI_T_Z[1], n_try).astype(np.float32)
        keep = is_inside_boi_tri(x, y, z)
        n_keep = int(keep.sum())
        take = min(n_keep, n - written)
        out[written:written+take, 0] = x[keep][:take]
        out[written:written+take, 1] = y[keep][:take]
        out[written:written+take, 2] = z[keep][:take]
        written += take
    return out
 
 
def sample_freestream_pts(n, rng):
    """
    Freestream = domínio menos: cilindro, BOI_C, BOI_T, todas as cascas adj_BC,
    cascas adj_cyl. Rejection rate ~12% → mult 1.4×.
    """
    out = np.empty((n, 3), dtype=np.float32)
    written = 0
    while written < n:
        n_try = int((n - written) * 1.4) + 256
        x = rng.uniform(X_MIN, X_MAX, n_try).astype(np.float32)
        y = rng.uniform(Y_MIN, Y_MAX, n_try).astype(np.float32)
        z = rng.uniform(Z_MIN, Z_MAX, n_try).astype(np.float32)
        in_cyl_solid = is_inside_cylinder(x, y, z)
        in_boi_c     = is_inside_boi_cyl(x, y, z)
        in_boi_t     = is_inside_boi_tri(x, y, z)
        in_adj_bc = (
            (y < Y_MIN + BC_TOL) | (y > Y_MAX - BC_TOL) |
            (x < X_MIN + BC_TOL) | (x > X_MAX - BC_TOL) |
            (z < Z_MIN + BC_TOL) | (z > Z_MAX - BC_TOL)
        )
        in_adj_cyl_lat = is_adj_cyl_lateral(x, y, z)
        in_adj_cyl_top = is_adj_cyl_top(x, y, z)
        keep = ~(in_cyl_solid | in_boi_c | in_boi_t | in_adj_bc
                 | in_adj_cyl_lat | in_adj_cyl_top)
        n_keep = int(keep.sum())
        take = min(n_keep, n - written)
        out[written:written+take, 0] = x[keep][:take]
        out[written:written+take, 1] = y[keep][:take]
        out[written:written+take, 2] = z[keep][:take]
        written += take
    return out
 

SAMPLERS = {
    "adj_inlet":    sample_adj_inlet_pts,
    "adj_outlet":   sample_adj_outlet_pts,
    "adj_xmin":     sample_adj_xmin_pts,
    "adj_xmax":     sample_adj_xmax_pts,
    "adj_ground":   sample_adj_ground_pts,
    "adj_top":      sample_adj_top_pts,
    "adj_cyl_lat":  sample_adj_cyl_lateral_pts,
    "adj_cyl_top":  sample_adj_cyl_top_pts,
    "boi_c":        sample_boi_cyl_pts,
    "boi_t":        sample_boi_tri_pts,
    "free":         sample_freestream_pts,
}
 
 
def sample_collocation():
    """
    Gera pool em layout determinístico de 11 regiões. Sem shuffle final.
    Tempo: uniforme em [TN_MIN, TN_MAX] (range real dos dados normalizados).
    """
    rng_coll = np.random.default_rng(SEED)
    pieces = []
    for r in COLLOCATION_REGIONS:
        pts = SAMPLERS[r["name"]](r["pool_size"], rng_coll)
        pieces.append(pts)
    xyz = np.concatenate(pieces, axis=0)
    assert len(xyz) == N_COLL, f"pool size mismatch: {len(xyz)} vs {N_COLL}"
 
    t = rng_coll.uniform(TN_MIN, TN_MAX, len(xyz)).astype(np.float32).reshape(-1, 1)
    xyz_n = xyz / L_REF
    return np.concatenate([xyz_n, t], axis=1).astype(np.float32)
 

print(f"Gerando pool de collocation...")
_t0 = time.time()
coll_np = sample_collocation()
print(f"Tempo de geração: {time.time()-_t0:.2f}s")
 
print(f"\nCollocation pool: {N_COLL:,} pontos (POOL_MULT={POOL_MULT})")
print(f"Per step total: {N_COLL_PER_STEP:,}")
print(f"\nGrupos:")
n_bc   = sum(r["per_step"] for r in COLLOCATION_REGIONS if r["name"].startswith("adj"))
n_boi  = sum(r["per_step"] for r in COLLOCATION_REGIONS if r["name"].startswith("boi"))
n_free = next(r["per_step"] for r in COLLOCATION_REGIONS if r["name"] == "free")
print(f"  BCs       (8 sub-regiões): {n_bc:>6,}  ({100*n_bc/N_COLL_PER_STEP:>4.1f}%)")
print(f"  Detail BOI (2 sub-regiões): {n_boi:>6,}  ({100*n_boi/N_COLL_PER_STEP:>4.1f}%)")
print(f"  Freestream                : {n_free:>6,}  ({100*n_free/N_COLL_PER_STEP:>4.1f}%)")
print(f"\nLayout do pool:")
for r in COLLOCATION_REGIONS:
    print(f"  {r['name']:<14} [{r['start']:>6d}, {r['end']:>6d})  "
          f"per_step={r['per_step']:>5d}  pool={r['pool_size']:>6d}")
print(f"\nTempo normalizado sorteado em: [{TN_MIN:.4f}, {TN_MAX:.4f}]")



# ================================================================================
# Visualização + Sanity Checks
 
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
 
# Desnormaliza para plotar em metros
xyz_phys = coll_np[:, :3] * L_REF
x_phys, y_phys, z_phys = xyz_phys[:, 0], xyz_phys[:, 1], xyz_phys[:, 2]
 

print(f"\n{'='*78}")
print(f"  SANITY CHECKS por região")
print(f"{'='*78}")
 
# Funções de verificação por região
def _check_slab(name, axis, side, tol):
    r = next(r for r in COLLOCATION_REGIONS if r["name"] == name)
    sl = slice(r["start"], r["end"])
    coord = {'x': x_phys, 'y': y_phys, 'z': z_phys}[axis][sl]
    AXIS = {'x': (X_MIN, X_MAX), 'y': (Y_MIN, Y_MAX), 'z': (Z_MIN, Z_MAX)}
    if side == 'min':
        ok = (coord >= AXIS[axis][0]) & (coord <= AXIS[axis][0] + tol)
    else:
        ok = (coord >= AXIS[axis][1] - tol) & (coord <= AXIS[axis][1])
    n_bad = int((~ok).sum())
    print(f"  {name:<14}  pts fora do slab: {n_bad}  (esperado: 0)")
    assert n_bad == 0, f"{name}: {n_bad} pontos fora do slab"
 
_check_slab("adj_inlet",   'y', 'min', BC_TOL)
_check_slab("adj_outlet",  'y', 'max', BC_TOL)
_check_slab("adj_xmin",    'x', 'min', BC_TOL)
_check_slab("adj_xmax",    'x', 'max', BC_TOL)
_check_slab("adj_ground",  'z', 'min', BC_TOL)
_check_slab("adj_top",     'z', 'max', BC_TOL)
 
# adj_cyl_lateral: r ∈ [R, R+CYL_TOL], z ∈ [0, H]
r_lat = next(r for r in COLLOCATION_REGIONS if r["name"] == "adj_cyl_lat")
sl = slice(r_lat["start"], r_lat["end"])
r_vals = np.sqrt((x_phys[sl] - XC_CYL)**2 + (y_phys[sl] - YC_CYL)**2)
ok = (r_vals >= R_CYL - 1e-3) & (r_vals <= R_CYL + CYL_TOL + 1e-3) & \
     (z_phys[sl] >= Z_MIN) & (z_phys[sl] <= H_CYL)
n_bad = int((~ok).sum())
print(f"  {'adj_cyl_lat':<14}  pts fora do anel: {n_bad}  (esperado: 0)")
assert n_bad == 0
 
# adj_cyl_top: r ∈ [0, R], z ∈ [H, H+CYL_TOL]
r_top = next(r for r in COLLOCATION_REGIONS if r["name"] == "adj_cyl_top")
sl = slice(r_top["start"], r_top["end"])
r_vals = np.sqrt((x_phys[sl] - XC_CYL)**2 + (y_phys[sl] - YC_CYL)**2)
ok = (r_vals <= R_CYL + 1e-3) & \
     (z_phys[sl] >= H_CYL - 1e-3) & (z_phys[sl] <= H_CYL + CYL_TOL + 1e-3)
n_bad = int((~ok).sum())
print(f"  {'adj_cyl_top':<14}  pts fora do disco: {n_bad}  (esperado: 0)")
assert n_bad == 0
 
# BOI_C: pts no bloco devem estar dentro de BOI_C e fora de cilindro/adj_cyl
r_bc = next(r for r in COLLOCATION_REGIONS if r["name"] == "boi_c")
sl = slice(r_bc["start"], r_bc["end"])
in_box = is_inside_boi_cyl(x_phys[sl], y_phys[sl], z_phys[sl])
in_cyl = is_inside_cylinder(x_phys[sl], y_phys[sl], z_phys[sl])
in_adj_cyl = (is_adj_cyl_lateral(x_phys[sl], y_phys[sl], z_phys[sl])
              | is_adj_cyl_top(x_phys[sl], y_phys[sl], z_phys[sl]))
n_bad = int((~in_box | in_cyl | in_adj_cyl).sum())
print(f"  {'boi_c':<14}  pts inválidos: {n_bad}  (esperado: 0)")
assert n_bad == 0
 
# BOI_T: dentro do prisma
r_bt = next(r for r in COLLOCATION_REGIONS if r["name"] == "boi_t")
sl = slice(r_bt["start"], r_bt["end"])
ok = is_inside_boi_tri(x_phys[sl], y_phys[sl], z_phys[sl])
n_bad = int((~ok).sum())
print(f"  {'boi_t':<14}  pts fora do prisma: {n_bad}  (esperado: 0)")
assert n_bad == 0
 
# Freestream: fora de tudo
r_free = next(r for r in COLLOCATION_REGIONS if r["name"] == "free")
sl = slice(r_free["start"], r_free["end"])
in_cyl = is_inside_cylinder(x_phys[sl], y_phys[sl], z_phys[sl])
in_boi_c = is_inside_boi_cyl(x_phys[sl], y_phys[sl], z_phys[sl])
in_boi_t = is_inside_boi_tri(x_phys[sl], y_phys[sl], z_phys[sl])
in_adj_bc = (
    (y_phys[sl] < Y_MIN + BC_TOL) | (y_phys[sl] > Y_MAX - BC_TOL) |
    (x_phys[sl] < X_MIN + BC_TOL) | (x_phys[sl] > X_MAX - BC_TOL) |
    (z_phys[sl] < Z_MIN + BC_TOL) | (z_phys[sl] > Z_MAX - BC_TOL)
)
in_adj_cyl = (is_adj_cyl_lateral(x_phys[sl], y_phys[sl], z_phys[sl])
              | is_adj_cyl_top(x_phys[sl], y_phys[sl], z_phys[sl]))
n_bad = int((in_cyl | in_boi_c | in_boi_t | in_adj_bc | in_adj_cyl).sum())
print(f"  {'free':<14}  pts em outras regiões: {n_bad}  (esperado: 0)")
assert n_bad == 0
 
print(f"  ✓ Todos os {len(COLLOCATION_REGIONS)} checks passaram.\n")
 
 
# ============================================================
# Plots em XY e XZ com 5 cores agrupadas → verificação visual

# adj_BC externas (6 faces) → azul
# adj_cyl (lat + top)       → vermelho
# BOI_C                     → laranja
# BOI_T                     → verde
# free                      → cinza
 
cor_bc      = (0.20, 0.45, 0.85)
cor_cyl     = (0.85, 0.20, 0.20)
cor_boi_c   = (0.95, 0.55, 0.10)
cor_boi_t   = (0.20, 0.65, 0.30)
cor_free    = (0.50, 0.50, 0.55)
 
# Slices agrupadas
def _region_slice(name):
    r = next(r for r in COLLOCATION_REGIONS if r["name"] == name)
    return slice(r["start"], r["end"])
 
# Índices das BCs externas concatenadas
_external_bc_names = ["adj_inlet", "adj_outlet", "adj_xmin", "adj_xmax",
                       "adj_ground", "adj_top"]
_ext_idx = np.concatenate([
    np.arange(*_region_slice(n).indices(N_COLL)) for n in _external_bc_names
])
_cyl_idx = np.concatenate([
    np.arange(*_region_slice("adj_cyl_lat").indices(N_COLL)),
    np.arange(*_region_slice("adj_cyl_top").indices(N_COLL)),
])
_boic_idx = np.arange(*_region_slice("boi_c").indices(N_COLL))
_boit_idx = np.arange(*_region_slice("boi_t").indices(N_COLL))
_free_idx = np.arange(*_region_slice("free").indices(N_COLL))
 
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
 
# --- XY (top) ---
ax = axes[0]
ax.scatter(x_phys[_free_idx], y_phys[_free_idx], s=0.3, alpha=0.20,
            c=[cor_free], rasterized=True, label=f'Free ({len(_free_idx)//1000}k)')
ax.scatter(x_phys[_ext_idx],  y_phys[_ext_idx],  s=0.5, alpha=0.45,
            c=[cor_bc],   rasterized=True, label=f'adj_BC ext. ({len(_ext_idx)//1000}k)')
ax.scatter(x_phys[_boit_idx], y_phys[_boit_idx], s=0.5, alpha=0.50,
            c=[cor_boi_t],rasterized=True, label=f'BOI_T ({len(_boit_idx)//1000}k)')
ax.scatter(x_phys[_boic_idx], y_phys[_boic_idx], s=0.5, alpha=0.50,
            c=[cor_boi_c],rasterized=True, label=f'BOI_C ({len(_boic_idx)//1000}k)')
ax.scatter(x_phys[_cyl_idx],  y_phys[_cyl_idx],  s=1.0, alpha=0.70,
            c=[cor_cyl],  rasterized=True, label=f'adj_cyl ({len(_cyl_idx)//1000}k)')
 
# Contornos
ax.add_patch(Rectangle((BOI_C_X[0], BOI_C_Y[0]),
                         BOI_C_X[1]-BOI_C_X[0], BOI_C_Y[1]-BOI_C_Y[0],
                         fill=False, edgecolor=cor_boi_c, lw=2, ls='--', zorder=5))
ax.add_patch(Polygon([BOI_T_V1, BOI_T_V2, BOI_T_V3],
                       fill=False, edgecolor=cor_boi_t, lw=2, ls='--', zorder=5))
ax.add_patch(Circle((XC_CYL, YC_CYL), R_CYL, fill=False, color='black', lw=1.5, zorder=6))
 
ax.annotate('', xy=(20, 120), xytext=(20, 30),
             arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax.text(25, 75, 'flow', fontsize=10, color='gray')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title(f'Top view (XY) — pool total: {N_COLL//1000}k pts')
ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=8, markerscale=8)
 
# --- XZ (side) ---
ax = axes[1]
ax.scatter(x_phys[_free_idx], z_phys[_free_idx], s=0.3, alpha=0.20,
            c=[cor_free], rasterized=True)
ax.scatter(x_phys[_ext_idx],  z_phys[_ext_idx],  s=0.5, alpha=0.45,
            c=[cor_bc],   rasterized=True)
ax.scatter(x_phys[_boit_idx], z_phys[_boit_idx], s=0.5, alpha=0.50,
            c=[cor_boi_t],rasterized=True)
ax.scatter(x_phys[_boic_idx], z_phys[_boic_idx], s=0.5, alpha=0.50,
            c=[cor_boi_c],rasterized=True)
ax.scatter(x_phys[_cyl_idx],  z_phys[_cyl_idx],  s=1.0, alpha=0.70,
            c=[cor_cyl],  rasterized=True)
ax.add_patch(Rectangle((XC_CYL - R_CYL, 0), 2*R_CYL, H_CYL,
                         fill=False, color='black', lw=1.5, zorder=6))
ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
ax.set_title('Side view (XZ)')
ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Z_MIN, Z_MAX)
 
plt.tight_layout()
plt.savefig('collocation_distribution.png', dpi=120)
plt.close()
print(f"Plot salvo: collocation_distribution.png\n")



# ============================================================
# Rede neural + Camada de Imposição Rídiga (Hard Constraint Layer)

class TanH(layers.Layer):
    def call(self, x):
        return tf.tanh(x)


class HardConstraintLayer(layers.Layer):
    """
    Impõe BCs arquiteturalmente via ansatz multiplicativo.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def smooth_max(a, b, k=50.0):
        """
        Smooth approximation de max(a, b). Com k=50, transição em escala ~0.02.
        """
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

        # --- Constantes (reduzidas para segurança numérica em fp32) ---
        mt       = 2000.0   # entre 500 (suave) e 5600 (dura)
        rate     = 500.0    # entre 100 e 3300
        k_smooth = 500.0    # entre 50 e 5000
        k_r2     = 20.0     # entre 5 e 50

        # --- 1. Funções de distância ---
        D_inlet  = tf.math.tanh(mt * (yn - YN_MIN))      # 0 em y = Y_MIN
        D_outlet = tf.math.tanh(mt * (YN_MAX - yn))      # 0 em y = Y_MAX
        D_ground = tf.math.tanh(mt * (zn - ZN_MIN))      # 0 em z = Z_MIN
        D_xmin   = tf.math.tanh(mt * (xn - XN_MIN))      # 0 em x = X_MIN
        D_xmax   = tf.math.tanh(mt * (XN_MAX - xn))      # 0 em x = X_MAX
        D_ztop   = tf.math.tanh(mt * (ZN_MAX - zn))      # 0 em z = Z_MAX

        r2_normalized = tf.square(xn - XC_N) + tf.square(yn - YC_N)
        d_radial_sq   = r2_normalized - R_N * R_N    # > 0 fora, < 0 dentro, 0 na parede
        d_top         = zn - H_N
        d_cyl_3d      = self.smooth_max(d_radial_sq, d_top, k=k_smooth)
        D_cyl         = tf.math.tanh(k_r2 * tf.maximum(d_cyl_3d, 0.0))

        # --- 2. Multiplicador único para u, v, w ---
        D_vel = D_inlet * D_ground * D_cyl * D_xmin * D_xmax * D_ztop

        # --- 3. Decays freestream ---
        decay_inlet = tf.exp(-rate * (yn - YN_MIN))
        decay_xmin  = tf.exp(-rate * (xn - XN_MIN))
        decay_xmax  = tf.exp(-rate * (XN_MAX - xn))
        decay_ztop  = tf.exp(-rate * (ZN_MAX - zn))
        decay_freestream = 1.0 - (1.0 - decay_inlet) \
                               * (1.0 - decay_xmin)  \
                               * (1.0 - decay_xmax)  \
                               * (1.0 - decay_ztop)

        # --- 4. Velocidade hard ansatz ---
        v_baseline = VN_INLET * decay_freestream * D_ground * D_cyl
        u_hard = un * D_vel
        v_hard = vn * D_vel + v_baseline
        w_hard = wn * D_vel

        # --- 5. Pressão hard ansatz: P = 0 em y = Y_MAX (saída) ---
        P_hard = Pn * D_outlet

        return tf.concat([u_hard, v_hard, w_hard, P_hard], axis=1)
    

def build_pinn(n_layers=6, n_neurons=256):
    """
    Constrói PINN com arquitetura MLP simples + Hard Constraints.
    Estrutura: Input(4) → Dense(256)+tanh × n_layers → Saída(256→4) → HardConstraint
    """
    inp = layers.Input(shape=(4,), name="xyzt")
    
    # Stack de camadas densas com tanh
    h = inp
    for i in range(n_layers):
        h = layers.Dense(
            n_neurons,
            activation="tanh",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name=f"dense_{i+1}",
        )(h)
    
    # Saída linear (sem ativação) — necessário pra pressão poder ser negativa
    # A última Dense antes da hard constraint
    raw_out = layers.Dense(
        4,                       
        activation='linear',     
        bias_initializer="zeros",
        name="raw_fields"
    )(h)
    
    # Aplica hard constraints (BCs forçadas arquiteturalmente)
    final_out = HardConstraintLayer(name="hard_constraints")([inp, raw_out])
    
    return Model(inp, final_out, name="PINN_MLP")



# ============================================================
# Build do modelo

MODEL_BLOCKS  = 6
MODEL_NEURONS = 256

model = build_pinn(MODEL_BLOCKS, MODEL_NEURONS)

print(f"Architecture: PINN ResNet")
print(f"  Residual blocks:    {MODEL_BLOCKS}")
print(f"  Neurons/layer:      {MODEL_NEURONS}")
print(f"  Total parameters:   {model.count_params():,}")
print(f"  Input shape:        (None, 4)")
print(f"  Output shape:       (None, 4)")

# Verificação rápida de que o modelo processa um batch de input sem produzir NaN/Inf
test_input = tf.constant(np.column_stack([
    np.random.uniform(XN_MIN, XN_MAX, 10),
    np.random.uniform(YN_MIN, YN_MAX, 10),
    np.random.uniform(ZN_MIN, ZN_MAX, 10),
    np.random.uniform(0, 1, 10),
]).astype(np.float32))

test_output = model(test_input, training=False)
n_finite = tf.reduce_all(tf.math.is_finite(test_output)).numpy()
print(f"\nSanity check:")
print(f"  Output shape: {test_output.shape}")
print(f"  All finite:   {n_finite}")
assert n_finite, "Model produces NaN/Inf on normalized input within valid domain!"
# Diagnóstico em camadas

# 1. Pega input de teste (range normalizado válido)
test_input = tf.constant(np.random.uniform(0, 1, (10, 4)).astype(np.float32))
print(f"Input: range [{test_input.numpy().min():.3f}, {test_input.numpy().max():.3f}]")
print(f"Input finite: {tf.reduce_all(tf.math.is_finite(test_input)).numpy()}")

# 2. Passa pelo modelo SEM a hard constraint
intermediate_model = Model(
    inputs=model.input,
    outputs=model.get_layer("raw_fields").output
)
raw_pred = intermediate_model(test_input, training=False).numpy()
print(f"\nRaw output (antes da hard constraint):")
print(f"  shape: {raw_pred.shape}")
print(f"  finite: {np.isfinite(raw_pred).all()}")
print(f"  range: [{raw_pred.min():.3f}, {raw_pred.max():.3f}]")
print(f"  mean: {raw_pred.mean():.3f}, std: {raw_pred.std():.3f}")

# 3. Passa pelo modelo COMPLETO (com hard constraint)
final_pred = model(test_input, training=False).numpy()
print(f"\nFinal output (após hard constraint):")
print(f"  shape: {final_pred.shape}")
print(f"  finite: {np.isfinite(final_pred).all()}")
print(f"  range: [{np.nanmin(final_pred):.3f}, {np.nanmax(final_pred):.3f}]")
print(f"  NaN count: {np.isnan(final_pred).sum()} / {final_pred.size}")
print(f"  Inf count: {np.isinf(final_pred).sum()} / {final_pred.size}")


# ============================================================
# Validação numérica das Hard Constraints
# (confirmação que as hard constraints produzem valores corretos nas BCs
# antes de iniciar o treino. Como o modelo ainda não foi treinado, os valores
# nas BCs devem vir direto da hard constraint arquitetural)

print("="*70)
print("VALIDATING HARD CONSTRAINTS (untrained model)")
print("="*70)

N_TEST = 1000

# --- Inlet (y=YN_MIN): esperado u=0, v=V_INF, w=0 ---
inlet_pts = tf.constant(np.column_stack([
    np.random.uniform(XN_MIN, XN_MAX, N_TEST),
    np.full(N_TEST, YN_MIN),
    np.random.uniform(ZN_MIN, ZN_MAX, N_TEST),
    np.random.uniform(0, 1, N_TEST),
]).astype(np.float32))

pred = model(inlet_pts, training=False).numpy()
u, v, w, P = pred[:, 0] * V_REF, pred[:, 1] * V_REF, pred[:, 2] * V_REF, pred[:, 3] * P_REF
print(f"\nInlet (y=Y_MIN):")
print(f"  u: mean={u.mean():+.4f}  std={u.std():.4f}  (expected: 0)")
print(f"  v: mean={v.mean():+.4f}  std={v.std():.4f}  (expected: {V_INF})")
print(f"  w: mean={w.mean():+.4f}  std={w.std():.4f}  (expected: 0)")

# --- Outlet (y=YN_MAX): esperado P=0 ---
outlet_pts = tf.constant(np.column_stack([
    np.random.uniform(XN_MIN, XN_MAX, N_TEST),
    np.full(N_TEST, YN_MAX),
    np.random.uniform(ZN_MIN, ZN_MAX, N_TEST),
    np.random.uniform(0, 1, N_TEST),
]).astype(np.float32))

pred = model(outlet_pts, training=False).numpy()
P = pred[:, 3] * P_REF
print(f"\nOutlet (y=Y_MAX):")
print(f"  P: mean={P.mean():+.4f}  std={P.std():.4f}  (expected: 0)")

# --- Ground (z=ZN_MIN): esperado u=v=w=0 (no-slip) ---
ground_pts = tf.constant(np.column_stack([
    np.random.uniform(XN_MIN, XN_MAX, N_TEST),
    np.random.uniform(YN_MIN, YN_MAX, N_TEST),
    np.full(N_TEST, ZN_MIN),
    np.random.uniform(0, 1, N_TEST),
]).astype(np.float32))

pred = model(ground_pts, training=False).numpy()
u, v, w = pred[:, 0] * V_REF, pred[:, 1] * V_REF, pred[:, 2] * V_REF
print(f"\nGround (z=Z_MIN):")
print(f"  u: mean={u.mean():+.4f}  std={u.std():.4f}  (expected: 0)")
print(f"  v: mean={v.mean():+.4f}  std={v.std():.4f}  (expected: 0)")
print(f"  w: mean={w.mean():+.4f}  std={w.std():.4f}  (expected: 0)")

# --- Side wall x=XN_MIN: esperado u=0, v=V_INF, w=0 (wall moving) ---
side_pts = tf.constant(np.column_stack([
    np.full(N_TEST, XN_MIN),
    np.random.uniform(YN_MIN, YN_MAX, N_TEST),
    np.random.uniform(ZN_MIN, ZN_MAX, N_TEST),
    np.random.uniform(0, 1, N_TEST),
]).astype(np.float32))

pred = model(side_pts, training=False).numpy()
u, v, w = pred[:, 0] * V_REF, pred[:, 1] * V_REF, pred[:, 2] * V_REF
print(f"\nSide wall (x=X_MIN, moving wall):")
print(f"  u: mean={u.mean():+.4f}  std={u.std():.4f}  (expected: 0)")
print(f"  v: mean={v.mean():+.4f}  std={v.std():.4f}  (expected: {V_INF})")
print(f"  w: mean={w.mean():+.4f}  std={w.std():.4f}  (expected: 0)")

# --- Top wall z=ZN_MAX: esperado u=0, v=V_INF, w=0 (wall moving) ---
top_pts = tf.constant(np.column_stack([
    np.random.uniform(XN_MIN, XN_MAX, N_TEST),
    np.random.uniform(YN_MIN, YN_MAX, N_TEST),
    np.full(N_TEST, ZN_MAX),
    np.random.uniform(0, 1, N_TEST),
]).astype(np.float32))

pred = model(top_pts, training=False).numpy()
u, v, w = pred[:, 0] * V_REF, pred[:, 1] * V_REF, pred[:, 2] * V_REF
print(f"\nTop wall (z=Z_MAX, moving wall):")
print(f"  u: mean={u.mean():+.4f}  std={u.std():.4f}  (expected: 0)")
print(f"  v: mean={v.mean():+.4f}  std={v.std():.4f}  (expected: {V_INF})")
print(f"  w: mean={w.mean():+.4f}  std={w.std():.4f}  (expected: 0)")

# --- Cylinder surface: esperado u=v=w=0 (no-slip) ---
theta = np.random.uniform(0, 2*np.pi, N_TEST)
cyl_pts = tf.constant(np.column_stack([
    XC_N + R_N * np.cos(theta),
    YC_N + R_N * np.sin(theta),
    np.random.uniform(ZN_MIN, H_N, N_TEST),
    np.random.uniform(0, 1, N_TEST),
]).astype(np.float32))

pred = model(cyl_pts, training=False).numpy()
u, v, w = pred[:, 0] * V_REF, pred[:, 1] * V_REF, pred[:, 2] * V_REF
print(f"\nCylinder surface (no-slip):")
print(f"  u: mean={u.mean():+.4f}  std={u.std():.4f}  (expected: 0)")
print(f"  v: mean={v.mean():+.4f}  std={v.std():.4f}  (expected: 0)")
print(f"  w: mean={w.mean():+.4f}  std={w.std():.4f}  (expected: 0)")

print("\n" + "="*70)
print("If any value differs significantly from expected, hard constraints have bugs.")
print("="*70)



# ============================================================
# Resíduos Físicos (PDE residuals)

# Computar constantes como tensores do tf
_L       = tf.constant(L_REF,    tf.float32)
_V       = tf.constant(V_REF,    tf.float32)
_P       = tf.constant(P_REF,    tf.float32)
_tR      = tf.constant(TIME_REF, tf.float32)
_RHO_INF = tf.constant(RHO_INF,  tf.float32)
_MU      = tf.constant(MU,       tf.float32)

# Escalas para resíduos adimensionalizados
SCALE_CONT = V_REF / L_REF                      # 1/s
SCALE_MOM  = RHO_INF * V_REF**2 / L_REF         # Pa/m

_SCALE_CONT = tf.constant(SCALE_CONT, tf.float32)
_SCALE_MOM  = tf.constant(SCALE_MOM,  tf.float32)


def physics_residuals(coll_pts, model):
    """
    Computa os resíduos das EDP das de Navier-Stokes incompressíveis nos pontos de colocação.

    Equações
        Continuidade:   ∇·V = 0 (pois ρ é constante)
        Momentum:     ρ(∂V/∂t + V·∇V) = -∇P + μ∇²V

    Retorna os resíduos adimensionalizados (O(1)).
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coll_pts)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coll_pts)

            pred = model(coll_pts, training=True)   # (N, 4)
            un = pred[:, 0:1]
            vn = pred[:, 1:2]
            wn = pred[:, 2:3]
            Pn = pred[:, 3:4]

            # Desnormalizar para obter campos físicos (com unidades)
            u = un * _V
            v = vn * _V
            w = wn * _V
            P = Pn * _P

        scale_grad = tf.stack([_L, _L, _L, _tR])

        du = tape1.gradient(u, coll_pts) / scale_grad   # (N, 4)
        dv = tape1.gradient(v, coll_pts) / scale_grad
        dw = tape1.gradient(w, coll_pts) / scale_grad
        dP = tape1.gradient(P, coll_pts) / scale_grad

        u_x, u_y, u_z, u_t = du[:, 0], du[:, 1], du[:, 2], du[:, 3]
        v_x, v_y, v_z, v_t = dv[:, 0], dv[:, 1], dv[:, 2], dv[:, 3]
        w_x, w_y, w_z, w_t = dw[:, 0], dw[:, 1], dw[:, 2], dw[:, 3]
        P_x, P_y, P_z      = dP[:, 0], dP[:, 1], dP[:, 2]

        del tape1

    # Laplacianos
    def lap(fx, fy, fz):
        """∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²"""
        fxx = tape2.gradient(fx, coll_pts)[:, 0] / _L
        fyy = tape2.gradient(fy, coll_pts)[:, 1] / _L
        fzz = tape2.gradient(fz, coll_pts)[:, 2] / _L
        return fxx + fyy + fzz

    lap_u = lap(u_x, u_y, u_z)
    lap_v = lap(v_x, v_y, v_z)
    lap_w = lap(w_x, w_y, w_z)

    del tape2

    u_flat = tf.squeeze(u, -1)
    v_flat = tf.squeeze(v, -1)
    w_flat = tf.squeeze(w, -1)

    # ─────────────────────────────────────────────────────────
    # Continuidade (com ρ constante → incompressível):
    #     ∇·V = 0
    # ─────────────────────────────────────────────────────────
    r_cont = u_x + v_y + w_z

    # ─────────────────────────────────────────────────────────
    # Momentum (Navier-Stokes incompressível, ρ constante):
    #     ρ(∂V/∂t + V·∇V) = -∇P + μ∇²V
    #
    # Resíduos na forma por componentes = LHS - RHS, tem que ser = 0):
    #     ρ(u_t + u·u_x + v·u_y + w·u_z) + P_x - μ∇²u = 0
    # ─────────────────────────────────────────────────────────
    r_mu = _RHO_INF * (u_t + u_flat*u_x + v_flat*u_y + w_flat*u_z) + P_x - _MU*lap_u
    r_mv = _RHO_INF * (v_t + u_flat*v_x + v_flat*v_y + w_flat*v_z) + P_y - _MU*lap_v
    r_mw = _RHO_INF * (w_t + u_flat*w_x + v_flat*w_y + w_flat*w_z) + P_z - _MU*lap_w

    # Adimensionalização por escalonamento
    return (r_cont / _SCALE_CONT,
            r_mu   / _SCALE_MOM,
            r_mv   / _SCALE_MOM,
            r_mw   / _SCALE_MOM)



# ============================================================
# Funções de perda

def mse(x):
    return tf.reduce_mean(tf.square(x))

def data_loss_grouped(batch, model,
                     w_z_cyl, w_z_bc, w_z_boi_c, w_z_boi_t, w_z_free):
    """
    Data loss SEGMENTADO por zona geométrica.
    
    Para cada zona, calcula MSE ponderado por componente (mesma proporção do Run 4):
        loss_zona = 1.0*l_u + 0.3*l_v + 1.0*l_w + 1.0*l_P
    
    Depois pondera as zonas:
        total = w_cyl * loss_cyl + w_bc * loss_bc + ...
    
    Retorna: 6 valores (5 losses por zona + total ponderado)
    """
    xyzt = batch[:, :4]
    tgt  = batch[:, 4:8]   # 4 componentes (não incluir gid)
    gid  = batch[:, 8]     # group_id como float32
    
    pred = model(xyzt, training=True)
    err = pred - tgt   # (N, 4)
    
    # Erro quadrático por componente
    sq_per_pt = (1.0 * tf.square(err[:, 0])     # u
                + 0.3 * tf.square(err[:, 1])    # v
                + 1.0 * tf.square(err[:, 2])    # w
                + 1.0 * tf.square(err[:, 3]))   # P
    # shape: (N,)
    
    def mean_zone(target_gid):
        mask = tf.equal(gid, target_gid)
        n = tf.reduce_sum(tf.cast(mask, tf.float32))
        masked = tf.where(mask, sq_per_pt, tf.zeros_like(sq_per_pt))
        summed = tf.reduce_sum(masked)
        return summed / tf.maximum(n, 1.0)
    
    l_cyl   = mean_zone(0.0)   # GID_CYL
    l_bc    = mean_zone(1.0)   # GID_BC
    l_boi_c = mean_zone(2.0)   # GID_BOI_C
    l_boi_t = mean_zone(3.0)   # GID_BOI_T
    l_free  = mean_zone(4.0)   # GID_FREE
    
    total_data = (w_z_cyl   * l_cyl
                  + w_z_bc    * l_bc
                  + w_z_boi_c * l_boi_c
                  + w_z_boi_t * l_boi_t
                  + w_z_free  * l_free)
    
    return total_data, l_cyl, l_bc, l_boi_c, l_boi_t, l_free

def physics_loss(coll_pts, model):
    r_c, r_mu, r_mv, r_mw = physics_residuals(coll_pts, model)
    return mse(r_c), mse(r_mu), mse(r_mv), mse(r_mw)



# ============================================================
# Pesos da Loss - agrupamento por zona
#
# Pesos por zona (data loss):
#   freestream   3.0   ← força v ≈ 17 m/s onde |V| é dominante
#   boi_c        1.5   ← detalhamento entorno cilindro
#   boi_t        1.0   ← wake distante
#   cilindro     0.5   ← hard constraint cuida
#   BCs          0.5   ← hard constraint cuida
#
# W["data"] reduzido de 100 para 70, para compensar aumento de magnitude global
#
# Pesos por componente (dentro do data_loss_grouped):
#   u: 1.0, v: 0.3, w: 1.0, P: 1.0

W = {
    "data":        70.0,
    "data_cyl":     0.5,
    "data_bc":      0.5,
    "data_boi_c":   1.5,
    "data_boi_t":   1.0,
    "data_free":    3.0,
    "cont":         1.0,
    "mom_u":        1.0,
    "mom_v":        1.0,
    "mom_w":        1.0,
}

for k, v in W.items():
    print(f"  W[{k:12s}] = {v}")



# ============================================================
# Preparo dos tensores para treinamento

BATCH_SIZE = 48_000  # = N_POINTS_PER_SNAP (1 snapshot inteiro por batch)
                     # 16k BC+cilindro + 16k detalhamento + 16k freestream

rng = np.random.default_rng(seed=SEED)

train_snap_indices_ordered = [i for i in range(n_snaps) if i not in val_snap_indices]
shuffled_snap_order = rng.permutation(train_snap_indices_ordered)

train_np_pieces = []
for i in shuffled_snap_order:
    _, start, end = snap_offsets[i]
    train_np_pieces.append(data_np[start:end])
train_np = np.concatenate(train_np_pieces, axis=0)

print(f"Snapshots in train (first 10 in new order): {shuffled_snap_order[:10].tolist()}")
print(f"train_np shape after pre-shuffle: {train_np.shape}")

train_tf = tf.data.Dataset.from_tensor_slices(train_np)
train_tf = train_tf.shuffle(buffer_size=min(len(train_np), 200_000),
                            seed=SEED, reshuffle_each_iteration=True)
train_tf = train_tf.batch(BATCH_SIZE, drop_remainder=True)
train_tf = train_tf.prefetch(tf.data.AUTOTUNE)

val_tensor  = tf.constant(val_np,  dtype=tf.float32)
coll_tensor = tf.Variable(coll_np, dtype=tf.float32)  # resampleable

steps_per_epoch = int(np.ceil(len(train_np) / BATCH_SIZE))
print(f"Batch size: {BATCH_SIZE}   TEÓRICO -> Steps/epoch: {steps_per_epoch}")

# --- Sanity check ---
sample_batch = next(iter(train_tf))
t_values = sample_batch[:, 3].numpy()
print(f"\nTemporal mixing diagnostic (first batch):")
print(f"  Unique timesteps in batch: {len(np.unique(t_values))} / {len(train_snap_indices_ordered)} train snapshots")
print(f"  Range of t (normalized): [{t_values.min():.4f}, {t_values.max():.4f}]")
print(f"  std of t in batch: {t_values.std():.4f}  (alto = bom mix temporal)")


# ================================================================================
# Train Step com sorteio estratificado (em 11 regiões)

# Pré-compute boundaries do pool como constantes do tf
_REGION_STARTS_TF = [tf.constant(r["start"], dtype=tf.int32) for r in COLLOCATION_REGIONS]
_REGION_ENDS_TF   = [tf.constant(r["end"],   dtype=tf.int32) for r in COLLOCATION_REGIONS]
_REGION_NSTEPS    = [r["per_step"] for r in COLLOCATION_REGIONS]

# --- Criação do optimizer ---
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# --- Extrai pesos do dict W (definido na Cell 12) ---
# --- Extrai pesos do dict W (definido na Cell 12) ---
w_data   = W["data"]
w_z_cyl  = W["data_cyl"]
w_z_bc   = W["data_bc"]
w_z_bc_c = W["data_boi_c"]
w_z_bc_t = W["data_boi_t"]
w_z_free = W["data_free"]
w_c      = W["cont"]
w_mu     = W["mom_u"]
w_mv     = W["mom_v"]
w_mw     = W["mom_w"]

print(f"\nPesos de zona (data_loss):")
print(f"  cyl={w_z_cyl}, bc={w_z_bc}, boi_c={w_z_bc_c}, "
      f"boi_t={w_z_bc_t}, free={w_z_free}")
print(f"Soma efetiva ponderada ≈ {0.08*w_z_cyl + 0.12*w_z_bc + 0.21*w_z_bc_c + 0.12*w_z_bc_t + 0.33*w_z_free:.2f}")
print(f"Pesos físicos: cont={w_c}, mom_u={w_mu}, mom_v={w_mv}, mom_w={w_mw}")
print(f"W[\"data\"] global = {w_data}\n")


@tf.function(jit_compile=True)
def train_step(batch, coll_full,
               w_data, w_z_cyl, w_z_bc, w_z_bc_c, w_z_bc_t, w_z_free,
               w_c, w_mu, w_mv, w_mw):
    """
    Train step com data loss SEGMENTADO em 5 zonas geométricas.
    
    Args:
        batch: (B, 9) — colunas: xn,yn,zn,tn, u,v,w,P, group_id
        coll_full: (Ncoll, 4) — pool de collocation
        w_data: peso global do data loss
        w_z_*: pesos por zona (cyl, bc, boi_c, boi_t, free)
        w_c, w_mu, w_mv, w_mw: pesos da física
    
    Returns 11 valores:
        total, l_d, l_cyl, l_bc, l_boi_c, l_boi_t, l_free,
        l_c, l_mu, l_mv, l_mw, gnorm
    """
    # Sorteio estratificado do collocation pool
    idx_pieces = []
    for start_tf, end_tf, n_step in zip(_REGION_STARTS_TF, _REGION_ENDS_TF, _REGION_NSTEPS):
        idx_piece = tf.random.uniform((n_step,), start_tf, end_tf, dtype=tf.int32)
        idx_pieces.append(idx_piece)
    idx = tf.concat(idx_pieces, axis=0)
    coll_sub = tf.gather(coll_full, idx)

    with tf.GradientTape() as tape:
        l_d, l_cyl, l_bc, l_boi_c, l_boi_t, l_free = data_loss_grouped(
            batch, model,
            w_z_cyl, w_z_bc, w_z_bc_c, w_z_bc_t, w_z_free,
        )
        l_c, l_mu, l_mv, l_mw = physics_loss(coll_sub, model)
        
        total = (w_data * l_d
                 + w_c  * l_c
                 + w_mu * l_mu + w_mv * l_mv + w_mw * l_mw)

    grads = tape.gradient(total, model.trainable_variables)
    grads, gnorm = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return (total, l_d, l_cyl, l_bc, l_boi_c, l_boi_t, l_free,
            l_c, l_mu, l_mv, l_mw, gnorm)

# ============================================================
# Loop de treino (otimizado para o Cluster)

import csv

DEBUG_MODE = "full"   # "fast" | "calibration" | "medium" | "full" 

LR_CONSTANT = 1e-4   # solicitado pelo orientador: LR constante

if DEBUG_MODE == "fast":
    N_EPOCHS         = 5
    LOG_EVERY        = 1
    SAVE_EVERY       = 999_999
    VAL_EVERY        = 1
    STEPS_PER_EPOCH  = 100
    CONV_ENABLED     = False

elif DEBUG_MODE == "calibration":
    N_EPOCHS         = 20
    LOG_EVERY        = 1            # Log toda época pra ver magnitudes
    SAVE_EVERY       = 999_999
    VAL_EVERY        = 5
    STEPS_PER_EPOCH  = 100
    CONV_ENABLED     = False

elif DEBUG_MODE == "medium":
    N_EPOCHS         = 450
    LOG_EVERY        = 25
    SAVE_EVERY       = 50
    VAL_EVERY        = 10
    STEPS_PER_EPOCH  = 100
    CONV_ENABLED     = True
    CONV_PATIENCE    = 100
    CONV_MIN_PCT     = 1.0
    CONV_WARMUP      = 200

elif DEBUG_MODE == "full":
    N_EPOCHS         = 10_000
    LOG_EVERY        = 50
    SAVE_EVERY       = 100
    VAL_EVERY        = 50
    STEPS_PER_EPOCH  = 100
    CONV_ENABLED     = True
    CONV_PATIENCE    = 200
    CONV_MIN_PCT     = 0.5
    CONV_WARMUP      = 500
    
RESAMPLE_COLL_EVERY = 999_999

# --- LR constante ---
optimizer.learning_rate.assign(LR_CONSTANT)
print(f"Learning rate fixado em {LR_CONSTANT:.1e}")

# --- Detecção de continuação ---
if 'history' not in globals() or len(history.get('epoch', [])) == 0:
    history = {
        "epoch": [], "total": [],
        "data":  [], "cont":  [], "mom_u": [], "mom_v": [], "mom_w": [],
        "phys_sum": [], "val": [], "lr": [], "gnorm": [],
    }
    epoch_offset = 0
    print("Iniciando treino do zero")
else:
    epoch_offset = max(history["epoch"])
    print(f"Continuando treino. Última época registrada: {epoch_offset}")

CHECKPOINT_DIR = "./pinn_checkpoints_segregated"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- CSV de resíduos por época ---
CSV_LOG_PATH = "training_log_segregated.csv"
if not os.path.exists(CSV_LOG_PATH) or epoch_offset == 0:
    # cria novo CSV com header
    with open(CSV_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "lr", "loss_total", "loss_data",
            "loss_data_cyl", "loss_data_bc", "loss_data_boi_c", "loss_data_boi_t", "loss_data_free",
            "loss_cont", "loss_mom_u", "loss_mom_v", "loss_mom_w",
            "loss_phys_sum", "loss_val", "gnorm", "elapsed_s"
        ])
    print(f"CSV de log criado: {CSV_LOG_PATH}")
else:
    print(f"CSV existente, fará append: {CSV_LOG_PATH}")

print(f"\n{'='*70}")
print(f"  Training PINN: {N_EPOCHS} new epochs (from ep {epoch_offset})")
print(f"  Total after: {epoch_offset + N_EPOCHS} epochs")
print(f"  LR: {float(optimizer.learning_rate.numpy()):.2e} (constante)")
print(f"  Train points: {len(train_np):,}   Val points: {len(val_np):,}")
print(f"  Steps/epoch:  {STEPS_PER_EPOCH} (de {steps_per_epoch} disponíveis)")
print(f"  Collocation:  {N_COLL:,}   Physics batch per step: {N_COLL_PER_STEP}")
print(f"  Pesos: data={w_data}  cont={w_c}  mom={w_mu}/{w_mv}/{w_mw}")
print(f"{'='*70}\n")


# ============================================================
# Monitoramento de convergência

class ConvergenceMonitor:
    """
    Detecta estagnação do treino observando uma métrica.
    Para quando a melhora relativa nas últimas `patience` avaliações é
    menor que `min_improvement_pct`.
    """
    def __init__(self, patience=200, min_improvement_pct=0.5, 
                 warmup_epochs=500, metric_name="val"):
        self.patience = patience
        self.min_improvement_pct = min_improvement_pct
        self.warmup_epochs = warmup_epochs
        self.metric_name = metric_name
        self.history = []
        self.best_value = float('inf')
        self.best_epoch = 0
        self.converged_epoch = None
    
    def update(self, epoch, value):
        """Returns True se convergiu."""
        self.history.append((epoch, value))
        
        if value < self.best_value:
            self.best_value = value
            self.best_epoch = epoch
        
        if epoch < self.warmup_epochs:
            return False
        if len(self.history) < self.patience:
            return False
        
        recent = self.history[-self.patience:]
        values = [v for _, v in recent]
        v_start = values[0]
        v_min_recent = min(values)
        
        if v_start <= 0:
            return False
        improvement_pct = 100.0 * (v_start - v_min_recent) / v_start
        
        if improvement_pct < self.min_improvement_pct:
            self.converged_epoch = epoch
            return True
        return False
    
    def status_str(self):
        if len(self.history) < self.patience:
            return f"[conv-mon] coletando dados ({len(self.history)}/{self.patience})"
        recent = self.history[-self.patience:]
        values = [v for _, v in recent]
        v_start = values[0]
        v_curr = values[-1]
        v_min = min(values)
        improvement_pct = 100.0 * (v_start - v_min) / max(v_start, 1e-12)
        return (f"[conv-mon] {self.metric_name} {v_start:.3e} → {v_curr:.3e} "
                f"(min={v_min:.3e}) | melhora {improvement_pct:.2f}% "
                f"em {self.patience} ép | threshold {self.min_improvement_pct}%")

best_val_loss = float('inf')
best_epoch = 0
best_ckpt_path = os.path.join(CHECKPOINT_DIR, "pinn_best_segregated.weights.h5")
print(f"Best checkpoint: {best_ckpt_path}")

CONV_ENABLED = (DEBUG_MODE == "full" or DEBUG_MODE == "medium")
if CONV_ENABLED:
    conv_monitor = ConvergenceMonitor(
        patience=200,           # Olha últimas 200 épocas
        min_improvement_pct=0.5, # Exige 0.5% de melhora
        warmup_epochs=500,       # Não para antes da ép 500
        metric_name="val",
    )

# --- Função de validação com jit-compilada (mais rápido) ---
@tf.function(jit_compile=True)
def evaluate_val(val_tensor):
    """
    Val não-segregada (igual ao Run 4) pra ter baseline consistente.
    val_tensor agora tem 9 colunas: [:,:4]=coords, [:,4:8]=targets, [:,8]=gid
    """
    pred = model(val_tensor[:, :4], training=False)
    tgt = val_tensor[:, 4:8]
    return tf.reduce_mean(tf.square(pred - tgt))


train_start = time.time()

for epoch in range(1, N_EPOCHS + 1):

    ep_total = tf.constant(0.0)
    ep_d     = tf.constant(0.0)
    ep_c     = tf.constant(0.0)
    ep_mu    = tf.constant(0.0)
    ep_mv    = tf.constant(0.0)
    ep_mw    = tf.constant(0.0)
    ep_gnorm = tf.constant(0.0)
    n_steps = 0

    ep_d_cyl   = tf.constant(0.0)
    ep_d_bc    = tf.constant(0.0)
    ep_d_boi_c = tf.constant(0.0)
    ep_d_boi_t = tf.constant(0.0)
    ep_d_free  = tf.constant(0.0)

    train_iter = iter(train_tf)
    
    while n_steps < STEPS_PER_EPOCH:
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        
        (total, l_d, l_cyl, l_bc, l_boi_c, l_boi_t, l_free,
        l_c, l_mu, l_mv, l_mw, gnorm) = train_step(
            batch, coll_tensor,
            w_data, w_z_cyl, w_z_bc, w_z_bc_c, w_z_bc_t, w_z_free,
            w_c, w_mu, w_mv, w_mw,
        )

        ep_total += total
        ep_d     += l_d
        ep_c     += l_c
        ep_mu    += l_mu
        ep_mv    += l_mv
        ep_mw    += l_mw
        ep_gnorm += gnorm

        ep_d_cyl   += l_cyl
        ep_d_bc    += l_bc
        ep_d_boi_c += l_boi_c
        ep_d_boi_t += l_boi_t
        ep_d_free  += l_free

        n_steps += 1
    
    # Médias da época
    avg_total = float(ep_total / n_steps) if n_steps > 0 else 0.0
    avg_data  = float(ep_d / n_steps) if n_steps > 0 else 0.0
    avg_cont  = float(ep_c / n_steps) if n_steps > 0 else 0.0
    avg_mu    = float(ep_mu / n_steps) if n_steps > 0 else 0.0
    avg_mv    = float(ep_mv / n_steps) if n_steps > 0 else 0.0
    avg_mw    = float(ep_mw / n_steps) if n_steps > 0 else 0.0
    avg_gnorm = float(ep_gnorm / n_steps) if n_steps > 0 else 0.0
    avg_phys_sum = avg_cont + avg_mu + avg_mv + avg_mw

    avg_d_cyl   = float(ep_d_cyl   / n_steps) if n_steps > 0 else 0.0
    avg_d_bc    = float(ep_d_bc    / n_steps) if n_steps > 0 else 0.0
    avg_d_boi_c = float(ep_d_boi_c / n_steps) if n_steps > 0 else 0.0
    avg_d_boi_t = float(ep_d_boi_t / n_steps) if n_steps > 0 else 0.0
    avg_d_free  = float(ep_d_free  / n_steps) if n_steps > 0 else 0.0

    lr_now = float(optimizer.learning_rate.numpy())
    
    history["epoch"].append(epoch + epoch_offset)
    history["total"].append(avg_total)
    history["data"].append(avg_data)
    history["cont"].append(avg_cont)
    history["mom_u"].append(avg_mu)
    history["mom_v"].append(avg_mv)
    history["mom_w"].append(avg_mw)
    history["phys_sum"].append(avg_phys_sum)
    history["lr"].append(float(optimizer.learning_rate.numpy()))
    history["gnorm"].append(avg_gnorm)
    
    # --- Validação (espaçada) ---
    if epoch % VAL_EVERY == 0 or epoch == 1:
        val_mse = float(evaluate_val(val_tensor))
        history["val"].append(val_mse)
    else:
        # Repete último valor pra manter mesmo tamanho (plotter pode mascarar duplicatas)
        history["val"].append(history["val"][-1] if history["val"] else float('nan'))
    
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        best_epoch = epoch + epoch_offset
        model.save_weights(best_ckpt_path)
        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"  ✓ Novo melhor val: {val_mse:.4e} → {os.path.basename(best_ckpt_path)}",
                   flush=True)

    if CONV_ENABLED and (epoch % VAL_EVERY == 0 or epoch == 1):
        converged = conv_monitor.update(epoch + epoch_offset, val_mse)
        
        if epoch % LOG_EVERY == 0:
            print(f"    {conv_monitor.status_str()}", flush=True)
        
        if converged:
            print(f"\n{'='*60}")
            print(f"CONVERGÊNCIA DETECTADA na época {epoch + epoch_offset}")
            print(f"  {conv_monitor.status_str()}")
            print(f"  Melhor valor: {conv_monitor.best_value:.4e} na ép {conv_monitor.best_epoch}")
            print(f"{'='*60}\n")
            
            final_path = os.path.join(CHECKPOINT_DIR, 
                                    f"pinn_converged_ep{epoch+epoch_offset:06d}.weights.h5")
            model.save_weights(final_path)
            print(f"Checkpoint final salvo: {final_path}")
            break

    # --- CSV log por época (em todas as épocas) ---
    elapsed = time.time() - train_start
    with open(CSV_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, lr_now, avg_total, avg_data,
            avg_d_cyl, avg_d_bc, avg_d_boi_c, avg_d_boi_t, avg_d_free,
            avg_cont, avg_mu, avg_mv, avg_mw,
            avg_phys_sum, val_mse, avg_gnorm, elapsed,
        ])
    
    # --- Console log (espaçado) ---
    if epoch % LOG_EVERY == 0 or epoch == 1:
        log_msg_main = (
            f"Ep {epoch:5d} | total {avg_total:.3e} | data {avg_data:.3e} "
            f"| cont {avg_cont:.3e} | mom_u {avg_mu:.3e} | mom_v {avg_mv:.3e} "
            f"| mom_w {avg_mw:.3e} | val {val_mse:.3e} | gnorm {avg_gnorm:.2f} "
            f"| {elapsed:.0f}s"
        )
        log_msg_zones = (
            f"          | d_cyl {avg_d_cyl:.3e} | d_bc {avg_d_bc:.3e} "
            f"| d_boi_c {avg_d_boi_c:.3e} | d_boi_t {avg_d_boi_t:.3e} "
            f"| d_free {avg_d_free:.3e}"
        )
        
        print(log_msg_main, flush=True)
        print(log_msg_zones, flush=True)
        
        with open("monitoramento_pinn.txt", "a") as f_log:
            f_log.write(log_msg_main + "\n")
            f_log.write(log_msg_zones + "\n")
    
    # --- Checkpoint (espaçado) ---
    if epoch % SAVE_EVERY == 0:
        ep_total_idx = epoch + epoch_offset
        path = os.path.join(CHECKPOINT_DIR, f"pinn_segregated_ep{ep_total_idx:06d}.weights.h5")
        model.save_weights(path)
        print(f"  checkpoint saved: {path}", flush=True)

print(f"\nTraining complete: {time.time()-train_start:.1f}s")
print(f"Total epochs trained: {epoch_offset + N_EPOCHS}")
print(f"CSV de resíduos: {CSV_LOG_PATH}")



# ============================================================
# Salvamento final do modelo e dos metadados

# Pesos finais no mesmo diretório dos checkpoints, com nome consistente
final_weights_path  = os.path.join(CHECKPOINT_DIR, "pinn_final_segregated.weights.h5")
final_metadata_path = os.path.join(CHECKPOINT_DIR, "pinn_final_segregated.metadata.json")
final_history_path  = os.path.join(CHECKPOINT_DIR, "pinn_final_segregated.history.json")

model.save_weights(final_weights_path)

# Metadata atualizado com configurações desta versão
metadata = dict(
    # Escalas de normalização
    L_ref     = float(L_REF),
    V_ref     = float(V_REF),
    P_ref     = float(P_REF),
    time_ref  = float(TIME_REF),

    # Geometria do domínio
    X_min=X_MIN, X_max=X_MAX,
    Y_min=Y_MIN, Y_max=Y_MAX,
    Z_min=Z_MIN, Z_max=Z_MAX,

    # Geometria do cilindro
    D_cyl=D_CYL, H_cyl=H_CYL,
    xc_cyl=XC_CYL, yc_cyl=YC_CYL,

    # BOIs
    BOI_C_X=list(BOI_C_X), BOI_C_Y=list(BOI_C_Y), BOI_C_Z=list(BOI_C_Z),
    BOI_T_V1=list(BOI_T_V1), BOI_T_V2=list(BOI_T_V2), BOI_T_V3=list(BOI_T_V3),
    BOI_T_Z=list(BOI_T_Z),

    # Tolerâncias geométricas
    BC_TOL=BC_TOL, CYL_TOL=CYL_TOL,

    # Escoamento e propriedades do ar
    V_inf=V_INF, P_op=P_OP, R_sp=R_SP, mu=MU, g=G_ACC, rho_inf=float(RHO_INF),

    # Janela temporal
    t_min   = float(times_all[0]),
    t_max   = float(times_all[-1]),
    n_snaps = int(n_snaps),

    # Configuração da arquitetura
    model_config=dict(
        type     = "MLP_simple",
        n_layers = 6,
        n_neurons= 256,
        n_params = int(model.count_params()),
    ),

    # Quotas de sampling
    sampling_quotas = QUOTAS,

    # Quotas de collocation
    collocation_per_step = {r["name"]: r["per_step"] for r in COLLOCATION_REGIONS},
    collocation_pool_total = N_COLL,

    # Pesos da loss
    weights = W,

    # Hyperparams de treino
    training=dict(
        batch_size      = BATCH_SIZE,
        steps_per_epoch = STEPS_PER_EPOCH,
        lr              = LR_CONSTANT,
        n_epochs        = max(history["epoch"]) if history["epoch"] else 0,
    ),
)
with open(final_metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

# History completo (curvas de loss por época) em JSON pra plot externo
with open(final_history_path, "w") as f:
    json.dump(history, f, indent=2)

print(f"Saved:")
print(f"  Weights:  {final_weights_path}")
print(f"  Metadata: {final_metadata_path}")
print(f"  History:  {final_history_path}")
print(f"  CSV log:  {CSV_LOG_PATH}   ({len(history['epoch'])} epochs)")



# ============================================================
# Plot: Loss - History 

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# ---- Subplot superior: Loss total + componentes principais ----
ax = axes[0]
ax.semilogy(history["epoch"], history["total"],    lw=2.0, label="total", color='black')
ax.semilogy(history["epoch"], history["data"],     lw=1.2, label="data", color='tab:blue')
ax.semilogy(history["epoch"], history["phys_sum"], lw=1.2, label="phys (sum)", color='tab:red')

# Validation: filtra os duplicados (mantém só onde mudou)
val_arr = np.array(history["val"])
val_ep  = np.array(history["epoch"])
mask = np.concatenate([[True], np.diff(val_arr) != 0])
ax.semilogy(val_ep[mask], val_arr[mask], 'o-', ms=3, lw=1.0,
             label="val", color='tab:green', alpha=0.8)
ax.set_ylabel("Loss (log)")
ax.set_title("PINN Training — Loss Total e Componentes Agregadas")
ax.legend(loc='upper right')
ax.grid(True, which="both", alpha=0.3)

# ---- Subplot inferior: Físicos individuais ----
ax = axes[1]
ax.semilogy(history["epoch"], history["cont"],  lw=1.2, label="continuidade", color='tab:purple')
ax.semilogy(history["epoch"], history["mom_u"], lw=1.0, label="mom u", color='tab:orange')
ax.semilogy(history["epoch"], history["mom_v"], lw=1.0, label="mom v", color='tab:brown')
ax.semilogy(history["epoch"], history["mom_w"], lw=1.0, label="mom w", color='tab:pink')
ax.set_xlabel("Epoch")
ax.set_ylabel("Physics residual (log)")
ax.set_title("Componentes Físicas Individuais")
ax.legend(loc='upper right')
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("training_log_segregated.csv.png", dpi=150)
plt.close()
print("Plot final salvo: training_log_segregated.csv.png")



# ============================================================
# Análise por fatias das predições da PINN vs CFD, para os snapshots de tempo selecionados
# (planos XZ ou XY) - plots_results/slices/

import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Diretórios ---
PLOTS_DIR = "plots_results"
SLICES_DIR = os.path.join(PLOTS_DIR, "slices")
SCATTER_DIR = os.path.join(PLOTS_DIR, "scatter")
HIST_DIR = os.path.join(PLOTS_DIR, "histograms")
SUMMARY_DIR = os.path.join(PLOTS_DIR, "summary")

for d in [SLICES_DIR, SCATTER_DIR, HIST_DIR, SUMMARY_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Plots serão salvos em: {PLOTS_DIR}/")


# --- Configurações de plotagem ---
PLANE = "xz"  # "xz" para vista lateral, "xy" para vista superior
T_SLICES_S = [10.0, 30.0, 58.0]
SLAB_TOL = 8.0
SHOW_FIGS = False  # False = só salvar PNGs sem abrir janela

if PLANE == "xz":
    SLICE_VALUES = [0, 100, 200, 250, 300, 400, 500, 600, 700]
    SLICE_LABEL = "y"
else:
    SLICE_VALUES = [10, 32, 65, 100, 150]
    SLICE_LABEL = "z"

def add_cylinder_xz(ax, y_slice):
    """Cilindro lateral (corte XZ). Visível se slice cruza o cilindro."""
    if abs(y_slice - YC_CYL) > R_CYL:
        return
    rect = mpatches.Rectangle(
        (XC_CYL - R_CYL, 0), 2 * R_CYL, H_CYL,
        linewidth=1.5, edgecolor='black', facecolor='white', zorder=10,
    )
    ax.add_patch(rect)


def add_cylinder_xy(ax, z_slice):
    """Cilindro como círculo (corte XY). Visível se slice abaixo do topo."""
    if z_slice > H_CYL:
        return
    circle = mpatches.Circle(
        (XC_CYL, YC_CYL), R_CYL,
        linewidth=1.5, edgecolor='black', facecolor='white', zorder=10,
    )
    ax.add_patch(circle)


def scatter_compare(ax_cfd, ax_pinn, h_coord, v_coord, val_cfd, val_pinn,
                      title_var, unit, cmap, plane, slice_value):
    """Plota CFD e PINN com escala de cor compartilhada."""
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

    for ax, sc, label in [(ax_cfd, sc_cfd, "CFD"),
                            (ax_pinn, sc_pinn, "PINN")]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{title_var} {label} [{unit}]", fontsize=10)
        ax.set_aspect('equal' if plane == "xy" else 'auto')
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)


# --- Plots ---
errors_by_slice = []
col_map_cached = None  # reutilizar entre snapshots

for t_target in T_SLICES_S:
    idx_t = np.argmin(np.abs(times_all - t_target))
    actual_t = times_all[idx_t]
    csv_path = snaps[idx_t][1]

    print(f"\n{'='*70}")
    print(f"Tempo: {t_target}s (snapshot mais próximo: t={actual_t:.1f}s)")
    print(f"{'='*70}")

    _d, col_map_cached = load_snapshot(csv_path, col_map=col_map_cached)
    x_cfd = _d['x']; y_cfd = _d['y']; z_cfd = _d['z']
    u_cfd = _d['u']; v_cfd = _d['v']; w_cfd = _d['w']
    P_cfd = _d['P']

    if P_cfd.mean() > 1e4:
        P_cfd = P_cfd - P_OP

    for slice_value in SLICE_VALUES:
        if PLANE == "xz":
            slab = np.abs(y_cfd - slice_value) < SLAB_TOL
            h_coord, v_coord = x_cfd, z_cfd
        else:
            slab = np.abs(z_cfd - slice_value) < SLAB_TOL
            h_coord, v_coord = x_cfd, y_cfd

        # Remoção do interior do cilindro
        inside_cyl = (
            ((x_cfd - XC_CYL)**2 + (y_cfd - YC_CYL)**2 <= R_CYL**2) &
            (z_cfd <= H_CYL)
        )
        valid = slab & ~inside_cyl
        n_valid = int(valid.sum())

        if n_valid < 10:
            print(f"  [skip] {SLICE_LABEL}={slice_value} — apenas {n_valid} pts")
            continue

        # Arrays do slice (fatias)
        h_s = h_coord[valid]
        v_s = v_coord[valid]
        x_s = x_cfd[valid]
        y_s = y_cfd[valid]
        z_s = z_cfd[valid]
        u_s = u_cfd[valid]
        v_vel_s = v_cfd[valid]
        w_s = w_cfd[valid]
        P_s = P_cfd[valid]
        Vmag_cfd_s = np.sqrt(u_s**2 + v_vel_s**2 + w_s**2)

        # PINN
        xn = (x_s / L_REF).astype(np.float32)
        yn = (y_s / L_REF).astype(np.float32)
        zn = (z_s / L_REF).astype(np.float32)
        tn = np.full_like(xn, actual_t / TIME_REF, dtype=np.float32)
        xyzt = tf.constant(np.column_stack([xn, yn, zn, tn]), dtype=tf.float32)
        pred = model(xyzt, training=False).numpy()
        u_p = pred[:, 0] * V_REF
        v_p = pred[:, 1] * V_REF
        w_p = pred[:, 2] * V_REF
        P_p = pred[:, 3] * P_REF
        Vmag_p_s = np.sqrt(u_p**2 + v_p**2 + w_p**2)

        # (|V| e P, CFD vs PINN)
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        scatter_compare(axes[0, 0], axes[0, 1], h_s, v_s,
                          Vmag_cfd_s, Vmag_p_s,
                          "|V|", "m/s", "viridis", PLANE, slice_value)
        scatter_compare(axes[1, 0], axes[1, 1], h_s, v_s,
                          P_s, P_p,
                          "P", "Pa", "coolwarm", PLANE, slice_value)

        plt.suptitle(
            f"CFD vs PINN — {PLANE.upper()} | {SLICE_LABEL}={slice_value}m | "
            f"t={actual_t:.1f}s | n={n_valid}",
            fontsize=11,
        )
        plt.tight_layout()

        # plots_results/slices/
        fname = (
            f"pinn_vs_cfd_{PLANE}_{SLICE_LABEL}{slice_value:04d}_"
            f"t{int(actual_t):03d}s.png"
        )
        fpath = os.path.join(SLICES_DIR, fname)
        plt.savefig(fpath, dpi=120, bbox_inches='tight')
        plt.close(fig)

        # Erros
        err_V = np.abs(Vmag_p_s - Vmag_cfd_s)
        err_P = np.abs(P_p - P_s)

        errors_by_slice.append({
            't': float(actual_t),
            'slice_label': SLICE_LABEL,
            'slice_value': slice_value,
            'n_pts': n_valid,
            'V_mae': float(err_V.mean()),
            'V_max': float(err_V.max()),
            'P_mae': float(err_P.mean()),
            'P_max': float(err_P.max()),
        })

        print(
            f"  {SLICE_LABEL}={slice_value:4d}m | "
            f"|V| MAE={err_V.mean():5.2f} m/s max={err_V.max():6.2f} | "
            f"P MAE={err_P.mean():5.0f} Pa max={err_P.max():6.0f}"
        )

# ============================================================
# Tabela final

print(f"\n{'='*70}")
print("RESUMO GLOBAL DE ERROS")
print(f"{'='*70}")
df_errors = pd.DataFrame(errors_by_slice)
print(df_errors.to_string(index=False, float_format='%.3f'))

print(f"\nMédias agregadas:")
print(f"  |V| MAE médio: {df_errors['V_mae'].mean():.3f} m/s")
print(f"  P MAE médio:   {df_errors['P_mae'].mean():.1f} Pa")

# Salva o resumo
df_errors.to_csv(os.path.join(SUMMARY_DIR, "errors_by_slice.csv"), index=False)
print(f"\nResumo salvo: {os.path.join(SUMMARY_DIR, 'errors_by_slice.csv')}")



# ============================================================
# Análise quantitativa (scatter PINN vs CFD - utilização do último snapshot completo)

# Carregamento do último snapshot (reusa col_map se existir)
last_t = float(times_all[-1])
last_csv_path = snaps[-1][1]
_d, _ = load_snapshot(last_csv_path,
                      col_map=col_map_cached if 'col_map_cached' in dir() else None)

x_cfd = _d['x']; y_cfd = _d['y']; z_cfd = _d['z']
u_cfd = _d['u']; v_cfd = _d['v']; w_cfd = _d['w']
P_cfd = _d['P']

if P_cfd.mean() > 1e4:
    P_cfd = P_cfd - P_OP

Vmag_cfd = np.sqrt(u_cfd**2 + v_cfd**2 + w_cfd**2)

# Avaliação da PINN nos mesmos pontos
n = len(x_cfd)
xyzt = np.column_stack([
    x_cfd / L_REF,
    y_cfd / L_REF,
    z_cfd / L_REF,
    np.full(n, last_t / TIME_REF, dtype=np.float32),
]).astype(np.float32)

pred = model(tf.constant(xyzt), training=False).numpy()
u_p = pred[:, 0] * V_REF
v_p = pred[:, 1] * V_REF
w_p = pred[:, 2] * V_REF
P_p = pred[:, 3] * P_REF
Vmag_p = np.sqrt(u_p**2 + v_p**2 + w_p**2)


# ----------
# Plot 1: Scatter PINN vs CFD (5 variáveis: u, v, w, |V|, P)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

datasets = [
    (u_cfd,    u_p,    'u',   'm/s', axes[0, 0]),
    (v_cfd,    v_p,    'v',   'm/s', axes[0, 1]),
    (w_cfd,    w_p,    'w',   'm/s', axes[0, 2]),
    (Vmag_cfd, Vmag_p, '|V|', 'm/s', axes[1, 0]),
    (P_cfd,    P_p,    'P',   'Pa',  axes[1, 1]),
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
    ax.set_title(
        f'{name}: MAE={mae:.2f} | RMSE={rmse:.2f} | bias={bias:+.2f} | r={corr:.3f}',
        fontsize=9,
    )
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

axes[1, 2].axis('off')

plt.suptitle(f'PINN vs CFD scatter — t={last_t:.1f}s | n={n} pts', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SCATTER_DIR, 'pinn_vs_cfd_scatter.png'),
             dpi=120, bbox_inches='tight')
plt.close()


# ----------
# Plot 2: Histograma de erros

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

error_datasets = [
    (u_p - u_cfd,         'u',   'm/s', axes[0, 0]),
    (v_p - v_cfd,         'v',   'm/s', axes[0, 1]),
    (w_p - w_cfd,         'w',   'm/s', axes[0, 2]),
    (Vmag_p - Vmag_cfd,   '|V|', 'm/s', axes[1, 0]),
    (P_p - P_cfd,         'P',   'Pa',  axes[1, 1]),
]

for errors, name, unit, ax in error_datasets:
    ax.hist(errors, bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='red', ls='--', lw=1.5, label='zero error')
    median = np.median(errors)
    ax.axvline(median, color='green', ls=':', lw=1.5,
                label=f'median={median:.2f}')

    pct25 = np.percentile(errors, 25)
    pct75 = np.percentile(errors, 75)
    iqr = pct75 - pct25

    ax.set_xlabel(f'Erro {name} (PINN - CFD) [{unit}]')
    ax.set_ylabel('Frequência')
    ax.set_title(f'{name}: Q1={pct25:.2f}, Q3={pct75:.2f}, IQR={iqr:.2f}',
                  fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[1, 2].axis('off')

plt.suptitle(f'Distribuição de erros PINN - CFD — t={last_t:.1f}s', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(HIST_DIR, 'pinn_vs_cfd_errors_hist.png'),
             dpi=120, bbox_inches='tight')
plt.close()


# ----------
# Tabela resumo

scales = {'u': V_REF, 'v': V_REF, 'w': V_REF, '|V|': V_REF, 'P': P_REF}

print(f"\n{'='*70}")
print(f"RESUMO PINN vs CFD em t={last_t:.1f}s (n={n} pts)")
print(f"{'='*70}")
print(f"\n{'Var':<8} {'MAE':>10} {'RMSE':>10} {'Bias':>10} {'Corr':>8} {'Rel.MAE':>10}")
print("-" * 70)

summary_rows = []
for cfd_vals, pinn_vals, name, unit, _ in datasets:
    mae = np.mean(np.abs(pinn_vals - cfd_vals))
    rmse = np.sqrt(np.mean((pinn_vals - cfd_vals)**2))
    bias = np.mean(pinn_vals - cfd_vals)
    corr = np.corrcoef(cfd_vals, pinn_vals)[0, 1]
    scale = scales.get(name, 1.0)
    rel_mae = 100 * mae / scale
    print(f"{name:<8} {mae:>9.3f}  {rmse:>9.3f}  {bias:>+9.3f}  {corr:>7.3f}  {rel_mae:>8.1f}%")
    summary_rows.append({
        'var': name, 'unit': unit,
        'MAE': float(mae), 'RMSE': float(rmse),
        'bias': float(bias), 'corr': float(corr),
        'rel_mae_pct': float(rel_mae),
    })

# Salvar o resumo
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(SUMMARY_DIR, "final_metrics_lastsnap.csv"),
                  index=False)
print(f"\nResumo salvo: {os.path.join(SUMMARY_DIR, 'final_metrics_lastsnap.csv')}")