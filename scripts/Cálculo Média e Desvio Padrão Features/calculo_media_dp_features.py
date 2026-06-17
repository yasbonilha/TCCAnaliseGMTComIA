# scripts/compute_normalization_stats.py
import numpy as np
import pandas as pd
import json
import glob
import os
import re

# ================================================================================
#                  FUNÇÕES IGUAL AO CÓDIGO DO CLUSTER
# ================================================================================

SNAPSHOT_DIR      = "/home/tmoraes/CSVs/"
OUTPUT_PATH = "/home/tmoraes/script_results/normalization_stats.json"
SNAPSHOT_EXT      = ".csv"
DT_PHYSICAL       = 0.1
TIMESTEP_IS_INDEX = True

SNAPSHOT_STRIDE   = 1            
SNAPSHOT_START    = 0
SNAPSHOT_MAX      = None

_TIME_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)(?=\D*$)")


NEEDED_COLS_PATTERNS = {
    "x": [r"x.coordinate", r"^x$"],
    "y": [r"y.coordinate", r"^y$"],
    "z": [r"z.coordinate", r"^z$"],
    "P": [r"^pressure$", r"static.pressure", r"^p$"],
    "u": [r"x.velocity", r"^u$"],
    "v": [r"y.velocity", r"^v$"],
    "w": [r"z.velocity", r"^w$"],
}


def parse_timestamp(filename: str,
                    dt_physical: float = DT_PHYSICAL,
                    timestep_is_index: bool = TIMESTEP_IS_INDEX):
    
    # Extrai o tempo físico [s] do nome do arquivo para futura ordenação temporal

    stem = os.path.splitext(os.path.basename(filename))[0]
    matches = _TIME_TOKEN_RE.findall(stem)
    if not matches:
        return None, None
    raw = float(matches[-1])
    t_phys = raw * dt_physical if timestep_is_index else raw

    #T_phys: tempo físico
    #T_raw : número timestep
    return float(t_phys), raw



def discover_snapshots(snapshot_dir: str,
                       ext: str = SNAPSHOT_EXT,
                       dt_physical: float = DT_PHYSICAL,
                       timestep_is_index: bool = TIMESTEP_IS_INDEX,
                       stride: int = 1,
                       start_idx: int = 0,
                       max_snapshots: int = None,
                       verbose: bool = True):

    # Lê o diretório e retorna uma lista do tempo físico e caminho em ordem cronológica

    if not os.path.isdir(snapshot_dir):
        raise NotADirectoryError(f"Snapshot directory not found: {snapshot_dir}")

    pattern = os.path.join(snapshot_dir, f"*{ext}")
    candidates = sorted(glob.glob(pattern))
    if verbose:
        print(f"Scanning '{snapshot_dir}' for *{ext}: "
              f"{len(candidates)} candidate(s)")

    # Separar timesteps e filtrar
    parsed = []
    skipped = []
    for path in candidates:
        t_phys, raw = parse_timestamp(path, dt_physical, timestep_is_index)
        if t_phys is None:
            skipped.append(os.path.basename(path))
            continue
        parsed.append((t_phys, path))

    # Ordenação cronológica
    parsed.sort(key=lambda s: s[0])

    # Aplicando intervalo (se houver)
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
        sep=r'\s+',           # um ou mais whitespace
        engine='python',      # regex sep precisa do engine python
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


snaps = discover_snapshots(SNAPSHOT_DIR, stride=1, start_idx=0, max_snapshots=None)
print(f"Encontrados {len(snaps)} snapshots")

# Acumuladores: para média e variância
# Welford online é mais estável que somar tudo e dividir depois.
# Mas pra simplificar e dado que temos só ~10⁸ pontos no total,
# sum + sum_of_squares funciona bem em float64.
keys = ['x', 'y', 'z', 'u', 'v', 'w', 'P']
n_total    = 0
sum_       = {k: 0.0 for k in keys}
sum_sq     = {k: 0.0 for k in keys}
min_       = {k: +np.inf for k in keys}
max_       = {k: -np.inf for k in keys}

col_map_cached = None
for i, (t_phys, path) in enumerate(snaps):
    if (i + 1) % 100 == 0:
        partial = {
            "n_processed": i + 1,
            "sum": {k: float(sum_[k]) for k in keys},
            "sum_sq": {k: float(sum_sq[k]) for k in keys},
            "n_total": n_total,
        }
        with open("/home/tmoraes/script_results/_partial.json", 'w') as f:
            json.dump(partial, f)

    if i % 50 == 0:
        print(f"  [{i+1}/{len(snaps)}] {os.path.basename(path)}")
    d, col_map_cached = load_snapshot(path, col_map=col_map_cached)
    
    # Converte pressão de absoluta para gauge se necessário
    if d['P'].mean() > 1e4:
        d['P'] = d['P'] - 101325.0
    
    n_local = len(d['x'])
    n_total += n_local
    for k in keys:
        arr = d[k].astype(np.float64)
        sum_[k]   += arr.sum()
        sum_sq[k] += np.square(arr).sum()
        min_[k]    = min(min_[k], float(arr.min()))
        max_[k]    = max(max_[k], float(arr.max()))

# Estatísticas finais
mean = {k: sum_[k] / n_total for k in keys}
var  = {k: sum_sq[k] / n_total - mean[k]**2 for k in keys}
std  = {k: float(np.sqrt(max(var[k], 1e-12))) for k in keys}

# Estatísticas temporais (vêm dos timestamps dos arquivos, não do conteúdo)
times = np.array([t for t, _ in snaps], dtype=np.float64)
t_mean = float(times.mean())
t_std  = float(times.std())

# Escala espacial ISOTRÓPICA: usa a média dos três desvios espaciais.
# (Outra opção é usar o std do eixo dominante, ou max — discuto abaixo.)
sigma_space_iso = float(np.mean([std['x'], std['y'], std['z']]))

stats = {
    "n_total_points": int(n_total),
    "n_snapshots":    len(snaps),
    "spatial": {
        # Médias por eixo (translação OK, não distorce geometria)
        "x_mean": float(mean['x']),
        "y_mean": float(mean['y']),
        "z_mean": float(mean['z']),
        # Std isotrópico (mesmo σ para os três eixos — preserva geometria)
        "sigma_space_iso": sigma_space_iso,
        # Stds anisotrópicos (só pra referência/debug, NÃO usar no treino)
        "x_std_aniso_debug": float(std['x']),
        "y_std_aniso_debug": float(std['y']),
        "z_std_aniso_debug": float(std['z']),
    },
    "temporal": {
        "t_mean": t_mean,
        "t_std":  t_std,
    },
    "outputs": {
        # Saídas: cada uma com sua estatística (não tem problema geométrico)
        "u_mean": float(mean['u']), "u_std": float(std['u']),
        "v_mean": float(mean['v']), "v_std": float(std['v']),
        "w_mean": float(mean['w']), "w_std": float(std['w']),
        "P_mean": float(mean['P']), "P_std": float(std['P']),
    },
    "bounds_physical": {
        "x_min": min_['x'], "x_max": max_['x'],
        "y_min": min_['y'], "y_max": max_['y'],
        "z_min": min_['z'], "z_max": max_['z'],
    },
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\nEstatísticas salvas em {OUTPUT_PATH}")
print(f"Resumo:")
print(f"  Pontos totais:        {n_total:,}")
print(f"  σ isotrópico espacial: {sigma_space_iso:.2f} m")
print(f"  Desvios anisotrópicos: x={std['x']:.1f}, y={std['y']:.1f}, z={std['z']:.1f}")
print(f"  v: mean={mean['v']:+.2f}, std={std['v']:.2f}")
print(f"  P: mean={mean['P']:+.1f}, std={std['P']:.1f}")