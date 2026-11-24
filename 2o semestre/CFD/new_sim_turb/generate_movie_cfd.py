#!/usr/bin/env python3
"""
generate_movie_cfd.py — Vídeos de CFD (Fluent) a partir de CSVs
================================================================

Versão CFD-only, adaptada de generate_movie.py (que comparava PINN vs CFD).

RODADA new_sim_turb (teto de mu_t/mu em 1e7, 211.579 células poliédricas):
  - pasta padrão: new_sim_turb/results/csv (t_00001.csv .. t_01000.csv)
  - lê só os t_*.csv (a pasta também tem o static_mesh.csv)
  - PRESSÃO ESTÁTICA por padrão: a coluna `pressure` do Fluent é a pressão
    MODIFICADA p + (2/3) rho k; aqui ela é convertida para p = pressure -
    (2/3) rho k, a mesma grandeza das figuras do relatório. --pressao
    modificada mantém a coluna como está.
  - corte horizontal em z = 32 m (o mesmo das figuras do relatório)
Toda a parte de rede neural (TensorFlow/PINN) foi removida: aqui só se lê os
CSVs exportados do Fluent, interpola-se para grades 2D e monta-se o vídeo.

Gera vídeos em dois cortes:
  - Vista superior: corte XY em z=32 m (meio do cilindro)
  - Vista lateral:  corte YZ em x=140 m (centro do cilindro)

Cada frame é um layout 1x2:
  | |V| [m/s] | P [Pa] |

INTERPOLAÇÃO (--interp):
  volume  (padrão) : triangula todos os centroides uma vez e avalia nos pontos
                     do plano de corte. Nada é descartado.
  2d               : recorta um slab e o projeta no plano. Na malha antiga
                     usava só 4% dos pontos e deixava o campo poligonal.
  3d               : triangula só o slab; quase coplanar, degenerado.

ESCALAS DE COR (--scale):
  global  (padrão) : |V| de 0 ao percentil 99,95 e p entre os percentis
                     0,05 e 99,95 de TODOS os frames desta simulação.
                     Constante ao longo do vídeo. (Com 1/99, como antes, a
                     estagnação, os ombros e a esteira saturavam.)
  cfd              : escala fixa V in [14,20] m/s, P in [-50,50] Pa —
                     mesma do vídeo CFD antigo, para comparação direta.
                     ATENÇÃO: nesta rodada |V| vai de ~2 a 28 m/s e P de
                     ~-290 a 190 Pa; essa escala satura. Use só para comparar.
  both             : gera os dois conjuntos (4 vídeos), reaproveitando o
                     mesmo cache de interpolação.

USO:
    # 0) Ver as colunas e os arquivos detectados, sem processar nada:
    python generate_movie_cfd.py --inspect

    # 1) Teste rápido com 5 frames:
    python generate_movie_cfd.py --test-mode

    # 2) Execução completa:
    python generate_movie_cfd.py

    # 3) Só REPLOTAR (ajuste estético) usando o cache de interpolação:
    python generate_movie_cfd.py --skip-cache

    # 4) Só recompilar os vídeos a partir dos PNGs já gerados:
    python generate_movie_cfd.py --skip-cache --skip-frames

PIPELINE:
    Fase 1: Interpolação dos CSVs para as grades 2D -> cache .npy (lento, 1x)
    Fase 2: Cálculo das escalas globais (percentis 1/99)
    Fase 3: Geração dos PNGs a partir do cache (replotável em minutos)
    Fase 4: Compilação dos vídeos com ffmpeg
"""

import argparse
import json
import re
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

from scipy.interpolate import LinearNDInterpolator


# ============================================================================
# CONFIGURAÇÃO — AJUSTE SE NECESSÁRIO
# ============================================================================
# Diretórios (podem ser sobrescritos por --csv-dir / --out-dir)
CSV_DIR   = Path("/home/ybonilha/simulations_CFD/new_sim_turb/results/csv")
OUT_DIR   = Path("/home/ybonilha/simulations_CFD/new_sim_turb/results/movie")

# Subpastas de trabalho (criadas dentro de OUT_DIR)
WORK_SUBDIR   = "_work"
CACHE_SUBNAME = "cfd_interp_cache"
FRAMES_SUBNAME = "frames"

# Preenchidos em main() a partir de OUT_DIR
WORK_DIR = None
CACHE_DIR = None
FRAMES_DIR = None
VIDEO_DIR = None

# Domínio físico
X_MIN, X_MAX = 0.0, 280.0
Y_MIN, Y_MAX = 0.0, 700.0
Z_MIN, Z_MAX = 0.0, 190.0

# Cilindro
XC, YC = 140.0, 200.0
R_CYL  = 27.0    # raio físico = D/2 = 54/2
H_CYL  = 65.0    # altura

# Cortes
Z_SLICE = 32.0   # vista superior (XY) -- mesmo corte das figuras do relatório
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

# Pressão: 'estatica' (padrão) converte a coluna do Fluent, que é a pressão
# MODIFICADA p + (2/3) rho k, para a estática. 'modificada' usa a coluna crua.
RHO = 1.225
PRESSAO = 'estatica'

# Escala fixa "CFD" (mesma do vídeo antigo)
V_CFD_RANGE = (14.0, 20.0)     # m/s
P_CFD_RANGE = (-50.0, 50.0)    # Pa

# Percentis usados na escala global
# 0,05/99,95: perto dos extremos reais. Com 1/99 (versão antiga) a escala
# cortava a estagnação, os ombros e o núcleo da esteira, que ocupam pouca área.
PCT_LO, PCT_HI = 0.05, 99.95

# Mapas de cor (--cmap). 'relatorio' (padrão): |V| sequencial (Blues, 0 = branco)
# e pressão divergente (RdBu_r) com o ZERO no centro, como nas figuras do
# relatório. 'jet': o mapa do vídeo antigo, para comparação lado a lado.
CMAP = 'relatorio'

# Subamostragem usada só no cálculo dos percentis (economiza RAM)
SCALE_STRIDE = 4

# Modo de interpolação. Sobrescrito por --interp. Ver build_interpolators().
#   'volume' (PADRÃO) : triangula a nuvem INTEIRA de 39.060 centroides, uma vez
#                       só, e avalia nos pontos 3D do plano de corte. Nenhum
#                       slab, nenhuma projeção, nenhum dado descartado.
#   '2d'              : projeta o slab no plano e triangula em 2D. É o que
#                       produzia o aspecto poligonal: usa 1694 dos 39.060
#                       pontos no corte XY (4,3%) e achata camadas empilhadas.
#   '3d'              : triangula só o slab em 3D. Quase coplanar, degenerado.
INTERP_MODE = 'volume'

# Filtro de spikes (--despike). 0 = desligado (padrão: dados crus).
# Se >0, células cujo |V| difere mais que este valor (m/s) da mediana dos 12
# vizinhos mais próximos são substituídas por essa mediana (em V e P) ANTES da
# interpolação. É uma alteração dos dados para fins de visualização — quando
# ligado, o valor aparece escrito no título do frame.
DESPIKE = 0.0
DESPIKE_K = 12   # nº de vizinhos usados na mediana


# ============================================================================
# LEITURA DOS CSVs DO FLUENT
# ============================================================================
def read_fluent_csv(path):
    """
    Lê um CSV exportado do Fluent, detectando o separador.
    O Fluent exporta ora separado por vírgula ("nodenumber, x-coordinate, ..."),
    ora por espaços, dependendo da opção de export usada.
    """
    with open(path, 'r', errors='replace') as f:
        header = f.readline()
    if header.count(',') >= 2:
        # leitor C do pandas: muito mais rápido que o 'python' em 211 mil linhas
        df = pd.read_csv(path, sep=',', skipinitialspace=True)
    else:
        df = pd.read_csv(path, sep=r'\s+', engine='python')
    df.columns = [str(c).strip().strip('"').strip() for c in df.columns]
    return df


def find_col(df, exact, contains=(), exclude=()):
    """
    Localiza uma coluna: primeiro por nome exato (case-insensitive),
    depois por substring, ignorando nomes que contenham algo de `exclude`.
    """
    lower = {c: c.lower().strip() for c in df.columns}
    for name in exact:
        for c, lc in lower.items():
            if lc == name:
                return c
    for frag in contains:
        for c, lc in lower.items():
            if frag in lc and not any(e in lc for e in exclude):
                return c
    return None


def detect_columns(df):
    """Devolve dict com os nomes das colunas relevantes (ou None)."""
    return {
        'x': find_col(df, ['x-coordinate', 'x coordinate', 'x_coord', 'x'],
                      ['x-coordinate', 'x-coord', 'x_coord']),
        'y': find_col(df, ['y-coordinate', 'y coordinate', 'y_coord', 'y'],
                      ['y-coordinate', 'y-coord', 'y_coord']),
        'z': find_col(df, ['z-coordinate', 'z coordinate', 'z_coord', 'z'],
                      ['z-coordinate', 'z-coord', 'z_coord']),
        'u': find_col(df, ['x-velocity', 'x velocity', 'u'],
                      ['x-velocity', 'x-vel']),
        'v': find_col(df, ['y-velocity', 'y velocity', 'v'],
                      ['y-velocity', 'y-vel']),
        'w': find_col(df, ['z-velocity', 'z velocity', 'w'],
                      ['z-velocity', 'z-vel']),
        'vmag': find_col(df, ['velocity-magnitude', 'velocity magnitude'],
                         ['velocity-magnitude', 'velocity magnitude']),
        'k': find_col(df, ['turb-kinetic-energy'], ['turb-kinetic']),
        'P': find_col(df, ['pressure', 'static-pressure', 'static pressure', 'p'],
                      ['pressure'],
                      exclude=('total', 'dynamic', 'coefficient', 'coeff')),
    }


def extract_fields(df):
    """
    Extrai x, y, z, |V| e P de um DataFrame do Fluent.
    Aceita tanto 'velocity-magnitude' direto quanto as componentes.
    """
    c = detect_columns(df)

    missing = [k for k in ('x', 'y', 'z', 'P') if c[k] is None]
    if missing:
        raise ValueError(
            f"Colunas não encontradas: {missing}. Colunas do arquivo: {list(df.columns)}")

    x = df[c['x']].values.astype(np.float64)
    y = df[c['y']].values.astype(np.float64)
    z = df[c['z']].values.astype(np.float64)
    P = df[c['P']].values.astype(np.float64)
    if PRESSAO == 'estatica':
        if c['k'] is None:
            raise ValueError("--pressao estatica precisa da coluna turb-kinetic-energy "
                             "(ou use --pressao modificada)")
        P = P - (2.0 / 3.0) * RHO * df[c['k']].values.astype(np.float64)

    if c['vmag'] is not None:
        V_mag = df[c['vmag']].values.astype(np.float64)
    else:
        if c['u'] is None or c['v'] is None:
            raise ValueError(
                "Sem 'velocity-magnitude' e sem as componentes x/y de velocidade. "
                f"Colunas do arquivo: {list(df.columns)}")
        u = df[c['u']].values.astype(np.float64)
        v = df[c['v']].values.astype(np.float64)
        w = (df[c['w']].values.astype(np.float64) if c['w'] is not None
             else np.zeros_like(u))
        V_mag = np.sqrt(u * u + v * v + w * w)

    return x, y, z, V_mag, P, c


def build_csv_index():
    """
    Mapeia índice de timestep -> caminho do CSV.

    Não assume o padrão 'timestep-0001.csv': pega o último número inteiro do
    nome do arquivo. Funciona com timestep-0350.csv, sol-350.csv, t_0350.csv...
    """
    # só os snapshots: a pasta da rodada nova também tem o static_mesh.csv
    files = sorted(CSV_DIR.glob("t_*.csv")) or sorted(CSV_DIR.glob("*.csv"))
    index = {}
    for f in files:
        nums = re.findall(r'\d+', f.stem)
        if not nums:
            continue
        index[int(nums[-1])] = f
    return index


# ============================================================================
# GRADES 2D REGULARES
# ============================================================================
def create_grids():
    """Cria grades 2D fixas e máscaras do cilindro. Retorna dict com tudo."""
    # Grade XY em z=Z_SLICE
    x_xy_1d = np.arange(X_MIN, X_MAX + DX, DX, dtype=np.float64)
    y_xy_1d = np.arange(Y_MIN, Y_MAX + DY, DY, dtype=np.float64)
    X_xy, Y_xy = np.meshgrid(x_xy_1d, y_xy_1d, indexing='xy')
    Z_xy = np.full_like(X_xy, Z_SLICE)
    mask_xy_cyl = ((X_xy - XC) ** 2 + (Y_xy - YC) ** 2 <= R_CYL ** 2)

    # Grade YZ em x=X_SLICE
    y_yz_1d = np.arange(Y_MIN, Y_MAX + DY, DY, dtype=np.float64)
    z_yz_1d = np.arange(Z_MIN, Z_MAX + DZ, DZ, dtype=np.float64)
    Y_yz, Z_yz = np.meshgrid(y_yz_1d, z_yz_1d, indexing='xy')
    X_yz = np.full_like(Y_yz, X_SLICE)
    # No corte YZ em x=140, o cilindro é um retângulo y in [173,227], z in [0,65]
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
            'extent': [Y_MIN, Y_MAX, X_MAX, X_MIN],
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
def despike_cells(pts3, V, P, thresh=None, k=None):
    """
    Substitui células isoladas com valor discrepante pela mediana dos vizinhos.

    Detecta pelo campo de velocidade: se |V_i - mediana(V dos k vizinhos)| >
    thresh, a célula é considerada spike e tanto V quanto P recebem a mediana
    dos vizinhos. Serve para malhas onde pontos isolados com valores espúrios
    dominam visualmente o campo interpolado.

    ATENÇÃO: isto altera os dados. Use só para visualização e deixe claro.
    """
    from scipy.spatial import cKDTree
    thresh = DESPIKE if thresh is None else thresh
    k = DESPIKE_K if k is None else k
    if thresh <= 0 or len(V) < k + 1:
        return V, P, 0

    tree = cKDTree(pts3)
    _, idx = tree.query(pts3, k=k + 1)
    nb = idx[:, 1:]
    V_med = np.median(V[nb], axis=1)
    bad = np.abs(V - V_med) > thresh
    if not bad.any():
        return V, P, 0
    V_out = V.copy()
    P_out = P.copy()
    V_out[bad] = V_med[bad]
    P_out[bad] = np.median(P[nb], axis=1)[bad]
    return V_out, P_out, int(bad.sum())


def _average_duplicates(pts2, *value_arrays, decimals=6):
    """Colapsa pontos 2D coincidentes, tirando a média dos valores."""
    key = np.round(pts2, decimals)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    uniq = np.zeros((len(counts), 2))
    np.add.at(uniq, inv, pts2)
    uniq /= counts[:, None]
    out = []
    for vals in value_arrays:
        acc = np.zeros(len(counts))
        np.add.at(acc, inv, vals)
        out.append(acc / counts)
    return uniq, out


# ---------------------------------------------------------------------------
# Triangulação do VOLUME, reaproveitada entre timesteps.
#
# A malha e ESTATICA: os centroides sao os mesmos em todos os arquivos.
# Entao a triangulacao de Delaunay so precisa ser construida UMA vez para a
# rodada inteira, e cada timestep so troca os valores nos vertices.
# LinearNDInterpolator aceita um objeto Delaunay pronto no lugar dos pontos --
# e o que torna isso barato: ~3 s uma vez, em vez de por frame e por corte.
# ---------------------------------------------------------------------------
_TRI_CACHE = {}


def get_volume_tri(pts3):
    """Delaunay 3D da nuvem inteira, construida uma vez e reusada."""
    from scipy.spatial import Delaunay
    key = (len(pts3), float(pts3[0, 0]), float(pts3[0, 1]),
           float(pts3[-1, 1]), float(pts3[-1, 2]))
    tri = _TRI_CACHE.get(key)
    if tri is None:
        t0 = time.time()
        tri = Delaunay(pts3)
        _TRI_CACHE[key] = tri
        print(f"  [INFO] Delaunay do volume: {len(pts3):,} pontos, "
              f"{len(tri.simplices):,} tetraedros, {time.time() - t0:.1f}s "
              f"(construida uma vez, reusada em todos os frames e cortes)")
    return tri


def build_volume_interpolators(pts3, V, P):
    """Interpoladores sobre a nuvem inteira, sem recortar slab nenhum.

    POR QUE ESTE E O MODO CERTO
        (Medidas da malha ANTIGA, 39.060 celulas -- a nova tem 211.579 celulas
        poliedricas e o mesmo argumento vale.) Fatiar-e-achatar descarta 96% das
        celulas: no corte XY, |z-32.5|<5 deixava 1694 dos 39.060 pontos. Sobre 1694 pontos espalhados de forma MUITO
        irregular -- 0,70 m colado no cilindro, 11,34 m no anel de transicao a
        r 50-100 m -- a interpolacao linear e linear POR TRIANGULO, e as arestas
        desses triangulos sao as quinas que aparecem no video.

        Pior: onde a malha e fina, a fatia de 10 m contem varias camadas de
        celulas (z de 27,5 a 37,1 em r 27-50). Projetar joga todas no mesmo
        plano e 120 dos 344 pontos ficam a menos de 0,3 m de outro -- lascas de
        Delaunay bem onde se quer detalhe. No corte YZ, com TOL_X=10, a fatia
        tem 20 m e sao 462 pontos quase coincidentes.

        Triangulando o volume, cada ponto do plano cai dentro de um tetraedro de
        celulas de verdade e a interpolacao e baricentrica em 3D -- que e o que
        um pos-processador de CFD faz. Nenhuma celula e descartada e nao ha
        projecao.

    RESSALVA HONESTA
        Isto nao inventa resolucao. A malha tem ~9,8 m de celula no campo
        distante, entao uma textura na escala da malha permanece -- ela e real,
        e o limite do que a simulacao resolve. O que some sao as facetas grandes
        e as lascas, que eram artefato do recorte.

        O buraco do cilindro fica DENTRO do fecho convexo, entao existem
        tetraedros atravessando-o. Os pontos ali sao mascarados por mask_cyl.
        Como a malha e refinada em volta (faces de 1 m e 3 camadas prismaticas
        na malha nova), os tetraedros vizinhos sao
        pequenos e nao atravessam os 53 m do buraco -- mas se algum dia a malha
        deixar de ser refinada ali, isso precisa ser reconferido.
    """
    tri = get_volume_tri(pts3)
    return (LinearNDInterpolator(tri, V, fill_value=np.nan),
            LinearNDInterpolator(tri, P, fill_value=np.nan),
            None)


def build_interpolators(pts3, V, P, plane, ts_idx=None, mode=None):
    """
    Constrói os interpoladores para um corte.

    mode='2d' (PADRÃO): projeta os pontos do slab no plano de corte e faz
        Delaunay 2D. É o correto para malhas em que o slab tem ~uma camada de
        células: a nuvem 3D fica quase coplanar e o Qhull, mesmo sem dar erro,
        produz tetraedros-lasca cujo resultado é ruído triangular no lugar do
        campo. Pontos que se projetam no mesmo lugar são mediados.

    mode='3d': triangulação 3D do slab (comportamento do script original).
        Só faz sentido se houver várias camadas de células dentro da
        tolerância. Se o Qhull falhar, cai automaticamente para 2D.

    Retorna (interp_V, interp_P, cols) — cols=None indica interpolação 3D.
    """
    mode = (mode or INTERP_MODE).lower()
    cols = (0, 1) if plane == 'xy' else (1, 2)

    if mode == '3d':
        try:
            return (LinearNDInterpolator(pts3, V, fill_value=np.nan),
                    LinearNDInterpolator(pts3, P, fill_value=np.nan),
                    None)
        except Exception:
            tag = f"ts={ts_idx} " if ts_idx is not None else ""
            print(f"  [INFO] {tag}corte {plane}: triangulação 3D degenerada, "
                  f"caindo para interpolação 2D no plano")

    pts2, (V2, P2) = _average_duplicates(pts3[:, cols], V, P)
    return (LinearNDInterpolator(pts2, V2, fill_value=np.nan),
            LinearNDInterpolator(pts2, P2, fill_value=np.nan),
            cols)


def cache_paths(ts_idx):
    """Retorna paths dos 4 arquivos .npy do cache de um timestep."""
    xy_dir = CACHE_DIR / f"xy_z{Z_SLICE:g}"
    yz_dir = CACHE_DIR / "yz_x140"
    return {
        'xy_V': xy_dir / f"V_t{ts_idx:04d}.npy",
        'xy_P': xy_dir / f"P_t{ts_idx:04d}.npy",
        'yz_V': yz_dir / f"V_t{ts_idx:04d}.npy",
        'yz_P': yz_dir / f"P_t{ts_idx:04d}.npy",
    }


def cache_exists(ts_idx):
    """Verifica se TODOS os 4 .npy do cache existem."""
    return all(p.exists() for p in cache_paths(ts_idx).values())


def load_from_cache(ts_idx, grids):
    """Carrega .npy do cache. Reaplica máscara do cilindro (NaN no interior)."""
    paths = cache_paths(ts_idx)
    V_xy = np.load(paths['xy_V'])
    P_xy = np.load(paths['xy_P'])
    V_yz = np.load(paths['yz_V'])
    P_yz = np.load(paths['yz_P'])
    V_xy[grids['xy']['mask_cyl']] = np.nan
    P_xy[grids['xy']['mask_cyl']] = np.nan
    V_yz[grids['yz']['mask_cyl']] = np.nan
    P_yz[grids['yz']['mask_cyl']] = np.nan
    return {'V_xy': V_xy, 'P_xy': P_xy, 'V_yz': V_yz, 'P_yz': P_yz}


def interpolate_snapshot(ts_idx, csv_path, grids):
    """
    Interpola UM snapshot do CFD para as duas grades 2D e salva em cache.

    Para acelerar, filtra pontos próximos ao plano de corte antes de interpolar.
    """
    df = read_fluent_csv(csv_path)
    x, y, z, V_mag, P, _ = extract_fields(df)

    # ---- Modo 'volume': sem slab, sem projecao, sem descartar celula -------
    # Os dois cortes usam a MESMA triangulacao (a malha e estatica), entao o
    # custo e uma Delaunay para a rodada inteira em vez de duas por frame.
    if (INTERP_MODE or '').lower() == 'volume':
        pts_all = np.column_stack([x, y, z])
        V_s, P_s, _ = despike_cells(pts_all, V_mag, P)
        iV, iP, _ = build_volume_interpolators(pts_all, V_s, P_s)

        out = {}
        for plane, shape_key in (('xy', 'X'), ('yz', 'Y')):
            g = grids[plane]
            q = np.column_stack([g['X'].ravel(), g['Y'].ravel(), g['Z'].ravel()])
            Vg = iV(q).reshape(g[shape_key].shape)
            Pg = iP(q).reshape(g[shape_key].shape)
            Vg[g['mask_cyl']] = np.nan
            Pg[g['mask_cyl']] = np.nan
            out['V_' + plane] = Vg
            out['P_' + plane] = Pg

        paths = cache_paths(ts_idx)
        np.save(paths['xy_V'], out['V_xy'].astype(np.float32))
        np.save(paths['xy_P'], out['P_xy'].astype(np.float32))
        np.save(paths['yz_V'], out['V_yz'].astype(np.float32))
        np.save(paths['yz_P'], out['P_yz'].astype(np.float32))
        return out

    # ---- Corte XY (z=Z_SLICE) ----
    mask_z = np.abs(z - Z_SLICE) < TOL_Z
    if mask_z.sum() < 4:
        raise ValueError(f"Poucos pontos perto de z={Z_SLICE} (TOL_Z={TOL_Z}): "
                         f"{mask_z.sum()}")
    pts_xy = np.column_stack([x[mask_z], y[mask_z], z[mask_z]])
    V_s, P_s, _ = despike_cells(pts_xy, V_mag[mask_z], P[mask_z])

    interp_V, interp_P, cols = build_interpolators(
        pts_xy, V_s, P_s, 'xy', ts_idx)

    grid_xy = grids['xy']
    query_xy = np.column_stack([
        grid_xy['X'].flatten(),
        grid_xy['Y'].flatten(),
        grid_xy['Z'].flatten()
    ])
    q = query_xy if cols is None else query_xy[:, cols]
    V_xy = interp_V(q).reshape(grid_xy['X'].shape)
    P_xy = interp_P(q).reshape(grid_xy['X'].shape)
    V_xy[grid_xy['mask_cyl']] = np.nan
    P_xy[grid_xy['mask_cyl']] = np.nan

    # ---- Corte YZ (x=X_SLICE) ----
    mask_x = np.abs(x - X_SLICE) < TOL_X
    if mask_x.sum() < 4:
        raise ValueError(f"Poucos pontos perto de x={X_SLICE} (TOL_X={TOL_X}): "
                         f"{mask_x.sum()}")
    pts_yz = np.column_stack([x[mask_x], y[mask_x], z[mask_x]])
    V_s, P_s, _ = despike_cells(pts_yz, V_mag[mask_x], P[mask_x])
    interp_V, interp_P, cols = build_interpolators(
        pts_yz, V_s, P_s, 'yz', ts_idx)

    grid_yz = grids['yz']
    query_yz = np.column_stack([
        grid_yz['X'].flatten(),
        grid_yz['Y'].flatten(),
        grid_yz['Z'].flatten()
    ])
    q = query_yz if cols is None else query_yz[:, cols]
    V_yz = interp_V(q).reshape(grid_yz['Y'].shape)
    P_yz = interp_P(q).reshape(grid_yz['Y'].shape)
    V_yz[grid_yz['mask_cyl']] = np.nan
    P_yz[grid_yz['mask_cyl']] = np.nan

    # Salva como float32 (metade do tamanho, precisão suficiente)
    paths = cache_paths(ts_idx)
    np.save(paths['xy_V'], V_xy.astype(np.float32))
    np.save(paths['xy_P'], P_xy.astype(np.float32))
    np.save(paths['yz_V'], V_yz.astype(np.float32))
    np.save(paths['yz_P'], P_yz.astype(np.float32))

    return {'V_xy': V_xy, 'P_xy': P_xy, 'V_yz': V_yz, 'P_yz': P_yz}


def run_phase1_cache(timesteps, csv_index, grids, force=False):
    """Fase 1: pré-computa interpolação do CFD para todos os timesteps."""
    (CACHE_DIR / f"xy_z{Z_SLICE:g}").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "yz_x140").mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print("FASE 1: Interpolando CFD para grades 2D")
    print(f"{'=' * 72}")
    print(f"  CSVs em:  {CSV_DIR}")
    print(f"  Cache em: {CACHE_DIR}")
    print(f"  Timesteps: {len(timesteps)}")

    t0 = time.time()
    n_err = 0
    for i, ts_idx in enumerate(timesteps):
        if cache_exists(ts_idx) and not force:
            continue
        try:
            interpolate_snapshot(ts_idx, csv_index[ts_idx], grids)
        except Exception as e:
            n_err += 1
            print(f"  [ERRO] ts={ts_idx} ({csv_index[ts_idx].name}): {e}")
            continue
        if (i + 1) % 10 == 0 or i == len(timesteps) - 1:
            elapsed = time.time() - t0
            eta = elapsed * (len(timesteps) - i - 1) / (i + 1)
            print(f"  [{i + 1:>3}/{len(timesteps)}] ts={ts_idx} "
                  f"elapsed={elapsed / 60:.1f}min eta={eta / 60:.1f}min")

    print(f"  Fase 1 concluída em {(time.time() - t0) / 60:.1f} min"
          + (f" ({n_err} erro(s))" if n_err else ""))


# ============================================================================
# FASE 2 — ESCALAS GLOBAIS
# ============================================================================
def run_phase2_scales(timesteps, grids):
    """Calcula percentis 1/99 dos campos para cada corte."""
    print(f"\n{'=' * 72}")
    print(f"FASE 2: Calculando escalas globais (percentis {PCT_LO}/{PCT_HI})")
    print(f"{'=' * 72}")

    buckets = {'V_xy': [], 'P_xy': [], 'V_yz': [], 'P_yz': []}
    n_used = 0

    for ts_idx in timesteps:
        if not cache_exists(ts_idx):
            continue
        data = load_from_cache(ts_idx, grids)
        for key in buckets:
            # subamostra só para o cálculo dos percentis (economiza RAM)
            buckets[key].append(data[key].flatten()[::SCALE_STRIDE])
        n_used += 1

    if n_used == 0:
        raise RuntimeError("Nenhum timestep em cache — rode a Fase 1 antes "
                           "(sem --skip-cache).")

    def pct(arr_list):
        arr = np.concatenate(arr_list)
        return (float(np.nanpercentile(arr, PCT_LO)),
                float(np.nanpercentile(arr, PCT_HI)))

    def vel(arr_list):
        # |V| começa sempre em 0: a estagnação e o núcleo da esteira ficam visíveis
        return (0.0, pct(arr_list)[1])

    scales = {
        'V_xy_global': vel(buckets['V_xy']),
        'P_xy_global': pct(buckets['P_xy']),
        'V_yz_global': vel(buckets['V_yz']),
        'P_yz_global': pct(buckets['P_yz']),
        'V_cfd_range': list(V_CFD_RANGE),
        'P_cfd_range': list(P_CFD_RANGE),
    }

    print(f"  Frames usados: {n_used}")
    for k, v in scales.items():
        print(f"  {k}: [{v[0]:.2f}, {v[1]:.2f}]")

    out_path = WORK_DIR / "scales.json"
    with open(out_path, 'w') as f:
        json.dump(scales, f, indent=2)
    print(f"  Salvo em: {out_path}")

    return scales


# ============================================================================
# FASE 3 — GERAÇÃO DOS FRAMES
# ============================================================================
def get_cylinder_patch(view):
    """Cria patch matplotlib para mostrar contorno do cilindro no plot."""
    if view == 'xy':
        # Vista DEITADA: horizontal=Y, vertical=X. Centro (YC, XC)=(200,140).
        return Circle((YC, XC), R_CYL, fill=False, edgecolor='black',
                      linewidth=1.5, linestyle='--')
    else:  # yz
        # Retângulo: y in [173, 227], z in [0, 65]
        return Rectangle((YC - R_CYL, 0), 2 * R_CYL, H_CYL,
                         fill=False, edgecolor='black',
                         linewidth=1.5, linestyle='--')


def plot_frame(t_phys, idx, cfd_data, grids, scales, view, scale_type, frames_dir):
    """
    Gera UM frame PNG para uma combinação (view, scale_type).

    Layout 1x2:  | |V| [m/s] | P [Pa] |
    Vista XY exibida deitada (horizontal=y, vertical=x p/ baixo) para casar
    com a orientação da vista YZ no slide.
    """
    grid = grids[view]

    V = cfd_data[f'V_{view}']
    P = cfd_data[f'P_{view}']

    # --- Orientação de exibição ---
    do_T = grid.get('transpose', False)
    origin = grid.get('origin', 'lower')
    if do_T:
        V, P = V.T, P.T

    # Define escala
    if scale_type == 'cfd':
        v_range = V_CFD_RANGE
        p_range = P_CFD_RANGE
    else:
        v_range = scales[f'V_{view}_global']
        p_range = scales[f'P_{view}_global']

    # --- Layout em polegadas: painéis com proporção física 1:1 e barra de cor
    #     com EXATAMENTE a altura do painel (a versão antiga usava
    #     constrained_layout, e a barra ocupava a altura inteira da figura).
    extent = grid['extent']
    phys_w = abs(extent[1] - extent[0])
    phys_h = abs(extent[3] - extent[2])
    PANEL_W = 7.0                                   # largura de cada painel (pol.)
    panel_h = PANEL_W * phys_h / phys_w
    L, GAP, CB_W, CB_PAD, R, B, T = 0.75, 1.35, 0.16, 0.12, 0.95, 0.65, 0.75
    fig_w = L + PANEL_W + GAP + PANEL_W + CB_PAD + CB_W + R
    fig_h = B + panel_h + T
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    axes, caxes = [], []
    for col in range(2):
        x0 = L + col * (PANEL_W + GAP)
        axes.append(fig.add_axes([x0 / fig_w, B / fig_h, PANEL_W / fig_w, panel_h / fig_h]))
        caxes.append(fig.add_axes([(x0 + PANEL_W + CB_PAD) / fig_w, B / fig_h,
                                   CB_W / fig_w, panel_h / fig_h]))

    from matplotlib.colors import Normalize, TwoSlopeNorm
    if CMAP == 'jet':
        cm_v = plt.cm.jet.copy(); cm_p = plt.cm.jet.copy()
        norm_v = Normalize(*v_range); norm_p = Normalize(*p_range)
    else:
        cm_v = plt.cm.Blues.copy(); cm_p = plt.cm.RdBu_r.copy()
        norm_v = Normalize(*v_range)
        # ASSIMÉTRICA, com o zero no centro (branco): cada lado usa a sua faixa
        # inteira de cor, então nem a estagnação (+) nem a sucção (-) ficam
        # apagadas. Custo: a mesma intensidade de cor vale módulos diferentes de
        # cada lado -- a barra mostra isso (meias-barras com espaçamentos
        # diferentes). Piso de 15% do outro lado: sem ele, um lado de poucos Pa
        # (ex.: -11 Pa no corte vertical) amplificaria ruído até a cor mais forte.
        lo, hi = p_range
        lo, hi = min(lo, -0.01), max(hi, 0.01)
        lo = min(lo, -0.15 * hi)
        hi = max(hi, -0.15 * lo)
        norm_p = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    for cm in (cm_v, cm_p):
        cm.set_bad('#9a9a9a', alpha=1.0)

    p_label = 'p estática [Pa]' if PRESSAO == 'estatica' else 'p modificada (p + ⅔ρk) [Pa]'
    panels = [(V, norm_v, cm_v, '|V| [m/s]'), (P, norm_p, cm_p, p_label)]

    for col, (data, norm, cm, label) in enumerate(panels):
        ax = axes[col]
        im = ax.imshow(data, extent=extent, origin=origin, norm=norm,
                       cmap=cm, interpolation='bilinear', aspect='equal')
        ax.set_title(label, fontsize=11)
        ax.add_patch(get_cylinder_patch(view))
        ax.set_xlabel(grid['xlabel'])
        ax.set_ylabel(grid['ylabel'])
        # setas nas pontas da barra quando há valores fora da escala
        finite = data[np.isfinite(data)]
        ext = 'neither'
        if finite.size:
            below, above = finite.min() < norm.vmin, finite.max() > norm.vmax
            ext = {(True, True): 'both', (True, False): 'min',
                   (False, True): 'max'}.get((below, above), 'neither')
        fig.colorbar(im, cax=caxes[col], extend=ext)

    view_name = f"Vista Superior (z={Z_SLICE:g} m)" if view == 'xy' else f"Vista Lateral (x={X_SLICE:g} m)"
    scale_name = "escala CFD" if scale_type == 'cfd' else "escala global"
    title = f'CFD  |  {view_name}  |  {scale_name}  |  t = {t_phys:.1f} s'
    if DESPIKE > 0:
        # dado alterado: tem que estar visível no próprio frame
        title += f'  |  despike {DESPIKE:g} m/s'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1 - 0.18 / fig_h)

    # SEM bbox_inches='tight' (causava dimensões ímpares no libx264).
    out_path = frames_dir / f'frame_{idx:04d}.png'
    fig.savefig(out_path, pil_kwargs={'optimize': True})
    plt.close(fig)


def run_phase3_frames(timesteps, grids, scales, scale_types):
    """Fase 3: gera todos os PNGs lendo os campos do cache."""
    print(f"\n{'=' * 72}")
    print("FASE 3: Geração de frames")
    print(f"{'=' * 72}")

    dirs = {}
    for view in ['xy', 'yz']:
        for st in scale_types:
            d = FRAMES_DIR / f'{view}_{st}_scale'
            d.mkdir(parents=True, exist_ok=True)
            # limpa PNGs antigos para não misturar numeração entre execuções
            for old in d.glob('frame_*.png'):
                old.unlink()
            dirs[(view, st)] = d

    t0 = time.time()
    n_written = 0   # numeração contígua: ffmpeg exige sequência sem buracos
    for i, ts_idx in enumerate(timesteps):
        t_phys = ts_idx * DT_CFD

        if not cache_exists(ts_idx):
            print(f"  [SKIP] ts={ts_idx}: cache ausente")
            continue

        cfd_data = load_from_cache(ts_idx, grids)

        for view in ['xy', 'yz']:
            for st in scale_types:
                plot_frame(t_phys, n_written, cfd_data, grids, scales,
                           view, st, dirs[(view, st)])
        n_written += 1

        if (i + 1) % 10 == 0 or i == len(timesteps) - 1:
            elapsed = time.time() - t0
            eta = elapsed * (len(timesteps) - i - 1) / (i + 1)
            print(f"  [{i + 1:>3}/{len(timesteps)}] ts={ts_idx} "
                  f"t={t_phys:.1f}s  elapsed={elapsed / 60:.1f}min "
                  f"eta={eta / 60:.1f}min")

    print(f"  Fase 3 concluída em {(time.time() - t0) / 60:.1f} min "
          f"({n_written} frames por vídeo)")


# ============================================================================
# FASE 4 — COMPILAÇÃO DOS VÍDEOS
# ============================================================================
def run_phase4_videos(scale_types):
    """Fase 4: chama ffmpeg para compilar os vídeos."""
    print(f"\n{'=' * 72}")
    print("FASE 4: Compilando vídeos com ffmpeg")
    print(f"{'=' * 72}")

    for view in ['xy', 'yz']:
        for st in scale_types:
            frames_dir = FRAMES_DIR / f'{view}_{st}_scale'
            frames_pattern = frames_dir / 'frame_%04d.png'
            video_path = VIDEO_DIR / f'video_cfd_{view}_{st}_scale.mp4'

            n_png = len(list(frames_dir.glob('frame_*.png')))
            if n_png == 0:
                print(f"  [SKIP] {video_path.name}: nenhum PNG em {frames_dir}")
                continue

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

            print(f"  Compilando: {video_path.name} ({n_png} frames)")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    size_mb = video_path.stat().st_size / 1024 / 1024
                    print(f"    OK ({size_mb:.1f} MB) -> {video_path}")
                else:
                    print("    ERRO no ffmpeg:")
                    print(result.stderr[-500:])
            except FileNotFoundError:
                print("    ERRO: ffmpeg não encontrado. Instale com:")
                print("      conda install -c conda-forge ffmpeg")
                return


# ============================================================================
# INSPEÇÃO (diagnóstico antes de rodar o pipeline)
# ============================================================================
def run_inspect(csv_index):
    print(f"\n{'=' * 72}")
    print("INSPEÇÃO")
    print(f"{'=' * 72}")
    print(f"  Pasta: {CSV_DIR}")
    print(f"  CSVs encontrados: {len(csv_index)}")
    if not csv_index:
        print("  Nenhum CSV. Verifique o caminho (--csv-dir).")
        return
    keys = sorted(csv_index)
    print(f"  Índices: {keys[0]} .. {keys[-1]}  "
          f"(t = {keys[0] * DT_CFD:.1f}s .. {keys[-1] * DT_CFD:.1f}s com DT_CFD={DT_CFD})")
    print(f"  Exemplos: {[csv_index[k].name for k in keys[:3]]}")

    first = csv_index[keys[0]]
    print(f"\n  Lendo {first.name} ...")
    df = read_fluent_csv(first)
    print(f"  Linhas: {len(df):,}")
    print(f"  Colunas: {list(df.columns)}")
    cols = detect_columns(df)
    print("\n  Mapeamento detectado:")
    for k, v in cols.items():
        print(f"    {k:>5} -> {v}")

    try:
        x, y, z, V, P, _ = extract_fields(df)
    except Exception as e:
        print(f"\n  [ERRO] {e}")
        return

    print("\n  Faixas nos dados:")
    print(f"    x: [{x.min():.2f}, {x.max():.2f}]   (config: [{X_MIN}, {X_MAX}])")
    print(f"    y: [{y.min():.2f}, {y.max():.2f}]   (config: [{Y_MIN}, {Y_MAX}])")
    print(f"    z: [{z.min():.2f}, {z.max():.2f}]   (config: [{Z_MIN}, {Z_MAX}])")
    print(f"    |V|: [{np.nanmin(V):.2f}, {np.nanmax(V):.2f}] m/s")
    print(f"    P:   [{np.nanmin(P):.2f}, {np.nanmax(P):.2f}] Pa")

    n_z = int((np.abs(z - Z_SLICE) < TOL_Z).sum())
    n_x = int((np.abs(x - X_SLICE) < TOL_X).sum())
    print("\n  Pontos perto dos planos de corte:")
    print(f"    |z - {Z_SLICE}| < {TOL_Z}: {n_z:,}")
    print(f"    |x - {X_SLICE}| < {TOL_X}: {n_x:,}")
    if n_z < 1000 or n_x < 1000:
        print("    AVISO: poucos pontos — considere aumentar TOL_Z / TOL_X.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    global CSV_DIR, OUT_DIR, WORK_DIR, CACHE_DIR, FRAMES_DIR, VIDEO_DIR, FPS
    global INTERP_MODE, DESPIKE, PRESSAO, CMAP

    parser = argparse.ArgumentParser(
        description="Gera vídeos de CFD (Fluent) a partir de CSVs.")
    parser.add_argument('--csv-dir', type=str, default=None,
                        help=f'Pasta dos CSVs (padrão: {CSV_DIR})')
    parser.add_argument('--out-dir', type=str, default=None,
                        help=f'Pasta de saída dos vídeos (padrão: {OUT_DIR})')
    parser.add_argument('--scale', choices=['global', 'cfd', 'both'],
                        default='global',
                        help="Escala de cor: 'global' (percentis 1/99 destes dados), "
                             "'cfd' (fixa 14-20 m/s, -50..50 Pa) ou 'both'. Padrão: global")
    parser.add_argument('--interp', choices=['volume', '2d', '3d'],
                        default=INTERP_MODE,
                        help="Interpolação: 'volume' triangula a nuvem inteira uma vez "
                             "e avalia no plano (recomendado — não descarta célula "
                             "nenhuma); '2d' projeta o slab no plano (era o padrão, "
                             "usa 4%% dos pontos e produz o aspecto poligonal); '3d' "
                             f"triangula só o slab. Padrão: {INTERP_MODE}")
    parser.add_argument('--despike', type=float, default=DESPIKE, metavar='M_S',
                        help='Substitui células cujo |V| difere mais que M_S m/s da '
                             'mediana dos 12 vizinhos pela mediana (em V e P), antes '
                             'de interpolar. ALTERA OS DADOS — fica escrito no frame. '
                             '0 = desligado (padrão)')
    parser.add_argument('--pressao', choices=['estatica', 'modificada'], default=PRESSAO,
                        help="Pressão mostrada: 'estatica' (padrão; coluna do Fluent menos "
                             "(2/3) rho k, como no relatório) ou 'modificada' (coluna crua)")
    parser.add_argument('--cmap', choices=['relatorio', 'jet'], default=CMAP,
                        help="Mapas de cor: 'relatorio' (padrão; Blues para |V| e RdBu_r "
                             "com zero no centro para p, como no relatório) ou 'jet' "
                             "(o do vídeo antigo)")
    parser.add_argument('--all-csv', action='store_true',
                        help='Usa todos os CSVs da pasta, em ordem, em vez de amostrar '
                             f't={T_START}..{T_END}s em {N_FRAMES} frames')
    parser.add_argument('--fps', type=int, default=FPS,
                        help=f'Frames por segundo do vídeo (padrão: {FPS})')
    parser.add_argument('--inspect', action='store_true',
                        help='Só inspeciona os CSVs (colunas, faixas) e sai')
    parser.add_argument('--test-mode', action='store_true',
                        help='Roda apenas 5 frames (validação rápida)')
    parser.add_argument('--skip-cache', action='store_true',
                        help='Pula Fase 1 (assume cache já existe)')
    parser.add_argument('--skip-frames', action='store_true',
                        help='Pula Fase 3 (só recompila vídeos)')
    parser.add_argument('--force-cache', action='store_true',
                        help='Recomputa cache mesmo se já existir')
    args = parser.parse_args()

    if args.csv_dir:
        CSV_DIR = Path(args.csv_dir)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    FPS = args.fps
    INTERP_MODE = args.interp
    DESPIKE = args.despike
    PRESSAO = args.pressao
    CMAP = args.cmap

    # O cache depende do modo de interpolação e do despike — cada configuração
    # tem a sua pasta, senão uma execução reaproveita .npy da outra.
    cache_name = CACHE_SUBNAME + f"_{INTERP_MODE}_p{PRESSAO}"
    if DESPIKE > 0:
        cache_name += f"_despike{DESPIKE:g}"

    WORK_DIR   = OUT_DIR / WORK_SUBDIR
    CACHE_DIR  = WORK_DIR / cache_name
    FRAMES_DIR = WORK_DIR / FRAMES_SUBNAME
    VIDEO_DIR  = OUT_DIR

    scale_types = ['cfd', 'global'] if args.scale == 'both' else [args.scale]

    if not CSV_DIR.is_dir():
        sys.exit(f"ERRO: pasta de CSVs não existe: {CSV_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    csv_index = build_csv_index()

    if args.inspect:
        run_inspect(csv_index)
        return

    print(f"CSVs em:          {CSV_DIR}  ({len(csv_index)} arquivos)")
    print(f"Vídeos em:        {VIDEO_DIR}")
    print(f"Arquivos de apoio: {WORK_DIR}")
    print(f"Escala(s):        {', '.join(scale_types)}")

    if not csv_index:
        sys.exit("ERRO: nenhum CSV encontrado. Rode com --inspect para diagnosticar.")

    # Grades
    print(f"\n{'=' * 72}")
    print("Criando grades 2D regulares")
    print(f"{'=' * 72}")
    grids = create_grids()

    # Seleção de timesteps
    if args.all_csv:
        timesteps = sorted(csv_index)
        print(f"\n  Usando todos os {len(timesteps)} CSVs da pasta")
    else:
        # mesma amostragem do script original (t=15..100 s, 192 frames)
        t_starts = np.linspace(T_START, T_END, N_FRAMES)
        timesteps = sorted(set(int(t) for t in np.round(t_starts / DT_CFD)))

        faltando = [t for t in timesteps if t not in csv_index]
        timesteps = [t for t in timesteps if t in csv_index]
        if faltando:
            print(f"\n  AVISO: {len(faltando)} timestep(s) sem CSV correspondente "
                  f"(ex.: {faltando[:5]}) — serão ignorados.")
        if not timesteps:
            sys.exit("ERRO: nenhum dos timesteps pedidos existe na pasta de CSVs.\n"
                     "      Use --all-csv para usar todos os arquivos existentes, ou\n"
                     "      rode --inspect para ver a numeração real e ajustar DT_CFD.")

    if args.test_mode:
        step = max(1, len(timesteps) // 5)
        timesteps = timesteps[::step][:5]
        print(f"  [TEST MODE] usando apenas {len(timesteps)} frames")

    print(f"  Timesteps selecionados: {len(timesteps)} "
          f"(t={timesteps[0] * DT_CFD:.1f}s a {timesteps[-1] * DT_CFD:.1f}s)")

    # FASE 1
    if not args.skip_cache:
        run_phase1_cache(timesteps, csv_index, grids, force=args.force_cache)
    else:
        print("\n[SKIP] Fase 1 (cache de interpolação)")

    # FASE 2 (só necessária se alguma saída usa escala global)
    scales = None
    if 'global' in scale_types:
        scales = run_phase2_scales(timesteps, grids)
    else:
        scales = {'V_cfd_range': list(V_CFD_RANGE), 'P_cfd_range': list(P_CFD_RANGE)}

    # FASE 3
    if not args.skip_frames:
        run_phase3_frames(timesteps, grids, scales, scale_types)
    else:
        print("\n[SKIP] Fase 3 (frames)")

    # FASE 4
    run_phase4_videos(scale_types)

    print(f"\n{'=' * 72}")
    print("CONCLUÍDO")
    print(f"{'=' * 72}")
    print(f"Vídeos em: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
