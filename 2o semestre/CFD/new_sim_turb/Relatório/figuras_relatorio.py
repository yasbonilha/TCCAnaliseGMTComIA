#!/usr/bin/env python3
"""
figuras_relatorio.py -- gera as figuras de resultados do relatorio ECA 514
a partir dos CSVs e do transcript da rodada new_sim_turb.

    python3 figuras_relatorio.py results
    python3 figuras_relatorio.py results --passo 1000 --media-cada 10 --workers 16

SAIDA (em results/meta/figuras/), na numeracao do relatorio v5:
    fig05_residuos.png        historico de residuos (transcript.txt)
    fig06_vel_z32.png         |V| e linhas de corrente no plano z = 32 m
    fig07_vel_x140.png        |V| no plano vertical x = 140 m
    fig08_pressao_z32.png     pressao estatica no plano z = 32 m
    fig09_cp.png              Cp em volta do cilindro x escoamento potencial
    fig10_strain_z32.png      taxa de deformacao (escala log) no plano z = 32 m
e imprime os numeros que o texto do relatorio usa (Cp na frente, Cp minimo,
Cp de base), para conferir contra a Tabela 12.

COMO OS CORTES SAO FEITOS
    O CSV e uma nuvem de centroides de celulas poliedricas. Para um corte em
    z = z0, entram as celulas cujo centroide esta a menos de ~meia aresta do
    plano (aresta estimada por cell-volume^(1/3), do static_mesh.csv), e o
    campo e interpolado linearmente (Delaunay 2D) numa grade regular de 1 m.
    Fora do fecho convexo usa-se o vizinho mais proximo. O interior do
    cilindro e mascarado.

PRESSAO
    A coluna `pressure` do Fluent e a pressao MODIFICADA p + (2/3) rho k
    (medido no job 6023 da rodada anterior). As figuras de pressao e o Cp usam
    a pressao ESTATICA, p = pressure - (2/3) rho k. O script imprime as duas
    para comparacao.

CP: INSTANTANEO E MEDIO
    O Cp e calculado no ultimo passo e na media temporal depois da primeira
    travessia (t > 41,2 s), lendo 1 a cada --media-cada passos. A media e a
    comparacao justa com o escoamento potencial; o instantaneo mostra a
    assimetria do desprendimento.
"""

import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Circle, Polygon, Rectangle
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

RHO, U, D = 1.225, 17.0, 54.0
R = D / 2
CX, CY, H = 140.0, 200.0, 65.0
QDIN = 0.5 * RHO * U ** 2
T_TRAV = 700.0 / U
DT = 0.1
BOI_C = (100, 160, 80, 80)                 # x0, y0, largura, altura (planta)
BOI_T = [(100, 240), (180, 240), (140, 600)]
BOI_Z = 100.0

# paleta categorica (ordem fixa) para as series de residuos
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, INK2, GRID = '#222222', '#555555', '#dddddd'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 11,
    'axes.titleweight': 'bold', 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})


def ler(p, **kw):
    d = pd.read_csv(p, skipinitialspace=True, **kw)
    d.columns = [c.strip() for c in d.columns]
    return d


def pasta_csv(res):
    return res if os.path.exists(os.path.join(res, 'static_mesh.csv')) else os.path.join(res, 'csv')


# ----------------------------------------------------------------- cortes
def corte(xyz, h, val, eixo, pos, g1, g2, frac=0.6):
    """Interpola `val` (n,) ou (n,k) no plano eixo=pos sobre a grade (g1, g2)."""
    ia = {'x': 0, 'y': 1, 'z': 2}[eixo]
    outros = [i for i in range(3) if i != ia]
    sel = np.abs(xyz[:, ia] - pos) < frac * h
    P = xyz[sel][:, outros]
    V = val[sel]
    G1, G2 = np.meshgrid(g1, g2)
    Q = np.c_[G1.ravel(), G2.ravel()]
    lin = LinearNDInterpolator(P, V)(Q)
    nn = NearestNDInterpolator(P, V)(Q)
    out = np.where(np.isnan(lin), nn, lin)
    shp = G1.shape + ((V.shape[1],) if V.ndim == 2 else ())
    return G1, G2, out.reshape(shp), sel.sum()


def contorno_planta(ax, boi=True):
    ax.add_patch(Circle((CX, CY), R, fc='#bdbdbd', ec=INK, lw=0.8, zorder=5))
    if boi:
        ax.add_patch(Rectangle(BOI_C[:2], *BOI_C[2:], fill=False, ec='white', lw=0.9, ls='--', zorder=6))
        ax.add_patch(Polygon(BOI_T, closed=True, fill=False, ec='white', lw=0.9, ls='--', zorder=6))


def fig_planta(G1, G2, F, titulo, rotulo, arq, cmap, norm=None, vel=None, boi=True):
    fig, ax = plt.subplots(figsize=(4.6, 9.2))
    m = ax.pcolormesh(G1, G2, F, cmap=cmap, norm=norm, shading='auto', rasterized=True)
    if vel is not None:
        ax.streamplot(G1[0], G2[:, 0], vel[..., 0], vel[..., 1], color='white',
                      linewidth=0.5, density=1.4, arrowsize=0.6)
    contorno_planta(ax, boi)
    ax.set_aspect('equal'); ax.set_xlim(0, 280); ax.set_ylim(0, 700)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)  →  sentido do escoamento')
    ax.set_title(titulo)
    cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.03); cb.set_label(rotulo)
    fig.savefig(arq); plt.close(fig)


# ----------------------------------------------------------------- residuos
def fig_residuos(transcript, arq):
    nomes, linhas, npassos = None, [], 0
    pat = re.compile(r'^\s*(\d+)((?:\s+\d(?:\.\d+)?e[-+]\d+){3,})')
    for ln in open(transcript, errors='replace'):
        if nomes is None and ln.strip().startswith('iter') and 'continuity' in ln:
            nomes = ln.split()[1:-1]          # tira 'iter' e 'time/iter'
        if 'Flow time' in ln:
            npassos += 1
        m = pat.match(ln)
        if m:
            linhas.append([int(m.group(1))] + [float(v) for v in m.group(2).split()])
    if not linhas:
        print('  residuos: nenhuma linha de iteracao no transcript'); return
    a = pd.DataFrame(linhas).drop_duplicates(0, keep='last').sort_values(0).values
    nomes = (nomes or [])[:a.shape[1] - 1] or ['r%d' % i for i in range(a.shape[1] - 1)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [2.2, 1]})
    for ax, sl in ((ax1, slice(None)), (ax2, slice(-120, None))):
        for j, n in enumerate(nomes):
            ax.semilogy(a[sl, 0], a[sl, j + 1], lw=0.8 if ax is ax1 else 1.4,
                        color=CAT[j % len(CAT)], label=n)
        ax.axhline(1e-3, color=INK2, lw=0.8, ls='--')
        ax.grid(True, color=GRID, lw=0.5); ax.set_xlabel('iteração')
    ax1.set_ylabel('resíduo escalado')
    ax1.set_title('Histórico completo (%d passos de tempo)' % npassos)
    ax2.set_title('Detalhe: últimas iterações')
    h_, l_ = ax1.get_legend_handles_labels()
    h_.append(plt.Line2D([], [], color=INK2, lw=0.8, ls='--')); l_.append('critério de convergência 10⁻³')
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(h_, l_, ncol=7, fontsize=8, frameon=False, loc='lower center')
    fig.savefig(arq); plt.close(fig)
    print('  residuos: %d iteracoes, series %s' % (len(a), ', '.join(nomes)))


# ----------------------------------------------------------------- Cp
def _ler_anel(args):
    caminho, idx = args
    d = ler(caminho, usecols=lambda c: c.strip() in ('pressure', 'turb-kinetic-energy'))
    return (d['pressure'].values[idx] - 2 / 3 * RHO * d['turb-kinetic-energy'].values[idx],
            d['pressure'].values[idx])


def cp_bins(theta, p, largura=10):
    b = np.arange(0, 180 + largura, largura)
    c = 0.5 * (b[1:] + b[:-1])
    v = np.array([np.median(p[(theta >= lo) & (theta < hi)]) if ((theta >= lo) & (theta < hi)).any() else np.nan
                  for lo, hi in zip(b[:-1], b[1:])])
    return c, v / QDIN


def main():
    ap = argparse.ArgumentParser(description='Figuras do relatorio ECA 514.')
    ap.add_argument('results')
    ap.add_argument('--passo', type=int, default=1000)
    ap.add_argument('--z', type=float, default=32.0, help='altura do corte horizontal (m)')
    ap.add_argument('--media-cada', dest='media_cada', type=int, default=10)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--anel', type=float, default=2.0, help='espessura do anel do Cp (m)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    csv = pasta_csv(a.results)
    base = a.results if csv != a.results else os.path.dirname(os.path.abspath(a.results))
    out = a.out or os.path.join(base, 'meta', 'figuras')
    os.makedirs(out, exist_ok=True)
    print('saida: %s' % out)

    # --- residuos
    tr = os.path.join(base, 'meta', 'transcript.txt')
    if os.path.exists(tr):
        fig_residuos(tr, os.path.join(out, 'fig05_residuos.png'))
    else:
        print('  transcript nao encontrado em %s' % tr)

    # --- malha e campo do passo
    m = ler(os.path.join(csv, 'static_mesh.csv'))
    t = ler(os.path.join(csv, 't_%05d.csv' % a.passo))
    assert np.array_equal(m['cellnumber'].values, t['cellnumber'].values), 't_ e static_mesh desalinhados'
    xyz = m[['x-coordinate', 'y-coordinate', 'z-coordinate']].values
    h = np.cbrt(m['cell-volume'].values)
    u, v, w = (t[c].values for c in ('x-velocity', 'y-velocity', 'z-velocity'))
    k = t['turb-kinetic-energy'].values
    pmod = t['pressure'].values
    pst = pmod - 2 / 3 * RHO * k
    Vmag = np.sqrt(u * u + v * v + w * w)
    tempo = a.passo * DT

    gx, gy, gz = np.arange(0, 280.5, 1.0), np.arange(0, 700.5, 1.0), np.arange(0, 190.5, 1.0)

    # --- fig 6: |V| + linhas de corrente em z
    G1, G2, F, n = corte(xyz, h, np.c_[Vmag, u, v], 'z', a.z, gx, gy)
    dentro = np.hypot(G1 - CX, G2 - CY) < R
    F[dentro] = np.nan
    fig_planta(G1, G2, F[..., 0], '|V| em z = %g m, t = %g s' % (a.z, tempo), '|V| (m/s)',
               os.path.join(out, 'fig06_vel_z32.png'), 'Blues', vel=F[..., 1:])
    print('  fig06: corte z=%g com %d celulas' % (a.z, n))

    # --- fig 8: pressao estatica em z
    G1, G2, P, _ = corte(xyz, h, pst, 'z', a.z, gx, gy)
    P[dentro] = np.nan
    lim = np.nanpercentile(np.abs(P), 99.5)
    fig_planta(G1, G2, P, 'Pressão estática em z = %g m, t = %g s' % (a.z, tempo), 'p (Pa, manométrica)',
               os.path.join(out, 'fig08_pressao_z32.png'), 'RdBu_r',
               norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))

    # --- fig 7: |V| em x = 140 (plano vertical)
    G1, G2, F, n = corte(xyz, h, Vmag, 'x', CX, gy, gz)
    F[(G1 > CY - R) & (G1 < CY + R) & (G2 < H)] = np.nan
    fig, ax = plt.subplots(figsize=(11, 3.6))
    mm = ax.pcolormesh(G1, G2, F, cmap='Blues', shading='auto', rasterized=True)
    ax.add_patch(Rectangle((CY - R, 0), D, H, fc='#bdbdbd', ec=INK, lw=0.8, zorder=5))
    ax.add_patch(Rectangle((160, 0), 80, BOI_Z, fill=False, ec='white', lw=0.9, ls='--', zorder=6))
    ax.add_patch(Rectangle((240, 0), 360, BOI_Z, fill=False, ec='white', lw=0.9, ls='--', zorder=6))
    ax.set_aspect('equal'); ax.set_xlim(0, 700); ax.set_ylim(0, 190)
    ax.set_xlabel('y (m)  →  sentido do escoamento'); ax.set_ylabel('z (m)')
    ax.set_title('|V| no plano vertical x = %g m, t = %g s' % (CX, tempo))
    cb = fig.colorbar(mm, ax=ax, shrink=0.8, pad=0.02); cb.set_label('|V| (m/s)')
    fig.savefig(os.path.join(out, 'fig07_vel_x140.png')); plt.close(fig)
    print('  fig07: corte x=%g com %d celulas' % (CX, n))

    # --- fig 10: taxa de deformacao em z
    dp = os.path.join(base, 'deriv', 'd_%05d.csv' % a.passo)
    if os.path.exists(dp):
        dd = ler(dp, usecols=lambda c: c.strip() in ('cellnumber', 'strain-rate-mag'))
        assert np.array_equal(dd['cellnumber'].values, m['cellnumber'].values)
        S = np.maximum(dd['strain-rate-mag'].values, 1e-4)
        G1, G2, Sg, _ = corte(xyz, h, np.log10(S), 'z', a.z, gx, gy)
        Sg = 10 ** Sg
        Sg[np.hypot(G1 - CX, G2 - CY) < R] = np.nan
        fig_planta(G1, G2, Sg, 'Taxa de deformação S em z = %g m, t = %g s' % (a.z, tempo), 'S (1/s)',
                   os.path.join(out, 'fig10_strain_z32.png'), 'Purples',
                   norm=LogNorm(vmin=np.nanpercentile(Sg, 5), vmax=np.nanpercentile(Sg, 99.9)))
        # fracao das celulas de S alto dentro das BOIs
        from matplotlib.path import Path
        alto = S > np.percentile(S, 99)
        boi = (((xyz[:, 0] > 100) & (xyz[:, 0] < 180) & (xyz[:, 1] > 160) & (xyz[:, 1] < 240)) |
               Path(BOI_T).contains_points(xyz[:, :2])) & (xyz[:, 2] < BOI_Z)
        print('  fig10: das celulas com S no 1%% mais alto, %.1f%% estao dentro das BOIs'
              % (100 * boi[alto].mean()))
    else:
        print('  d_%05d.csv nao encontrado; fig10 pulada' % a.passo)

    # --- fig 9: Cp em volta do cilindro (meia altura: 15 < z < 50 m)
    r = np.hypot(xyz[:, 0] - CX, xyz[:, 1] - CY)
    anel = np.where((r > R) & (r < R + a.anel) & (xyz[:, 2] > 15) & (xyz[:, 2] < 50))[0]
    # theta: 0 no ponto de estagnacao (frente, y < CY), 180 atras; dobra os dois lados
    th = np.degrees(np.arccos(np.clip((CY - xyz[anel, 1]) / r[anel], -1, 1)))
    lado = np.sign(xyz[anel, 0] - CX)
    c_i, cp_i = cp_bins(th, pst[anel])
    c_ip, cp_ip = cp_bins(th[lado > 0], pst[anel][lado > 0])
    c_im, cp_im = cp_bins(th[lado < 0], pst[anel][lado < 0])

    arqs = sorted(glob.glob(os.path.join(csv, 't_*.csv')))
    passos = np.array([int(re.search(r't_(\d+)\.csv$', f).group(1)) for f in arqs])
    usar = [f for f, s in zip(arqs, passos) if s * DT > T_TRAV and s % a.media_cada == 0]
    print('  Cp medio: lendo %d arquivos (t > %.1f s, 1 a cada %d passos)...' % (len(usar), T_TRAV, a.media_cada), flush=True)
    with Pool(a.workers) as pool:
        res = pool.map(_ler_anel, [(f, anel) for f in usar])
    pmed = np.mean([r_[0] for r_ in res], axis=0)
    pmed_mod = np.mean([r_[1] for r_ in res], axis=0)
    c_m, cp_m = cp_bins(th, pmed)
    _, cp_m_mod = cp_bins(th, pmed_mod)

    tt = np.linspace(0, 180, 361)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(tt, 1 - 4 * np.sin(np.radians(tt)) ** 2, color=INK2, lw=1.2, ls='--', label='escoamento potencial 2D: 1 − 4 sen²θ')
    ax.plot(c_m, cp_m, color=CAT[0], lw=2, marker='o', ms=5, label='CFD, média temporal (t > %.0f s)' % T_TRAV)
    ax.plot(c_ip, cp_ip, color=CAT[1], lw=1.2, marker='^', ms=4, alpha=0.9, label='CFD, t = %g s, lado +x' % tempo)
    ax.plot(c_im, cp_im, color=CAT[2], lw=1.2, marker='v', ms=4, alpha=0.9, label='CFD, t = %g s, lado −x' % tempo)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlim(0, 180); ax.set_xticks(range(0, 181, 30))
    ax.set_ylim(-3.2, 1.3)
    ax.set_xlabel('θ (graus a partir do ponto de estagnação)')
    ax.set_ylabel('C_p = (p − p∞) / (½ρU²)')
    ax.set_title('Coeficiente de pressão em volta do cilindro (15 m < z < 50 m)')
    ax.grid(True, color=GRID, lw=0.5)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig09_cp.png')); plt.close(fig)

    def at(c, cp, ang):
        return cp[np.argmin(np.abs(c - ang))]
    print('\nNUMEROS PARA O TEXTO (Cp com pressao estatica; entre colchetes, com a modificada)')
    print('  celulas no anel: %d (r de %.0f a %.0f m, 15 < z < 50 m)' % (len(anel), R, R + a.anel))
    print('  Cp medio frente (theta~5)  : %+.2f  [%+.2f]' % (at(c_m, cp_m, 5), at(c_m, cp_m_mod, 5)))
    print('  Cp medio minimo            : %+.2f em theta = %.0f  [%+.2f]'
          % (np.nanmin(cp_m), c_m[np.nanargmin(cp_m)], np.nanmin(cp_m_mod)))
    print('  Cp medio base (theta~175)  : %+.2f  [%+.2f]' % (at(c_m, cp_m, 175), at(c_m, cp_m_mod, 175)))
    print('  Cp instantaneo minimo +x/-x: %+.2f / %+.2f' % (np.nanmin(cp_ip), np.nanmin(cp_im)))
    print('  potencial 2D: frente +1, minimo -3 em 90 graus, base +1')
    print('\nfiguras em %s' % out)


if __name__ == '__main__':
    sys.exit(main())
