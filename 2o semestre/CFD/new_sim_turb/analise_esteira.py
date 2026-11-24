#!/usr/bin/env python3
"""
analise_esteira.py -- o desprendimento de vortices sobreviveu ao teto de 1e7?

    python3 analise_esteira.py results
    python3 analise_esteira.py results \
        --ref /home/ybonilha/simulations_CFD/simples_turbulenta/results_boi/csv
    python3 analise_esteira.py results --workers 16 --png esteira.png

POR QUE ESTE SCRIPT EXISTE (handoff 2, segundo risco)
    O teto de 1e5 prendia 24% das celulas da esteira. Solto em 1e7, mu_t
    sobe (no job 6125 chegou a 5,1e5 -- cinco vezes o teto antigo) e pode
    difundir o campo a ponto de matar o desprendimento. O .out do solve nao
    responde isso: ele so olha o ULTIMO passo. A pergunta e temporal.

    Era para ser um monitor de ponto durante o job. Nao precisou: os t_*.csv
    saem a cada passo, entao a serie temporal num ponto ja esta no disco, e
    ler daqui nao arrisca nenhum prompt novo no journal.

O QUE ELE FAZ
    1. Acha, no static_mesh.csv, a celula mais proxima de cada ponto de
       sonda (padrao: tres pontos na esteira, ver PONTOS).
    2. Le essas linhas em todos os t_*.csv (mesma sessao = mesma ordem de
       linhas; o cellnumber e conferido em cada arquivo).
    3. Descarta a primeira travessia (L/U = 700/17 = 41,2 s) e mede, em cada
       ponto: media, desvio, frequencia dominante (FFT e cruzamentos por zero),
       Strouhal St = f D / U.
    4. Veredito por ponto: sinal OSCILA ou virou RETA.

    COMO LER
      - x-velocity (componente TRANSVERSAL) na linha de centro (x = 140)
        oscila em f; y-velocity ali oscila em 2f. Por isso a frequencia sai da
        x-velocity.
      - Esperado: St ~ 0,2 -> f ~ 0,063 Hz -> T ~ 15,9 s. Com 59 s uteis sao
        ~3,7 periodos: a FFT tem resolucao grossa (1/59 s = 0,017 Hz), entao o
        periodo por cruzamentos por zero e a estimativa mais fina.
      - "RETA": desvio padrao da x-velocity < 0,05 m/s depois da 1a travessia.
        Uma esteira viva de cilindro a 17 m/s oscila com amplitude de m/s.

    --ref aponta para a rodada ANTIGA (teto 1e5) e faz a mesma analise nos
    mesmos pontos, lado a lado. E a comparacao que responde se o teto mudava
    a dinamica, e nao so a estatistica de nu_t.
"""

import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

U, D, DT = 17.0, 54.0, 0.1
T_TRAVESSIA = 700.0 / U            # 41,2 s
# Pontos de sonda (x, y, z) em m. O primeiro e o do handoff.
PONTOS = [(140.0, 300.0, 32.0),     # linha de centro, 1,85 D atras do eixo
          (170.0, 300.0, 32.0),     # fora do eixo, mesma estacao
          (140.0, 400.0, 32.0)]     # linha de centro, mais a jusante
CAMPOS = ['x-velocity', 'y-velocity', 'pressure']
LIMIAR_RETA = 0.05                 # m/s, desvio da x-velocity


def ler(p, **kw):
    d = pd.read_csv(p, skipinitialspace=True, **kw)
    d.columns = [c.strip() for c in d.columns]
    return d


def _ler_linhas(args):
    """Le um t_*.csv inteiro (o pandas nao pula linhas mais barato que isso)
    e devolve so as linhas das sondas."""
    caminho, idx, cell_esperado = args
    d = ler(caminho, usecols=lambda c: c.strip() in ['cellnumber'] + CAMPOS)
    if not np.array_equal(d['cellnumber'].values[idx], cell_esperado):
        return caminho, None
    return caminho, d[CAMPOS].values[idx]


def pasta_csv(results):
    """Aceita tanto a pasta da rodada (results/) quanto a propria csv/."""
    if os.path.exists(os.path.join(results, 'static_mesh.csv')):
        return results
    return os.path.join(results, 'csv')


def serie(results, pontos, workers):
    csv = pasta_csv(results)
    m = ler(os.path.join(csv, 'static_mesh.csv'))
    P = m[['x-coordinate', 'y-coordinate', 'z-coordinate']].values
    idx, info = [], []
    for p in pontos:
        dist = np.linalg.norm(P - np.array(p), axis=1)
        i = int(dist.argmin())
        idx.append(i)
        info.append((p, P[i], dist[i]))
    idx = np.array(idx)
    cells = m['cellnumber'].values[idx]

    arqs = sorted(glob.glob(os.path.join(csv, 't_*.csv')))
    passos = np.array([int(re.search(r't_(\d+)\.csv$', a).group(1)) for a in arqs])
    with Pool(workers) as pool:
        out = dict(pool.map(_ler_linhas, [(a, idx, cells) for a in arqs], chunksize=4))
    ruins = [a for a in arqs if out[a] is None]
    if ruins:
        print('  *** %d arquivo(s) com cellnumber diferente do static_mesh nas sondas;'
              ' ignorados. Primeiro: %s' % (len(ruins), ruins[0]))
    ok = [i for i, a in enumerate(arqs) if out[a] is not None]
    V = np.stack([out[arqs[i]] for i in ok])           # (tempo, ponto, campo)
    return passos[ok] * DT, V, info


def frequencia(t, s):
    """Frequencia dominante por FFT (com janela) e por cruzamentos por zero."""
    s = s - s.mean()
    if s.std() == 0:
        return np.nan, np.nan, 0
    n = len(s)
    F = np.abs(np.fft.rfft(s * np.hanning(n)))
    f = np.fft.rfftfreq(n, t[1] - t[0])
    f_fft = f[1:][F[1:].argmax()]
    z = np.where(np.diff(np.sign(s)) > 0)[0]           # cruzamentos subindo
    if len(z) >= 2:
        # interpolacao linear do instante de cada cruzamento
        tz = t[z] - s[z] * (t[z + 1] - t[z]) / (s[z + 1] - s[z])
        f_zc = 1.0 / np.diff(tz).mean()
    else:
        f_zc = np.nan
    return f_fft, f_zc, max(len(z) - 1, 0)


def analisa(nome, t, V, info):
    print('\n' + '=' * 72)
    print('%s   (%d instantes, t = %.1f .. %.1f s)' % (nome, len(t), t[0], t[-1]))
    print('=' * 72)
    m = t > T_TRAVESSIA
    print('  descartada a 1a travessia (t <= %.1f s); restam %.1f s'
          % (T_TRAVESSIA, t[m][-1] - t[m][0] if m.any() else 0))
    res = []
    for j, (p, c, dist) in enumerate(info):
        u, v, pr = V[m, j, 0], V[m, j, 1], V[m, j, 2]
        f_fft, f_zc, nper = frequencia(t[m], u)
        reta = u.std() < LIMIAR_RETA
        print('\n  sonda (%g, %g, %g) -> celula em (%.1f, %.1f, %.1f), a %.2f m'
              % (p + tuple(c) + (dist,)))
        print('    x-velocity  media %+7.3f  desvio %6.3f  min %+7.2f  max %+7.2f'
              % (u.mean(), u.std(), u.min(), u.max()))
        print('    y-velocity  media %+7.3f  desvio %6.3f' % (v.mean(), v.std()))
        print('    pressure    media %+7.2f  desvio %6.2f' % (pr.mean(), pr.std()))
        if reta:
            print('    VEREDITO: RETA. Desvio da x-velocity < %.2f m/s: nao ha'
                  ' desprendimento neste ponto.' % LIMIAR_RETA)
        else:
            print('    f FFT %.4f Hz (St %.3f)   f cruzamentos %.4f Hz (St %.3f, T %.1f s,'
                  ' %d periodos)'
                  % (f_fft, f_fft * D / U, f_zc, f_zc * D / U,
                     1 / f_zc if f_zc else np.nan, nper))
            print('    VEREDITO: OSCILA.')
        res.append(dict(u_std=u.std(), f_zc=f_zc, reta=reta))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('results', help='pasta da rodada (com csv/)')
    ap.add_argument('--ref', help='pasta da rodada antiga (teto 1e5), mesmo formato')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--png', help='salva a serie temporal num PNG')
    ap.add_argument('--out', default=None,
                    help='CSV com a serie (padrao: <results>/meta/esteira_sondas.csv)')
    a = ap.parse_args()

    corridas = [('NOVA (teto 1e7): %s' % a.results, a.results)]
    if a.ref:
        corridas.append(('ANTIGA (teto 1e5): %s' % a.ref, a.ref))

    todas = []
    for nome, pasta in corridas:
        print('lendo %s ...' % pasta, flush=True)
        t, V, info = serie(pasta, PONTOS, a.workers)
        todas.append((nome, t, V, info, analisa(nome, t, V, info)))

    # serie salva junto do dado (so a da rodada nova)
    nome, t, V, info, _ = todas[0]
    out = a.out or os.path.join(a.results, 'meta', 'esteira_sondas.csv')
    cols = {'t': t}
    for j, (p, _, _) in enumerate(info):
        for k, c in enumerate(CAMPOS):
            cols['%s@%g_%g_%g' % ((c,) + p)] = V[:, j, k]
    pd.DataFrame(cols).to_csv(out, index=False)
    print('\nserie salva em %s' % out)

    if len(todas) == 2:
        print('\n' + '=' * 72)
        print('COMPARACAO, sonda a sonda (x-velocity apos a 1a travessia)')
        print('=' * 72)
        print('  %-22s %18s %18s' % ('sonda', 'desvio nova/antiga', 'St nova/antiga'))
        for j, (p, _, _) in enumerate(todas[0][3]):
            n, o = todas[0][4][j], todas[1][4][j]
            print('  %-22s %8.3f / %-8.3f %8.3f / %-8.3f'
                  % ('(%g,%g,%g)' % p, n['u_std'], o['u_std'],
                     n['f_zc'] * D / U, o['f_zc'] * D / U))
        print('  Amplitude muito menor na nova = o teto antigo segurava a difusao;')
        print('  amplitude e St parecidos = o teto mudava a estatistica de nu_t, nao a')
        print('  dinamica da esteira.')

    if a.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(PONTOS), 1, figsize=(10, 2.6 * len(PONTOS)),
                               sharex=True)
        for j, p in enumerate(PONTOS):
            for nome, t, V, _, _ in todas:
                ax[j].plot(t, V[:, j, 0], lw=1, label=nome.split(':')[0])
            ax[j].axvline(T_TRAVESSIA, color='0.6', ls='--', lw=0.8)
            ax[j].set_ylabel('u (m/s)')
            ax[j].set_title('x-velocity em (%g, %g, %g) m' % p, fontsize=10)
        ax[0].legend(fontsize=8)
        ax[-1].set_xlabel('t (s)   [linha tracejada: fim da 1a travessia]')
        fig.tight_layout()
        fig.savefig(a.png, dpi=130)
        print('figura salva em %s' % a.png)

    # codigo de saida: 1 se a sonda do handoff (a primeira) virou reta
    return 1 if todas[0][4][0]['reta'] else 0


if __name__ == '__main__':
    sys.exit(main())
