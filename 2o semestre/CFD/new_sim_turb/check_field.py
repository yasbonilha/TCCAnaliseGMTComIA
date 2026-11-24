#!/usr/bin/env python3
"""
check_field.py -- diz se um CSV exportado contem um campo de escoamento
FISICAMENTE COERENTE, ou apenas valores plausiveis em posicoes erradas.

    ./check_field.py results/csv/t_01000.csv
    ./check_field.py results/csv/static_mesh.csv        (teste geometrico)
    ./check_field.py t_01000.csv --cyl 140 200 26.5

POR QUE ESTE SCRIPT EXISTE
    O CSV do passo 1000 tem estatisticas que parecem boas: |V| de 0,14 a 28 m/s,
    pressao de -299 a +185 Pa (a pressao de estagnacao a 17 m/s e +177),
    refluxo em 2,4% das celulas, k ate 24,6. Sao exatamente os numeros de uma
    esteira de cilindro desenvolvida.

    E, no entanto, olhando ONDE cada valor esta, nao ha esteira nenhuma: colado
    no cilindro a velocidade mediana e 17,3 m/s na frente, no ombro e atras, e
    a componente transversal u e 0,00. O escoamento nao desvia do obstaculo.

    Ou seja, as duas leituras se contradizem -- e e essa contradicao que este
    script mede. As checagens abaixo separam "o campo esta errado" de "o campo
    esta certo e o arquivo nao sabe onde cada celula fica".

OS QUATRO TESTES

 1. GEOMETRIA (so em static_mesh.csv, e o mais forte de todos)
    cell-wall-distance e distancia geometrica pura. As faces z=0, z=190, x=0 e
    x=280 sao paredes, entao para TODA celula vale, sem excecao e sem precisar
    saber nada do cilindro:
        cell-wall-distance <= min(z, 190-z, x, 280-x)
    Uma unica violacao prova que a linha combina a coordenada de uma celula com
    o dado de outra. Nao envolve CFD nenhum: e aritmetica.

 2. CONDICAO DE CONTORNO
    A entrada e velocity-inlet uniforme de 17 m/s. A celula encostada nela nao
    tem como estar longe disso -- nao ha nada entre a face e o centroide, e o
    cilindro esta a 200 m dali. A saida impoe p = 0. Valores impossiveis
    nessas duas faixas sao a assinatura de linha trocada.

 3. COERENCIA INTERNA DAS COLUNAS
    O modelo SST garante  nu_t = a1*k / max(a1*w, S*F2)  <=  k/w.
    Se essa desigualdade vale em 100% das linhas, as sete colunas de campo de
    cada linha vem toda da MESMA celula. Isso separa "colunas embaralhadas
    entre si" de "bloco de campo desalinhado do bloco de coordenadas".

 4. ESTRUTURA ESPACIAL
    Esteira, camada limite e camada cisalhante sao regioes CONECTADAS. Se as
    celulas anomalas forem isoladas -- vizinhas normais em volta -- nao e
    escoamento, e ruido de enderecamento.

TESTE 5 (rodada new_sim_turb) -- O TETO DE mu_t/mu
    Na rodada anterior o teto de 1e5 prendia 4,3% das celulas (24,1% na caixa
    da esteira, boi_t). Esta rodada sobe o teto para 1e7, e o handoff faz duas
    previsoes falseaveis, que este teste mede no proprio CSV:
      1. ~0% das celulas no teto (se nao, 1e7 tambem e baixo);
      2. nu_t na esteira entre ~5 e ~10 m2/s (se for 100+ em todo lugar, o
         teto nao era o problema).
    "No teto" = mu_t/mu >= teto*(1-1e-6). O limiar e impresso junto, porque
    comparar fracoes com limiares diferentes ja enganou este projeto (handoff
    8.3). Se houver static_mesh.csv na mesma pasta, a fracao sai tambem
    PONDERADA POR VOLUME (handoff 8.2: "% de celulas" nao e "% do dominio").

HIPOTESE QUE MOTIVOU ISTO
    O solver rodou com 32 ranks; o export rodou com 8. O log do Fluent avisa
    "reading 32 partition grid onto 8 compute node machine. Combining every 4
    partitions" e logo depois "bandwidth reduction using Reverse Cuthill-McKee".
    A malha e REPARTICIONADA e RENUMERADA na leitura. Se o bloco de coordenadas
    e o bloco de escalares forem montados em ordens diferentes, cada linha fica
    com coordenada de uma celula e campo de outra -- o que reproduz TODOS os
    sintomas acima de uma vez so.

    Se for isso, exportar com o mesmo numero de ranks do solve (ou com -t1)
    resolve. E o que o export_ranks.slurm testa.
"""

import argparse
import sys

import numpy as np
import pandas as pd

# Caixa do dominio. y_min e entrada e y_max e saida: NAO sao paredes, e por
# isso ficam de fora do limite de distancia a parede.
#
# Sao DEFAULTS, nao constantes: --domain sobrescreve. Uma malha nova pode ter
# extensao diferente, e cravar o valor errado aqui faz o Teste 1 acusar
# violacao que nao existe -- foi o que aconteceu no job 5479, quando 1e-6 m de
# tolerancia sobre um limite arredondado produziu 21,6% de falso positivo.
XMIN, XMAX = 0.0, 280.0
YMIN, YMAX = 0.0, 700.0
ZMIN, ZMAX = 0.0, 190.0
UIN = 17.0
RHO = 1.225
MU = 1.7894e-5
VISC_MAX = 1e7          # teto de mu_t/mu desta rodada (--visc-max)
# Caixa de refino da esteira (boi_t): prisma triangular, z de 0 a 100 m.
BOI_T = ((100.0, 240.0), (180.0, 240.0), (140.0, 600.0))
BOI_T_ZMAX = 100.0
BOI_C = (100.0, 180.0, 160.0, 240.0, 100.0)    # xmin xmax ymin ymax zmax


def dentro_triangulo(x, y, tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    d1 = (x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)
    d2 = (x - x3) * (y2 - y3) - (x2 - x3) * (y - y3)
    d3 = (x - x1) * (y3 - y1) - (x3 - x1) * (y - y1)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def acha_cilindro(x, y, z):
    """Procura o furo vertical na nuvem de centroides. Restringe a z baixo
    porque o cilindro nao vai ate o teto: na projecao 2D de altura inteira as
    celulas ACIMA dele tapam o buraco e ele desaparece."""
    from scipy.spatial import cKDTree
    m = z < 40
    if m.sum() < 100:
        return None
    T = cKDTree(np.c_[x[m], y[m]])
    gx = np.arange(XMIN + 5, XMAX - 4, 2.5)
    gy = np.arange(YMIN + 5, YMAX - 4, 2.5)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    dist, _ = T.query(np.c_[GX.ravel(), GY.ravel()])
    i = dist.argmax()
    cx, cy, R = GX.ravel()[i], GY.ravel()[i], dist.max()
    if R < 15:                      # nada muito maior que o espacamento
        return None
    # Altura: sobe fatia a fatia a partir do chao e para na PRIMEIRA que tem
    # celulas dentro do raio -- e ali que acaba o solido. Pegar o maximo das
    # fatias vazias erraria: acima do cilindro existem fatias vazias por acaso,
    # so porque a malha ali e grossa.
    r = np.hypot(x - cx, y - cy)
    H = 0.0
    for h in np.arange(0, ZMAX, 5):
        if ((r < R * .9) & (z >= h) & (z < h + 5)).sum() > 0:
            break
        H = h + 5
    return cx, cy, R, H


def secao(t):
    print('\n' + '=' * 68)
    print(t)
    print('=' * 68)


def main():
    global XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX, VISC_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--cyl', nargs=3, type=float, default=None,
                    metavar=('CX', 'CY', 'R'))
    ap.add_argument('--domain', nargs=6, type=float, default=None,
                    metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'),
                    help='extensao do dominio (do /mesh/check). Sem isto usa '
                         '%g %g %g %g %g %g' % (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX))
    ap.add_argument('--visc-max', dest='visc_max', type=float, default=VISC_MAX,
                    help='teto de mu_t/mu usado no solve (padrao %g)' % VISC_MAX)
    a = ap.parse_args()
    VISC_MAX = a.visc_max

    if a.domain:
        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = a.domain

    d = pd.read_csv(a.csv, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    print('arquivo : %s' % a.csv)
    print('linhas  : %d' % len(d))
    print('colunas : %s' % ', '.join(d.columns))

    x = d['x-coordinate'].values
    y = d['y-coordinate'].values
    z = d['z-coordinate'].values
    problemas = []

    # Extensao observada vs. a assumida. Impressa SEMPRE, porque os limites do
    # dominio sao entrada dos testes 1 e 2 e uma malha nova pode te-los mudado.
    # No job 5479 um limite arredondado por 1 cm gerou 21,6% de falso positivo.
    print('\nextensao dos centroides (a assumida esta entre parenteses):')
    for nome, v, lo, hi in (('x', x, XMIN, XMAX), ('y', y, YMIN, YMAX),
                            ('z', z, ZMIN, ZMAX)):
        aviso = ''
        if v.min() < lo - 1 or v.max() > hi + 1:
            aviso = '   <-- FORA do dominio assumido; use --domain'
        print('  %s: %8.2f .. %8.2f   (%g .. %g)%s'
              % (nome, v.min(), v.max(), lo, hi, aviso))

    # ---- 1. geometria -----------------------------------------------------
    if 'cell-wall-distance' in d.columns:
        secao('TESTE 1 -- GEOMETRIA (nao envolve CFD)')
        dw = d['cell-wall-distance'].values
        teto = np.minimum.reduce([z - ZMIN, ZMAX - z, x - XMIN, XMAX - x])
        exc = dw - teto
        # TOLERANCIA, e por que ela precisa ser generosa.
        # A primeira versao deste teste usava 1e-6 m e acusou 21,6% das celulas
        # no job 5479 -- FALSO POSITIVO INTEIRO. Os limites 0/280/0/190 sao os
        # numeros ARREDONDADOS do /mesh/check; a parede de verdade pode estar em
        # 279,99. Toda celula cuja parede mais proxima e lateral passa entao a
        # "violar" por 1 cm. O sintoma era visivel na propria saida: nos piores
        # casos d e o teto batiam ate a segunda decimal.
        # Corrupcao de endereco desloca a distancia a parede em DEZENAS de metros,
        # nao em centimetros. Por isso 1,0 m, e por isso a distribuicao e impressa:
        # o leitor julga a escala em vez de confiar num binario.
        TOL = 1.0
        print('  cell-wall-distance <= min(z-%g, %g-z, x-%g, %g-x)' % (ZMIN, ZMAX, XMIN, XMAX))
        print('  (y_min e entrada e y_max e saida: nao entram, nao sao paredes)')
        print('  excesso sobre o teto: p50 %+.3f  p99 %+.3f  MAXIMO %+.3f m'
              % (np.median(exc), np.quantile(exc, .99), exc.max()))
        print('  celulas com excesso > %.1f m: %d de %d (%.2f%%)'
              % (TOL, (exc > TOL).sum(), len(dw), 100 * (exc > TOL).mean()))
        if exc.max() <= TOL:
            print('  VEREDITO: coerente. O excesso maximo (%.3f m) e da ordem do erro'
                  % exc.max())
            print('  de arredondamento dos limites do dominio, nao de troca de linha.')
        else:
            i = np.argsort(exc)[::-1][:5]
            print('  piores casos:')
            for j in i:
                print('    x=%7.2f y=%7.2f z=%7.2f -> d=%8.2f  teto %8.2f  excesso %+.2f'
                      % (x[j], y[j], z[j], dw[j], teto[j], exc[j]))
            print('\n  VEREDITO: excesso grande demais para ser arredondamento.')
            problemas.append('cell-wall-distance incompativel com as coordenadas')

    tem_vel = 'x-velocity' in d.columns
    if not tem_vel:
        secao('RESUMO')
        print('  arquivo sem campos de velocidade; so o teste 1 se aplica.')
        return 1 if problemas else 0

    u = d['x-velocity'].values
    v = d['y-velocity'].values
    w = d['z-velocity'].values
    V = np.sqrt(u * u + v * v + w * w)
    p = d['pressure'].values
    k = d.get('turb-kinetic-energy')
    om = d.get('specific-diss-rate')
    mut = d.get('viscosity-turb')

    # ---- 2. condicoes de contorno -----------------------------------------
    secao('TESTE 2 -- CONDICOES DE CONTORNO')
    ent = y < YMIN + 5
    sai = y > YMAX - 10
    print('  ENTRADA (velocity-inlet uniforme, v=%g, u=w=0), celulas a menos de 5 m:' % UIN)
    print('    n=%d   v_y: min %.2f  med %.2f  max %.2f' % (ent.sum(), v[ent].min(), np.median(v[ent]), v[ent].max()))
    ruim = (np.abs(v[ent] - UIN) > 2) | (np.abs(u[ent]) > 1)
    print('    incompativeis com a BC: %d (%.0f%%)' % (ruim.sum(), 100 * ruim.mean()))
    if (v[ent] < 0).any():
        print('    *** %d celulas com REFLUXO encostadas na entrada. Impossivel:'
              % (v[ent] < 0).sum())
        print('        a face impoe +%g m/s e o cilindro esta a 200 m dali.' % UIN)
        problemas.append('refluxo colado na entrada')
    print('  SAIDA (pressure-outlet, p=0), celulas a menos de 10 m:')
    print('    n=%d   p: min %.1f  med %.1f  max %.1f' % (sai.sum(), p[sai].min(), np.median(p[sai]), p[sai].max()))
    if np.abs(p[sai]).max() > 50:
        print('    *** p de ate %.0f Pa colado numa face que impoe p=0.' % np.abs(p[sai]).max())
        problemas.append('pressao impossivel na saida')

    # ---- 3. coerencia entre colunas ---------------------------------------
    if k is not None and om is not None and mut is not None:
        secao('TESTE 3 -- AS 7 COLUNAS DE CAMPO VEM DA MESMA CELULA?')
        r = (mut.values / RHO) / (k.values / om.values)
        print('  nu_t / (k/omega):  mediana %.6f   max %.6f' % (np.median(r), r.max()))
        print('  violacoes de nu_t <= k/omega: %d' % (r > 1 + 1e-6).sum())
        if (r > 1 + 1e-6).sum() == 0:
            print('  VEREDITO: as colunas de campo sao coerentes ENTRE SI.')
            print('  Entao o problema, se houver, esta entre o bloco de campo e o de')
            print('  coordenadas -- nao entre as colunas de campo.')

    # ---- 5. teto de mu_t/mu -----------------------------------------------
    if mut is not None:
        import os
        secao('TESTE 5 -- TETO DE mu_t/mu = %g (previsoes do handoff)' % VISC_MAX)
        razao = mut.values / MU
        nut = mut.values / RHO
        no_teto = razao >= VISC_MAX * (1 - 1e-6)
        perto = razao >= VISC_MAX * (1 - 1e-3)
        print('  limiar "no teto": mu_t/mu >= teto*(1-1e-6) = %.6g' % (VISC_MAX * (1 - 1e-6)))
        print('  mu_t/mu: p50 %.3g  p99 %.3g  max %.4g   (teto %.4g)'
              % (np.median(razao), np.quantile(razao, .99), razao.max(), VISC_MAX))
        print('  celulas no teto: %d de %d (%.3f%%)   [a 1e-3 do teto: %d]'
              % (no_teto.sum(), len(razao), 100 * no_teto.mean(), perto.sum()))
        # ponderado por volume, se houver static_mesh.csv ao lado
        vol = None
        sm = os.path.join(os.path.dirname(os.path.abspath(a.csv)), 'static_mesh.csv')
        if os.path.exists(sm) and os.path.abspath(sm) != os.path.abspath(a.csv):
            m = pd.read_csv(sm, skipinitialspace=True)
            m.columns = [c.strip() for c in m.columns]
            if 'cell-volume' in m.columns and len(m) == len(d) and \
               np.array_equal(m['cellnumber'].values, d['cellnumber'].values):
                vol = m['cell-volume'].values
                print('  ponderado por VOLUME (static_mesh.csv): %.3f%% do dominio no teto'
                      % (100 * vol[no_teto].sum() / vol.sum()))
            else:
                print('  static_mesh.csv ao lado nao alinha com este arquivo '
                      '(ou nao tem cell-volume): sem ponderacao por volume')
        regioes = [
            ('boi_c (caixa do cilindro)',
             (x > BOI_C[0]) & (x < BOI_C[1]) & (y > BOI_C[2]) & (y < BOI_C[3]) & (z < BOI_C[4])),
            ('boi_t (esteira)', dentro_triangulo(x, y, BOI_T) & (z < BOI_T_ZMAX)),
        ]
        print('\n  %-27s %7s %9s %9s %9s %9s' % ('regiao', 'celulas', '% teto',
                                                'nu_t p50', 'nu_t p90', 'nu_t max'))
        for nome, m in [('dominio inteiro', np.ones(len(d), bool))] + regioes:
            if m.sum():
                print('  %-27s %7d %8.3f%% %9.3g %9.3g %9.3g'
                      % (nome, m.sum(), 100 * no_teto[m].mean(), np.median(nut[m]),
                         np.quantile(nut[m], .9), nut[m].max()))
        print('  (nu_t em m2/s; teto em nu_t = %.4g m2/s)' % (VISC_MAX * MU / RHO))
        print('\n  PREVISAO 1: ~0%% no teto           -> medido %.3f%%' % (100 * no_teto.mean()))
        mt = regioes[1][1]
        if mt.sum():
            print('  PREVISAO 2: nu_t esteira 5-10 m2/s -> p90 em boi_t %.3g m2/s'
                  % np.quantile(nut[mt], .9))
        print('  (So vale apos o escoamento se desenvolver: num teste de 20 passos')
        print('   a esteira ainda nem existe. Leia isto no fim da producao.)')
        if no_teto.mean() > 0.01:
            print('  *** mais de 1% das celulas no teto: o teto ainda morde.')
            problemas.append('mu_t/mu no teto em %.2f%% das celulas' % (100 * no_teto.mean()))

    # ---- 4. estrutura espacial --------------------------------------------
    secao('TESTE 4 -- AS ANOMALIAS FORMAM ESTRUTURA OU SAO CELULAS ISOLADAS?')
    from scipy.spatial import cKDTree
    P = np.c_[x, y, z]
    T = cKDTree(P)
    lento = V < 8
    if lento.sum() > 10:
        _, nb = T.query(P[lento], k=7)
        viz = lento[nb[:, 1:]].sum(1)
        print('  celulas com V<8: %d (%.1f%% do dominio)' % (lento.sum(), 100 * lento.mean()))
        print('  vizinhos lentos (de 6): media %.2f' % viz.mean())
        print('  totalmente isoladas (0 vizinhos lentos): %d (%.0f%%)'
              % ((viz == 0).sum(), 100 * (viz == 0).mean()))
        print('  esperado num campo real: uma esteira e CONECTADA, quase toda celula')
        print('  lenta tem 4-6 vizinhas lentas. Isolamento alto = ruido de endereco.')
        if (viz == 0).mean() > 0.3:
            problemas.append('celulas anomalas isoladas, sem estrutura conectada')

    # ---- cilindro ---------------------------------------------------------
    cil = tuple(a.cyl) + (0,) if a.cyl else acha_cilindro(x, y, z)
    if cil:
        cx, cy, R, H = cil
        secao('O CILINDRO: encontrado em x=%.0f  y=%.0f  raio=%.1f  altura~%.0f' % (cx, cy, R, H))
        rr = np.hypot(x - cx, y - cy)
        print('  celulas dentro de r<%.0f e z<%.0f: %d  (tem que ser 0: e o furo da malha)'
              % (R, max(H, 10), ((rr < R) & (z < max(H, 10))).sum()))
        anel = (rr > R) & (rr < R + 14) & (z < max(H, 40))
        ang = np.degrees(np.arctan2(x - cx, cy - y))
        print('\n  setor          n     V med    u med     p med   p max')
        # angulo: 0 = frente (encara o vento), + para o lado +x, +-180 = tras.
        # TRAS usa |ang| >= 155 para pegar as duas metades (155 a 180 e -180 a -155);
        # antes so pegava a metade +x.
        setores = [('FRENTE', np.abs(ang) < 25),
                   ('OMBRO +x', (ang >= 65) & (ang < 115)),
                   ('OMBRO -x', (ang >= -115) & (ang < -65)),
                   ('TRAS', np.abs(ang) >= 155)]
        for nome, sel in setores:
            m = anel & sel
            if m.sum() > 2:
                print('  %-10s %5d  %7.2f  %7.2f  %8.2f %7.1f'
                      % (nome, m.sum(), np.median(V[m]), np.median(u[m]), np.median(p[m]), p[m].max()))
        print('\n  ESPERADO a %g m/s:  FRENTE V->0 e p->+%.0f Pa;  OMBRO V->~%.0f m/s;'
              % (UIN, 0.5 * RHO * UIN ** 2, 2 * UIN))
        print('                     TRAS V baixo com refluxo.')
        frente = anel & (np.abs(ang) < 25)
        if frente.sum() > 5 and np.median(V[frente]) > 0.8 * UIN:
            print('\n  *** Nao ha estagnacao na frente do cilindro nem desvio lateral')
            print('      (u mediano ~0). O escoamento atravessa o obstaculo como se')
            print('      ele nao existisse, o que nenhum solver produz.')
            problemas.append('nenhuma interacao com o cilindro')

    # ---- veredito ---------------------------------------------------------
    secao('VEREDITO')
    if not problemas:
        print('  Nenhuma incoerencia detectada. O campo parece fisicamente valido.')
        return 0
    for s in problemas:
        print('  - %s' % s)
    so_teto = all(p.startswith('mu_t/mu no teto') for p in problemas)
    if so_teto:
        print('\n  O unico problema e o teto: o campo e coerente, mas mu_t/mu ainda')
        print('  encosta em %g. Handoff 2, previsao 1: o teto ainda e baixo, ou ha'
              % VISC_MAX)
        print('  outro problema. Veja no TESTE 5 em que regiao isso acontece.')
        return 1
    print('\n  Os testes 1-4 acusaram incoerencia. Nesta rodada o CSV e escrito de')
    print('  dentro do solve, entao "valor certo na posicao errada" (job 5479) nao')
    print('  deveria acontecer: desconfie primeiro do journal (prompt desalinhado)')
    print('  e leia o transcript em meta/.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
