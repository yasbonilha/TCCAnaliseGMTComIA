#!/usr/bin/env python3
"""
make_solve_csv_journal.py -- gera o journal que RESOLVE e escreve CSV no mesmo
passe, sem nunca gravar e reler .dat.h5.            [rodada new_sim_turb]

O QUE MUDA NESTA RODADA (ver HANDOFF_novo_cfd.md)
    Na fisica, UMA coisa so:

        Maximum Turb. Viscosity Ratio:  1e5 (default do Fluent)  ->  1e7

    O journal antigo NUNCA definiu esse teto -- o 1e5 era o default. Por isso
    a mudanca e ACRESCENTAR um comando (/solve/set/limits), e nao editar um.
    Medido no dado antigo: 4,3% das celulas no teto em t=300 s e 24,1% na caixa
    de refino da esteira (boi_t). A escala fisica da esteira e
    nu_t ~ 4,6 a 9,2 m2/s, ou seja mu_t/mu ~ 3e5 a 6e5 -- acima do teto antigo.

    Todo o resto e SO EXPORTACAO, nao toca na solucao:
      - static_mesh.csv ganha cell-volume                      (handoff 3.3)
      - deriv/d_NNNNN.csv: 9 derivadas de velocidade +
        strain-rate-mag, em TODO snapshot (mesmo passo do t_)  (handoff 3.1)
      - wall/<zona>_NNNNN.csv: y+ e tensao de parede por zona  (handoff 3.4)
      - bc/inlet_*.csv: k, omega, mu_t, TI e mu_t/mu NAS FACES
        do inlet, lidos do solver                              (handoff 3.2)
      - meta/: transcript completo (residuos e contagem de
        iteracoes do SIMPLE), report summary e setup.cas.h5    (handoff 3.5)

    Os t_NNNNN.csv ficam IDENTICOS em formato: mesmas 11 colunas, mesma
    ordem, mesmos nomes de arquivo. As derivadas vao em arquivo SEPARADO
    justamente para nao mexer nisso -- o alinhamento com o t_ do mesmo passo
    e por cellnumber (mesma sessao, mesma numeracao), e as coordenadas vem
    junto em todo arquivo para conferir.

O TETO E UMA SEQUENCIA DE PROMPTS -- ORDEM MEDIDA EM SESSAO INTERATIVA
    /solve/set/limits nao tem subcomandos: ele pergunta varios limites em
    sequencia, e a ordem NAO esta documentada. Ela foi lida numa sessao
    interativa do Fluent 25.2 (05/10/2026), com k-omega SST e sem equacao de
    energia -- exatamente este caso:

        > /solve/set/limits
        1 Minimum allowable absolute pressure (in [Pa]) [1]                ,
        2 Maximum allowable absolute pressure (in [Pa]) [50000000000]      ,
        3 Minimum allowable k (in [m^2/s^2]) [1e-14]                       ,
        4 Minimum allowable omega (in [s^-1]) [9.999999999999999e-21]      ,
        5 Maximum allowable turbulent/laminar viscosity ratio [100000]     1e7

    Sao 5 prompts e o teto e o ULTIMO; ha 4 antes dele (--limits-skip 4).
    Nao ha prompt de temperatura: ele so aparece com a equacao de energia.
    Se um dia ligar energia (ou trocar o modelo), a contagem muda -- refaca
    a leitura interativa.

    Cada ',' aceita o default de um prompt (Text Command List, 2.2). Se a
    contagem estivesse errada, 1e7 cairia noutro limite (por exemplo omega
    minimo = 1e7, o que destruiria a solucao em silencio), ou o teto ficaria
    em 1e5 sem aviso. Por isso, logo depois de definir, este journal roda
    /solve/set/limits DE NOVO so com linhas em branco (aceita tudo como
    esta). O log mostra cada prompt com o valor ATUAL entre colchetes, e
    tem que aparecer:

        Maximum allowable turbulent/laminar viscosity ratio [10000000]

    (o Fluent escreve por extenso: 5e10 aparece como 50000000000). Os outros
    quatro tem que continuar com os defaults acima. O solve_csv.slurm confere.
    Linhas em branco a mais depois do comando sao inofensivas: no prompt de
    comando, branco nao faz nada; num prompt pendente, aceita o default.

POR QUE UM LACO DESENROLADO, E NAO execute-commands NEM transient-export
    (inalterado da rodada anterior) Cada bloco ja apareceu em log de producao:

        /solve/dual-time-iterate <n> <inner>      avanca n passos
        /file/export/ascii <arquivo>              escreve o estado atual
        ()                                        Surfaces: dominio inteiro
        yes                                       Delimiter/Comma? -> virgula
        <campos, um por linha>                    ASCII scalar(N)>
        ()                                        FECHA a lista (branco nao fecha)
        yes                                       Cell-Centered? -> centroide

    FORMAS NOVAS nesta rodada, conferidas em sessao interativa (05/10/2026):
      - nomes de campo novos: CONFIRMADOS. (inquire-cell-functions) devolveu
        a lista oficial de campos deste build, e todos os usados aqui estao
        nela com exatamente esta grafia: dx-velocity-dx ... dz-velocity-dz,
        strain-rate-mag, cell-volume, turb-intensity, viscosity-ratio,
        y-plus, wall-shear, x/y/z-wall-shear. Um export do dominio com as
        derivadas, strain-rate-mag, cell-volume, turb-intensity e
        viscosity-ratio rodou ate "Done." com a sequencia de prompts de
        sempre (Surfaces / Delimiter / ASCII scalar(N) / Cell-Centered).
      - /report/summary: "write to files? [no]" -> yes ->
        "report summary filename [\"report.sum\"]" -> nome. Confirmado.
      - export com SUPERFICIE nomeada: CONFIRMADO. A sequencia e
            Surfaces(1) [()] inlet
            Surfaces(2) [()] ()          <- fecha a lista de superficies
        e o resto igual. Com Cell-Centered? yes sai o valor de FACE:
        inlet -> 3965 linhas, ground -> 5933 (contra 211.579 celulas), com
        y = 0 no inlet e z = 0 no ground. O cabecalho trouxe exatamente os
        campos pedidos, nos dois casos.

    O QUE O TESTE DO INLET MOSTROU (ja serve de verificacao 3.2 parcial):
        viscosity-ratio = 10, turb-kinetic-energy = 1.08375 nas faces --
        exatamente TI = 5%, mu_t/mu = 10 do handoff.

    ATENCAO, turb-intensity NAO E A TI LOCAL. O teste devolveu 0.85 onde a TI
    e 5%: o Fluent divide sqrt(2k/3) pela VELOCIDADE DE REFERENCIA dos
    Reference Values (default 1 m/s), nao pela velocidade local nem por 17.
    Conta: sqrt(2*1.08375/3) = 0.85 = 0.05 * 17. Para ter a TI fisica, use
        TI = turb-intensity / 17
    ou calcule direto de k. O campo fica no export porque e barato e prova a
    conta, mas nao o leia como fracao.

    E MAIS (job 6124): nas FACES do inlet, turb-intensity e viscosity-ratio
    sao calculados com a celula vizinha, nao com a BC. No init coincidem; no
    fim do solve deram TI 1,70% e mu_t/mu 9,24, com k = 1.08375 e
    omega = 7419.21 identicos ao imposto. A BC se le de k e omega:
        TI = sqrt(2k/3) / 17 = 5%      mu_t/mu = rho k / (omega mu) = 10

MONITOR DA ESTEIRA
    Nao ha monitor de ponto dentro do Fluent: a forma TUI de report-definition
    + report-file nunca rodou neste build, e um prompt desalinhado ali no meio
    de 1000 passos e o pior lugar para descobrir isso. Os t_NNNNN.csv ja saem
    a cada passo; o monitor_esteira.py le a celula mais proxima de
    (140, 300, 32) m desses arquivos ENQUANTO o job roda, e tambem a contagem
    de iteracoes do SIMPLE no transcript. Mesmo termometro, risco zero.

USO
    ./make_solve_csv_journal.py --steps 20 --results results_test \\
        --out JOURNAL_solve_csv_TEST
    ./make_solve_csv_journal.py --steps 1000 --results results \\
        --out JOURNAL_solve_csv

    --steps N          passos de tempo no total.
    --every N          exporta 1 t_*.csv a cada N passos (padrao 1).
    --deriv-every N    derivadas a cada N passos (padrao 1 = junto de todo
                       t_*.csv; disco no cluster nao e restricao; 0 desliga).
    --wall-every N     y+ / tensao de parede a cada N passos (padrao 10; 0 desliga).
    --visc-max X       teto de mu_t/mu (padrao 1e7).
    --limits-skip N    quantos prompts de /solve/set/limits vem ANTES do
                       teto (padrao 4, medido em sessao interativa).
    --inner N          iteracoes internas por passo (padrao 50, como antes).
    --dt X             passo de tempo (padrao 0.1).
    --autosave N       .dat.h5 a cada N passos SO como restart (padrao 200; 0 desliga).
"""

import argparse
import os

WORKDIR = '/home/ybonilha/simulations_CFD/new_sim_turb'
# Mesmo arquivo de malha da rodada anterior (211.579 celulas), copiado para a
# pasta nova. A malha NAO muda nesta rodada (handoff 4 e 5).
MESH = os.path.join(WORKDIR, 'geometria_malha', 'malha_simples_boi.msh.h5')
N_CELULAS = 211579

# Zonas -- iguais a rodada anterior. Ver a explicacao longa no gerador antigo:
# BC aplicada a nome inexistente deixa prompt pendente e desalinha o journal.
ZONA_INLET   = 'inlet'
ZONA_OUTLET  = 'outlet'
ZONAS_MOVEIS = ['lateral_min', 'lateral_max', 'top']
# Paredes FIXAS: nao recebem comando de BC (no-slip e o default), mas entram
# na exportacao de y+ e tensao de parede.
ZONAS_FIXAS  = ['ground', 'cylinder_wall', 'cylinder_top']

VISC_MAX = 1e7
# Prompts de /solve/set/limits ANTES de "Maximum allowable turbulent/laminar
# viscosity ratio": p_abs min, p_abs max, k min, omega min. MEDIDO em sessao
# interativa (ver docstring). Vale para SST sem energia.
LIMITS_SKIP = 4
# Linhas em branco depois do eco de conferencia: cobre todos os prompts que
# existirem, mesmo que sejam mais do que o esperado. Branco a mais e no-op.
LIMITS_ECHO_BLANKS = 12

# --- campos ------------------------------------------------------------------
# t_NNNNN.csv: NAO MUDAR. Formato identico a rodada anterior (handoff 5).
CAMPOS = ['x-velocity', 'y-velocity', 'z-velocity', 'pressure',
          'turb-kinetic-energy', 'specific-diss-rate', 'viscosity-turb']
# static_mesh.csv: cell-volume e novo (handoff 3.3).
CAMPOS_ESTATICOS = ['cell-wall-distance', 'cell-volume']
# Categoria Derivatives (handoff 3.1). Nome d<i>-velocity-d<j> = d u_i / d x_j.
# Nomes CONFIRMADOS por (inquire-cell-functions), 05/10/2026.
CAMPOS_DERIV = ['dx-velocity-dx', 'dx-velocity-dy', 'dx-velocity-dz',
                'dy-velocity-dx', 'dy-velocity-dy', 'dy-velocity-dz',
                'dz-velocity-dx', 'dz-velocity-dy', 'dz-velocity-dz',
                'strain-rate-mag']
# Faces de parede (handoff 3.4).
# Nomes CONFIRMADOS por (inquire-cell-functions), 05/10/2026.
CAMPOS_PAREDE = ['y-plus', 'wall-shear',
                 'x-wall-shear', 'y-wall-shear', 'z-wall-shear']
# Faces do inlet (handoff 3.2): o que o solver de fato impos.
# Nomes CONFIRMADOS por (inquire-cell-functions), 05/10/2026.
# turb-intensity = sqrt(2k/3) / U_ref, com U_ref = 1 m/s (Reference Values),
# NAO a TI local: o teste deu 0.85 para TI = 5%. TI fisica = coluna / 17.
CAMPOS_INLET = ['y-velocity', 'turb-kinetic-energy', 'specific-diss-rate',
                'viscosity-turb', 'turb-intensity', 'viscosity-ratio']

# Subpastas de saida. O Fluent NAO cria pasta: exportar para pasta inexistente
# falha e deixa prompt pendente. Este gerador cria todas.
SUBDIRS = ['csv', 'deriv', 'wall', 'bc', 'meta', 'dat_cas']

# Custo: medido na malha antiga (job 5495), 7.070.046 bytes para 39.060
# celulas x 11 colunas -> ~16,5 bytes por valor. Linear em primeira ordem.
BYTES_POR_VALOR = 7070046 / (39060 * 11)


def bloco_export(caminho, campos, superficie=None):
    """A ordem destas linhas veio dos logs 5293/5294 e nao e negociavel.
    superficie=None -> dominio inteiro ('()'), a forma ja verificada.
    superficie='nome' -> a lista de superficies recebe o nome e e fechada por
    '()'. FORMA NOVA: a conferir na sessao interativa."""
    if superficie is None:
        sup = ['()']                     # Surfaces(1) [()] -> dominio inteiro
    else:
        sup = [superficie, '()']         # uma superficie, e fecha a lista
    # Caminho SEM aspas, como na forma verificada (sem espacos no caminho).
    return (['/file/export/ascii %s' % caminho] + sup +
            ['yes'] +                    # Delimiter/Comma? -> virgula
            list(campos) +
            ['()',                       # fecha ASCII scalar(N)> -- branco NAO fecha
             'yes'])                     # Cell-Centered? -> centroide / face


def num(x):
    """1e7, nao 1e+07: e o que se le no log e no journal."""
    return ('%g' % x).replace('e+0', 'e').replace('e+', 'e')


def cruzou(passo, n, cada):
    """True se o intervalo (passo-n, passo] contem um multiplo de 'cada'.
    Funciona mesmo quando --every nao divide --deriv-every."""
    return cada > 0 and passo // cada > (passo - n) // cada


def main():
    ap = argparse.ArgumentParser(
        description='Gera o journal de solve+CSV da rodada new_sim_turb.')
    ap.add_argument('--out', default='JOURNAL_solve_csv')
    ap.add_argument('--results', default='results')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--deriv-every', dest='deriv_every', type=int, default=1)
    ap.add_argument('--wall-every', dest='wall_every', type=int, default=10)
    ap.add_argument('--inner', type=int, default=50)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--autosave', type=int, default=200)
    ap.add_argument('--visc-max', dest='visc_max', type=float, default=VISC_MAX)
    ap.add_argument('--limits-skip', dest='limits_skip', type=int,
                    default=LIMITS_SKIP)
    ap.add_argument('--mesh', default=MESH)
    ap.add_argument('--ncells', type=int, default=N_CELULAS,
                    help='so para a estimativa de disco')
    ap.add_argument('--inlet', default=ZONA_INLET)
    ap.add_argument('--outlet', default=ZONA_OUTLET)
    ap.add_argument('--moving-wall', dest='moving_wall',
                    default=','.join(ZONAS_MOVEIS))
    ap.add_argument('--fixed-wall', dest='fixed_wall',
                    default=','.join(ZONAS_FIXAS),
                    help='paredes fixas: sem comando de BC, so exportacao')
    ap.add_argument('--fields', default=None,
                    help='sobrescreve os campos do t_*.csv (NAO recomendado)')
    ap.add_argument('--no-mkdir', dest='mkdir', action='store_false')
    a = ap.parse_args()

    if a.every < 1:
        ap.error('--every precisa ser >= 1')

    campos = a.fields.split() if a.fields else list(CAMPOS)
    moveis = [t.strip() for t in a.moving_wall.split(',') if t.strip()]
    fixas = [t.strip() for t in a.fixed_wall.split(',') if t.strip()]
    paredes = fixas + moveis
    D = {s: os.path.join(a.results, s) for s in SUBDIRS}

    if a.mkdir:
        for p in D.values():
            os.makedirs(p, exist_ok=True)

    L = []
    W = L.append

    W('; =====================================================================')
    W(';  JOURNAL_solve_csv -- rodada new_sim_turb.')
    W(';  GERADO por make_solve_csv_journal.py. Nao edite a mao.')
    W(';')
    W(';  Resolve %d passos de %g s; t_*.csv a cada %d passo(s), direto da'
      % (a.steps, a.dt, a.every))
    W(';  memoria do solver (nunca relendo .dat.h5).')
    W(';')
    W(';  UNICA MUDANCA NA FISICA em relacao a rodada anterior (job 5518):')
    W(';      Maximum Turb. Viscosity Ratio  1e5 (default)  ->  %s' % num(a.visc_max))
    W(';  Todo o resto que e novo aqui e exportacao.')
    W('; =====================================================================')
    W('/file/set-tui-version "25.2"')
    W('/file/confirm-overwrite? no')
    W('; Transcript: TUDO que o Fluent imprime -- inclusive residuos e o numero')
    W('; de iteracoes do SIMPLE por passo -- vai junto dos CSV (handoff 3.5).')
    W('; O monitor_esteira.py le a contagem de iteracoes daqui durante o job.')
    W('/file/start-transcript "%s"' % os.path.join(D['meta'], 'transcript.txt'))
    W('; As duas linhas do webserver vem do journal original (diff minimo).')
    W('(putenv "FLUENT_WEBSERVER_TOKEN" "your_secure_token_here")')
    W('/serve/start-web-server "Fluent WebUI" 5000 0 "" ""')
    W('; ---------------------------------------------------------------------')
    W('; Malha: caminho ABSOLUTO.')
    W('; ---------------------------------------------------------------------')
    W('/file/read-case "%s"' % a.mesh)
    W('; ---------------------------------------------------------------------')
    W('; Fisica -- IDENTICA a rodada anterior, linha por linha.')
    W('; ---------------------------------------------------------------------')
    W('/define/models/unsteady-2nd-order? yes')
    W('/define/models/viscous/kw-sst? yes')
    W('; BCs por NOME de zona. Nome errado = prompt pendente = journal')
    W('; desalinhado em silencio. Confira com o probe antes de rodar.')
    W('/define/boundary-conditions/set/velocity-inlet %s () vmag no 17 quit' % a.inlet)
    W('/define/boundary-conditions/set/pressure-outlet %s () prevent-reverse-flow? yes quit'
      % a.outlet)
    W('; Parede movel: um comando por zona (forma ja verificada).')
    for z in moveis:
        W('/define/boundary-conditions/set/wall %s () motion-bc yes motion-bc-moving '
          'relative? no rotating? no components? no vmag no 17 ni 0 nj 1 quit' % z)
    W('; %s: parede fixa no-slip = default, sem comando.' % ', '.join(fixas))
    W('; ---------------------------------------------------------------------')
    W('; NUMERICA -- SIMPLE (20) e momento Third-Order MUSCL (6). k e omega')
    W('; ficam no default (2nd Order Upwind, conforme o summary da rodada')
    W('; anterior), assim como wall-omega-treatment = correlation. Nao se')
    W('; acrescenta comando para eles: mudar uma variavel so.')
    W('; ---------------------------------------------------------------------')
    W('(ti-menu-load-string "/solve/set/p-v-coupling 20")')
    W('')
    W('(ti-menu-load-string "/solve/set/discretization-scheme/mom 6")')
    W('')
    W('; ---------------------------------------------------------------------')
    W('; A MUDANCA DESTA RODADA: teto de mu_t/mu.')
    W(';')
    W('; /solve/set/limits pergunta os limites EM SEQUENCIA. Cada "," abaixo')
    W('; aceita o default de um prompt (Text Command List, 2.2). Sao %d antes'
      % a.limits_skip)
    W('; do teto (p_abs min, p_abs max, k min, omega min) -- ordem lida em')
    W('; sessao interativa do 25.2. O teto e o 5o e ULTIMO prompt.')
    W('; ---------------------------------------------------------------------')
    W('/solve/set/limits')
    for _ in range(a.limits_skip):
        W(',')
    W(num(a.visc_max))
    W('')
    W('')
    W('')
    W('; ECO DE CONFERENCIA: o mesmo comando, so com linhas em branco. Nao muda')
    W('; nada; imprime cada limite com o valor ATUAL entre colchetes. No log tem')
    W('; que aparecer:  Maximum allowable turbulent/laminar viscosity ratio [%d]'
      % int(round(a.visc_max)))
    W('; e TODOS os outros com o default de antes. O solve_csv.slurm confere.')
    W('/solve/set/limits')
    for _ in range(LIMITS_ECHO_BLANKS):
        W('')

    if a.autosave > 0:
        W('; ---------------------------------------------------------------------')
        W('; Autosave a cada %d passos, SO como ponto de restart. Reler case+data' % a.autosave)
        W('; noutra sessao devolveu campo PERMUTADO no job 5479. Se precisar,')
        W('; valide com check_field.py antes de confiar. O dado bom e o CSV.')
        W('; ---------------------------------------------------------------------')
        W('/file/auto-save/root-name "%s"' % os.path.join(D['dat_cas'], 'restart'))
        W('/file/auto-save/data-frequency %d' % a.autosave)
        W('/file/auto-save/case-frequency if-mesh-is-modified')

    W('/solve/set/time-step %g' % a.dt)
    W('/solve/initialize/hyb-initialization')
    W('; ---------------------------------------------------------------------')
    W('; REGISTRO DO QUE O SOLVER FEZ (handoff 3.5): summary no log, o mesmo')
    W('; summary em arquivo, e o .cas (so case, sem dados -- nada e relido).')
    W('; ---------------------------------------------------------------------')
    W('(ti-menu-load-string "/report/summary")')
    W('')
    W('; "write to files? [no]" -> yes ; "report summary filename" -> caminho')
    W('(ti-menu-load-string "/report/summary yes %s")'
      % os.path.join(D['meta'], 'report_summary.txt'))
    W('')
    W('/file/write-case "%s"' % os.path.join(D['meta'], 'setup.cas.h5'))
    W('(ti-menu-load-string "/parallel/timer/reset")')
    W('')
    W('; ---------------------------------------------------------------------')
    W('; MALHA ESTATICA -- coordenadas, distancia a parede e VOLUME, uma vez so,')
    W('; DENTRO desta sessao (alinhado celula a celula com os t_*.csv).')
    W('; cell-volume e novo: permite ponderar "% de celulas" por volume.')
    W('; ---------------------------------------------------------------------')
    for ln in bloco_export(os.path.join(D['csv'], 'static_mesh.csv'), CAMPOS_ESTATICOS):
        W(ln)
    W('')
    W('; ---------------------------------------------------------------------')
    W('; CONDICAO DE CONTORNO DO INLET, LIDA DO SOLVER (handoff 3.2).')
    W('; Valores nas FACES do inlet = o que foi imposto, antes de qualquer')
    W('; decaimento. Substitui a verificacao (c), que comparava a BC com')
    W('; valores ja decaidos a 0,2 m da entrada. Repetido no fim do solve.')
    W('; ---------------------------------------------------------------------')
    for ln in bloco_export(os.path.join(D['bc'], 'inlet_faces_init.csv'),
                           CAMPOS_INLET, superficie=a.inlet):
        W(ln)
    W('')
    W('; ---------------------------------------------------------------------')
    W('; LACO PRINCIPAL, desenrolado. Cada bloco: avanca %d passo(s), escreve o' % a.every)
    W('; t_*.csv; a cada %s passos tambem derivadas, e a cada %s y+/parede.'
      % (a.deriv_every or '-', a.wall_every or '-'))
    W('; ---------------------------------------------------------------------')

    passo = 0
    n_csv = n_der = n_wall = 0
    while passo < a.steps:
        n = min(a.every, a.steps - passo)
        passo += n
        W('; --- passo %05d ---' % passo)
        W('/solve/dual-time-iterate %d %d' % (n, a.inner))
        for ln in bloco_export(os.path.join(D['csv'], 't_%05d.csv' % passo), campos):
            W(ln)
        n_csv += 1
        if cruzou(passo, n, a.deriv_every):
            for ln in bloco_export(os.path.join(D['deriv'], 'd_%05d.csv' % passo),
                                   CAMPOS_DERIV):
                W(ln)
            n_der += 1
        if cruzou(passo, n, a.wall_every):
            for z in paredes:
                for ln in bloco_export(os.path.join(D['wall'], '%s_%05d.csv' % (z, passo)),
                                       CAMPOS_PAREDE, superficie=z):
                    W(ln)
            n_wall += 1
        W('')

    W('; ---------------------------------------------------------------------')
    W('; Inlet de novo no fim. k e omega nas faces sao o valor IMPOSTO e tem que')
    W('; bater com o inlet_faces_init.csv. turb-intensity e viscosity-ratio NAO')
    W('; batem: sao derivados com a celula vizinha e mostram o valor ja decaido')
    W('; (job 6124: mu_t/mu 9,24 e TI 1,70% no fim, com k e omega identicos).')
    W('; ---------------------------------------------------------------------')
    for ln in bloco_export(os.path.join(D['bc'], 'inlet_faces_end.csv'),
                           CAMPOS_INLET, superficie=a.inlet):
        W(ln)
    W('')
    W('(ti-menu-load-string "/parallel/timer/usage")')
    W('')
    W('/file/stop-transcript')
    W('/exit yes')

    with open(a.out, 'w', newline='\n') as fh:
        fh.write('\n'.join(L) + '\n')

    # --- resumo --------------------------------------------------------------
    gb = lambda ncol, nfiles: nfiles * a.ncells * ncol * BYTES_POR_VALOR / 1024 ** 3
    g_t = gb(4 + len(campos), n_csv)
    g_d = gb(4 + len(CAMPOS_DERIV), n_der)
    print('%s escrito: %d linhas' % (a.out, len(L)))
    print('  passos       : %d de %g s = %g s de tempo fisico'
          % (a.steps, a.dt, a.steps * a.dt))
    print('  teto mu_t/mu : %s   (%d prompt(s) de /solve/set/limits antes dele)'
          % (num(a.visc_max), a.limits_skip))
    print('  t_*.csv      : %d (1 a cada %d)  ~%.1f GB' % (n_csv, a.every, g_t))
    print('  d_*.csv      : %d (1 a cada %s)  ~%.1f GB'
          % (n_der, a.deriv_every or 'nunca', g_d))
    print('  wall         : %d instantes x %d zonas (%s)'
          % (n_wall, len(paredes), ' '.join(paredes)))
    print('  saida        : %s/{%s}' % (a.results, ','.join(SUBDIRS)))
    print('  malha        : %s' % a.mesh)
    print('  zonas        : inlet=%s outlet=%s moveis=%s'
          % (a.inlet, a.outlet, ','.join(moveis)))
    print('  (estimativa de disco: base %d celulas, ~%.1f bytes por valor)'
          % (a.ncells, BYTES_POR_VALOR))
    if not os.path.exists(a.mesh):
        print('  AVISO: malha nao encontrada em %s' % a.mesh)


if __name__ == '__main__':
    main()
