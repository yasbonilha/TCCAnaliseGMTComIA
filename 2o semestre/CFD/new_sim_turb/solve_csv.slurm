#!/bin/bash
#SBATCH --partition=cpu-9654
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=new_turb
#SBATCH --output=simulation_logs/new_turb_%j.out
#SBATCH --error=simulation_logs/new_turb_%j.err

# ---------------------------------------------------------------------------
# solve_csv.slurm -- rodada new_sim_turb: teto de mu_t/mu em 1e7, CSV escrito
# de dentro do solve, mais derivadas, y+, BC do inlet e registro do solver.
#
#     cd /home/ybonilha/simulations_CFD/new_sim_turb
#     sbatch solve_csv.slurm 20             # TESTE      -> results_test/
#     sbatch solve_csv.slurm 1000           # producao   -> results/
#     sbatch solve_csv.slurm 1000 1 --wall-every 1   # opcoes extras vao ao gerador
#
# ORDEM OBRIGATORIA: primeiro o de 20 passos. A producao SE RECUSA a rodar
# enquanto results_test/ nao provar que o teto pegou (ver "PORTAO" abaixo).
# Para pular o portao conscientemente:  FORCE=1 sbatch solve_csv.slurm 1000
#
# O QUE O TESTE DE 20 PASSOS TEM QUE MOSTRAR (secao VERIFICACAO, no fim do .out)
#   1. TETO: o eco do /solve/set/limits com
#        Maximum allowable turbulent/laminar viscosity ratio [10000000]
#      e o report_summary.txt com "Maximum Turb. Viscosity Ratio 1e+07".
#      Os outros quatro limites com os defaults (1, 5e10, 1e-14, 1e-20).
#   2. ARQUIVOS: 20 t_, 20 d_, 2 instantes x 6 zonas de parede, 2 do inlet.
#   3. SIMPLE: iteracoes por passo bem abaixo de 50 (antes: 13,8).
#   4. check_field.py: estagnacao na frente do cilindro (~+177 Pa).
#   5. DERIVADAS: S calculado das 9 derivadas contra strain-rate-mag.
#
# --time=24:00:00. A rodada anterior (so os t_, 211.579 celulas) cabia em 12 h
# com folga de ~2x sobre a projecao de ~5,4 h de escrita. Agora as derivadas
# saem em TODO passo e mais que dobram a escrita (11 -> 25 colunas por passo),
# e mu_t 100x maior pode pedir mais iteracoes do SIMPLE. 24 h mantem a mesma
# folga. Se a particao nao aceitar 24 h, o sbatch recusa na hora -- nao a meio
# caminho. O job termina quando termina; o limite nao custa nada.
#
# --ntasks=32 igual aos jobs 5187/5518, para o custo continuar comparavel.
# ---------------------------------------------------------------------------

set -u

WORKDIR=/home/ybonilha/simulations_CFD/new_sim_turb
MESH=$WORKDIR/geometria_malha/malha_simples_boi.msh.h5
STEPS=${1:-1000}
EVERY=${2:-1}
shift 2 2>/dev/null || shift $#
EXTRA="$*"          # repassado ao gerador: --wall-every, --mesh, --inlet...
INNER=50            # iteracoes internas maximas por passo (igual ao gerador)

if [ "$STEPS" -le 50 ]; then
    RESULTS=results_test
    JOURNAL=JOURNAL_solve_csv_TEST
else
    RESULTS=results
    JOURNAL=JOURNAL_solve_csv
fi
TRANSCRIPT=$RESULTS/meta/transcript.txt

echo "## new_sim_turb iniciado $(date) em $(hostname -s), job ${SLURM_JOB_ID:-?}"
echo "## passos: $STEPS   t_ a cada: $EVERY   saida: $RESULTS   extra: ${EXTRA:-nenhum}"

cd "$WORKDIR" || { echo "## ERRO: nao consegui entrar em $WORKDIR"; exit 1; }
mkdir -p simulation_logs

[ -f "$MESH" ] || { echo "## ERRO: malha nao encontrada em $MESH"; ls -la geometria_malha; exit 1; }
ls -la "$MESH" | sed 's/^/## malha: /'
for f in make_solve_csv_journal.py check_journal.py check_field.py; do
    [ -f "$f" ] || { echo "## ERRO: $f nao esta em $WORKDIR"; exit 1; }
done

# --- PORTAO: producao so depois de um teste que provou o teto ----------------
# O teto e a UNICA mudanca de fisica desta rodada, e ele entra por uma
# sequencia de prompts. Se a contagem estivesse errada, 1000 passos sairiam
# com o teto antigo -- ou com omega minimo = 1e7 -- sem erro nenhum na tela.
# O teste de 20 passos custa minutos; descobrir isso depois custa o dia.
TETO_RE='turbulent/laminar viscosity ratio \[(10000000|1e\+?0?7|1(\.0+)?e\+?0?7)\]'
if [ "$RESULTS" = results ] && [ "${FORCE:-0}" != 1 ]; then
    if ! grep -qE "$TETO_RE" results_test/meta/transcript.txt 2>/dev/null; then
        echo "## ERRO: results_test/meta/transcript.txt nao mostra o teto em 1e7."
        echo "##   Rode antes:  sbatch solve_csv.slurm 20   e leia a VERIFICACAO."
        echo "##   Para pular conscientemente:  FORCE=1 sbatch solve_csv.slurm $STEPS"
        exit 1
    fi
    echo "## portao: results_test mostrou o teto em 1e7 -- seguindo para producao"
fi

# --- limpeza -----------------------------------------------------------------
# Arquivo que ja existe faz o Fluent perguntar "OK to overwrite?" e comer a
# linha seguinte do journal como resposta (job 5239). E CSV velho misturado
# com novo passa despercebido.
#   TESTE: apaga o que houver -- e descartavel por definicao.
#   PRODUCAO: NAO apaga. Sao dezenas de GB e pode ser a unica copia de uma
#   rodada boa; apagar dentro de um batch e o tipo de coisa que da errado calado.
N_OLD=$(find "$RESULTS" -type f 2>/dev/null | wc -l)
if [ "$N_OLD" -gt 0 ]; then
    if [ "$RESULTS" = results_test ]; then
        echo "## Removendo $N_OLD arquivo(s) antigo(s) de $RESULTS"
        rm -rf "$RESULTS"
    else
        echo "## ERRO: $RESULTS/ ja tem $N_OLD arquivo(s) ($(du -sh "$RESULTS" | cut -f1))."
        echo "##   Renomeie ou apague voce mesmo antes de rodar a producao:"
        echo "##     mv $WORKDIR/$RESULTS $WORKDIR/${RESULTS}_old_\$(date +%Y%m%d)"
        exit 1
    fi
fi

echo "## Espaco livre:"; df -h . | tail -1

# --- journal -----------------------------------------------------------------
echo "## Gerando o journal (o gerador cria as subpastas de $RESULTS):"
python3 make_solve_csv_journal.py --steps "$STEPS" --every "$EVERY" \
    --inner "$INNER" --results "$RESULTS" --out "$JOURNAL" --mesh "$MESH" $EXTRA \
    || { echo "## ERRO: gerador falhou"; exit 1; }

echo "## Conferindo o journal:"
python3 check_journal.py "$JOURNAL" || {
    echo "## ERRO: check_journal.py reprovou. Nao vou submeter ao Fluent."
    exit 1
}

# --- registro: o que rodou fica junto do dado (handoff 3.5) ------------------
# $0 dentro de um job SLURM e a copia do script que o sbatch guardou, ou seja,
# exatamente o que esta rodando -- mesmo que o arquivo na pasta mude depois.
cp make_solve_csv_journal.py check_journal.py check_field.py "$JOURNAL" "$RESULTS/meta/"
cp "$0" "$RESULTS/meta/solve_csv.slurm" 2>/dev/null
{
    echo "job       ${SLURM_JOB_ID:-?}"
    echo "inicio    $(date -Iseconds)"
    echo "no        $(hostname -s)"
    echo "ranks     ${SLURM_NTASKS:-32}"
    echo "passos    $STEPS   every $EVERY   inner $INNER"
    echo "extra     ${EXTRA:-nenhum}"
    echo "malha     $MESH"
    echo "md5 malha $(md5sum "$MESH" | cut -d' ' -f1)"
} > "$RESULTS/meta/run_info.txt"

# --- ambiente ----------------------------------------------------------------
# set -- antes do source: o activate do conda le os parametros posicionais
# deste script e tentaria ativar um ambiente chamado "$1".
set --
source $HOME/miniconda3/bin/activate
conda activate pinn_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
module load ansys/v25.2
for dd in /opt/ohpc/pub/libs \
          /opt/ohpc/pub/apps/ansys_inc/v252/CEI/apex252/machines/linux_2.6_64/Mesa-13.0.5/lib; do
    [ -e "$dd/libGL.so.1" ] && { export LD_LIBRARY_PATH=$dd:$LD_LIBRARY_PATH; break; }
done

NPROC=${SLURM_NTASKS:-32}
echo "## Rodando Fluent com -t$NPROC"
T0=$SECONDS
# < /dev/null: journal abortado nao deixa o Fluent preso lendo stdin (job 5293).
fluent 3ddp -gu -driver null -t$NPROC -i "$WORKDIR/$JOURNAL" < /dev/null
RC=$?
DT=$(( SECONDS - T0 ))
echo "## Fluent: ${DT}s = $(( DT / 3600 ))h$(( DT % 3600 / 60 ))min (codigo $RC)"
echo "fim       $(date -Iseconds)   fluent ${DT}s   codigo $RC" >> "$RESULTS/meta/run_info.txt"

# ===========================================================================
echo
echo "############################################################"
echo "## VERIFICACAO"
echo "############################################################"
OK=1

# --- 1. teto -----------------------------------------------------------------
echo
echo "## 1. TETO DE mu_t/mu"
echo "##   prompts do /solve/set/limits no transcript (o 1o bloco define, o 2o ecoa):"
grep -E 'allowable (absolute pressure|k |omega|turbulent/laminar)' "$TRANSCRIPT" 2>/dev/null \
    | sed 's/^/##     /'
if grep -qE "$TETO_RE" "$TRANSCRIPT" 2>/dev/null; then
    echo "##   [ok]    eco mostra o teto em 1e7"
else
    echo "##   [FALHA] eco NAO mostra o teto em 1e7 -- a rodada NAO serve"
    OK=0
fi
echo "##   report_summary.txt:"
grep -iE 'Absolute Pressure|Turb. Kinetic|Spec. Dissipation|Viscosity Ratio' \
    "$RESULTS/meta/report_summary.txt" 2>/dev/null | sed 's/^/##     /' \
    || echo "##     (report_summary.txt nao encontrado)"
if grep -qiE 'Maximum Turb. Viscosity Ratio +(10000000|1e\+0?7)' \
        "$RESULTS/meta/report_summary.txt" 2>/dev/null; then
    echo "##   [ok]    summary confirma 1e7"
else
    echo "##   [FALHA] summary nao confirma 1e7"
    OK=0
fi
# O Fluent avisa quando o limitador age: "turbulent viscosity limited to
# viscosity ratio of X in N cells". Previsao do handoff: ~0 celulas com 1e7.
N_LIM=$(grep -ciE 'viscosity limited to viscosity ratio' "$TRANSCRIPT" 2>/dev/null)
echo "##   avisos de mu_t limitado pelo teto: ${N_LIM:-0} (previsao: ~0)"
grep -iE 'viscosity limited to viscosity ratio' "$TRANSCRIPT" 2>/dev/null \
    | tail -3 | sed 's/^/##     /'

# --- 2. arquivos -------------------------------------------------------------
echo
echo "## 2. ARQUIVOS"
N_T=$(ls -1 "$RESULTS"/csv/t_*.csv 2>/dev/null | wc -l)
N_D=$(ls -1 "$RESULTS"/deriv/d_*.csv 2>/dev/null | wc -l)
N_W=$(ls -1 "$RESULTS"/wall/*.csv 2>/dev/null | wc -l)
N_B=$(ls -1 "$RESULTS"/bc/*.csv 2>/dev/null | wc -l)
N_ESP=$(( (STEPS + EVERY - 1) / EVERY ))
echo "##   t_*.csv  : $N_T   (esperado $N_ESP)"
echo "##   d_*.csv  : $N_D   (esperado $N_ESP, salvo --deriv-every)"
echo "##   wall     : $N_W   (6 zonas x instantes de --wall-every)"
echo "##   bc inlet : $N_B   (esperado 2: init e end)"
echo "##   static   : $(ls "$RESULTS"/csv/static_mesh.csv 2>/dev/null || echo FALTA)"
echo "##   meta     : $(ls "$RESULTS"/meta | tr '\n' ' ')"
echo "##   disco    : $(du -sh "$RESULTS" | cut -f1)"
[ "$N_T" -eq "$N_ESP" ] || OK=0

# --- 3 a 5: analise em python ------------------------------------------------
echo
python3 - "$RESULTS" "$TRANSCRIPT" "$INNER" <<'PY' || OK=0
import glob, os, re, sys
import numpy as np
import pandas as pd

res, transcript, inner = sys.argv[1], sys.argv[2], int(sys.argv[3])
falhou = False

def ler(p):
    d = pd.read_csv(p, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    return d

# ---- 3. iteracoes do SIMPLE por passo ----------------------------------------
# Linha de iteracao: "   157  2.3456e-03  1.2e-04 ..." (6 residuos no SST);
# fim do passo: "Flow time =". Exigir >= 3 residuos exclui as linhas da
# hyb-initialization ("   1   1.000000e+00"), que tem um so.
# Iteracoes do passo = ultimo numero de iteracao do passo MENOS o do passo
# anterior. Contar linhas erra por 1: o Fluent reimprime, no inicio de cada
# passo, a ultima iteracao do passo anterior (visto no job 6124).
it_re = re.compile(r"^\s*(\d+)(\s+\d(\.\d+)?e[-+]\d+){3,}")
por_passo, ult, ant = [], None, 0
try:
    for ln in open(transcript, errors='replace'):
        mt = it_re.match(ln)
        if mt:
            ult = int(mt.group(1))
        elif 'Flow time' in ln and ult is not None:
            por_passo.append(ult - ant)
            ant = ult
except OSError:
    print('##   transcript nao encontrado')
if por_passo:
    a = np.array(por_passo)
    print('##   passos: %d   media %.1f   p50 %d   p95 %d   max %d'
          % (len(a), a.mean(), np.median(a), np.quantile(a, .95), a.max()))
    no_teto = (a >= inner).sum()
    print('##   passos que bateram no teto de %d: %d (%.1f%%)'
          % (inner, no_teto, 100 * no_teto / len(a)))
    if no_teto:
        print('##   [ATENCAO] passo que bate no teto NAO convergiu -- o dado desse')
        print('##   instante e de qualidade inferior. Ver handoff 2, risco 1.')
    print('##   ultimos 10 passos: %s' % ' '.join(map(str, a[-10:])))
else:
    print('##   nao achei linhas de iteracao no transcript (formato mudou?)')

# ---- inlet: BC lida do solver ----------------------------------------------
# k e omega nas faces do inlet SAO o valor imposto (constantes no tempo).
# turb-intensity e viscosity-ratio NAO: sao campos derivados, e na face o
# Fluent os calcula com a celula vizinha. No init coincidem (campo uniforme);
# depois do solve refletem o valor JA DECAIDO junto a entrada (job 6124:
# mu_t/mu 9,24 e TI 1,70% com k e omega identicos ao imposto). Por isso a BC
# sai de k e omega: TI = sqrt(2k/3)/17 e mu_t/mu = rho*k/(omega*mu).
print('\n## INLET -- valores impostos, lidos nas faces (handoff 3.2)')
RHO, MU, U = 1.225, 1.7894e-5, 17.0
for nome in ('inlet_faces_init.csv', 'inlet_faces_end.csv'):
    p = os.path.join(res, 'bc', nome)
    if not os.path.exists(p):
        print('##   %s: FALTA' % nome); falhou = True; continue
    d = ler(p)
    k, w = d['turb-kinetic-energy'].median(), d['specific-diss-rate'].median()
    print('##   %-22s faces %5d  k %.6g  omega %.6g  -> TI %.4f%%  mu_t/mu %.4f'
          '  (v %.4g; dispersao de k %.1e)'
          % (nome, len(d), k, w, 100 * np.sqrt(2 * k / 3) / U, RHO * k / (w * MU),
             d['y-velocity'].median(), d['turb-kinetic-energy'].std()))
    print('##   %-22s colunas derivadas (celula vizinha): TI %.4f%%  mu_t/mu %.4f'
          % ('', 100 * d['turb-intensity'].median() / U, d['viscosity-ratio'].median()))
print('##   esperado (de k e omega): TI 5%, mu_t/mu 10, v 17. As colunas derivadas')
print('##   so batem no init; no end mostram o valor ja decaido junto a entrada.')

# ---- 5. derivadas: S das 9 componentes contra strain-rate-mag ----------------
print('\n## 5. DERIVADAS -- S = sqrt(2 Sij Sij) das 9 componentes vs strain-rate-mag')
ds = sorted(glob.glob(os.path.join(res, 'deriv', 'd_*.csv')))
if ds:
    d = ler(ds[-1])
    g = np.empty((len(d), 3, 3))
    for i, a in enumerate('xyz'):
        for j, b in enumerate('xyz'):
            g[:, i, j] = d['d%s-velocity-d%s' % (a, b)].values
    Sij = 0.5 * (g + g.transpose(0, 2, 1))
    S = np.sqrt(2 * (Sij ** 2).sum(axis=(1, 2)))
    F = d['strain-rate-mag'].values
    m = F > 1e-8
    r = S[m] / F[m]
    print('##   %s: %d celulas' % (os.path.basename(ds[-1]), len(d)))
    print('##   S_calc / strain-rate-mag: p1 %.6f  p50 %.6f  p99 %.6f'
          % tuple(np.quantile(r, [.01, .5, .99])))
    if abs(np.median(r) - 1) < 1e-3:
        print('##   [ok] strain-rate-mag do Fluent E sqrt(2 Sij Sij) -- o S do')
        print('##   limitador do SST pode ser lido direto da coluna.')
    else:
        print('##   [ATENCAO] definicao diferente; razao mediana %.4f. Use o S'
              % np.median(r))
        print('##   calculado das 9 derivadas, nao a coluna.')
    t_mesmo = os.path.join(res, 'csv', os.path.basename(ds[-1]).replace('d_', 't_'))
    if os.path.exists(t_mesmo):
        t = ler(t_mesmo)
        alinh = (np.array_equal(t['cellnumber'].values, d['cellnumber'].values) and
                 np.allclose(t['x-coordinate'].values, d['x-coordinate'].values))
        print('##   d_ e t_ do mesmo passo alinhados linha a linha: %s'
              % ('sim' if alinh else 'NAO'))
        falhou |= not alinh
else:
    print('##   nenhum d_*.csv')

# ---- parede ------------------------------------------------------------------
print('\n## PAREDE -- y+ no ultimo instante exportado')
ws = sorted(glob.glob(os.path.join(res, 'wall', '*.csv')))
if ws:
    ult = max(re.search(r'_(\d+)\.csv$', p).group(1) for p in ws)
    for p in sorted(x for x in ws if x.endswith('_%s.csv' % ult)):
        d = ler(p)
        print('##   %-30s faces %6d  y+ p50 %9.3g  max %9.3g  tau_w p50 %9.3g'
              % (os.path.basename(p), len(d), d['y-plus'].median(),
                 d['y-plus'].max(), d['wall-shear'].median()))

# ---- volume no static_mesh ---------------------------------------------------
p = os.path.join(res, 'csv', 'static_mesh.csv')
if os.path.exists(p):
    d = ler(p)
    V = d['cell-volume']
    print('\n## static_mesh.csv: %d celulas, volume total %.4g m3 '
          '(caixa 280x700x190 = %.4g, menos o cilindro)'
          % (len(d), V.sum(), 280 * 700 * 190))
    if (V <= 0).any():
        print('##   [FALHA] %d celulas com volume <= 0' % (V <= 0).sum())
        falhou = True

sys.exit(1 if falhou else 0)
PY

# --- 4. coerencia do campo ---------------------------------------------------
ULTIMO=$(ls -1 "$RESULTS"/csv/t_*.csv 2>/dev/null | sort -V | tail -1)
if [ -n "${ULTIMO:-}" ]; then
    echo
    echo "## 4. check_field.py no static_mesh e no ultimo passo ($ULTIMO)"
    python3 check_field.py "$RESULTS/csv/static_mesh.csv" || OK=0
    python3 check_field.py "$ULTIMO" || OK=0
fi

echo
echo "############################################################"
if [ "$OK" = 1 ] && [ "$RC" = 0 ]; then
    echo "## RESULTADO: tudo conferiu."
    [ "$RESULTS" = results_test ] && echo "## Pode rodar a producao: sbatch solve_csv.slurm 1000"
else
    echo "## RESULTADO: ha [FALHA] ou codigo de saida $RC acima -- leia antes de seguir."
fi
echo "############################################################"
echo "## terminado $(date), codigo $RC"
exit $RC
