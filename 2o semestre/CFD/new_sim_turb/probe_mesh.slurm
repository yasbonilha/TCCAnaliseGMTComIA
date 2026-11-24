#!/bin/bash
#SBATCH --partition=gpu-amd
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=teste10
#SBATCH --output=simulation_logs/teste10_%j.out
#SBATCH --error=simulation_logs/teste10_%j.err

# ---------------------------------------------------------------------------
# probe_mesh.slurm -- inspeciona a malha nova antes de escrever o journal.
#
#     cd /home/ybonilha/simulations_CFD/simples_turbulenta
#     sbatch probe_mesh.slurm
#     sbatch probe_mesh.slurm /caminho/para/outra.msh.h5
#
# NAO RODA SOLVER. Le a malha, imprime zonas, contagem de celulas, extensao e
# qualidade, e projeta custo de disco e de escrita para varias cadencias.
# Leva segundos.
#
# POR QUE ANTES E NAO JUNTO
# Duas razoes, e as duas ja custaram caro neste projeto:
#
#  - As BCs sao aplicadas por NOME de zona. Se o Ansys renomeou alguma ao gerar
#    a malha com body of influence, o comando falha, deixa um prompt pendente,
#    e o Fluent come a proxima linha do journal como resposta. Tudo depois anda
#    um lugar sem nenhuma mensagem de erro.
#
#  - O numero de celulas decide a cadencia de gravacao, e a decisao ficou
#    apertada: na malha antiga (39.060 celulas) cada CSV custava 7 MB e 3,6 s.
#    Refinar por 10x levaria 1000 passos a ~70 GB e ~10 h so de escrita.
#    Escolher --every sem esse numero e chute.
# ---------------------------------------------------------------------------

set -u

WORKDIR=/home/ybonilha/simulations_CFD/new_sim_turb/
MESH=${1:-$WORKDIR/geometria_malha/malha_simples_boi.msh.h5}

echo "## PROBE de malha iniciado $(date) em $(hostname -s), job $SLURM_JOB_ID"
echo "## malha: $MESH"

cd "$WORKDIR" || { echo "## ERRO: nao consegui entrar em $WORKDIR"; exit 1; }
mkdir -p simulation_logs

if [ ! -f "$MESH" ]; then
    echo "## ERRO: malha nao encontrada em $MESH"
    echo "##   Conteudo de $(dirname "$MESH"):"
    ls -la "$(dirname "$MESH")" 2>/dev/null | sed 's/^/##   /'
    exit 1
fi
ls -la "$MESH" | sed 's/^/## /'

[ -f "$WORKDIR/PROBE_mesh" ] || { echo "## ERRO: PROBE_mesh nao esta em $WORKDIR"; exit 1; }
# sed 's/\r$//' porque os arquivos costumam vir do Windows por copia e cola, e
# um \r numa linha em branco faz ela deixar de ser vazia -- exatamente o que
# desalinha os prompts.
sed -e 's/\r$//' -e "s|@MESH@|$MESH|" "$WORKDIR/PROBE_mesh" > "$WORKDIR/PROBE_mesh.jou"

set --
source $HOME/miniconda3/bin/activate
conda activate pinn_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
module load ansys/v25.2
for dd in /opt/ohpc/pub/libs \
          /opt/ohpc/pub/apps/ansys_inc/v252/CEI/apex252/machines/linux_2.6_64/Mesa-13.0.5/lib; do
    [ -e "$dd/libGL.so.1" ] && { export LD_LIBRARY_PATH=$dd:$LD_LIBRARY_PATH; break; }
done

echo
echo "############################################################"
echo "## FLUENT"
echo "############################################################"
LOG=$WORKDIR/simulation_logs/probe_mesh_${SLURM_JOB_ID}.txt
# < /dev/null: journal abortado nao encerra o Fluent, ele cai no prompt
# interativo e fica lendo stdin para sempre (foi o que consumiu o job 5293).
timeout -k 30 600 fluent 3ddp -gu -driver null -t${SLURM_NTASKS:-4} \
    -i "$WORKDIR/PROBE_mesh.jou" < /dev/null 2>&1 | tee "$LOG"
echo "## fluent terminou com codigo ${PIPESTATUS[0]}"

# --- resumo legivel ---------------------------------------------------------
echo
echo "############################################################"
echo "## RESUMO"
echo "############################################################"

NCELL=$(grep -oE '^ *[0-9]+ cells,' "$LOG" | grep -oE '[0-9]+' | head -1)
echo "## celulas : ${NCELL:-NAO ENCONTRADO}"
echo
echo "## ZONAS (confira os nomes contra os que o journal usa):"
grep -E '^ +[0-9]+ (polygonal|polyhedral|triangular|quadrilateral).* faces,  zone id' "$LOG" \
    | sed 's/^/##   /'
echo
echo "##   O journal de solve aplica BC nestes nomes (malha com BOI):"
echo "##     inlet  outlet  lateral_min  lateral_max  top"
echo "##   ground, cylinder_wall e cylinder_top ficam no default (parede fixa)."
for Z in inlet outlet lateral_min lateral_max top; do
    if grep -qE "^\s+$Z\s*$|$Z faces" "$LOG"; then
        echo "##     [ok]     $Z encontrado"
    else
        echo "##     [FALTA]  $Z NAO encontrado -- passe o nome certo ao gerador:"
        echo "##                make_solve_csv_journal.py --inlet X --outlet Y --moving-wall Z"
    fi
done

echo
echo "## EXTENSAO DO DOMINIO (do /mesh/check):"
grep -A3 -iE 'domain extents|x-coordinate:' "$LOG" | sed 's/^/##   /' | head -12

echo
echo "## AVISOS DE QUALIDADE (vazio e bom sinal):"
grep -iE 'negative|degenerate|left-handed|WARNING.*mesh|failed' "$LOG" | sed 's/^/##   /' | head -10

# --- projecao de custo ------------------------------------------------------
# Escala medida na malha antiga (job 5495): 39.060 celulas -> 7.070.046 bytes e
# 3,6 s por CSV. Isso da ~181 bytes e ~92 us por celula por arquivo. A escala e
# linear em primeira aproximacao; o tempo real tem tambem um custo fixo de
# abertura de arquivo, entao a projecao tende a SUPERESTIMAR malhas pequenas e
# subestimar um pouco as grandes.
if [ -n "${NCELL:-}" ]; then
    echo
    echo "## PROJECAO PARA 1000 PASSOS (base: 39.060 cel -> 7,07 MB e 3,6 s por CSV)"
    python3 - "$NCELL" <<'PY'
import sys
n = int(sys.argv[1])
b_cell = 7070046 / 39060.0
s_cell = 3.6 / 39060.0
mb = n * b_cell / 1024**2
s  = n * s_cell
print("##   %d celulas  ->  %.1f MB e %.1f s por CSV" % (n, mb, s))
print("##")
print("##   %-8s %10s %14s" % ("--every", "arquivos", "disco"))
print("##   %-8s %10s %14s %14s" % ("", "", "", "tempo de escrita"))
for e in (1, 2, 5, 10, 20, 50):
    k = 1000 // e
    tot_gb = k * mb / 1024
    tot_h = k * s / 3600
    print("##   %-8d %10d %11.1f GB %11.1f h" % (e, k, tot_gb, tot_h))
print("##")
print("##   Criterio fisico para a cadencia: o desprendimento de vortices tem")
print("##   f = St*U/D = 0,2*17/53 ~ 0,064 Hz, ou seja periodo de ~15,6 s.")
print("##   Com dt = 0,1 s:  --every 5 da 31 amostras por ciclo; --every 10 da 15;")
print("##   --every 20 da 8. Abaixo de ~8 por ciclo o vortice comeca a serrilhar.")
PY
fi

cat <<'EOF'

## ==========================================================================
## PROXIMOS PASSOS
##   1. Confira os nomes de zona acima. Se algum divergir, me diga qual e o
##      nome novo -- o gerador aceita --inlet / --outlet / --moving-wall.
##   2. Com o numero de celulas e a tabela de custo, a gente escolhe --every.
##   3. Ai roda: sbatch solve_csv.slurm 20   (teste de 20 passos)
##      e so depois a producao.
## ==========================================================================
EOF
echo "## PROBE terminado $(date)"
