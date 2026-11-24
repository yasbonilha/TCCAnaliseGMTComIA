#!/bin/bash
#SBATCH --partition=gpu-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=movie_cfd
#SBATCH --output=simulation_logs/movie_%j.out
#SBATCH --error=simulation_logs/movie_%j.err

# ---------------------------------------------------------------------------
# movie.slurm -- gera os videos do escoamento (generate_movie_cfd.py) com
# TODOS os snapshots da rodada new_sim_turb (--all-csv: 1000 quadros, 0,1 s a
# 100 s; a 24 fps, ~42 s de video).
#
#     cd /home/ybonilha/simulations_CFD/new_sim_turb
#     sbatch movie.slurm                      # pipeline completo
#     sbatch movie.slurm --skip-cache         # so replota (cache ja existe)
#     sbatch movie.slurm --skip-cache --cmap jet --scale both
#
# Argumentos extras vao direto para o generate_movie_cfd.py (que ja recebe
# --all-csv aqui). Saida: results/movie/video_cfd_{xy,yz}_global_scale.mp4
#
# RECURSOS
#   O script e serial (um processo), por isso 1 tarefa e poucos nucleos.
#   Memoria: a triangulacao de Delaunay da malha inteira (211.579 pontos,
#   ~1,4 milhao de tetraedros) fica residente o job todo, mais os
#   interpoladores e, na fase 2, uma amostra de todos os quadros -- 48 GB da
#   folga larga. Tempo: a fase 1 (ler 1000 CSVs de ~38 MB e interpolar dois
#   cortes) domina; a estimativa e de algumas horas, e 12 h da margem. Se o
#   job cair no meio, rode de novo: quadros ja interpolados ficam em cache e
#   sao pulados.
# ---------------------------------------------------------------------------

set -u
WORKDIR=/home/ybonilha/simulations_CFD/new_sim_turb
EXTRA="$*"

echo "## movie iniciado $(date) em $(hostname -s), job ${SLURM_JOB_ID:-?}"
echo "## argumentos extras: ${EXTRA:-nenhum}"
cd "$WORKDIR" || { echo "## ERRO: nao consegui entrar em $WORKDIR"; exit 1; }
mkdir -p simulation_logs

[ -f generate_movie_cfd.py ] || { echo "## ERRO: generate_movie_cfd.py nao esta em $WORKDIR"; exit 1; }
N_CSV=$(ls -1 results/csv/t_*.csv 2>/dev/null | wc -l)
echo "## snapshots em results/csv: $N_CSV"
[ "$N_CSV" -gt 0 ] || { echo "## ERRO: nenhum t_*.csv em results/csv"; exit 1; }

# --- ambiente ----------------------------------------------------------------
# set -- antes do source: o activate do conda le os parametros posicionais.
set --
source $HOME/miniconda3/bin/activate
conda activate pinn_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# numpy/scipy: usar os nucleos pedidos, sem disputar com outros jobs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MPLBACKEND=Agg

# ffmpeg: sem ele as fases 1-3 rodam, mas o video nao e montado.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "## AVISO: ffmpeg nao encontrado no pinn_env. Os quadros (PNG) serao"
    echo "##   gerados, mas os videos nao. Instale e recompile so os videos com:"
    echo "##     conda install -n pinn_env -c conda-forge ffmpeg"
    echo "##     sbatch movie.slurm --skip-cache --skip-frames"
fi

df -h . | tail -1 | sed 's/^/## disco: /'

# --- execucao ----------------------------------------------------------------
T0=$SECONDS
# python -u: log sem buffer, para acompanhar com tail -f simulation_logs/movie_<job>.out
python3 -u generate_movie_cfd.py --all-csv $EXTRA
RC=$?
DT=$(( SECONDS - T0 ))
echo
echo "## generate_movie_cfd.py: $(( DT / 3600 ))h$(( DT % 3600 / 60 ))min (codigo $RC)"

echo "## videos:"
ls -lh results/movie/*.mp4 2>/dev/null | sed 's/^/##   /' || echo "##   nenhum"
echo "## terminado $(date), codigo $RC"
exit $RC
