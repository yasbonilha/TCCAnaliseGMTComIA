#!/bin/bash
#SBATCH --partition=gpu-nvidia
#SBATCH --nodelist=gn01
# O gn01 tem 3 A40. --gres=gpu:a40:1 faz o SLURM entregar uma placa que
# nenhum outro job está usando: memória de GPU não é dividida. Com a sst3 e
# a siren1 ainda rodando, a ema1 fica com a terceira placa.
# O que os jobs dividem é a RAM e os CPUs do nó: cada um pede 128 GB e 16
# CPUs. Se não couberem os três pedidos, este job fica PENDENTE na fila,
# mas não derruba os outros. Conferir antes do sbatch:
#   scontrol show node gn01     (RealMemory / AllocMem, CPUTot / CPUAlloc)
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Mesmo limite da siren1. A rede de treino é a mesma (~35,9 s/época medidos
# na calibração da siren1, job 6434), e o EMA soma ao passo só uma média
# ponderada de 1,3 M de pesos. O custo novo é a validação das DUAS redes a
# cada VAL_EVERY épocas: pela estimativa, ~1% do tempo, mas isso NÃO está
# medido. A calibração (DEBUG_MODE = "calibration", Cell 15) imprime o
# tempo por época e a coluna val_time: refazer a conta com eles ANTES de
# submeter o "full". Sem a validação, 50.000 × 35,9 s ≈ 20,8 dias, mais
# ~30 min de leitura dos CSVs. Se a conta passar de ~25 dias, a folga até
# 27 dias fica abaixo de ~8% e o job corre o risco de morrer antes da
# Cell 16 (que salva os pesos finais e a metadata).
#SBATCH --time=27-00:00:00
#SBATCH --job-name=pinn_ema1
# logs/ só recebe o .out e o .err do SLURM. Tem que EXISTIR antes do sbatch:
# o SLURM abre estes arquivos antes de o script rodar, então um mkdir aqui
# dentro chegaria tarde demais (e com logs/ faltando o job perde o .out).
#   mkdir -p /home/tmoraes/treino_turb/10-29_ema/logs
#SBATCH --output=/home/tmoraes/treino_turb/10-29_ema/logs/pinn_ema1_%j.out
#SBATCH --error=/home/tmoraes/treino_turb/10-29_ema/logs/pinn_ema1_%j.err

echo "## Job iniciado em $(date)"
echo "## Node: $(hostname -s), Job ID: $SLURM_JOB_ID"
# Qual A40 coube a este job — para conferir que não é a da sst3 nem a da siren1
echo "## GPU(s) alocada(s): ${CUDA_VISIBLE_DEVICES:-nenhuma}"

# Setup ambiente
source $HOME/miniconda3/bin/activate
conda activate pinn_env
module load cuda/12.3

if [ -z "$CUDA_HOME" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
fi

export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)" 2>/dev/null))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export TF_CPP_MIN_LOG_LEVEL=2

# O script grava tudo com caminho ABSOLUTO (TRAIN_DIR/saida/, Cell 3), então
# o diretório de onde o sbatch é chamado não decide mais nada. O cd fica só
# para o cwd do Python ser previsível.
cd /home/tmoraes/treino_turb/10-29_ema

# Executa apenas o script Python
python -u /home/tmoraes/treino_turb/10-29_ema/pinn_ema_1029.py
rc=$?

# O código de saída tem que ser guardado ANTES do echo: o $(date) na mesma
# linha roda primeiro e zera o $?, e a versão antiga imprimia "exit code: 0"
# mesmo quando o Python falhava. O exit $rc faz o SLURM marcar o job como
# FAILED quando ele falha (sacct mostra o estado certo).
echo "## Job finalizado em $(date), exit code: $rc"
exit $rc
