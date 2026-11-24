#!/usr/bin/env python3
"""
inferencia_sst.py — Inferência + plots da PINN k-ω SST

Adaptado de inferencia_sem2.py (versão laminar). O que mudou, e por quê:

  1. SEIS saídas (u, v, w, P, k, ω), não quatro. k e ω saem em espaço LOG:
         k = exp(kn·LNK_DP + LNK_MEAN) − K_EPS
         ω = exp(ωn·LNW_DP + LNW_MEAN)
     Desnormalizar k e ω com z-score comum devolve lixo silenciosamente.

  2. CONSTANTES DE NORMALIZAÇÃO DIFERENTES. O treino SST rodou com
     AUTO_NORM_STATS ligado e MEDIU as constantes no próprio dataset. As do
     script laminar (V_MEAN=14.322, P_MEAN=−32.789, …) são de outra malha e
     estão erradas aqui — até 8,5% em P e 196% em LNK_MEAN. Ver o bloco
     CONSTANTS abaixo, que traz a proveniência de cada número.

  3. HardConstraintLayer REESCRITA. A do laminar diverge da que a rede usou
     em quatro pontos, e cada um deles muda o campo previsto:
         - CYL_RIM_FIX: max duro com as duas folgas na mesma escala, em vez
           do smooth_max. Sem isso D_cyl ≈ 0,98 na quina do cilindro, ou
           seja, quase nenhuma restrição justo onde o escoamento separa;
         - saídas de k e ω, com a BC de inlet por mistura convexa;
         - tf.stack + reshape no lugar do tf.concat (mesmo valor; o concat
           quebra o XLA no treino — aqui é só para ficar idêntico);
         - smooth_max com k=75, não 50.

  4. Plots de turbulência: k, ω e ν_t, em escala log, contra as colunas
     turb-kinetic-energy, specific-diss-rate e viscosity-turb do CSV — que
     o laminar não tinha. ν_t da PINN sai da fórmula SST INTEIRA (S por
     autodiff, F2, limitador de cisalhamento e teto), não de k/ω.

  5. CSV do Fluent com VÍRGULA e 11 colunas; arquivos t_00002.csv. O
     laminar lia sep=r'\\s+' e timestep-XXXX.csv.

  6. training_log_sst.csv tem 47 colunas com nomes novos (total, data,
     phys_sum, cont, mom_u…, k, omega, val_mse4, val_mse6, val_aligned,
     frac_*, mu_t_p50). Os do laminar (loss_total, loss_data, …) não
     existem mais.

  7. [19/10/2026] PERFIS DE RODADA (sst3). Caminhos, constantes de
     normalização e teto de ν_t deixaram de ser globais soltos e viraram um
     PERFIL por RUN_TAG (seção 1). O perfil é escolhido pelo NOME do
     checkpoint, como a BC do inlet já era. Motivo: na sst3 os três mudaram
     juntos — dado do CFD novo (job 6125), constantes medidas nele (job
     PINN 6389) e teto 1e7 —, e qualquer um esquecido dá figura errada SEM
     erro (ver o cabeçalho da seção 1). Saída em 10-18_simples/
     inferencia_<tag>/, fora de saida/, que é só do treino.

  8. [27/10/2026] RODADA siren1 (camadas SIREN, pasta 10-19_siren/). O
     perfil passa a dizer também a ARQUITETURA (tanh ou SIREN + ω0) e a
     PASTA DA RODADA, porque a siren1 treinou em outra pasta. A seção 7
     ganhou uma cópia da SirenLayer da Cell 9 do treino, e o metadata
     confere tipo e ω0 contra o perfil (seção 4). Ver a seção 1 para o
     motivo: uma SIREN com o ω0 errado carrega os pesos sem erro nenhum.
     Dado, constantes, BCs e teto são os da sst3 — mesmo CFD (job 6125).

AUTOCONFERÊNCIA: o script roda, antes de qualquer plot, o mesmo teste de
fronteira da Cell 9 do treino (u=v=w=0 nas paredes, P=0 no outlet, k e ω
nos valores de inlet). Se a cópia da HardConstraintLayer divergir da do
treino, ele PARA aqui em vez de gerar 40 figuras erradas. Foi exatamente
esse o defeito da réplica do laminar. O que ele NÃO pega: constantes de
normalização erradas (a BC é montada e lida com as mesmas constantes, então
qualquer conjunto passa). Isso é responsabilidade do perfil e do metadata.

Uso:
    python inferencia_sst.py --list                   # checkpoints da sst3
    python inferencia_sst.py                          # checkpoint mais recente
    python inferencia_sst.py --checkpoint pinn_sst3_000200.weights.h5
    python inferencia_sst.py --checkpoint pinn_sst3_cal_best.weights.h5
    python inferencia_sst.py --run-tag sst3_cal --list
    python inferencia_sst.py --run-tag siren1 --list  # checkpoints da siren1
    python inferencia_sst.py --run-tag siren1         # siren1, mais recente
    python inferencia_sst.py --checkpoint pinn_siren1_005000.weights.h5
    python inferencia_sst.py --no-slices --no-nut     # rápido
    python inferencia_sst.py --selftest-only
O metadata da Cell 16 (pinn_<tag>.metadata.json, ao lado dos pesos) é lido
sozinho quando existe; --metadata aponta outro, --no-metadata desliga.
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, SymLogNorm
import tensorflow as tf
from tensorflow.keras import layers, Model


# ==============================================================================
# 1. PERFIS DE RODADA — caminhos, constantes e teto, por RUN_TAG
# ------------------------------------------------------------------------------
# [19/10/2026] Até a sst2 isto eram globais soltos apontando para
# treino_29-09. Na sst3 TRÊS coisas mudaram juntas, e cada uma, esquecida,
# gera figura errada sem erro nenhum:
#
#   - O DADO DO CFD. treino_29-09/csv é o job 5518 (teto 1e5); a sst3
#     treinou no job 6125 (teto 1e7). Mesma malha, mesmos nomes de arquivo:
#     o script leria o CFD antigo sem reclamar. As duas rodadas de CFD
#     divergem depois de t ≈ 40 s, então os slices de 58 e 100 s
#     comparariam a PINN com um escoamento fora de fase.
#   - AS CONSTANTES DE NORMALIZAÇÃO. As de saída mudaram com o dado: com as
#     da sst2, k sai até 19% abaixo do correto na esteira (kn ≈ 2,2) e v
#     até 0,25 m/s deslocado. O self-test não pega (ver seção 12).
#   - O TETO DE ν_t. Com 1e5 o cálculo de ν_t da PINN corta em 1,46 m²/s,
#     e o dado novo chega a 7,47 (p90 de 4,74 na esteira): o scatter
#     mostraria um patamar e a tabela um "ramo do teto" que a rede não tem.
#
# Por isso cada rodada é um PERFIL, e o perfil é escolhido pelo NOME do
# checkpoint (pinn_<tag>_NNNNNN.weights.h5), como a BC do inlet já era. Um
# checkpoint de rodada sem perfil ABORTA, em vez de cair num padrão.
#
# Por enquanto só os perfis da sst3. A sst3_cal é a calibração (job 6389):
# mesmo dado e mesmas constantes — a Cell 4 amostra de forma determinística —,
# com pastas próprias.
#
# Pastas: o treino escreve em <pasta da rodada>/saida/ (só ele); a
# inferência escreve em <pasta da rodada>/inferencia_<tag>/, ao lado, nunca
# dentro.
#
# [27/10/2026] siren1. Duas coisas entraram no perfil:
#
#   - A PASTA DA RODADA. A siren1 treinou em 10-19_siren/, não em
#     10-18_simples/ (as duas rodaram juntas e diagnosticos/ não tem tag no
#     nome). Até aqui TREINO_DIR/SAIDA_DIR eram globais fixos; agora cada
#     perfil leva treino_dir e saida_dir, e SAIDA_DIR passa a ser o do
#     perfil em uso (apply_profile).
#   - A ARQUITETURA. Os pesos da sst3 e da siren1 têm o MESMO número de
#     parâmetros (1.318.918). Carregar um no build_pinn do outro dá erro
#     (o .weights.h5 do Keras 3 indexa pela classe da camada — dense ×
#     siren_layer), mas carregar a siren1 numa SIREN com ω0 errado NÃO dá:
#     mesmas formas, mesmos caminhos, campo errado em silêncio. Por isso o
#     ω0 vem do perfil e o metadata o confere (seção 4).
#     Os valores de "type" são os que a Cell 16 de cada rodada escreve em
#     model_config["type"], para a conferência ser comparação direta.
#
# As constantes da siren1 são as da sst3, e não uma cópia: mesmo CFD, e a
# calibração da siren1 (job 6434) mediu Δ = 0,0% em todas contra elas.
#
# [02/11/2026] inferencia_ema.py — SÓ A RODADA ema1. Este arquivo saiu do
# inferencia_sst.py e serve a UMA rodada: os perfis sst3, sst3_cal e siren1
# saíram daqui (o inferencia_sst.py continua com eles, intocado). Cada
# rodada tem o seu script de inferência, porque a inferência muda junto com
# o que a rodada testa.
#
# A ema1 tem DUAS redes com a MESMA arquitetura (SIREN 6×512, ω0 = 10),
# treinadas no mesmo laço (pinn_ema_1029.py):
#   ema1_siren   a rede de treino, a que o Adam atualiza
#   ema1_ema     o EMA dos pesos dela, θ_ema ← β·θ_ema + (1 − β)·θ, a que
#                o treino escolhe como best oficial
# São dois perfis porque o perfil continua sendo escolhido pelo NOME do
# checkpoint (pinn_ema1_siren_* × pinn_ema1_ema_*, Cell 3 do treino), e o
# nome é a ÚNICA coisa que separa as duas redes: os pesos de uma carregam
# na outra sem erro (mesma classe, mesmas formas). O campo `rede` diz qual
# das duas o perfil lê, e `par` aponta para o perfil da outra, que é como
# a comparação acha a rede irmã (seções 10 e 16).
#
# Tudo o mais é igual nos dois perfis: o mesmo dado (CFD 6125), as mesmas
# constantes (a calibração da ema1, job 6577, mediu Δ = 0,0% em todas contra
# _CONST_SST3), a mesma pasta de checkpoints e o MESMO training_log, que
# traz as colunas das duas redes (as do EMA com "_ema" no fim, Cell 15).
#
# TRÊS PASTAS DE SAÍDA, dentro de 10-29_ema/inferencia_ema1/, fora de
# saida/ (que é só do treino). Uma pasta-mãe só, para a inferência inteira
# da rodada se arquivar ou apagar de uma vez:
#   inferencia_ema1/
#   ├── inferencia_ema1_siren/       figuras da rede de treino
#   ├── inferencia_ema1_ema/         figuras do EMA
#   └── inferencia_ema1_comparacao/  EMA × rede de treino (loss e MAE/méd|y|)
#
# `ema` guarda o β e os passos por época do treino. Servem à seção 13 para
# calcular a partir de que época a inicialização aleatória já saiu do EMA
# (β^n < 1%), que é a marca nos gráficos de comparação. Se o metadata
# existir, o β dele é conferido contra este (seção 4).
# ==============================================================================
TREINO_DIR = "/home/tmoraes/treino_turb/10-29_ema"   # pasta da ema1
SAIDA_DIR  = os.path.join(TREINO_DIR, "saida")      # o que o TREINO escreve
CFD_6125_CSV = "/home/tmoraes/treino_turb/new_sim_turb/results/csv"

# Arquitetura (Cell 9 do treino). As duas redes da ema1 saem do mesmo
# build_pinn, então é a mesma nos dois perfis.
_ARCH_SIREN = dict(type="SIREN", n_layers=6, n_neurons=512,
                   omega_0=10.0)     # decisão do orientador, 19/10/2026

# Constantes de normalização MEDIDAS pela Cell 4 do treino no dado do job
# 6125, impressas pela calibração (job PINN 6389) e coladas no fallback da
# Cell 4. 6 casas, como o .out imprime: erro < 1e-5 relativo por campo. O
# metadata da Cell 16, quando existe, as substitui com precisão total e
# avisa se alguma divergir (seção 4). K_EPS e T_MEAN/T_DP são exatos.
# O nome continua _CONST_SST3 porque é de onde elas vêm; a ema1 usa o mesmo
# CFD e a calibração dela (job 6577) mediu Δ = 0,0% em todas.
_CONST_SST3 = dict(
    # --- entrada: z-score isotrópico no espaço, z-score no tempo ---
    X_MEAN = 140.003410,
    Y_MEAN = 290.804396,
    Z_MEAN = 66.353356,
    DP_ISO = 101.319715,
    T_MEAN = 50.1,
    T_DP   = 28.838631497813,
    # --- saídas u, w: escala fixa;  v, P: z-score ---
    V_REF  = 5.0,
    V_MEAN = 15.047482,     # sst2: 15.092712
    V_DP   = 7.070827,      # sst2:  7.005018
    P_MEAN = -35.608335,    # sst2: −35.563905
    P_DP   = 95.521821,     # sst2:  95.561854
    # --- saídas k, ω: log ANTES do z-score ---
    LNK_MEAN = -2.123991,   # sst2: −2.183332
    LNK_DP   = 2.223361,    # sst2:  2.157926
    LNW_MEAN = 1.089561,    # sst2:  1.096495
    LNW_DP   = 1.856329,    # sst2:  1.857515
    # K_EPS = K_EPS_FRAC · K_INLET = 0,01 · 1,08375  (exato)
    K_EPS = 0.0108375,
)

# EMA do treino (Cells 9, 14 e 15 do pinn_ema_1029.py)
_EMA_EMA1 = dict(
    beta            = 0.999,   # BETA_EMA, decisão do orientador (29/10/2026)
    steps_per_epoch = 100,     # STEPS_PER_EPOCH; o EMA anda 1 vez por passo
)

# Comum aos dois perfis da ema1 — ver o cabeçalho desta seção
_CKPT_DIR_EMA1  = os.path.join(SAIDA_DIR, "pinn_ema1_checkpoints")
_CSV_LOG_EMA1   = os.path.join(SAIDA_DIR, "historico", "training_log_ema1.csv")
_INF_DIR_EMA1   = os.path.join(TREINO_DIR, "inferencia_ema1")   # pasta-mãe
_COMP_DIR_EMA1  = os.path.join(_INF_DIR_EMA1, "inferencia_ema1_comparacao")

RUNS = {
    "ema1_siren": dict(
        descricao      = "ema1, rede de treino (SIREN ω0=10, CFD 6125)",
        rede           = "siren",
        par            = "ema1_ema",
        cfd_job_id     = 6125,
        snapshot_dir   = CFD_6125_CSV,
        saida_dir      = SAIDA_DIR,
        checkpoint_dir = _CKPT_DIR_EMA1,
        csv_log        = _CSV_LOG_EMA1,
        output_dir     = os.path.join(_INF_DIR_EMA1, "inferencia_ema1_siren"),
        comparacao_dir = _COMP_DIR_EMA1,
        constants      = _CONST_SST3,
        visc_ratio_max = 1.0e7,    # turb-viscosity/max-limit do job 6125
        hard_bc_inlet  = True,     # HARD_BC_K/OMEGA_INLET da Cell 9
        arch           = _ARCH_SIREN,
        ema            = _EMA_EMA1,
    ),
    "ema1_ema": dict(
        descricao      = f"ema1, EMA dos pesos (β = {_EMA_EMA1['beta']}), "
                         f"SIREN ω0=10, CFD 6125",
        rede           = "ema",
        par            = "ema1_siren",
        cfd_job_id     = 6125,
        snapshot_dir   = CFD_6125_CSV,
        saida_dir      = SAIDA_DIR,
        checkpoint_dir = _CKPT_DIR_EMA1,
        csv_log        = _CSV_LOG_EMA1,
        output_dir     = os.path.join(_INF_DIR_EMA1, "inferencia_ema1_ema"),
        comparacao_dir = _COMP_DIR_EMA1,
        constants      = _CONST_SST3,
        visc_ratio_max = 1.0e7,
        hard_bc_inlet  = True,
        arch           = _ARCH_SIREN,
        ema            = _EMA_EMA1,
    ),
}
# O EMA é o padrão porque é nele que a inferência "oficial" roda (é o best
# do treino). A seção 16 roda as duas redes de qualquer jeito, salvo --rede.
RUN_TAG_PADRAO = "ema1_ema"

# Valores de partida; main() os refaz com apply_profile() depois de saber
# de qual rodada é o checkpoint.
RUN_TAG        = RUN_TAG_PADRAO
CHECKPOINT_DIR = RUNS[RUN_TAG]["checkpoint_dir"]
SNAPSHOT_DIR   = RUNS[RUN_TAG]["snapshot_dir"]
CSV_LOG_PATH   = RUNS[RUN_TAG]["csv_log"]
OUTPUT_DIR     = RUNS[RUN_TAG]["output_dir"]
COMPARACAO_DIR = RUNS[RUN_TAG]["comparacao_dir"]

SLICES_DIR   = os.path.join(OUTPUT_DIR, "slices")
SCATTER_DIR  = os.path.join(OUTPUT_DIR, "scatter")
HIST_DIR     = os.path.join(OUTPUT_DIR, "histograms")
SUMMARY_DIR  = os.path.join(OUTPUT_DIR, "summary")
LOSSHIST_DIR = os.path.join(OUTPUT_DIR, "loss_history")


# ==============================================================================
# 2. GEOMETRIA E ESCOAMENTO  (Cell 2 do treino — não mudou)
# ==============================================================================
X_MIN, X_MAX = 0.0, 280.0
Y_MIN, Y_MAX = 0.0, 700.0
Z_MIN, Z_MAX = 0.0, 190.0

D_CYL  = 54.0
H_CYL  = 65.0
R_CYL  = D_CYL / 2.0
XC_CYL = 140.0
YC_CYL = 200.0

V_INF   = 17.0
P_OP    = 101325.0
RHO_INF = 1.225
MU      = 1.7894e-5
NU      = MU / RHO_INF          # 1,4607e-05 m²/s

DT_PHYSICAL = 0.1               # s por índice de arquivo


# ==============================================================================
# 3. CONSTANTES DE NORMALIZAÇÃO
# ------------------------------------------------------------------------------
# Vêm do PERFIL (seção 1). Esta cópia é só o valor de partida; main() chama
# apply_profile() com o perfil do checkpoint e, se houver metadata, ele
# manda (seção 4).
#
# HISTÓRICO: até a sst2 os números estavam aqui, medidos no job 6083 sobre o
# dado antigo (CFD 5518). Na sst3 as constantes de SAÍDA mudaram com o dado
# (as espaciais e as de tempo não: mesma malha, mesma janela). Os valores da
# sst2 ficam comentados ao lado dos da sst3, em _CONST_SST3.
# ==============================================================================
CONSTANTS = dict(RUNS[RUN_TAG_PADRAO]["constants"])

# Arquitetura da rede (perfil, seção 1). apply_profile() a troca; o
# metadata, quando existe, só CONFERE — não sobrescreve (seção 4).
ARCH = dict(RUNS[RUN_TAG_PADRAO]["arch"])

# --- BC de turbulência no inlet (Cell 2/Cell 4 do treino) ---
TI_INLET         = 0.05
VISC_RATIO_INLET = 10.0
K_INLET          = 1.083750       # m²/s²
OMEGA_INLET      = 7419.21347     # 1/s
# LIGADAS nas rodadas "sst", "sst2" e "sst3" (o perfil traz o valor, seção
# 1; --hard-bc-inlet on/off sobrescreve). Chegaram a ser desligadas em
# 01/10/2026 com base no checkpoint 400 (erro de 1,36 dex em k na faixa do
# inlet) e voltaram no mesmo dia: no checkpoint 2200 o erro ali já era 0,13
# dex, e com a escala local de r_k o inlet responde por 0,6% do resíduo.
#
# O ansatz é propriedade dos PESOS. Se um dia sair uma rodada com elas
# desligadas, o perfil dela leva hard_bc_inlet=False. O self-test acusa se
# estiver errado — as linhas "inlet k" e "inlet ω" falham por ordens de
# grandeza.
HARD_BC_K_INLET     = True
HARD_BC_OMEGA_INLET = True

# --- constantes do ansatz (Cell 9 do treino) ---
HC_MT       = 135.0
HC_RATE     = 72.0
HC_K_SMOOTH = 75.0
HC_K_R2     = 238.0
CYL_RIM_FIX = True

# --- SST (Cell 2) — só usadas no cálculo de ν_t ---
SST_BETA_STAR  = 0.09
SST_A1         = 0.31
SST_SHEAR_LIMITER = True
# Teto do solver: vem do perfil. 1e5 até a sst2 (ν_t ≤ 1,4607 m²/s); 1e7 na
# sst3 (ν_t ≤ 146,07 m²/s), em que nenhuma célula do CFD chega a ele.
VISC_RATIO_MAX = RUNS[RUN_TAG_PADRAO]["visc_ratio_max"]
NU_T_MAX       = VISC_RATIO_MAX * NU          # 146,07 m²/s na sst3
MUT_MAX        = RHO_INF * NU_T_MAX           # 178,94 Pa·s na sst3
D_WALL_FLOOR   = 1.0e-3
OMEGA_FLOOR    = 1.0e-3
S_SQRT_EPS     = 1.0e-16

N_OUTPUTS   = 6
COMP_ORDER  = ["u", "v", "w", "P", "k", "omega"]


# ==============================================================================
# 4. METADATA — override opcional, com conferência
# ------------------------------------------------------------------------------
# [02/11/2026] ema1. A Cell 16 do pinn_ema_1029.py escreve UM metadata POR
# REDE, com o sufixo da rede (pinn_ema1_ema.metadata.json e
# pinn_ema1_siren.metadata.json). A busca por pinn_<tag>.metadata.json ao
# lado dos pesos (seção 16) já acha o certo, porque a tag do perfil É esse
# nome. As duas metadatas são iguais, a não ser pelo campo
# ema["rede_destes_pesos"].
#
# Por isso a conferência nova (_check_ema_metadata) é o campo `ema`:
#   - rede_destes_pesos tem que ser a `rede` do perfil, e run_tag + "_" +
#     rede tem que dar a tag do perfil. Senão o metadata é da OUTRA rede ou
#     de outra rodada, e ABORTA. É a mesma lógica do ω0 (_check_arch_metadata):
#     pesos das duas redes carregam um no lugar do outro sem erro, então o
#     que não for conferido aqui passa calado;
#   - sem o bloco `ema`, o metadata não é de uma rodada com EMA: ABORTA;
#   - β e passos por época: o metadata MANDA (é o que o treino usou), e uma
#     divergência contra o perfil é impressa. Eles só alimentam a marca da
#     época em que a inicialização sai do EMA (seção 13), não a inferência;
#   - se o checkpoint for de uma época ALÉM do fim do treino que escreveu o
#     metadata, avisa: é o caso de a pasta ainda ter o metadata da
#     calibração (20 épocas) enquanto o full roda. As constantes são as
#     mesmas (a Cell 4 é determinística), então as figuras saem certas, mas
#     o resumo do metadata (best, n de atualizações) não é deste treino.
# ==============================================================================
# β e passos por época do EMA. Partem do perfil (seção 1) e o metadata, se
# existir, os substitui. A seção 13 os lê para calcular a época da marca.
EMA_CFG = dict(RUNS[RUN_TAG_PADRAO]["ema"])


def apply_metadata(path, ckpt=None):
    """
    Lê o pinn_<tag>.metadata.json da Cell 16 e sobrescreve CONSTANTS.

    O metadata é a fonte autoritativa: foi escrito pelo mesmo processo que
    treinou a rede. Enquanto ele não existe (treino em andamento), valem os
    valores hardcoded acima. Qualquer divergência acima de 1e-4 relativo é
    impressa — é o sinal de que os hardcoded ficaram velhos.

    ckpt: o checkpoint que vai ser lido, só para o aviso de metadata de um
    treino mais curto (ver o cabeçalho desta seção). Opcional.
    """
    global K_INLET, OMEGA_INLET, TI_INLET, VISC_RATIO_INLET
    global HARD_BC_K_INLET, HARD_BC_OMEGA_INLET
    global VISC_RATIO_MAX, NU_T_MAX, MUT_MAX, SST_SHEAR_LIMITER

    with open(path, encoding="utf-8") as f:
        meta = json.load(f)

    nz = meta.get("normalizacao", {})
    print(f"\n  [metadata] {path}")
    print(f"  [metadata] criado em {meta.get('criado_em', '?')}  "
          f"run_tag={meta.get('run_tag', '?')}")

    if nz.get("K_NORM_MODE") not in (None, "logz"):
        raise SystemExit(f"K_NORM_MODE = {nz['K_NORM_MODE']!r}: este script só "
                         f"implementa 'logz'. Ver denormalize_outputs.")

    divergentes = []
    for key in CONSTANTS:
        if key in nz:
            novo, velho = float(nz[key]), float(CONSTANTS[key])
            den = max(abs(velho), 1e-12)
            if abs(novo - velho) / den > 1e-4:
                divergentes.append((key, velho, novo))
            CONSTANTS[key] = novo

    if divergentes:
        print(f"  [metadata] >>> {len(divergentes)} constante(s) divergiram do "
              f"hardcoded — o metadata manda:")
        for k_, v_, n_ in divergentes:
            print(f"               {k_:<10s} {v_:>14.6f} -> {n_:>14.6f}")
    else:
        print(f"  [metadata] todas as constantes batem com o hardcoded.")

    tb = meta.get("turbulencia", {})
    if tb:
        K_INLET          = float(tb.get("k_inlet", K_INLET))
        OMEGA_INLET      = float(tb.get("omega_inlet", OMEGA_INLET))
        TI_INLET         = float(tb.get("TI_inlet", TI_INLET))
        VISC_RATIO_INLET = float(tb.get("visc_ratio_inlet", VISC_RATIO_INLET))
        SST_SHEAR_LIMITER = bool(tb.get("shear_limiter", SST_SHEAR_LIMITER))
        VISC_RATIO_MAX   = float(tb.get("visc_ratio_max", VISC_RATIO_MAX))
        NU_T_MAX         = VISC_RATIO_MAX * NU
        MUT_MAX          = RHO_INF * NU_T_MAX

    hc = meta.get("hard_constraints", meta.get("rede", {}))
    for name, var in (("HARD_BC_K_INLET", "HARD_BC_K_INLET"),
                      ("HARD_BC_OMEGA_INLET", "HARD_BC_OMEGA_INLET")):
        if name in hc:
            globals()[var] = bool(hc[name])

    _check_arch_metadata(meta.get("model_config", {}))
    _check_ema_metadata(meta, ckpt)

    _derive()
    return meta


def _check_arch_metadata(mc):
    """
    [27/10/2026] Confere a arquitetura do perfil contra o model_config da
    Cell 16. ABORTA em vez de sobrescrever: se divergir, ou o perfil está
    errado ou o metadata é de outra rodada, e nos dois casos não há como
    saber qual dos dois é o certo daqui.

    O caso que importa é o ω0: uma SIREN com ω0 diferente do treino carrega
    os pesos sem erro (mesmas formas, mesmos caminhos no .h5) e devolve um
    campo errado — o self-test não pega, porque as BCs são impostas pelo
    ansatz e saem certas com qualquer ω0. Tipo e tamanho errados já davam
    erro no load_weights; conferir aqui só antecipa a mensagem.
    """
    if not mc:
        print(f"  [metadata] sem model_config — arquitetura do perfil, "
              f"sem conferência: {_arch_str(ARCH)}")
        return
    problemas = []
    for key in ("type", "n_layers", "n_neurons"):
        if key in mc and mc[key] != ARCH.get(key):
            problemas.append(f"{key}: perfil {ARCH.get(key)!r} × "
                             f"metadata {mc[key]!r}")
    if ARCH.get("type") == "SIREN" or mc.get("type") == "SIREN":
        w_p, w_m = ARCH.get("omega_0"), mc.get("omega_0")
        if w_m is None or w_p is None or float(w_p) != float(w_m):
            problemas.append(f"omega_0: perfil {w_p!r} × metadata {w_m!r}")
        sc = mc.get("input_scaling", "none")
        if not str(sc).startswith("none"):
            problemas.append(f"input_scaling = {sc!r}: este script só "
                             f"implementa entrada SEM reescala (seção 7)")
    if problemas:
        raise SystemExit(
            "ERRO: a arquitetura do perfil não é a que o metadata descreve:\n"
            + "".join(f"    {p}\n" for p in problemas)
            + "  Corrigir o perfil (seção 1) ou apontar o --metadata certo. "
              "Um ω0 errado carrega os pesos sem erro e dá figuras erradas.")
    print(f"  [metadata] arquitetura confere com o perfil: {_arch_str(ARCH)}")


def _check_ema_metadata(meta, ckpt=None):
    """
    [02/11/2026] Confere o bloco `ema` da Cell 16 do pinn_ema_1029.py contra
    o perfil em uso (ver o cabeçalho desta seção). ABORTA se o metadata for
    de outra rede ou não tiver EMA; atualiza EMA_CFG com β e passos por
    época do treino.
    """
    prof = RUNS[RUN_TAG]
    em = meta.get("ema")
    if not em:
        raise SystemExit(
            f"ERRO: o metadata não tem o bloco 'ema' — não é de uma rodada com "
            f"EMA (pinn_ema_1029.py). O perfil '{RUN_TAG}' espera um.\n"
            f"  Apontar o --metadata certo, ou --no-metadata para usar só o "
            f"perfil.")

    rede_meta = em.get("rede_destes_pesos")
    tag_meta  = f"{meta.get('run_tag', '?')}_{rede_meta}"
    problemas = []
    if rede_meta != prof["rede"]:
        problemas.append(f"rede: perfil {prof['rede']!r} × metadata {rede_meta!r}")
    if tag_meta != RUN_TAG:
        problemas.append(f"tag: perfil {RUN_TAG!r} × metadata {tag_meta!r} "
                         f"(run_tag + '_' + rede_destes_pesos)")
    if problemas:
        raise SystemExit(
            "ERRO: o metadata não é o desta rede:\n"
            + "".join(f"    {p_}\n" for p_ in problemas)
            + "  EMA e rede de treino têm as mesmas formas: os pesos de uma "
              "carregam na outra sem erro.\n"
              "  Conferir o --checkpoint e o --metadata.")

    # β e passos por época: o metadata manda
    _b_meta = em.get("beta")
    _s_meta = meta.get("treino", {}).get("steps_per_epoch")
    for chave, v_meta in (("beta", _b_meta), ("steps_per_epoch", _s_meta)):
        if v_meta is None:
            continue
        v_prof = prof["ema"][chave]
        if float(v_meta) != float(v_prof):
            print(f"  [metadata] >>> {chave}: perfil {v_prof} × metadata "
                  f"{v_meta} — o metadata manda")
        EMA_CFG[chave] = type(v_prof)(v_meta)

    print(f"  [metadata] EMA confere com o perfil: rede '{rede_meta}', "
          f"β = {EMA_CFG['beta']}, {EMA_CFG['steps_per_epoch']} passos/época, "
          f"aquecimento: {em.get('aquecimento', '?')}")
    print(f"  [metadata] best do treino: EMA na época "
          f"{em.get('best_epoch_ema', '?')} (val_aligned "
          f"{em.get('best_val_aligned_ema', float('nan')):.4g}), rede de "
          f"treino na época {em.get('best_epoch_siren', '?')} (val_aligned "
          f"{em.get('best_val_aligned_siren', float('nan')):.4g})")

    # metadata de um treino mais curto que o checkpoint (calibração na pasta)
    n_ep_meta = meta.get("treino", {}).get("n_epochs")
    m_ep = _CKPT_EPOCH_RE.search(os.path.basename(ckpt or ""))
    if n_ep_meta is not None and m_ep and int(m_ep.group(1)) > int(n_ep_meta):
        print(f"  [metadata] >>> o checkpoint é da época {int(m_ep.group(1))}, "
              f"mas o metadata é de um treino de {n_ep_meta} épocas "
              f"(calibração?).")
        print(f"                 As constantes são as mesmas, então as "
              f"figuras saem certas; o resumo acima é daquele treino, não "
              f"deste.")


def _arch_str(a):
    s = f"{a['type']} {a['n_layers']}×{a['n_neurons']}"
    return s + (f", ω0={a['omega_0']:g}" if a["type"] == "SIREN" else ", tanh")


# ==============================================================================
# 5. GRANDEZAS DERIVADAS
# ==============================================================================
def _derive():
    """Recalcula tudo que depende de CONSTANTS. Chamar após qualquer override."""
    g = globals()
    for k_, v_ in CONSTANTS.items():
        g[k_] = v_

    g["XC_N"] = (XC_CYL - CONSTANTS["X_MEAN"]) / CONSTANTS["DP_ISO"]
    g["YC_N"] = (YC_CYL - CONSTANTS["Y_MEAN"]) / CONSTANTS["DP_ISO"]
    g["R_N"]  = R_CYL / CONSTANTS["DP_ISO"]
    g["H_N"]  = (H_CYL - CONSTANTS["Z_MEAN"]) / CONSTANTS["DP_ISO"]

    g["XN_MIN"] = (X_MIN - CONSTANTS["X_MEAN"]) / CONSTANTS["DP_ISO"]
    g["XN_MAX"] = (X_MAX - CONSTANTS["X_MEAN"]) / CONSTANTS["DP_ISO"]
    g["YN_MIN"] = (Y_MIN - CONSTANTS["Y_MEAN"]) / CONSTANTS["DP_ISO"]
    g["YN_MAX"] = (Y_MAX - CONSTANTS["Y_MEAN"]) / CONSTANTS["DP_ISO"]
    g["ZN_MIN"] = (Z_MIN - CONSTANTS["Z_MEAN"]) / CONSTANTS["DP_ISO"]
    g["ZN_MAX"] = (Z_MAX - CONSTANTS["Z_MEAN"]) / CONSTANTS["DP_ISO"]

    # Valores NORMALIZADOS das BCs de turbulência — exatamente como a Cell 4
    # os calcula. É o que a HardConstraintLayer mistura.
    g["KN_INLET"]   = (float(np.log(K_INLET + CONSTANTS["K_EPS"]))
                       - CONSTANTS["LNK_MEAN"]) / CONSTANTS["LNK_DP"]
    g["LNWN_INLET"] = (float(np.log(OMEGA_INLET))
                       - CONSTANTS["LNW_MEAN"]) / CONSTANTS["LNW_DP"]


_derive()


def apply_profile(tag):
    """
    Carrega o perfil da rodada `tag` (seção 1): constantes de normalização,
    teto de ν_t, BC dura do inlet e — desde a siren1 — arquitetura e pasta
    saida/ da rodada. Os outros caminhos ficam com main(), porque a linha
    de comando pode sobrescrevê-los um a um.
    """
    global RUN_TAG, VISC_RATIO_MAX, NU_T_MAX, MUT_MAX
    global HARD_BC_K_INLET, HARD_BC_OMEGA_INLET, ARCH, SAIDA_DIR
    if tag not in RUNS:
        raise SystemExit(
            f"ERRO: não há perfil para a rodada '{tag}'. Perfis: "
            f"{', '.join(RUNS)}.\n  Cada rodada tem dado de CFD, constantes "
            f"e teto próprios (seção 1); usar os de outra daria figuras "
            f"erradas sem erro nenhum.")
    prof = RUNS[tag]
    RUN_TAG = tag
    CONSTANTS.clear()
    CONSTANTS.update(prof["constants"])
    VISC_RATIO_MAX = float(prof["visc_ratio_max"])
    NU_T_MAX       = VISC_RATIO_MAX * NU
    MUT_MAX        = RHO_INF * NU_T_MAX
    HARD_BC_K_INLET = HARD_BC_OMEGA_INLET = bool(prof["hard_bc_inlet"])
    ARCH      = dict(prof["arch"])
    SAIDA_DIR = prof["saida_dir"]
    _derive()
    return prof


# ==============================================================================
# 6. DISTÂNCIA À PAREDE — analítica (Cell 7 do treino)
#
# Conferida contra o cell-wall-distance do Fluent no job 6083: correlação
# 1.000000, erro mediano 0,0000 m, p99 0,0098 m. Só é usada aqui para o ν_t.
# inlet (y=0) e outlet (y=700) NÃO são parede.
# ==============================================================================
def distance_to_cylinder(x, y, z):
    r  = np.sqrt((x - XC_CYL)**2 + (y - YC_CYL)**2)
    dr = np.maximum(r - R_CYL, 0.0)
    dz = np.maximum(z - H_CYL, 0.0)
    return np.sqrt(dr*dr + dz*dz)


def wall_distance(x, y, z, apply_floor=True):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)
    d = np.stack([z - Z_MIN, Z_MAX - z, x - X_MIN, X_MAX - x,
                  distance_to_cylinder(x, y, z)], axis=-1).min(axis=-1)
    return np.maximum(d, D_WALL_FLOOR) if apply_floor else d


def is_inside_cylinder(x, y, z):
    return (((x - XC_CYL)**2 + (y - YC_CYL)**2 <= R_CYL**2) & (z <= H_CYL))


# ==============================================================================
# 7. HARDCONSTRAINTLAYER — cópia da Cell 9 do treino
#
# Cada diferença em relação a esta função muda o campo previsto em todo o
# domínio, e nada avisa. O self-test da seção 12 é o que protege.
# ==============================================================================
def hc_smooth_max(a, b, k=None):
    k = HC_K_SMOOTH if k is None else k
    m = tf.maximum(a, b)
    return m + (1.0 / k) * tf.math.log(tf.exp(k * (a - m)) + tf.exp(k * (b - m)))


def hc_distance_functions(xn, yn, zn):
    D_inlet  = tf.math.tanh(HC_MT * (yn - YN_MIN))
    D_outlet = tf.math.tanh(HC_MT * (YN_MAX - yn))
    D_ground = tf.math.tanh(HC_MT * (zn - ZN_MIN))
    D_xmin   = tf.math.tanh(HC_MT * (xn - XN_MIN))
    D_xmax   = tf.math.tanh(HC_MT * (XN_MAX - xn))
    D_ztop   = tf.math.tanh(HC_MT * (ZN_MAX - zn))

    r2_normalized = tf.square(xn - XC_N) + tf.square(yn - YC_N)
    d_radial_sq   = r2_normalized - R_N * R_N
    d_top         = zn - H_N
    if CYL_RIM_FIX:
        d_cyl_3d = tf.maximum(d_radial_sq, (2.0 * R_N) * d_top)
    else:
        d_cyl_3d = hc_smooth_max(d_radial_sq, d_top, k=HC_K_SMOOTH)
    D_cyl = tf.math.tanh(HC_K_R2 * tf.maximum(d_cyl_3d, 0.0))

    D_vel = D_inlet * D_ground * D_cyl * D_xmin * D_xmax * D_ztop

    decay_inlet = tf.exp(-HC_RATE * (yn - YN_MIN))
    decay_xmin  = tf.exp(-HC_RATE * (xn - XN_MIN))
    decay_xmax  = tf.exp(-HC_RATE * (XN_MAX - xn))
    decay_ztop  = tf.exp(-HC_RATE * (ZN_MAX - zn))
    decay_freestream = 1.0 - (1.0 - decay_inlet) * (1.0 - decay_xmin) \
                           * (1.0 - decay_xmax) * (1.0 - decay_ztop)

    return dict(D_inlet=D_inlet, D_outlet=D_outlet, D_ground=D_ground,
                D_cyl=D_cyl, D_vel=D_vel, decay_freestream=decay_freestream)


class HardConstraintLayer(layers.Layer):
    """6 saídas. Idêntica à do treino, inclusive no stack+reshape."""

    def call(self, inputs):
        xyzt, raw_pred = inputs

        xn = xyzt[:, 0:1]
        yn = xyzt[:, 1:2]
        zn = xyzt[:, 2:3]

        un  = raw_pred[:, 0:1]
        vn  = raw_pred[:, 1:2]
        wn  = raw_pred[:, 2:3]
        Pn  = raw_pred[:, 3:4]
        kn  = raw_pred[:, 4:5]
        omn = raw_pred[:, 5:6]

        D = hc_distance_functions(xn, yn, zn)
        D_inlet  = D["D_inlet"]
        D_outlet = D["D_outlet"]
        D_ground = D["D_ground"]
        D_cyl    = D["D_cyl"]
        D_vel    = D["D_vel"]
        decay_freestream = D["decay_freestream"]

        u_hard = un * D_vel
        w_hard = wn * D_vel

        v_phys_raw  = vn * CONSTANTS["V_DP"] + CONSTANTS["V_MEAN"]
        v_phys_hard = (v_phys_raw * D_vel
                       + V_INF * decay_freestream * D_ground * D_cyl)
        v_hard = (v_phys_hard - CONSTANTS["V_MEAN"]) / CONSTANTS["V_DP"]

        P_phys_raw  = Pn * CONSTANTS["P_DP"] + CONSTANTS["P_MEAN"]
        P_phys_hard = P_phys_raw * D_outlet
        P_hard = (P_phys_hard - CONSTANTS["P_MEAN"]) / CONSTANTS["P_DP"]

        k_hard  = (kn * D_inlet + KN_INLET * (1.0 - D_inlet)
                   if HARD_BC_K_INLET else kn)
        om_hard = (omn * D_inlet + LNWN_INLET * (1.0 - D_inlet)
                   if HARD_BC_OMEGA_INLET else omn)

        return tf.reshape(
            tf.stack([u_hard, v_hard, w_hard, P_hard, k_hard, om_hard],
                     axis=1),
            (-1, N_OUTPUTS))


class SirenLayer(layers.Layer):
    """
    [27/10/2026] Cópia da SirenLayer da Cell 9 do treino da siren1:
    sin(ω0·(W·x + b)).

    Dois detalhes que NÃO podem mudar, porque o .weights.h5 do Keras 3
    guarda cada variável pelo caminho derivado da CLASSE e pela ordem de
    criação (layers/siren_layer/vars/0, .../vars/1), não pelo nome dado à
    camada (siren_1 …):
      - o nome da classe: SirenLayer;
      - a ordem do add_weight: kernel, depois bias.
    Renomear a classe dá erro no load_weights; inverter a ordem também
    (formas diferentes). A inicialização não importa aqui — os pesos são
    sobrescritos —, mas fica igual à do treino para a rede recém-montada
    ser a mesma.
    """
    def __init__(self, units, is_first=False, omega_0=30.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.is_first = is_first
        self.omega_0 = float(omega_0)

    def build(self, input_shape):
        fan_in = int(input_shape[-1])
        if self.is_first:
            limit = 1.0 / fan_in
        else:
            limit = float(np.sqrt(6.0 / fan_in)) / self.omega_0
        self.w = self.add_weight(
            shape=(fan_in, self.units),
            initializer=tf.keras.initializers.RandomUniform(minval=-limit,
                                                            maxval=limit),
            trainable=True, name="kernel")
        self.b = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True, name="bias")

    def call(self, inputs):
        return tf.sin(self.omega_0 * (tf.matmul(inputs, self.w) + self.b))


def build_pinn(n_layers=6, n_neurons=512, arch=None):
    """
    arch=None usa o ARCH do perfil em uso (seção 1). "MLP_simple" é a rede
    tanh da sst3; "SIREN" é a da siren1, com a ENTRADA SEM REESCALA (o
    z-score da Cell 4 vai direto na 1ª camada, como no treino).
    """
    arch = ARCH if arch is None else arch
    inp = layers.Input(shape=(4,), name="xyzt")
    h = inp
    if arch["type"] == "MLP_simple":
        for i in range(n_layers):
            h = layers.Dense(n_neurons, activation="tanh",
                             kernel_initializer="glorot_uniform",
                             bias_initializer="zeros",
                             name=f"dense_{i+1}")(h)
        nome = "PINN_MLP"
    elif arch["type"] == "SIREN":
        for i in range(n_layers):
            h = SirenLayer(units=n_neurons, is_first=(i == 0),
                           omega_0=arch["omega_0"], name=f"siren_{i+1}")(h)
        nome = "PINN_SIREN"
    else:
        raise SystemExit(f"arquitetura desconhecida: {arch['type']!r} "
                         f"(seção 1: 'MLP_simple' ou 'SIREN')")
    raw_out = layers.Dense(N_OUTPUTS, activation="linear",
                           bias_initializer="zeros", name="raw_fields")(h)
    final_out = HardConstraintLayer(name="hard_constraints")([inp, raw_out])
    return Model(inp, final_out, name=nome)


# ==============================================================================
# 8. NORMALIZAÇÃO / DESNORMALIZAÇÃO
# ==============================================================================
def normalize_inputs(x_phys, y_phys, z_phys, t_phys):
    x_phys = np.atleast_1d(x_phys).astype(np.float64).ravel()
    y_phys = np.atleast_1d(y_phys).astype(np.float64).ravel()
    z_phys = np.atleast_1d(z_phys).astype(np.float64).ravel()
    n = len(x_phys)

    if np.isscalar(t_phys):
        t_arr = np.full(n, float(t_phys))
    else:
        t_arr = np.atleast_1d(t_phys).astype(np.float64).ravel()
        assert len(t_arr) == n, f"len(t)={len(t_arr)} != len(x)={n}"

    C = CONSTANTS
    xn = (x_phys - C["X_MEAN"]) / C["DP_ISO"]
    yn = (y_phys - C["Y_MEAN"]) / C["DP_ISO"]
    zn = (z_phys - C["Z_MEAN"]) / C["DP_ISO"]
    tn = (t_arr  - C["T_MEAN"]) / C["T_DP"]
    return np.column_stack([xn, yn, zn, tn]).astype(np.float32)


def denormalize_outputs(pred):
    """
    Saída da rede (N, 6) -> unidades físicas.

    k e ω passam por EXPONENCIAL, não por z-score. Em float32 a exponencial
    de um ln ω normalizado perto de +4,2 (o inlet) já é ~7400 e a conta
    perde dígitos, então as contas de k e ω vão em float64.
    """
    C = CONSTANTS
    p = np.asarray(pred, np.float64)
    u = p[:, 0] * C["V_REF"]
    v = p[:, 1] * C["V_DP"] + C["V_MEAN"]
    w = p[:, 2] * C["V_REF"]
    P = p[:, 3] * C["P_DP"] + C["P_MEAN"]
    k = np.exp(p[:, 4] * C["LNK_DP"] + C["LNK_MEAN"]) - C["K_EPS"]
    om = np.exp(p[:, 5] * C["LNW_DP"] + C["LNW_MEAN"])
    # k pode sair levemente negativo quando kn está no piso: exp(...)−K_EPS.
    # É o mesmo −1e−09 que o log do treino reporta. Cortar em 0 é honesto:
    # k < 0 não existe, e o valor é ruído de fp.
    k = np.maximum(k, 0.0)
    return u, v, w, P, k, om


def _roundtrip_check():
    """
    normalize/denormalize de k e ω têm que ser inversas. Se alguém mexer em
    K_EPS ou nos μ/σ, isto acusa antes dos plots.
    """
    C = CONSTANTS
    k_t  = np.array([1e-4, K_INLET, 1.0, 50.0], np.float64)
    om_t = np.array([1e-2, 1.0, OMEGA_INLET, 1e5], np.float64)
    kn   = (np.log(np.maximum(k_t, 0.0) + C["K_EPS"]) - C["LNK_MEAN"]) / C["LNK_DP"]
    omn  = (np.log(np.maximum(om_t, OMEGA_FLOOR)) - C["LNW_MEAN"]) / C["LNW_DP"]
    fake = np.zeros((len(k_t), 6))
    fake[:, 4], fake[:, 5] = kn, omn
    _, _, _, _, k_b, om_b = denormalize_outputs(fake)
    ok_k  = np.allclose(k_b, k_t, rtol=1e-6, atol=1e-8)
    ok_om = np.allclose(om_b, om_t, rtol=1e-6)
    assert ok_k and ok_om, (
        f"round-trip de k/ω falhou — as constantes de normalização não são "
        f"consistentes.\n  k:  {k_t} -> {k_b}\n  ω: {om_t} -> {om_b}")
    return True


# ==============================================================================
# 9. LOADER DE CSV  (Cell 3/5 do treino)
# ==============================================================================
NEEDED_COLS_PATTERNS = {
    "x":     [r"x.coordinate", r"^x$"],
    "y":     [r"y.coordinate", r"^y$"],
    "z":     [r"z.coordinate", r"^z$"],
    "P":     [r"^pressure$", r"static.pressure", r"^p$"],
    "u":     [r"x.velocity", r"^u$"],
    "v":     [r"y.velocity", r"^v$"],
    "w":     [r"z.velocity", r"^w$"],
    "k":     [r"turb.kinetic.energy", r"^k$", r"^tke$"],
    "omega": [r"specific.diss.rate", r"^omega$", r"^w_sst$"],
    "mut":   [r"viscosity.turb", r"turb.*viscosity", r"^mut$"],
}

NON_SNAPSHOT_BASENAMES = {"static_mesh.csv"}
_TIME_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _norm_col(s):
    return re.sub(r"[\"'﻿]", "", s).strip().lower()


def detect_delimiter(path, n_probe=2048):
    """
    O Fluent pergunta "Delimiter/Comma?" no export. O job 5518 respondeu
    vírgula; rodadas antigas responderam espaço. Detectar é mais barato que
    uma constante que pode mentir — o script laminar tinha sep=r'\\s+' fixo
    e leria este CSV como uma coluna só.
    """
    with open(path, "r") as fh:
        head = fh.read(n_probe).splitlines()
    if not head:
        return r"\s+"
    first = head[0]
    return "," if first.count(",") >= 2 else r"\s+"


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
            raise KeyError(f"coluna '{key}' não encontrada. Padrões: {patterns}. "
                           f"Disponíveis: {list(norm_map.keys())}")
        mapping[key] = found
    return mapping


def load_snapshot(path, col_map=None, sep=None):
    """
    Devolve dict com x, y, z, u, v, w, P, k, omega, mut (μ_t DINÂMICA, Pa·s).

    A ordem das colunas no CSV do Fluent é
        cellnumber, x, y, z, viscosity-turb, specific-diss-rate,
        turb-kinetic-energy, pressure, z-velocity, y-velocity, x-velocity
    ou seja, INVERTIDA em relação ao journal e com u por último. Resolver
    por NOME, nunca por posição.
    """
    sep = sep if sep is not None else detect_delimiter(path)
    df = pd.read_csv(path, sep=sep, engine="python",
                     skipinitialspace=True, dtype=np.float32)
    # normalizar ANTES de descartar: o cabeçalho do Fluent vem " cellnumber"
    df.columns = [_norm_col(c) for c in df.columns]
    df = df.drop(columns=["cellnumber"], errors="ignore")
    if col_map is None:
        col_map = resolve_column_names(df.columns)
    out = {k: df[col_map[k]].to_numpy(dtype=np.float32, copy=False)
           for k in NEEDED_COLS_PATTERNS}
    return out, col_map


def discover_snapshots(snapshot_dir, dt_physical=DT_PHYSICAL):
    """
    t_00002.csv -> t = 2 × 0,1 s. Mesma regra do treino: o ÚLTIMO número do
    nome é o índice do passo. static_mesh.csv é excluído pelo nome.
    """
    files = sorted(glob.glob(os.path.join(snapshot_dir, "*.csv")))
    snaps = []
    for f in files:
        base = os.path.basename(f)
        if base in NON_SNAPSHOT_BASENAMES:
            continue
        m = _TIME_TOKEN_RE.findall(os.path.splitext(base)[0])
        if not m:
            continue
        snaps.append((float(m[-1]) * dt_physical, f))
    snaps.sort(key=lambda s: s[0])
    return snaps


# ==============================================================================
# 10. CHECKPOINTS
# ==============================================================================
# Propositalmente SEM o RUN_TAG: a PASTA é que diz de qual rodada são os
# arquivos. Amarrar o regex ao tag fazia um --checkpoint-dir apontando para
# outra rodada devolver "nenhum checkpoint" com a pasta cheia.
# [19/10/2026] O tag pode ter "_" (sst3_cal): [A-Za-z0-9]+ não casava.
#
# [02/11/2026] ema1: UMA PASTA, DUAS REDES. pinn_ema1_checkpoints/ tem
#   pinn_ema1_siren_NNNNNN / _best / final   (tag ema1_siren)
#   pinn_ema1_ema_NNNNNN   / _best / final   (tag ema1_ema)
# "O mais recente" pela época sozinha pegaria qualquer uma das duas (as duas
# têm a mesma época mais alta). Por isso list_checkpoints e
# find_latest_checkpoint ganharam o filtro `tag`, e há funções novas para
# o PAR (a mesma época nas duas redes):
#   partner_checkpoint   troca o sufixo da rede no nome (_ema <-> _siren);
#   checkpoint_label     "época 5000", "best" ou "final", para os títulos;
#   checkpoint_table     época -> {rede: arquivo}, para o --list;
#   find_latest_pair     a época mais alta em que AS DUAS existem;
#   check_pair           confere que o par é da mesma rodada, de redes
#                        diferentes e da mesma época.
# O BEST NÃO TEM PAR DE MESMA ÉPOCA: cada rede tem o próprio best (Cell 15),
# escolhido pela própria validação. check_pair aceita best × best e devolve
# um aviso, que a comparação (seções 13, 15 e 16) leva para os títulos.
# ==============================================================================
_CKPT_EPOCH_RE = re.compile(r"pinn_.+?_(\d+)\.weights\.h5$")


def list_checkpoints(checkpoint_dir, tag=None):
    """
    Checkpoints NUMERADOS da pasta, em ordem de época. tag (p.ex.
    "ema1_ema") deixa só os daquela rede; None, como antes, lista todos.
    """
    files = glob.glob(os.path.join(checkpoint_dir, "pinn_*.weights.h5"))
    if tag is not None:
        files = [f for f in files if tag_from_checkpoint(f) == tag]
    numbered = [(int(m.group(1)), f) for f in files
                if (m := _CKPT_EPOCH_RE.search(os.path.basename(f)))]
    numbered.sort()
    return [f for _, f in numbered]


# O TAG VEM DO NOME DO ARQUIVO, não de uma opção de linha de comando.
# "pinn_sst_002200.weights.h5" é da rodada "sst", onde a BC dura do inlet
# estava LIGADA; "pinn_sst2_000500.weights.h5" é da rodada seguinte, onde
# ela está desligada. O ansatz é uma propriedade dos PESOS, então deixar
# isso depender de uma flag que o usuário pode esquecer é convite a gerar
# 40 figuras erradas. (O self-test pegaria, mas depois do susto.)
# [19/10/2026] Casa também o tag com "_" (pinn_sst3_cal_best) e o arquivo
# FINAL da Cell 16 (pinn_sst3.weights.h5, sem sufixo), que a versão anterior
# não reconhecia — devolvia tag None e o checkpoint final rodava sem perfil.
_CKPT_TAG_RE = re.compile(r"pinn_(.+?)(?:_(?:\d+|best))?\.weights\.h5$")


def tag_from_checkpoint(path):
    """Tag da rodada a partir do nome do .h5, ou None se não reconhecer."""
    m = _CKPT_TAG_RE.search(os.path.basename(path or ""))
    return m.group(1) if m else None


def resolve_checkpoint(arg, checkpoint_dir, base_dir=None, tag=None):
    """
    Acha o .h5 mesmo quando o --checkpoint-dir aponta para outra rodada.

    Foi exatamente o que aconteceu quando o RUN_TAG padrão virou "sst2":
    passar --checkpoint pinn_sst_002200.weights.h5 juntava o nome com
    .../pinn_sst2_checkpoints/ e dava "não encontrado". Agora o nome do
    arquivo manda. (base_dir é 10-18_simples/saida/, onde o treino da sst3
    põe as pastas pinn_<tag>_checkpoints.)

    [27/10/2026] base_dir=None usa o SAIDA_DIR do perfil em uso. E, se o
    tag do nome tiver perfil, tenta também a pasta de checkpoints DELE: a
    siren1 está em 10-19_siren/, então nem a pasta irmã nem o base_dir da
    sst3 a achariam sem --run-tag siren1.

    [02/11/2026] tag: com arg=None, o mais recente DAQUELA rede (seção 10,
    cabeçalho). Sem tag, como antes.
    """
    base_dir = SAIDA_DIR if base_dir is None else base_dir
    if arg is None:
        return find_latest_checkpoint(checkpoint_dir, tag=tag), []
    if os.path.isabs(arg):
        return (arg if os.path.exists(arg) else None), [arg]

    tentativas = [os.path.join(checkpoint_dir, arg)]
    tag = tag_from_checkpoint(arg)
    if tag:
        # a pasta irmã da que foi pedida — funciona mesmo se o SAIDA_DIR
        # deste script estiver apontando para outro lugar
        irma = os.path.join(os.path.dirname(os.path.abspath(checkpoint_dir)),
                            f"pinn_{tag}_checkpoints", arg)
        tentativas.append(irma)
        tentativas.append(os.path.join(base_dir, f"pinn_{tag}_checkpoints", arg))
        if tag in RUNS:
            tentativas.append(os.path.join(RUNS[tag]["checkpoint_dir"], arg))
    tentativas.append(os.path.abspath(arg))

    vistos, unicas = set(), []
    for t in tentativas:
        if t not in vistos:
            vistos.add(t); unicas.append(t)
    for t in unicas:
        if os.path.exists(t):
            return t, unicas
    return None, unicas


def find_latest_checkpoint(checkpoint_dir, tag=None):
    """
    O numerado de época mais alta (da rede `tag`, se dada). Sem numerados,
    cai no final e depois no best daquela rede: é o caso da calibração, em
    que SAVE_EVERY = 999.999 e só existem esses dois.
    """
    files = list_checkpoints(checkpoint_dir, tag=tag)
    if files:
        return files[-1]
    if tag is not None:
        for nome in (f"pinn_{tag}.weights.h5", f"pinn_{tag}_best.weights.h5"):
            cand = os.path.join(checkpoint_dir, nome)
            if os.path.exists(cand):
                return cand
        return None
    outros = sorted(glob.glob(os.path.join(checkpoint_dir,
                                           "pinn_*.weights.h5")))
    return outros[-1] if outros else None


def partner_checkpoint(path):
    """
    O arquivo da OUTRA rede da mesma rodada e do mesmo tipo (mesma época,
    best ou final): pinn_ema1_ema_005000 <-> pinn_ema1_siren_005000. Troca
    só a tag no nome, na mesma pasta. Devolve (caminho, tag_do_par) — o
    caminho pode não existir; quem chama confere. (None, None) se a tag do
    arquivo não tiver perfil com par.
    """
    tag = tag_from_checkpoint(path)
    if tag not in RUNS or not RUNS[tag].get("par"):
        return None, None
    tag_par = RUNS[tag]["par"]
    nome = os.path.basename(path)
    assert nome.startswith(f"pinn_{tag}"), nome
    nome_par = f"pinn_{tag_par}" + nome[len(f"pinn_{tag}"):]
    return os.path.join(os.path.dirname(path), nome_par), tag_par


def checkpoint_label(path):
    """'época 5000', 'best' ou 'final' — o que vai nos títulos e logs."""
    nome = os.path.basename(path or "")
    m = _CKPT_EPOCH_RE.search(nome)
    if m:
        return f"época {int(m.group(1))}"
    if nome.endswith("_best.weights.h5"):
        return "best"
    return "final"


def checkpoint_table(checkpoint_dir):
    """
    {época: {rede: arquivo}} dos numerados, mais as chaves "best" e
    "final". rede é o campo `rede` do perfil ("siren"/"ema"); arquivos sem
    perfil entram com a tag crua, para o --list não esconder nada.
    """
    tab = {}
    for f in glob.glob(os.path.join(checkpoint_dir, "pinn_*.weights.h5")):
        tag = tag_from_checkpoint(f)
        rede = RUNS[tag]["rede"] if tag in RUNS else (tag or "?")
        m = _CKPT_EPOCH_RE.search(os.path.basename(f))
        chave = int(m.group(1)) if m else checkpoint_label(f)
        tab.setdefault(chave, {})[rede] = f
    return tab


def find_latest_pair(checkpoint_dir, tag):
    """
    (arquivo de `tag`, arquivo do par) na época mais alta em que AS DUAS
    redes têm checkpoint. A Cell 15 grava a rede de treino e depois o EMA;
    se o treino estiver no meio dessa época, a mais alta ainda não tem par
    e fica para a próxima. Sem numerados, tenta final × final e depois
    best × best. (None, None) se não achar par nenhum.
    """
    tag_par = RUNS[tag]["par"]
    meus = {int(_CKPT_EPOCH_RE.search(os.path.basename(f)).group(1)): f
            for f in list_checkpoints(checkpoint_dir, tag=tag)}
    deles = {int(_CKPT_EPOCH_RE.search(os.path.basename(f)).group(1)): f
             for f in list_checkpoints(checkpoint_dir, tag=tag_par)}
    comuns = sorted(set(meus) & set(deles))
    if comuns:
        return meus[comuns[-1]], deles[comuns[-1]]
    for suf in ("", "_best"):
        a = os.path.join(checkpoint_dir, f"pinn_{tag}{suf}.weights.h5")
        b = os.path.join(checkpoint_dir, f"pinn_{tag_par}{suf}.weights.h5")
        if os.path.exists(a) and os.path.exists(b):
            return a, b
    return None, None


def check_pair(path_a, path_b):
    """
    Confere que os dois checkpoints formam um par comparável. ABORTA se
    forem da mesma rede, de rodadas diferentes ou de tipos diferentes
    (época × best, época 5000 × época 4900). Devolve um aviso (string)
    para best × best, que tem épocas diferentes por construção, ou "" se
    o par é da mesma época. A conferência de que as redes carregadas são
    DIFERENTES (|Δ saída| > 0) fica na seção 16, depois de carregar.
    """
    tag_a, tag_b = tag_from_checkpoint(path_a), tag_from_checkpoint(path_b)
    erros = []
    if tag_a not in RUNS or tag_b not in RUNS:
        erros.append(f"tag sem perfil: {tag_a!r}, {tag_b!r}")
    elif RUNS[tag_a].get("par") != tag_b:
        erros.append(f"{tag_a!r} e {tag_b!r} não são as duas redes da mesma "
                     f"rodada (o par de {tag_a!r} é {RUNS[tag_a].get('par')!r})")
    lab_a, lab_b = checkpoint_label(path_a), checkpoint_label(path_b)
    if lab_a != lab_b:
        erros.append(f"tipos diferentes: {lab_a} × {lab_b}")
    if erros:
        raise SystemExit("ERRO: os dois checkpoints não formam um par:\n"
                         + "".join(f"    {e}\n" for e in erros)
                         + f"    {path_a}\n    {path_b}")
    if lab_a == "best":
        return ("best × best: cada rede tem o PRÓPRIO best, escolhido pela "
                "própria validação, em geral em épocas diferentes")
    return ""


def wait_until_stable(path, tries=6, pause=2.0):
    """
    O treino ainda está rodando e reescreve o 'best' a cada melhora. Ler no
    meio da escrita dá um .h5 truncado. Esperar o tamanho parar de mudar
    custa segundos e evita um erro confuso de HDF5.
    """
    last = -1
    for _ in range(tries):
        try:
            cur = os.path.getsize(path)
        except OSError:
            cur = -1
        if cur == last and cur > 0:
            return True
        last = cur
        time.sleep(pause)
    return False


# ==============================================================================
# 11. INFERÊNCIA
# ==============================================================================
def predict_fields(model, x, y, z, t, batch=200_000):
    """Campos físicos nos pontos dados. Em lotes: 211.579 × 6 cabe, mas um
    snapshot inteiro com autodiff não."""
    n = len(np.atleast_1d(x).ravel())
    outs = []
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        xyzt = normalize_inputs(np.ravel(x)[sl], np.ravel(y)[sl],
                                np.ravel(z)[sl],
                                t if np.isscalar(t) else np.ravel(t)[sl])
        outs.append(model(tf.constant(xyzt), training=False).numpy())
    pred = np.concatenate(outs, axis=0)
    u, v, w, P, k, om = denormalize_outputs(pred)
    return dict(u=u, v=v, w=w, P=P, k=k, omega=om, raw=pred)


# --- constantes TF para o ν_t ---
def _tf_consts():
    C = CONSTANTS
    return dict(
        DP_ISO = tf.constant(C["DP_ISO"], tf.float32),
        V_REF  = tf.constant(C["V_REF"],  tf.float32),
        V_DP   = tf.constant(C["V_DP"],   tf.float32),
        V_MEAN = tf.constant(C["V_MEAN"], tf.float32),
        LNK_DP = tf.constant(C["LNK_DP"], tf.float32),
        LNK_MEAN = tf.constant(C["LNK_MEAN"], tf.float32),
        LNW_DP = tf.constant(C["LNW_DP"], tf.float32),
        LNW_MEAN = tf.constant(C["LNW_MEAN"], tf.float32),
    )


def _mu_t_batch(model, pts, dw):
    """
    μ_t pela fórmula SST INTEIRA, nos moldes da Cell 10:

        μ_t = min( (ρk/ω) / max(1, S·F2/(a1·ω)) , VISC_RATIO_MAX·μ )
        F2  = tanh(Φ2²),  Φ2 = max( 2√k/(β*ωy), 500ν/(y²ω) )
        S   = √(2 S_ij S_ij)

    NÃO é k/ω. No job 6083 o limitador de cisalhamento corta 2,36% das
    células e o teto pega 5,36% — comparar k/ω com a coluna viscosity-turb
    erra justamente nessas.

    O k aqui é o k SEM o −K_EPS (é o que o resíduo usa): exp(kn·σ+μ). A
    diferença é 0,0108 m²/s², visível só onde k já é desprezível.
    """
    T = _tf_consts()
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(pts)
        pr = model(pts, training=False)
        u = pr[:, 0:1] * T["V_REF"]
        v = pr[:, 1:2] * T["V_DP"] + T["V_MEAN"]
        w = pr[:, 2:3] * T["V_REF"]
    du = tape.gradient(u, pts) / T["DP_ISO"]
    dv = tape.gradient(v, pts) / T["DP_ISO"]
    dwv = tape.gradient(w, pts) / T["DP_ISO"]
    del tape

    ux, uy, uz = du[:, 0], du[:, 1], du[:, 2]
    vx, vy, vz = dv[:, 0], dv[:, 1], dv[:, 2]
    wx, wy, wz = dwv[:, 0], dwv[:, 1], dwv[:, 2]

    S12 = 0.5 * (uy + vx)
    S13 = 0.5 * (uz + wx)
    S23 = 0.5 * (vz + wy)
    S2 = 2.0 * (ux*ux + vy*vy + wz*wz
                + 2.0 * (S12*S12 + S13*S13 + S23*S23))

    pr = model(pts, training=False)
    k  = tf.exp(pr[:, 4] * T["LNK_DP"] + T["LNK_MEAN"])
    om = tf.exp(pr[:, 5] * T["LNW_DP"] + T["LNW_MEAN"])

    y  = tf.maximum(dw, D_WALL_FLOOR)
    phi2 = tf.maximum(2.0 * tf.sqrt(tf.maximum(k, 0.0))
                      / (SST_BETA_STAR * om * y),
                      500.0 * NU / (tf.square(y) * om))
    F2 = tf.math.tanh(tf.square(phi2))

    mu_base = RHO_INF * k / om
    if SST_SHEAR_LIMITER:
        S = tf.sqrt(S2 + S_SQRT_EPS)
        denom = tf.maximum(1.0, S * F2 / (SST_A1 * om))
        mu_lim = mu_base / denom
    else:
        denom = tf.ones_like(mu_base)
        mu_lim = mu_base
    mu_t = tf.minimum(mu_lim, MUT_MAX)
    return (mu_t.numpy(), F2.numpy(), tf.sqrt(S2 + S_SQRT_EPS).numpy(),
            (denom > 1.0 + 1e-6).numpy(), (mu_lim > MUT_MAX).numpy())


def predict_mu_t(model, x, y, z, t, batch=50_000, verbose=False):
    """μ_t [Pa·s] da PINN, em lotes (o autodiff dobra a memória)."""
    x = np.ravel(x); y = np.ravel(y); z = np.ravel(z)
    n = len(x)
    dw_all = wall_distance(x, y, z).astype(np.float32)
    mu, f2, S, lim, cap = [], [], [], [], []
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        xyzt = normalize_inputs(x[sl], y[sl], z[sl],
                                t if np.isscalar(t) else np.ravel(t)[sl])
        a, b, c, d, e = _mu_t_batch(model, tf.constant(xyzt),
                                    tf.constant(dw_all[sl]))
        mu.append(a); f2.append(b); S.append(c); lim.append(d); cap.append(e)
        if verbose and (i // batch) % 2 == 0:
            print(f"      μ_t {min(i + batch, n):>7,}/{n:,}", flush=True)
    return dict(mu_t=np.concatenate(mu), F2=np.concatenate(f2),
                S=np.concatenate(S), limited=np.concatenate(lim),
                capped=np.concatenate(cap), d_wall=dw_all)


# ==============================================================================
# 12. SELF-TEST DAS HARD CONSTRAINTS
#
# O mesmo teste da Cell 9 do treino. Existe porque a réplica da
# HardConstraintLayer no script laminar tinha DIVERGIDO em três pontos
# (smooth_max sem subtrair o máximo, D_vel multiplicando por D_outlet,
# decay_inlet invertido) e ninguém percebeu: os plots saíam, só saíam
# errados. Se este teste falhar, o resto não vale nada.
# ==============================================================================
def selftest_hard_constraints(model, n=400):
    """
    Confere o ansatz nas 8 fronteiras, como a Cell 9 do treino.

    ATENÇÃO ÀS PAREDES MÓVEIS. No caso do Fluent, top, x_min e x_max são
    paredes com velocidade 17 m/s em +y (é assim que o escoamento livre não
    desenvolve camada limite nas laterais da caixa); só o CHÃO e o CILINDRO
    são paredes fixas. O ansatz reproduz isso sem nenhum ramo especial:

        v = v_rede·D_vel + V_INF·decay_fs·D_ground·D_cyl

    Na parede móvel D_vel = 0 e decay = 1, sobrando V_INF; no chão e no
    cilindro D_ground ou D_cyl zera o segundo termo também. Esperar v = 0
    em todas as paredes é o erro fácil — e foi o que esta função fez na
    primeira versão.

    Os pontos de teste ficam a MARGEM metros das outras fronteiras: no
    inlet junto ao chão, por exemplo, v vale 17·D_ground < 17 de verdade, e
    comparar com 17 ali acusaria um defeito que não existe.
    """
    rng = np.random.default_rng(0)
    MARGEM = 10.0
    TOL_V  = 1.0e-3       # m/s, para u, w e v em parede fixa
    TOL_REL = 1.0e-3      # relativo, para v=V_INF, k e ω

    def campos(x, y, z):
        return predict_fields(model, x, y, z, t=50.0)

    print("\n" + "=" * 78)
    print("  SELF-TEST DAS HARD CONSTRAINTS")
    print("=" * 78)

    checks = []

    # ---------------- inlet (y = Y_MIN) ----------------
    x = rng.uniform(X_MIN + MARGEM, X_MAX - MARGEM, n)
    z = rng.uniform(Z_MIN + MARGEM, Z_MAX - MARGEM, n)
    # longe do cilindro também: ele toca o plano y=0? não, mas D_cyl<1 perto
    f = campos(x, np.full(n, Y_MIN), z)
    checks += [("inlet u", np.abs(f["u"]).max(), 0.0, TOL_V),
               ("inlet w", np.abs(f["w"]).max(), 0.0, TOL_V),
               ("inlet v", f["v"].mean(), V_INF, TOL_REL * V_INF)]
    if HARD_BC_K_INLET:
        checks.append(("inlet k", f["k"].mean(), K_INLET, TOL_REL * K_INLET))
    if HARD_BC_OMEGA_INLET:
        checks.append(("inlet ω", f["omega"].mean(), OMEGA_INLET,
                       TOL_REL * OMEGA_INLET))

    # ---------------- outlet (P = 0) ----------------
    x = rng.uniform(X_MIN + MARGEM, X_MAX - MARGEM, n)
    z = rng.uniform(Z_MIN + MARGEM, Z_MAX - MARGEM, n)
    f = campos(x, np.full(n, Y_MAX), z)
    checks.append(("outlet P", np.abs(f["P"]).max(), 0.0, 1.0e-3))

    # ---------------- paredes ----------------
    # (nome, x, y, z, v_esperado)  — v_esperado = V_INF nas MÓVEIS
    xx = rng.uniform(X_MIN + MARGEM, X_MAX - MARGEM, n)
    yy = rng.uniform(Y_MIN + MARGEM, Y_MAX - MARGEM, n)
    zz = rng.uniform(Z_MIN + MARGEM, Z_MAX - MARGEM, n)
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    rr = R_CYL * np.sqrt(rng.uniform(0.0, 1.0, n))

    paredes = [
        ("chão (fixa)",      xx, yy, np.full(n, Z_MIN), 0.0),
        ("topo (móvel)",     xx, yy, np.full(n, Z_MAX), V_INF),
        ("x_min (móvel)",    np.full(n, X_MIN), yy, zz, V_INF),
        ("x_max (móvel)",    np.full(n, X_MAX), yy, zz, V_INF),
        ("cil. lateral",     XC_CYL + R_CYL*np.cos(th),
                             YC_CYL + R_CYL*np.sin(th),
                             rng.uniform(MARGEM, H_CYL - 1.0, n), 0.0),
        ("cil. topo",        XC_CYL + rr*np.cos(th),
                             YC_CYL + rr*np.sin(th),
                             np.full(n, H_CYL), 0.0),
        ("cil. QUINA",       XC_CYL + R_CYL*np.cos(th),
                             YC_CYL + R_CYL*np.sin(th),
                             np.full(n, H_CYL), 0.0),
    ]

    for nome, px, py, pz, v_esp in paredes:
        f = campos(px, py, pz)
        checks.append((f"{nome} u", np.abs(f["u"]).max(), 0.0, TOL_V))
        checks.append((f"{nome} w", np.abs(f["w"]).max(), 0.0, TOL_V))
        if v_esp == 0.0:
            # tanh(238·max(d,0)) não é exatamente 0 em fp32 sobre a
            # superfície: o resíduo medido é ~6e-6, que vira 17·6e-6 ≈ 1e-4
            # em v. É o mesmo número que a Cell 9 do treino reporta.
            checks.append((f"{nome} v", np.abs(f["v"]).max(), 0.0, 2.0e-3))
        else:
            checks.append((f"{nome} v", f["v"].mean(), v_esp, TOL_REL * v_esp))

    falhas = []
    print(f"  {'teste':<22s} {'obtido':>15s} {'esperado':>15s} {'tol':>10s}")
    print("  " + "-" * 74)
    for nome, obtido, esperado, tol in checks:
        ok = abs(obtido - esperado) <= tol
        if not ok:
            falhas.append(nome)
        print(f"  {nome:<22s} {obtido:>15.6g} {esperado:>15.6g} "
              f"{tol:>10.1e}  {'OK' if ok else 'FALHOU'}")
    print("  " + "-" * 74)

    if falhas:
        print(f"  {len(falhas)} de {len(checks)} FALHARAM: {falhas}")
        print("  A HardConstraintLayer deste script NÃO é a mesma que a rede")
        print("  usou no treino, ou a BC dura do inlet está trocada.")
        print("  Qualquer plot gerado agora estaria errado — por isso para aqui.")
        raise SystemExit(2)

    print(f"  {len(checks)}/{len(checks)} passaram, em 8 fronteiras.")
    print("  O ansatz bate com o do treino.")
    # As constantes de normalização NÃO são testadas aqui: a BC é montada
    # (KN_INLET, V_INF·decay) e lida com as MESMAS constantes, então
    # qualquer conjunto passa. A versão anterior dizia o contrário. Quem as
    # garante é o perfil (seção 1) e, quando existe, o metadata (seção 4).
    print("  (constantes de normalização: não verificáveis por este teste —")
    print("   vêm do perfil e, quando existe, do metadata)")
    print("=" * 78)
    return True

# ==============================================================================
# 13. LOSS HISTORY  (47 colunas do training_log_sst.csv)
# ------------------------------------------------------------------------------
# [02/11/2026] ema1: o training_log_ema1.csv tem 56 colunas — as 47 de
# sempre, que são TODAS da rede de treino, e no fim as do EMA:
#   val_mse6_ema, val_mse4_ema, val_aligned_ema, val_<zona>_ema, val_time
# (Cell 15 do pinn_ema_1029.py).
#
# O EMA NÃO TEM LOSS DE TREINO. total, data, os resíduos da física, |g| e
# as máscaras são da rede de treino: o EMA nunca passa pela fita do
# gradiente, só acompanha os pesos. Do EMA só existe VALIDAÇÃO. Daí:
#   - pasta siren: loss_history.png e diagnostics.png como sempre;
#   - pasta ema:   loss_history.png com a validação do EMA, e as curvas de
#                  treino em CINZA, rotuladas como da rede de treino (são
#                  referência, não do EMA). Sem diagnostics.png: tudo nele é
#                  da rede de treino e já está na pasta siren;
#   - comparação:  plot_loss_comparison, as duas validações juntas.
#
# ÉPOCAS DE VALIDAÇÃO. A validação roda a cada VAL_EVERY épocas e o CSV
# repete o último valor no meio. A coluna val_time é > 0 só nas épocas em
# que a validação rodou de fato, então ela é usada para achar esses pontos
# (_val_epochs); sem ela, cai no _changed de sempre.
#
# A MARCA DA INICIALIZAÇÃO NO EMA (escolha explícita, pedida em 02/11/2026).
# O EMA nasce igual à rede de treino ALEATÓRIA e, depois de n passos, ainda
# guarda β^n daqueles pesos. Enquanto essa fração é grande, o EMA sai pior
# que a rede de treino por um motivo que não tem nada a ver com a média —
# ele está arrastando a inicialização. A comparação EMA × rede de treino só
# mede o efeito do EMA depois que essa fração ficou pequena.
#   Critério escolhido: β^n < EMA_RESIDUO_LIMIAR = 1%.
#   n = ln(0,01)/ln(β) = 4.603 passos para β = 0,999; com 100 passos por
#   época, época ≈ 46.
# A marca é uma linha vertical nas figuras do EMA e da comparação, com a
# conta escrita na legenda, e a região antes dela fica sombreada. As
# médias impressas e gravadas em CSV pela comparação usam SÓ as épocas
# depois da marca. β e passos por época vêm de EMA_CFG (perfil, ou o
# metadata quando existe — seção 4). Trocar o limiar é uma linha aqui.
# ==============================================================================
GRAD_CLIP_NORM = 5.0     # Cell 14 do treino

EMA_RESIDUO_LIMIAR = 0.01    # fração da inicialização que ainda "vale" (1%)

PHYS_KEYS  = ["cont", "mom_u", "mom_v", "mom_w", "k", "omega"]
ZONE_KEYS  = ["cyl", "bc", "boi_c", "boi_t", "free"]


def _changed(series):
    """Máscara dos pontos em que a série mudou. A validação roda a cada
    VAL_EVERY épocas e o valor é repetido no meio — plotar tudo desenha
    degraus que parecem patamar de treino."""
    v = np.asarray(series, float)
    return np.concatenate([[True], np.diff(v) != 0])


def ema_marca_inicializacao():
    """
    Época a partir da qual a inicialização aleatória pesa menos que
    EMA_RESIDUO_LIMIAR no EMA (ver o cabeçalho desta seção). Devolve
    (época, passos, texto) — o texto é o que vai nas legendas e no log.
    """
    beta = float(EMA_CFG["beta"])
    spe  = int(EMA_CFG["steps_per_epoch"])
    n    = float(np.log(EMA_RESIDUO_LIMIAR) / np.log(beta))
    ep   = n / spe
    txt  = (f"inicialização < {100*EMA_RESIDUO_LIMIAR:g}% no EMA: "
            f"β^n < {EMA_RESIDUO_LIMIAR:g} com β={beta:g} "
            f"→ n={n:,.0f} passos = época {ep:.0f}")
    return ep, n, txt


def _marcar_inicializacao(ax, ep_marca, txt, ep_max):
    """Linha vertical na época da marca e sombra antes dela."""
    ax.axvline(ep_marca, color="black", ls=":", lw=1.2, label=txt)
    ax.axvspan(0, min(ep_marca, ep_max), color="gray", alpha=0.12, lw=0)


def _read_log(csv_path):
    """CSV do treino, com o separador detectado. None se não existir."""
    if not os.path.exists(csv_path):
        print(f"  [skip] CSV de log não encontrado: {csv_path}")
        return None
    with open(csv_path) as fh:
        first = fh.readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(csv_path, sep=sep)
    print(f"  loss history: {len(df)} épocas, {len(df.columns)} colunas "
          f"(sep={sep!r})")
    return df


def _val_epochs(df, col):
    """Máscara das épocas em que a validação RODOU (ver o cabeçalho)."""
    if "val_time" in df:
        return (df["val_time"] > 0).to_numpy()
    return _changed(df[col])


def _tem_colunas_ema(df):
    falta = [c for c in ("val_aligned_ema", "val_mse4_ema", "val_mse6_ema")
             if c not in df.columns]
    if falta:
        print(f"  >>> o CSV não tem as colunas do EMA {falta}: não é o "
              f"training_log de uma rodada com EMA (pinn_ema_1029.py).")
        return False
    return True


def plot_loss_history(csv_path, output_dir, rede="siren"):
    """
    rede="siren": as figuras de sempre (loss_history.png, diagnostics.png).
    rede="ema": só loss_history.png, com a validação do EMA (ver o
    cabeçalho desta seção).
    """
    df = _read_log(csv_path)
    if df is None:
        return None

    faltando = [c for c in ["epoch", "total", "data", "phys_sum"]
                if c not in df.columns]
    if faltando:
        print(f"  >>> colunas ausentes: {faltando}")
        print(f"      Colunas do arquivo: {list(df.columns)}")
        print(f"      Se os nomes forem loss_total/loss_data/..., este é o CSV")
        print(f"      da PINN LAMINAR, não o do SST.")
        return None

    if rede == "ema":
        return _plot_loss_history_ema(df, output_dir)

    ep = df["epoch"]

    # ---------------- figura 1: perdas ----------------
    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)

    ax = axes[0]
    ax.semilogy(ep, df["total"], lw=2.0, color="black",
                label="total (objetivo ponderado)")
    ax.semilogy(ep, df["data"], lw=1.2, color="tab:blue",
                label="data (resíduo médio, sem peso)")
    ax.semilogy(ep, df["phys_sum"], lw=1.2, color="tab:red",
                label="física (soma das contribuições)")
    if "val_aligned" in df:
        m = _changed(df["val_aligned"])
        ax.semilogy(ep[m], df["val_aligned"][m], "o-", ms=3.5, lw=1.0,
                    color="tab:green", alpha=0.9,
                    label="val_aligned (= mesma fórmula do data)")
    if "val_mse4" in df:
        # Fica uma ordem de grandeza abaixo das outras (MSE neutro em 4
        # componentes normalizadas, sem peso de zona nem de componente),
        # então encosta no rodapé do eixo. Em cinza claro e pontilhado
        # fino ela simplesmente sumia. Roxo tracejado-pontilhado: destoa
        # das outras quatro (preto, azul, vermelho, verde) e o traço
        # continua avisando que ela NÃO é comparável com 'data'.
        m = _changed(df["val_mse4"])
        _v4 = float(df["val_mse4"].iloc[-1])
        ax.semilogy(ep[m], df["val_mse4"][m], color="tab:purple",
                    ls=(0, (4, 1.5, 1, 1.5)), lw=1.6, alpha=0.95, zorder=5,
                    label=f"val MSE/4 — outra escala, só p/ comparar com a "
                          f"laminar (fim: {_v4:.3g})")
    ax.set_ylabel("loss (log)")
    ax.set_title("PINN k-ω SST — treino\n"
                 "val_aligned é a única curva comparável com 'data'; "
                 "val_mse4 existe só para comparar com a versão laminar",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    if "val_aligned" in df:
        d_fin = float(df["data"].iloc[-1])
        v_fin = float(df["val_aligned"].iloc[-1])
        # y = 0,14: entre a val_mse4 (rodapé) e o data (~0,4). Em 0,05 a
        # caixa cobria justamente a val_mse4.
        ax.text(0.015, 0.14,
                f"época {int(ep.iloc[-1])}: data={d_fin:.3f}  "
                f"val_al={v_fin:.3f}  (razão {v_fin/max(d_fin,1e-30):.2f})\n"
                f"razão ≈ 1 => generaliza; >> 1 => decorou o treino",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    ax = axes[1]
    cores = dict(cont="tab:purple", mom_u="tab:orange", mom_v="tab:brown",
                 mom_w="tab:pink", k="tab:cyan", omega="tab:red")
    for key in PHYS_KEYS:
        if key in df:
            ax.semilogy(ep, df[key], lw=1.2, color=cores[key], label=key)
    ax.set_ylabel("resíduo físico (log)")
    ax.set_title("Resíduos por equação — k e ω são os termos novos do SST",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    for z in ZONE_KEYS:
        if f"data_{z}" in df:
            ax.semilogy(ep, df[f"data_{z}"], lw=1.1, label=f"treino {z}")
    for z in ZONE_KEYS:
        if f"val_{z}" in df:
            m = _changed(df[f"val_{z}"])
            ax.semilogy(ep[m], df[f"val_{z}"][m], "--", lw=0.9, alpha=0.6,
                        label=f"val {z}")
    ax.set_ylabel("data loss por zona (log)")
    ax.set_title("Zonas — tracejado é validação. Afastamento entre cheio e "
                 "tracejado na MESMA zona é overfit local", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[3]
    for c in COMP_ORDER:
        if f"comp_{c}" in df:
            ax.semilogy(ep, df[f"comp_{c}"], lw=1.1, label=c)
    ax.set_ylabel("data loss por componente (log)")
    ax.set_xlabel("época")
    ax.set_title("Componentes (sem o peso de DATA_COMP_W) — k e ω em espaço "
                 "normalizado log", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(output_dir, "loss_history.png")
    plt.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  salvo: {out1}")

    # ---------------- figura 2: diagnósticos ----------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    ax = axes[0]
    for key, lbl in [("frac_active_k", "resíduo de k ativo"),
                     ("frac_active_om", "resíduo de ω ativo"),
                     ("frac_in_wall", "na faixa de parede"),
                     ("frac_prod_lim", "production limiter ativo"),
                     ("frac_visc_cap", "ν_t no teto")]:
        if key in df:
            ax.plot(ep, 100.0 * df[key], lw=1.2, label=lbl)
    ax.set_ylabel("% dos pontos de colocação")
    ax.set_title("Máscaras do resíduo. 'ν_t no teto' subindo muito significa "
                 "que a rede está encostando no limite do solver", fontsize=10)
    ax.legend(loc="center right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "mu_t_p50" in df:
        ax.semilogy(ep, df["mu_t_p50"], lw=1.3, color="tab:green",
                    label="μ_t mediana [Pa·s]")
        ax.axhline(MUT_MAX, color="red", ls="--", lw=0.9,
                   label=f"teto do solver ({MUT_MAX:.3f} Pa·s)")
        ax.axhline(MU, color="gray", ls=":", lw=0.9,
                   label=f"μ molecular ({MU:.2e})")
    ax.set_ylabel("μ_t (log)")
    ax.set_title("Viscosidade turbulenta mediana prevista", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    if "gnorm" in df:
        ax.semilogy(ep, df["gnorm"], lw=1.0, color="tab:gray",
                    label="|g| antes do clip")
        ax.axhline(GRAD_CLIP_NORM, color="red", ls="--", lw=0.9,
                   label=f"clip = {GRAD_CLIP_NORM}")
    if "skipped" in df and float(np.nanmax(df["skipped"])) > 0:
        ax2 = ax.twinx()
        ax2.plot(ep, df["skipped"], lw=1.0, color="tab:red", alpha=0.6)
        ax2.set_ylabel("passos descartados (acum.)", color="tab:red")
    ax.set_ylabel("norma do gradiente (log)")
    ax.set_xlabel("época")
    ax.set_title("Estabilidade. |g| sempre acima do clip significa que todo "
                 "passo está sendo cortado — o LR efetivo é outro",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(output_dir, "diagnostics.png")
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  salvo: {out2}")
    return out1


def _plot_loss_history_ema(df, output_dir):
    """
    loss_history.png da pasta do EMA: validação do EMA em cor, curvas de
    treino da rede de treino em cinza (referência). Dois painéis: o geral
    e as zonas. Os resíduos da física e as componentes ficam de fora — são
    só da rede de treino e já estão na pasta siren.
    """
    if not _tem_colunas_ema(df):
        return None
    ep = df["epoch"]
    ep_marca, _, txt_marca = ema_marca_inicializacao()
    print(f"  marca: {txt_marca}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True)

    ax = axes[0]
    # as três em cinza (não são do EMA), separadas pelo traço
    for col, lbl, lw, ls_ in (("total", "total", 1.6, "-"),
                              ("data", "data", 1.1, "-"),
                              ("phys_sum", "física", 1.1, "--")):
        ax.semilogy(ep, df[col], lw=lw, ls=ls_, color="0.65",
                    label=f"{lbl} — rede de treino (referência)")
    m = _val_epochs(df, "val_aligned_ema")
    ax.semilogy(ep[m], df["val_aligned_ema"][m], "o-", ms=3.5, lw=1.0,
                color="tab:green", label="val_aligned_ema (= mesma fórmula do data)")
    _v4 = float(df["val_mse4_ema"].iloc[-1])
    ax.semilogy(ep[m], df["val_mse4_ema"][m], color="tab:purple",
                ls=(0, (4, 1.5, 1, 1.5)), lw=1.6, alpha=0.95, zorder=5,
                label=f"val_mse4_ema — outra escala (fim: {_v4:.3g})")
    _marcar_inicializacao(ax, ep_marca, txt_marca, float(ep.max()))
    ax.set_ylabel("loss (log)")
    ax.set_title(f"PINN k-ω SST — EMA dos pesos (β = {EMA_CFG['beta']:g})\n"
                 "O EMA não tem loss de treino: em cinza, as da rede de "
                 "treino, só como referência", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    d_fin = float(df["data"].iloc[-1])
    v_fin = float(df["val_aligned_ema"].iloc[-1])
    ax.text(0.015, 0.14,
            f"época {int(ep.iloc[-1])}: data (rede de treino)={d_fin:.3f}  "
            f"val_al_ema={v_fin:.3f}  (razão {v_fin/max(d_fin,1e-30):.2f})",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    ax = axes[1]
    for z in ZONE_KEYS:
        if f"data_{z}" in df:
            ax.semilogy(ep, df[f"data_{z}"], lw=0.8, color="0.7")
    for i_, z in enumerate(ZONE_KEYS):
        if f"val_{z}_ema" in df:
            ax.semilogy(ep[m], df[f"val_{z}_ema"][m], "--", lw=1.1,
                        color=f"C{i_}", label=f"val {z} (EMA)")
    ax.plot([], [], color="0.7", lw=0.8,
            label="treino por zona — rede de treino (referência)")
    _marcar_inicializacao(ax, ep_marca, txt_marca, float(ep.max()))
    ax.set_ylabel("data loss por zona (log)")
    ax.set_xlabel("época")
    ax.set_title("Zonas — tracejado é a validação do EMA; cinza é o treino "
                 "da rede de treino", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "loss_history.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  salvo: {out}")
    return out


def plot_loss_comparison(csv_path, output_dir):
    """
    [02/11/2026] Pasta de comparação: as validações das duas redes juntas,
    só nas épocas em que a validação rodou. Quatro painéis:
      (a) val_aligned das duas;
      (b) razão val_aligned_ema / val_aligned (< 1: EMA melhor);
      (c) val_mse4 e val_mse6 das duas;
      (d) validação por zona das duas.
    Grava também comparacao_validacao.csv (uma linha por validação) e
    imprime as médias da razão DEPOIS da marca da inicialização.
    """
    df = _read_log(csv_path)
    if df is None or not _tem_colunas_ema(df):
        return None
    m  = _val_epochs(df, "val_aligned")
    dv = df[m].reset_index(drop=True)
    ep = dv["epoch"]
    ep_marca, n_marca, txt_marca = ema_marca_inicializacao()
    print(f"  marca: {txt_marca}")
    razao = dv["val_aligned_ema"] / dv["val_aligned"]
    C_S, C_E = "tab:blue", "tab:orange"      # rede de treino, EMA

    fig, axes = plt.subplots(4, 1, figsize=(13, 17), sharex=True)

    ax = axes[0]
    ax.semilogy(ep, dv["val_aligned"], "o-", ms=3, lw=1.1, color=C_S,
                label="val_aligned — rede de treino")
    ax.semilogy(ep, dv["val_aligned_ema"], "s-", ms=3, lw=1.1, color=C_E,
                label="val_aligned_ema — EMA")
    _marcar_inicializacao(ax, ep_marca, txt_marca, float(ep.max()))
    ax.set_ylabel("val_aligned (log)")
    ax.set_title(f"EMA (β = {EMA_CFG['beta']:g}) × rede de treino — só "
                 f"validação (o EMA não tem loss de treino)\n"
                 f"Antes da linha pontilhada o EMA ainda arrasta a "
                 f"inicialização: a comparação não vale ali", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(ep, razao, "o-", ms=3, lw=1.1, color="black",
            label="val_aligned_ema / val_aligned")
    ax.axhline(1.0, color="red", ls="--", lw=1.0,
               label="1 = empate  (abaixo: EMA melhor)")
    _marcar_inicializacao(ax, ep_marca, txt_marca, float(ep.max()))
    _pos = ep > ep_marca
    if _pos.any():
        # o eixo segue a parte que vale: antes da marca a razão chega a 5
        # e achata o resto
        _r = razao[_pos]
        _lo, _hi = float(_r.min()), float(_r.max())
        _pad = 0.1 * max(_hi - _lo, 0.02)
        ax.set_ylim(min(_lo, 1.0) - _pad, max(_hi, 1.0) + _pad)
        ax.text(0.015, 0.05,
                f"depois da época {ep_marca:.0f}: razão média "
                f"{float(_r.mean()):.3f}, última {float(_r.iloc[-1]):.3f}, "
                f"EMA melhor em {100*float((_r < 1).mean()):.0f}% das "
                f"validações",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    ax.set_ylabel("razão (linear)")
    ax.set_title("Razão EMA / rede de treino na mesma época — o gráfico que "
                 "diz se o EMA está ajudando", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for col, ls_ in (("val_mse4", "-"), ("val_mse6", "--")):
        ax.semilogy(ep, dv[col], ls=ls_, lw=1.1, color=C_S,
                    label=f"{col} — rede de treino")
        ax.semilogy(ep, dv[col + "_ema"], ls=ls_, lw=1.1, color=C_E,
                    label=f"{col}_ema — EMA")
    _marcar_inicializacao(ax, ep_marca, txt_marca, float(ep.max()))
    ax.set_ylabel("MSE neutro (log)")
    ax.set_title("val_mse4 (u, v, w, P) e val_mse6 (+ k, ω), sem peso de "
                 "zona nem de componente", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[3]
    for i_, z in enumerate(ZONE_KEYS):
        if f"val_{z}" in dv and f"val_{z}_ema" in dv:
            ax.semilogy(ep, dv[f"val_{z}"], "-", lw=1.0, color=f"C{i_}",
                        label=f"{z}")
            ax.semilogy(ep, dv[f"val_{z}_ema"], "--", lw=1.0, color=f"C{i_}")
    ax.plot([], [], "k-", lw=1.0, label="cheio: rede de treino")
    ax.plot([], [], "k--", lw=1.0, label="tracejado: EMA")
    _marcar_inicializacao(ax, ep_marca, txt_marca, float(ep.max()))
    ax.set_ylabel("val por zona (log)")
    ax.set_xlabel("época")
    ax.set_title("Validação por zona — mesma cor é a mesma zona", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "loss_history_comparacao.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  salvo: {out}")

    # --- tabela: uma linha por validação, com a marca explícita ---
    cols = ["epoch", "val_aligned", "val_aligned_ema"]
    tab = dv[cols + [c for c in ("val_mse4", "val_mse4_ema",
                                 "val_mse6", "val_mse6_ema") if c in dv]].copy()
    tab.insert(3, "razao_ema_sobre_treino", razao)
    tab["depois_da_marca"] = ep > ep_marca
    out_csv = os.path.join(output_dir, "comparacao_validacao.csv")
    salvar_csv(tab, out_csv)
    print(f"  tabela: {out_csv}")
    if _pos.any():
        print(f"  razão val_aligned_ema / val_aligned depois da época "
              f"{ep_marca:.0f}: média {float(razao[_pos].mean()):.3f}, "
              f"última {float(razao.iloc[-1]):.3f} "
              f"(< 1 = EMA melhor; {int(_pos.sum())} validações)")
    else:
        print(f"  >>> nenhuma validação depois da época {ep_marca:.0f}: "
              f"o treino é curto demais para comparar (calibração?).")
    return out


# ==============================================================================
# 14. SLICES CFD × PINN
# ------------------------------------------------------------------------------
# [02/11/2026] ema1. Duas mudanças:
#
#   - plot_slices ganhou `rotulo`, que vai no título de cada figura ("[EMA
#     (β=0.999), época 5000]"). As pastas siren e ema têm slices com os
#     MESMOS nomes de arquivo; fora da pasta, só o título as distingue.
#
#   - plot_diff_slices (nova, OPCIONAL — só roda com --diff-slices, seção
#     16; decisão de 02/11/2026). Mostra ONDE no campo o EMA muda a
#     previsão e se a mudança ajuda. Para cada slice, uma linha por campo
#     (|V|, P, k, ω, ν_t) e duas colunas:
#       (1) EMA − rede de treino: a diferença entre as duas PREVISÕES, sem
#           o CFD. Diz onde o EMA mexe;
#       (2) |erro do EMA| − |erro da rede de treino|, os dois contra o CFD.
#           Azul = o EMA ficou mais perto do CFD ali; vermelho = mais longe.
#     Para k, ω e ν_t, as duas colunas são em log10 (razão EMA/treino, e
#     |log10(PINN/CFD)|), porque o campo varre ordens de grandeza.
#     Grava as figuras em <comparação>/slices_diferenca/ e uma tabela por
#     slice com a fração de pontos em que o EMA ficou mais perto do CFD.
#     É CARO: a inferência das duas redes nos pontos dos slices é refeita
#     aqui, e o ν_t pela fórmula SST precisa de autodiff (--no-nut usa k/ω).
# ==============================================================================
def add_cylinder_xz(ax, y_slice):
    if abs(y_slice - YC_CYL) > R_CYL:
        return
    meia_corda = np.sqrt(max(R_CYL**2 - (y_slice - YC_CYL)**2, 0.0))
    ax.add_patch(mpatches.Rectangle(
        (XC_CYL - meia_corda, 0.0), 2 * meia_corda, H_CYL,
        linewidth=1.2, edgecolor="black", facecolor="white", zorder=10))


def add_cylinder_xy(ax, z_slice):
    if z_slice > H_CYL:
        return
    ax.add_patch(mpatches.Circle((XC_CYL, YC_CYL), R_CYL,
                                 linewidth=1.2, edgecolor="black",
                                 facecolor="white", zorder=10))


# ==============================================================================
# ESCALA DOS CAMPOS DE TURBULÊNCIA NOS PLOTS
# ------------------------------------------------------------------------------
# A rede treina em ln k e ln ω, e k varre ~1e-10 a 37 m²/s² neste caso. Num
# colorbar LINEAR isso faz 99,97% do domínio virar uma mancha de uma só cor:
# a figura fica correta e ilegível ao mesmo tempo. O log existe por isso.
#
# Mas a escala linear responde outra pergunta — "o valor em m²/s² está
# certo onde ele é grande?" — e é nela que o campo é comparável a qualquer
# outro resultado de CFD. Então o default é gerar AS DUAS, em arquivos
# separados, em vez de escolher.
#
#   "log"    -> só log10 (comportamento anterior)
#   "linear" -> só escala natural
#   "both"   -> as duas (arquivos *_lin.png a mais)
# ==============================================================================
ESCALA_K_OM = "both"

# ==============================================================================
# [23/11/2026] ν_t FORA DAS FIGURAS — orientação do orientador
# ------------------------------------------------------------------------------
# Não se sabe ainda o quanto a acurácia de ν_t importa para a análise, então
# ele sai de TODAS as figuras de comparação PINN × CFD e EMA × rede de
# treino:
#   - slices (seção 14): a linha de ν_t, na figura em log e na *_lin.png;
#   - slices da diferença (seção 14, --diff-slices): a linha de ν_t;
#   - scatter e histogramas (seção 15): os painéis de ν_t e o painel "ν_t
#     por RAMO do modelo";
#   - comparação MAE/méd|y| (seção 15): as barras de ν_t.
# Os NÚMEROS de ν_t continuam nas tabelas e nos CSVs (tabelas A e B,
# errors_by_slice.csv, diferenca_por_slice.csv, comparacao_MAE_med_abs.csv)
# e no relatório "ν_t POR REGIÃO" do console: tirar das figuras não apaga o
# registro, e a decisão pode ser revista sem rodar a inferência de novo.
# O cálculo de ν_t pela fórmula SST (autodiff, a parte cara) continua
# rodando por causa dessas tabelas; --no-nut o troca por k/ω.
#
# True volta ao comportamento anterior (ν_t em todas as figuras).
# ==============================================================================
PLOT_NU_T = False

def _row(axrow, h, v, cfd, pinn, nome, unidade, plano, slice_value,
         log=False, cmap="viridis"):
    """CFD | PINN | diferença, com a mesma escala de cor nos dois primeiros."""
    finite = np.isfinite(cfd) & np.isfinite(pinn)
    if log:
        pos = finite & (cfd > 0) & (pinn > 0)
        if pos.sum() < 10:
            log = False
    if log:
        lo = max(np.percentile(cfd[pos], 1), 1e-12)
        hi = np.percentile(np.concatenate([cfd[pos], pinn[pos]]), 99)
        norm = LogNorm(vmin=lo, vmax=max(hi, lo * 10))
        kw = dict(norm=norm)
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = np.log10(np.where(pinn > 0, pinn, np.nan)
                            / np.where(cfd > 0, cfd, np.nan))
        diff_lbl = f"log10(PINN/CFD) de {nome}"
        dmax = np.nanpercentile(np.abs(diff), 99) if np.isfinite(diff).any() else 1.0
        dmax = max(float(dmax), 1e-3)
        dkw = dict(vmin=-dmax, vmax=+dmax)
    else:
        lo = float(np.nanmin([cfd.min(), pinn.min()]))
        hi = float(np.nanmax([cfd.max(), pinn.max()]))
        kw = dict(vmin=lo, vmax=hi)
        diff = pinn - cfd
        diff_lbl = f"PINN − CFD de {nome} [{unidade}]"
        dmax = max(float(np.nanpercentile(np.abs(diff), 99)), 1e-9)
        dkw = dict(vmin=-dmax, vmax=+dmax)

    for ax, val, titulo, c, extra in [
            (axrow[0], cfd,  f"{nome} CFD [{unidade}]",  cmap, kw),
            (axrow[1], pinn, f"{nome} PINN [{unidade}]", cmap, kw),
            (axrow[2], diff, diff_lbl, "coolwarm", dkw)]:
        sc = ax.scatter(h, v, c=val, cmap=c, s=3, zorder=2, **extra)
        if plano == "xz":
            add_cylinder_xz(ax, slice_value)
            ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
            ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Z_MIN, Z_MAX)
        else:
            add_cylinder_xy(ax, slice_value)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
            ax.set_aspect("equal")
        ax.set_title(titulo, fontsize=9)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)


def plot_slices(model, snaps, t_targets, plano="xz", slab_tol=8.0,
                com_nut=True, rotulo=None):
    """rotulo: identifica a rede nos títulos (ver o cabeçalho da seção)."""
    _rot = f" [{rotulo}]" if rotulo else ""
    if plano == "xz":
        slice_values = [0, 100, 200, 250, 300, 400, 500, 600, 700]
        slice_label = "y"
    else:
        slice_values = [10, 32, 65, 100, 150]
        slice_label = "z"

    times_all = np.array([t for t, _ in snaps])
    linhas = []
    col_map = None

    for t_target in t_targets:
        idx_t = int(np.argmin(np.abs(times_all - t_target)))
        actual_t = float(times_all[idx_t])
        csv_path = snaps[idx_t][1]
        print(f"\n  t={t_target}s (real {actual_t:.1f}s) — "
              f"{os.path.basename(csv_path)}")
        d, col_map = load_snapshot(csv_path, col_map=col_map)

        P_cfd_all = d["P"]
        if P_cfd_all.mean() > 1e4:
            P_cfd_all = P_cfd_all - P_OP

        for slice_value in slice_values:
            coord = d["y"] if plano == "xz" else d["z"]
            slab = np.abs(coord - slice_value) < slab_tol
            valid = slab & ~is_inside_cylinder(d["x"], d["y"], d["z"])
            n_valid = int(valid.sum())
            if n_valid < 10:
                continue

            xs, ys, zs = d["x"][valid], d["y"][valid], d["z"][valid]
            us, vs, ws = d["u"][valid], d["v"][valid], d["w"][valid]
            Ps  = P_cfd_all[valid]
            ks  = d["k"][valid]
            oms = d["omega"][valid]
            nut_cfd = d["mut"][valid] / RHO_INF       # coluna é μ_t DINÂMICA

            h = xs
            vcoord = zs if plano == "xz" else ys
            Vmag_cfd = np.sqrt(us**2 + vs**2 + ws**2)

            f = predict_fields(model, xs, ys, zs, actual_t)
            Vmag_p = np.sqrt(f["u"]**2 + f["v"]**2 + f["w"]**2)

            if com_nut:
                mt = predict_mu_t(model, xs, ys, zs, actual_t)
                nut_p = mt["mu_t"] / RHO_INF
            else:
                nut_p = f["k"] / np.maximum(f["omega"], OMEGA_FLOOR)

            # ESCALA_K_OM governa as três linhas de turbulência; com
            # "both" a figura principal fica em log (como sempre) e sai uma
            # segunda figura *_lin.png com as mesmas três em escala natural.
            _lg = (ESCALA_K_OM != "linear")
            nrows = 5 if PLOT_NU_T else 4       # [23/11/2026] ν_t fora
            fig, axes = plt.subplots(nrows, 3, figsize=(16, 4.0 * nrows))
            _row(axes[0], h, vcoord, Vmag_cfd, Vmag_p, "|V|", "m/s",
                 plano, slice_value, log=False, cmap="viridis")
            _row(axes[1], h, vcoord, Ps, f["P"], "P", "Pa",
                 plano, slice_value, log=False, cmap="coolwarm")
            _row(axes[2], h, vcoord, ks, f["k"], "k", "m²/s²",
                 plano, slice_value, log=_lg, cmap="magma")
            _row(axes[3], h, vcoord, oms, f["omega"], "ω", "1/s",
                 plano, slice_value, log=_lg, cmap="cividis")
            if PLOT_NU_T:
                _row(axes[4], h, vcoord, nut_cfd, nut_p, "ν_t", "m²/s",
                     plano, slice_value, log=_lg, cmap="plasma")

            plt.suptitle(
                f"CFD × PINN SST{_rot} — {plano.upper()} | "
                f"{slice_label}={slice_value} m | t={actual_t:.1f} s | "
                f"n={n_valid}"
                + ("" if (com_nut or not PLOT_NU_T)
                   else "   [ν_t da PINN = k/ω, sem limitador nem teto: "
                        "--no-nut]"),
                fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.975])
            fname = (f"pinn_vs_cfd_{plano}_{slice_label}{slice_value:04d}"
                     f"_t{int(round(actual_t)):03d}s.png")
            fpath = os.path.join(SLICES_DIR, fname)
            plt.savefig(fpath, dpi=110, bbox_inches="tight")
            plt.close(fig)

            if ESCALA_K_OM == "both":
                _nr2 = 3 if PLOT_NU_T else 2    # [23/11/2026] ν_t fora
                fig2, ax2 = plt.subplots(_nr2, 3, figsize=(16, 4.0 * _nr2))
                _row(ax2[0], h, vcoord, ks, f["k"], "k", "m²/s²",
                     plano, slice_value, log=False, cmap="magma")
                _row(ax2[1], h, vcoord, oms, f["omega"], "ω", "1/s",
                     plano, slice_value, log=False, cmap="cividis")
                if PLOT_NU_T:
                    _row(ax2[2], h, vcoord, nut_cfd, nut_p, "ν_t", "m²/s",
                         plano, slice_value, log=False, cmap="plasma")
                plt.suptitle(
                    f"CFD × PINN SST{_rot} — turbulência em ESCALA LINEAR — "
                    f"{plano.upper()} | {slice_label}={slice_value} m | "
                    f"t={actual_t:.1f} s | n={n_valid}", fontsize=12)
                plt.tight_layout(rect=[0, 0, 1, 0.965])
                plt.savefig(os.path.join(SLICES_DIR,
                                         fname.replace(".png", "_lin.png")),
                            dpi=110, bbox_inches="tight")
                plt.close(fig2)

            def _mae(a, b):
                return float(np.mean(np.abs(a - b)))

            def _logmae(a, b):
                m = (a > 0) & (b > 0)
                return (float(np.mean(np.abs(np.log10(b[m]) - np.log10(a[m]))))
                        if m.sum() else float("nan"))

            linhas.append(dict(
                t=actual_t, slice_label=slice_label, slice_value=slice_value,
                n_pts=n_valid,
                V_mae=_mae(Vmag_cfd, Vmag_p),
                V_max=float(np.abs(Vmag_p - Vmag_cfd).max()),
                P_mae=_mae(Ps, f["P"]),
                P_max=float(np.abs(f["P"] - Ps).max()),
                k_logmae=_logmae(ks, f["k"]),
                om_logmae=_logmae(oms, f["omega"]),
                nut_logmae=_logmae(nut_cfd, nut_p)))

            print(f"    {slice_label}={slice_value:4d} m | n={n_valid:>6d} | "
                  f"|V| MAE={linhas[-1]['V_mae']:6.2f} | "
                  f"P MAE={linhas[-1]['P_mae']:7.1f} | "
                  f"k dex={linhas[-1]['k_logmae']:5.2f} | "
                  f"ω dex={linhas[-1]['om_logmae']:5.2f} | "
                  f"ν_t dex={linhas[-1]['nut_logmae']:5.2f}", flush=True)

    df = pd.DataFrame(linhas)
    out = os.path.join(SUMMARY_DIR, "errors_by_slice.csv")
    salvar_csv(df, out)
    print(f"\n  resumo: {out}")
    if len(df):
        print(f"    |V| MAE médio : {df['V_mae'].mean():.3f} m/s")
        print(f"    P   MAE médio : {df['P_mae'].mean():.1f} Pa")
        print(f"    k   erro médio: {df['k_logmae'].mean():.2f} dex "
              f"(1 dex = fator 10)")
        print(f"    ω   erro médio: {df['om_logmae'].mean():.2f} dex")
        print(f"    ν_t erro médio: {df['nut_logmae'].mean():.2f} dex")
    return df


def _eixos_slice(ax, plano, slice_value):
    """Limites, rótulos e o cilindro — o mesmo que _row faz por painel."""
    if plano == "xz":
        add_cylinder_xz(ax, slice_value)
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
        ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Z_MIN, Z_MAX)
    else:
        add_cylinder_xy(ax, slice_value)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_aspect("equal")


def _painel_simetrico(ax, h, v, val, titulo, plano, slice_value):
    """Mapa em coolwarm com escala simétrica no p99 de |val|."""
    fin = np.isfinite(val)
    dmax = (float(np.nanpercentile(np.abs(val[fin]), 99))
            if fin.any() else 1.0)
    dmax = max(dmax, 1e-12)
    sc = ax.scatter(h, v, c=val, cmap="coolwarm", s=3, zorder=2,
                    vmin=-dmax, vmax=+dmax)
    _eixos_slice(ax, plano, slice_value)
    ax.set_title(titulo, fontsize=9)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)


def plot_diff_slices(model_siren, model_ema, snaps, t_targets, output_dir,
                     plano="xz", slab_tol=8.0, com_nut=True,
                     rotulo_siren="rede de treino", rotulo_ema="EMA",
                     aviso_par=""):
    """
    [02/11/2026] Slices da diferença EMA − rede de treino (ver o cabeçalho
    desta seção). Mesmos slices e instantes de plot_slices. Só roda com
    --diff-slices (seção 16).
    """
    os.makedirs(output_dir, exist_ok=True)
    if plano == "xz":
        slice_values = [0, 100, 200, 250, 300, 400, 500, 600, 700]
        slice_label = "y"
    else:
        slice_values = [10, 32, 65, 100, 150]
        slice_label = "z"
    times_all = np.array([t for t, _ in snaps])
    linhas, col_map = [], None
    _sup2 = f"\n>>> {aviso_par}" if aviso_par else ""

    for t_target in t_targets:
        idx_t = int(np.argmin(np.abs(times_all - t_target)))
        actual_t = float(times_all[idx_t])
        csv_path = snaps[idx_t][1]
        print(f"\n  [diferença] t={t_target}s (real {actual_t:.1f}s) — "
              f"{os.path.basename(csv_path)}")
        d, col_map = load_snapshot(csv_path, col_map=col_map)
        P_cfd_all = d["P"]
        if P_cfd_all.mean() > 1e4:
            P_cfd_all = P_cfd_all - P_OP

        for slice_value in slice_values:
            coord = d["y"] if plano == "xz" else d["z"]
            slab = np.abs(coord - slice_value) < slab_tol
            valid = slab & ~is_inside_cylinder(d["x"], d["y"], d["z"])
            n_valid = int(valid.sum())
            if n_valid < 10:
                continue
            xs, ys, zs = d["x"][valid], d["y"][valid], d["z"][valid]
            h = xs
            vcoord = zs if plano == "xz" else ys

            cfd = dict(
                V=np.sqrt(d["u"][valid]**2 + d["v"][valid]**2
                          + d["w"][valid]**2),
                P=P_cfd_all[valid], k=d["k"][valid], om=d["omega"][valid],
                nut=d["mut"][valid] / RHO_INF)
            prev = {}
            for nome, net in (("siren", model_siren), ("ema", model_ema)):
                f = predict_fields(net, xs, ys, zs, actual_t)
                if com_nut:
                    nut = predict_mu_t(net, xs, ys, zs, actual_t)["mu_t"] / RHO_INF
                else:
                    nut = f["k"] / np.maximum(f["omega"], OMEGA_FLOOR)
                prev[nome] = dict(
                    V=np.sqrt(f["u"]**2 + f["v"]**2 + f["w"]**2),
                    P=f["P"], k=f["k"], om=f["omega"], nut=nut)

            campos = [("V", "|V|", "m/s", False), ("P", "P", "Pa", False),
                      ("k", "k", "m²/s²", True), ("om", "ω", "1/s", True),
                      ("nut", "ν_t", "m²/s", True)]
            # [23/11/2026] ν_t sai da FIGURA, mas continua no CSV e no log
            campos_fig = [c_ for c_ in campos if PLOT_NU_T or c_[0] != "nut"]
            fig, axes = plt.subplots(len(campos_fig), 2,
                                     figsize=(12, 4.0 * len(campos_fig)))
            eixos = {c_[0]: ax_ for c_, ax_ in zip(campos_fig, axes)}
            linha = dict(t=actual_t, slice_label=slice_label,
                         slice_value=slice_value, n_pts=n_valid)
            for ch, nome, un, log in campos:
                ps, pe, c = prev["siren"][ch], prev["ema"][ch], cfd[ch]
                if log:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ok = (ps > 0) & (pe > 0) & (c > 0)
                        dif = np.where(ok, np.log10(pe / ps), np.nan)
                        e_s = np.where(ok, np.abs(np.log10(ps / c)), np.nan)
                        e_e = np.where(ok, np.abs(np.log10(pe / c)), np.nan)
                    t1 = f"log10(EMA / treino) de {nome}"
                    t2 = (f"|err EMA| − |err treino| de {nome} [dex]\n"
                          f"azul: EMA mais perto do CFD")
                else:
                    dif = pe - ps
                    e_s, e_e = np.abs(ps - c), np.abs(pe - c)
                    t1 = f"EMA − treino de {nome} [{un}]"
                    t2 = (f"|err EMA| − |err treino| de {nome} [{un}]\n"
                          f"azul: EMA mais perto do CFD")
                ganho = e_e - e_s
                if ch in eixos:
                    axrow = eixos[ch]
                    _painel_simetrico(axrow[0], h, vcoord, dif, t1, plano,
                                      slice_value)
                    _painel_simetrico(axrow[1], h, vcoord, ganho, t2, plano,
                                      slice_value)
                fin = np.isfinite(ganho)
                linha[f"{ch}_frac_ema_melhor"] = (float(np.mean(ganho[fin] < 0))
                                                  if fin.any() else float("nan"))
                linha[f"{ch}_delta_mae"] = (float(np.mean(ganho[fin]))
                                            if fin.any() else float("nan"))
                linha[f"{ch}_dif_abs_media"] = float(np.nanmean(np.abs(dif)))
            linhas.append(linha)

            plt.suptitle(
                f"EMA − rede de treino — {plano.upper()} | "
                f"{slice_label}={slice_value} m | t={actual_t:.1f} s | "
                f"n={n_valid}\n{rotulo_ema}  ×  {rotulo_siren}"
                + ("" if (com_nut or not PLOT_NU_T)
                   else "   [ν_t = k/ω: --no-nut]") + _sup2,
                fontsize=11)
            plt.tight_layout(rect=[0, 0, 1, 0.955 if aviso_par else 0.965])
            fname = (f"diferenca_ema_treino_{plano}_{slice_label}"
                     f"{slice_value:04d}_t{int(round(actual_t)):03d}s.png")
            plt.savefig(os.path.join(output_dir, fname), dpi=110,
                        bbox_inches="tight")
            plt.close(fig)
            print(f"    {slice_label}={slice_value:4d} m | n={n_valid:>6d} | "
                  f"EMA mais perto do CFD em: "
                  + "  ".join(f"{nm} {100*linha[ch + '_frac_ema_melhor']:.0f}%"
                              for ch, nm, _, _ in campos), flush=True)

    df = pd.DataFrame(linhas)
    if aviso_par and len(df):
        df["aviso"] = aviso_par
    out = os.path.join(output_dir, "diferenca_por_slice.csv")
    salvar_csv(df, out)
    print(f"\n  resumo: {out}")
    if len(df):
        print(f"  fração média de pontos com o EMA mais perto do CFD: "
              + "  ".join(f"{nm} {100*df[ch + '_frac_ema_melhor'].mean():.0f}%"
                          for ch, nm in (("V", "|V|"), ("P", "P"), ("k", "k"),
                                         ("om", "ω"), ("nut", "ν_t"))))
    return df


# ==============================================================================
# 15. SCATTER, HISTOGRAMAS E TABELA
# ------------------------------------------------------------------------------
# [02/11/2026] ema1. As figuras e a tabela de cada rede são as de sempre,
# geradas uma vez por rede, cada uma na sua pasta (seção 16). Três mudanças:
#
#   - OS TÍTULOS DIZEM QUAL REDE É ("[EMA (β=0.999), época 5000]" ou
#     "[rede de treino, época 5000]"). As duas pastas têm figuras com os
#     mesmos nomes de arquivo e, fora da pasta, nada mais as distinguiria.
#     O rótulo vem do perfil em uso (_rotulo_rede) mais o checkpoint, que a
#     seção 16 passa em `rotulo`.
#   - plot_scatter_and_histograms devolve a tabela com o instante e o n de
#     pontos em df.attrs, para a comparação conferir que as duas redes
#     foram medidas NO MESMO SNAPSHOT e nos mesmos pontos (a amostragem é
#     a mesma: default_rng(0) sobre o mesmo CSV).
#   - plot_metric_comparison (nova), para a pasta de comparação: MAE/méd|y|
#     das duas redes lado a lado, por variável, mais um CSV com as duas
#     colunas e a diferença. Só MAE/méd|y|, por decisão de 02/11/2026: é a
#     métrica principal da tabela (B). Dois painéis, porque as linhas log10
#     de k, ω e ν_t têm outra leitura (MAE em dex sobre a média de |log10|,
#     não um erro percentual do campo físico — ver o rodapé da tabela B).
#
# [23/11/2026] ν_t fora das figuras desta seção (scatter, histogramas, painel
# de ramos e barras da comparação MAE/méd|y|), por orientação do orientador.
# As tabelas e CSVs continuam com ν_t. Chave e motivo: PLOT_NU_T, seção 14.
# ==============================================================================
# ==============================================================================
# ESCRITA DE CSV — separador, decimal e codificação
# ------------------------------------------------------------------------------
# Dois defeitos reais, os dois meus, que apareceram ao abrir o
# final_metrics_snapshot.csv no Excel em português:
#
# (1) "A pressão está na casa dos quadrilhões."
#     O arquivo trazia MAE = 4.320142665740032 Pa. Num Excel em pt-BR o ponto
#     é separador de MILHAR, e 15 dígitos depois do ponto são exatamente 5
#     grupos de 3 — agrupamento válido. O Excel lê 4.320.142.665.740.032.
#     Só a linha do P estourou porque é a única com valor > 1 cujos dígitos
#     formam agrupamento válido; as outras têm 16 dígitos (não múltiplo de 3)
#     e viram texto.
#     Causa de fundo: eu escrevia o float64 inteiro, 16-17 algarismos
#     significativos, numa métrica de ERRO. Seis bastam, e seis também
#     tornam o agrupamento improvável.
#
# (2) "ω e ν aparecem como interrogação."
#     O pandas escreve UTF-8 sem BOM; o Excel, ao abrir um .csv com duplo
#     clique, assume a codepage do sistema. Resultado: mojibake.
#
# Correções: nomes ASCII na coluna `var` (omega, nu_t), BOM na codificação,
# arredondamento para 6 algarismos, e separador/decimal configuráveis.
CSV_LOCALE  = "pt_BR"        # "pt_BR" -> sep=';' decimal=','   |   "en_US" -> ',' e '.'
CSV_SIGFIGS = 6              # algarismos significativos nas métricas

# Símbolo bonito -> nome ASCII. O símbolo continua no console e nas figuras,
# onde renderiza bem; só o CSV usa o nome por extenso.
VAR_ASCII = {"ω": "omega", "ν_t": "nu_t", "|V|": "V_mag"}


def _sig(x, n=None):
    """Arredonda para n algarismos significativos, preservando int/str/NaN."""
    n = CSV_SIGFIGS if n is None else n
    if not isinstance(x, (int, float, np.floating, np.integer)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    v = float(x)
    if not np.isfinite(v) or v == 0.0:
        return v
    return float(np.round(v, -int(np.floor(np.log10(abs(v)))) + (n - 1)))


def salvar_csv(df, caminho):
    """
    Grava o DataFrame de um jeito que abre certo no Excel pt-BR E continua
    legível por pandas (`pd.read_csv(p, sep=';', decimal=',')`).
    """
    d = df.copy()
    if "var" in d.columns:
        d["var"] = d["var"].map(lambda v: VAR_ASCII.get(v, v))
    for c in d.columns:
        if d[c].dtype.kind in "fc":
            d[c] = d[c].map(_sig)
    sep, dec = ((";", ",") if CSV_LOCALE == "pt_BR" else (",", "."))
    d.to_csv(caminho, index=False, sep=sep, decimal=dec,
             encoding="utf-8-sig")
    return caminho


def _lim_robusto(cc, pp, q=0.2, folga=0.05):
    """
    Limites de eixo por PERCENTIL, não por min/max.

    Motivo, medido: no CSV deste caso k vai de 1e-10 a 37,8 e μ_t de
    9,1e-15 a 1,79. Em log10 isso é 11,6 e 14,3 DÉCADAS de eixo, enquanto
    99% dos pontos cabem em ~4. Com min/max, a nuvem toda vira um borrão no
    canto superior direito e o gráfico não mostra nada — foi o que
    aconteceu, e é fácil confundir com um efeito físico (o teto de ν_t, por
    exemplo). O teto apareceria como um PATAMAR horizontal em
    log10(1,4607) = +0,165, não como um canto.

    Recorta só o ENQUADRAMENTO. As métricas continuam sobre todos os pontos,
    e o número de pontos fora do quadro é anotado na figura.
    """
    v = np.concatenate([np.asarray(cc, float), np.asarray(pp, float)])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0, 0
    lo, hi = np.percentile(v, [q, 100.0 - q])
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max()) or 1.0
    d = (hi - lo) * folga
    lo, hi = lo - d, hi + d
    fora = int(np.sum((np.asarray(cc) < lo) | (np.asarray(cc) > hi)
                      | (np.asarray(pp) < lo) | (np.asarray(pp) > hi)))
    return float(lo), float(hi), fora


def _painel_scatter(ax, c, p, nome, un, log, sig_ref=None):
    """Um painel do scatter. Devolve (cc, pp) já no espaço usado."""
    if log:
        m = (c > 0) & (p > 0)
        cc, pp = np.log10(np.asarray(c)[m]), np.log10(np.asarray(p)[m])
        rot = f"log10 {nome}"
    else:
        cc, pp = np.asarray(c, float), np.asarray(p, float)
        rot = f"{nome} [{un}]"
    ax.scatter(cc, pp, s=1, alpha=0.25, c="steelblue")
    lo, hi, n_fora = _lim_robusto(cc, pp)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.4, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    # as MÉTRICAS usam todos os pontos; só o enquadramento é recortado
    mae  = float(np.mean(np.abs(pp - cc)))
    rmse = float(np.sqrt(np.mean((pp - cc)**2)))
    bias = float(np.mean(pp - cc))
    corr = float(np.corrcoef(cc, pp)[0, 1]) if len(cc) > 2 else float("nan")
    sig  = float(np.std(cc)) if sig_ref is None else sig_ref
    rel  = f" | {100*mae/sig:.0f}% de σ" if sig > 0 else ""
    # ------------------------------------------------------------------
    # MÉTRICA PRINCIPAL (decisão do orientador): MAE dividido pela MÉDIA DO
    # MÓDULO da variável. Vem PRIMEIRO, sozinha na linha de cima do título.
    #
    # Por que o módulo e não a média simples: u, w e P têm média ~0 — os
    # dois primeiros por simetria do escoamento, a pressão por ser
    # manométrica — e dividir por ~0 faz a porcentagem explodir e trocar
    # de sinal sem que nada físico mude. A média do módulo é a magnitude
    # típica do campo e não sofre cancelamento. Nos campos que não trocam
    # de sinal (v, |V|, k, ω, ν_t) os dois denominadores são o mesmo
    # número. Ver a coluna |méd|/σ na tabela (B).
    # ------------------------------------------------------------------
    mu_abs = float(np.mean(np.abs(cc)))
    pct = 100.0 * mae / mu_abs if mu_abs > 0 else float("nan")
    cab = "—" if not np.isfinite(pct) else f"{pct:.1f}%"
    ax.set_xlabel(f"{rot} CFD"); ax.set_ylabel(f"{rot} PINN")
    # Em log10 a conta é sobre log10(y): NÃO é erro percentual do campo
    # físico, e o rótulo diz isso em vez de deixar subentendido. O número
    # com a leitura física está no painel linear da mesma variável.
    ax.set_title(f"{nome}:  MAE/méd|y| = {cab}"
                 + ("   [em dex, não no campo físico]" if log else "")
                 + f"\nMAE={mae:.3g}{' dex' if log else ''}{rel} | "
                   f"RMSE={rmse:.3g} | viés={bias:+.3g} | r={corr:.3f}",
                 fontsize=9.5)
    if n_fora:
        ax.text(0.98, 0.02, f"{n_fora:,} pts fora do quadro\n"
                            f"(métricas incluem todos)",
                transform=ax.transAxes, fontsize=6.5, ha="right",
                va="bottom", color="dimgray")
    if nome == "ν_t":
        if log:
            ax.axhline(np.log10(NU_T_MAX), color="red", ls=":", lw=1.0)
        else:
            ax.axhline(NU_T_MAX, color="red", ls=":", lw=1.0)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    return cc, pp


def _painel_hist(ax, c, p, nome, un, log):
    if log:
        m = (c > 0) & (p > 0)
        err = np.log10(np.asarray(p)[m]) - np.log10(np.asarray(c)[m])
        rot = f"log10(PINN/CFD) de {nome} [dex]"
    else:
        err = np.asarray(p, float) - np.asarray(c, float)
        rot = f"erro {nome} (PINN − CFD) [{un}]"
    _lo, _hi, _fora = _lim_robusto(err, err, q=0.2)
    ax.hist(err, bins=120, range=(_lo, _hi), alpha=0.75,
            edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="red", ls="--", lw=1.3)
    med = float(np.median(err))
    q1, q3 = np.percentile(err, [25, 75])
    ax.axvline(med, color="green", ls=":", lw=1.3, label=f"mediana={med:+.3g}")
    ax.set_xlabel(rot); ax.set_ylabel("frequência")
    ax.set_title(f"{nome}: Q1={q1:+.3g}  Q3={q3:+.3g}  IQR={q3-q1:.3g}",
                 fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def _grade(n_paineis, ncol, largura=16.0, altura_painel=5.0):
    nrow = int(np.ceil(n_paineis / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(largura, altura_painel * nrow))
    axes = np.atleast_1d(axes).ravel()
    return fig, axes


def _painel_ramo(ax, nut_c, nut_p, k_eff, om_p):
    """ν_t colorido pelo RAMO do modelo. Um patamar horizontal no ν_t
    (PINN parada enquanto o CFD varia décadas) tem três causas possíveis e
    elas são indistinguíveis numa nuvem monocromática: teto, limitador de
    cisalhamento, ou k/ω já errado. Colorir pelo ramo resolve em um olhar."""
    teto_p, lim_p = _ramos_constitutivos(nut_p, k_eff, om_p)
    mok = (nut_c > 0) & (nut_p > 0)
    cc, pp = np.log10(nut_c[mok]), np.log10(nut_p[mok])
    livre = (~teto_p & ~lim_p)[mok]
    for msk, cor, rot in ((livre, "tab:green", "livre: ν_t = k/ω"),
                          (lim_p[mok], "tab:orange", "limitador de cisalhamento"),
                          (teto_p[mok], "tab:red", "teto do solver")):
        if msk.sum():
            ax.scatter(cc[msk], pp[msk], s=1, alpha=0.3, c=cor,
                       label=f"{rot}  ({100*msk.mean():.0f}%)")
    lo, hi, _ = _lim_robusto(cc, pp)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax.axhline(np.log10(NU_T_MAX), color="red", ls=":", lw=1.0)
    ax.text(lo, np.log10(NU_T_MAX), " teto", color="red", fontsize=7,
            va="bottom")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("log10 ν_t CFD"); ax.set_ylabel("log10 ν_t PINN")
    ax.set_title("ν_t por RAMO do modelo (classificação da PINN)", fontsize=9)
    leg = ax.legend(loc="upper left", fontsize=7, markerscale=6)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def _rotulo_rede():
    """'EMA (β=0.999)' ou 'rede de treino', pelo perfil em uso."""
    if RUNS.get(RUN_TAG, {}).get("rede") == "ema":
        return f"EMA (β={EMA_CFG['beta']:g})"
    return "rede de treino"


def plot_scatter_and_histograms(model, snaps, t_target=None, com_nut=True,
                                max_pts=250_000, rotulo=None):
    """
    rotulo: o que identifica a rede nos títulos (ver o cabeçalho desta
    seção). None usa só _rotulo_rede(), sem o checkpoint.
    """
    _rot = rotulo or _rotulo_rede()
    if t_target is None:
        actual_t, csv_path = snaps[-1]
    else:
        times_all = np.array([t for t, _ in snaps])
        i = int(np.argmin(np.abs(times_all - t_target)))
        actual_t, csv_path = snaps[i]

    print(f"\n  scatter/histogramas em t={actual_t:.1f}s "
          f"({os.path.basename(csv_path)})")
    d, _ = load_snapshot(csv_path)

    P_cfd = d["P"]
    if P_cfd.mean() > 1e4:
        P_cfd = P_cfd - P_OP

    n_all = len(d["x"])
    if n_all > max_pts:
        sel = np.random.default_rng(0).choice(n_all, max_pts, replace=False)
        sel.sort()
    else:
        sel = np.arange(n_all)

    x, y, z = d["x"][sel], d["y"][sel], d["z"][sel]
    u_c, v_c, w_c = d["u"][sel], d["v"][sel], d["w"][sel]
    P_c = P_cfd[sel]
    k_c, om_c = d["k"][sel], d["omega"][sel]
    nut_c = d["mut"][sel] / RHO_INF
    V_c = np.sqrt(u_c**2 + v_c**2 + w_c**2)
    n = len(x)

    f = predict_fields(model, x, y, z, actual_t)
    V_p = np.sqrt(f["u"]**2 + f["v"]**2 + f["w"]**2)
    mt = None
    if com_nut:
        mt = predict_mu_t(model, x, y, z, actual_t, verbose=True)
        nut_p = mt["mu_t"] / RHO_INF
    else:
        nut_p = f["k"] / np.maximum(f["omega"], OMEGA_FLOOR)

    # --------------------------------------------------------------------
    # DOIS GRUPOS DE VARIÁVEIS, DUAS FIGURAS
    # Com as 8 variáveis numa grade 3x3 cada painel ficava pequeno demais
    # para enxergar a nuvem. A divisão também é conceitual: o primeiro
    # grupo é exatamente o que a PINN LAMINAR já previa, então é a figura
    # que compara os dois trabalhos; o segundo é o que a turbulência
    # acrescentou.
    # (cfd, pinn, nome, unidade, log?)
    # --------------------------------------------------------------------
    base = [
        (u_c,   f["u"],     "u",    "m/s",   False),
        (v_c,   f["v"],     "v",    "m/s",   False),
        (w_c,   f["w"],     "w",    "m/s",   False),
        (V_c,   V_p,        "|V|",  "m/s",   False),
        (P_c,   f["P"],     "P",    "Pa",    False),
    ]
    turb_raw = [
        (k_c,   f["k"],     "k",    "m²/s²", None),
        (om_c,  f["omega"], "ω",    "1/s",   None),
        (nut_c, nut_p,      "ν_t",  "m²/s",  None),
    ]
    quer_log = ESCALA_K_OM in ("log", "both")
    quer_lin = ESCALA_K_OM in ("linear", "both")
    turb_log = [(c, p, nm, un, True)  for c, p, nm, un, _ in turb_raw]
    turb_lin = [(c, p, nm, un, False) for c, p, nm, un, _ in turb_raw]
    # [23/11/2026] as FIGURAS de turbulência ficam sem ν_t (PLOT_NU_T); a
    # tabela, mais abaixo, continua com turb_log/turb_lin completos
    _fig_ok = (lambda cp: PLOT_NU_T or cp[2] != "ν_t")
    turb_log_fig = [cp for cp in turb_log if _fig_ok(cp)]
    turb_lin_fig = [cp for cp in turb_lin if _fig_ok(cp)]
    _sem_nut = "" if PLOT_NU_T else " (k e ω; ν_t fora das figuras)"

    # ---------------- scatter: figura 1, variáveis da laminar -------------
    fig, axes = _grade(6, 3, largura=16.0, altura_painel=5.2)
    for cp, ax in zip(base, axes):
        _painel_scatter(ax, *cp)
    axes[5].axis("off")
    plt.suptitle(f"PINN SST [{_rot}] × CFD — variáveis do escoamento médio "
                 f"(as mesmas da PINN laminar)\nt={actual_t:.1f}s | "
                 f"n={n:,} pontos", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out = os.path.join(SCATTER_DIR, "pinn_vs_cfd_scatter_base.png")
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"  salvo: {out}")

    # ---------------- scatter: figura 2, turbulência em log ---------------
    if quer_log:
        if PLOT_NU_T:
            fig, axes = _grade(4, 2, largura=11.5, altura_painel=5.4)
        else:
            fig, axes = _grade(2, 2, largura=11.5, altura_painel=5.4)
        for cp, ax in zip(turb_log_fig, axes):
            _painel_scatter(ax, *cp)
        if PLOT_NU_T:
            if mt is not None:
                _painel_ramo(axes[3], nut_c, nut_p,
                             f["k"] + CONSTANTS["K_EPS"], f["omega"])
            else:
                axes[3].axis("off")
        plt.suptitle(f"PINN SST [{_rot}] × CFD — campos de turbulência{_sem_nut}, log10\n"
                     f"é o espaço em que a rede foi treinada | "
                     f"t={actual_t:.1f}s | n={n:,}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.945])
        out = os.path.join(SCATTER_DIR, "pinn_vs_cfd_scatter_turb.png")
        plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
        print(f"  salvo: {out}")

    # ---------------- scatter: figura 3, turbulência linear ---------------
    if quer_lin:
        _nc = len(turb_lin_fig)
        fig, axes = _grade(_nc, _nc, largura=16.0 * _nc / 3 if _nc < 3 else 16.0,
                           altura_painel=5.4)
        for cp, ax in zip(turb_lin_fig, axes):
            _painel_scatter(ax, *cp)
        plt.suptitle(f"PINN SST [{_rot}] × CFD — campos de turbulência{_sem_nut}, escala linear\n"
                     f"t={actual_t:.1f}s | n={n:,}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(SCATTER_DIR, "pinn_vs_cfd_scatter_turb_lin.png")
        plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
        print(f"  salvo: {out}")

    # ---------------- histogramas, mesma divisão --------------------------
    fig, axes = _grade(6, 3, largura=16.0, altura_painel=3.7)
    for cp, ax in zip(base, axes):
        _painel_hist(ax, *cp)
    axes[5].axis("off")
    plt.suptitle(f"Distribuição do erro [{_rot}] — escoamento médio — "
                 f"t={actual_t:.1f}s", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HIST_DIR, "pinn_vs_cfd_errors_hist_base.png")
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"  salvo: {out}")

    for grupo, sufixo, rotulo in ((turb_log_fig, "turb", "log10"),
                                  (turb_lin_fig, "turb_lin", "escala linear")):
        if (rotulo == "log10" and not quer_log) or \
           (rotulo != "log10" and not quer_lin):
            continue
        _nc = len(grupo)
        fig, axes = _grade(_nc, _nc, largura=16.0 * _nc / 3 if _nc < 3 else 16.0,
                           altura_painel=3.9)
        for cp, ax in zip(grupo, axes):
            _painel_hist(ax, *cp)
        plt.suptitle(f"Distribuição do erro [{_rot}] — turbulência{_sem_nut} ({rotulo}) — "
                     f"t={actual_t:.1f}s", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(HIST_DIR,
                           f"pinn_vs_cfd_errors_hist_{sufixo}.png")
        plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
        print(f"  salvo: {out}")

    # ---------------- tabela ----------------
    # Escalas FÍSICAS do problema, não as de normalização. V_REF=5 é um
    # número da rede, não do escoamento; usar V_INF.
    P_SCALE = 0.5 * RHO_INF * V_INF**2            # ≈ 177 Pa
    escalas = {"u": V_INF, "v": V_INF, "w": V_INF, "|V|": V_INF, "P": P_SCALE,
               "k": K_INLET, "ω": OMEGA_INLET, "ν_t": NU_T_MAX}

    campos = list(base) + (turb_log if quer_log else []) \
                        + (turb_lin if quer_lin else [])

    linhas = []
    for c, p, nome, un, log in campos:
        n_c0 = n_p0 = 0
        if log:
            m = (c > 0) & (p > 0)
            n_c0 = int(np.sum(np.asarray(c) <= 0))
            n_p0 = int(np.sum(np.asarray(p) <= 0))
            cc, pp = np.log10(np.asarray(c)[m]), np.log10(np.asarray(p)[m])
            unid = "dex"
        else:
            cc, pp = np.asarray(c, float), np.asarray(p, float)
            unid = un
        err = pp - cc
        ae = np.abs(err)
        mae = float(ae.mean()); rmse = float(np.sqrt(np.mean(err**2)))
        bias = float(err.mean()); med = float(np.median(ae))
        corr = float(np.corrcoef(cc, pp)[0, 1]) if len(cc) > 2 else float("nan")
        mu_c  = float(cc.mean())
        sig_c = float(cc.std())
        sstot = float(np.sum((cc - mu_c)**2))
        r2 = float(1.0 - np.sum(err**2) / sstot) if sstot > 0 else float("nan")
        rng_ = float(cc.max() - cc.min())
        nrmse = 100.0 * rmse / rng_ if rng_ > 0 else float("nan")

        # ---- os denominadores do erro relativo -------------------------
        # MÉDIA (MAE/méd): o erro percentual pedido pelo orientador — MAE da
        #   variável dividido pela média daquela variável. É SEMPRE
        #   calculado, sem filtro. O cuidado na leitura está na coluna
        #   |méd|/σ ao lado: quando ela é << 1 a média é pequena perante a
        #   dispersão e a porcentagem fica instável (u e w têm média ~0 por
        #   simetria; a P depende de onde se põe o zero — aqui P_OP já foi
        #   subtraída). A linha fica marcada com '*' nesse caso, mas o
        #   número é impresso do mesmo jeito.
        # MÉDIA DO MÓDULO (MAE/méd|y|): mesma ideia, dividindo pela média
        #   de |valor|. Não degenera quando o campo é simétrico em torno de
        #   zero, e é o número a usar no lugar do anterior quando |méd|/σ
        #   for pequeno.
        # σ: erro contra o SINAL que havia para prever. RMSE/σ = 100% é
        #   exatamente o escore de quem chutasse a média em todo ponto.
        # escala física: fixa a priori, não depende do dataset, então
        #   compara entre rodadas e contra a laminar.
        # Em log10 a "escala" é 1 dex: 10% dela = fator 1,26.
        esc = 1.0 if log else escalas.get(nome, rng_)
        mu_abs = float(np.mean(np.abs(cc)))
        mae_sigma  = 100.0 * mae / sig_c if sig_c > 0 else float("nan")
        rmse_sigma = 100.0 * rmse / sig_c if sig_c > 0 else float("nan")
        mae_esc    = 100.0 * mae / esc if esc > 0 else float("nan")
        mae_media  = 100.0 * mae / abs(mu_c) if abs(mu_c) > 0 else float("nan")
        mae_med_abs = 100.0 * mae / mu_abs if mu_abs > 0 else float("nan")
        razao_ms   = abs(mu_c) / sig_c if sig_c > 0 else float("nan")
        a10 = float(100.0 * np.mean(ae < 0.10 * esc))
        a20 = float(100.0 * np.mean(ae < 0.20 * esc))

        n_fora = int(n - len(cc))
        # ORDEM DAS COLUNAS DO CSV: MAE_media_abs_pct é a MÉTRICA PRINCIPAL
        # e por isso vem logo depois de var/unidade/espaco — é a coluna D da
        # planilha. O dict preserva a ordem de inserção, e é ela que o
        # to_csv usa; mexer aqui é o que muda a planilha.
        linhas.append(dict(
            var=nome, unidade=unid, espaco="log10" if log else "linear",
            MAE_media_abs_pct=mae_med_abs,
            MAE=mae, RMSE=rmse, bias=bias, median_abs_err=med,
            R2=r2, corr=corr,
            media_CFD=mu_c, sigma_CFD=sig_c, escala=esc,
            MAE_sigma_pct=mae_sigma, RMSE_sigma_pct=rmse_sigma,
            MAE_escala_pct=mae_esc,
            MAE_media_pct=mae_media,
            media_abs_CFD=mu_abs, abs_media_sobre_sigma=razao_ms,
            NRMSE_pct=nrmse, acc_10pct=a10, acc_20pct=a20,
            n_usado=len(cc), n_total=int(n), n_excluido=n_fora,
            pct_excluido=100.0 * n_fora / max(n, 1),
            n_zero_cfd=n_c0, n_zero_pinn=n_p0))

    def _pc(v, w=9):
        """Porcentagem em largura fixa. Um '—' quando o denominador não
        serve, e notação científica quando o número estoura a coluna — era
        o que embolava 'MAE/méd' com a coluna seguinte."""
        if not np.isfinite(v):
            t = "—"
        elif abs(v) >= 1.0e4:
            t = f"{v:.2g}%"
        else:
            t = f"{v:.1f}%"
        return f"{t:>{w}}"

    def _rr(v, w=9):
        """R² em largura fixa. Um R² muito negativo (modelo pior que chutar
        a média) tem dezenas de dígitos e invadia a coluna vizinha."""
        if not np.isfinite(v):
            t = "—"
        elif v < -99.0:
            t = "<-99"
        else:
            t = f"{v:+.3f}"
        return f"{t:>{w}}"

    print(f"\n  {'='*96}")
    print(f"  (A) ERRO ABSOLUTO [{_rot}] — t={actual_t:.1f}s | n={n:,}")
    print(f"  {'='*96}")
    print(f"  {'var':<6}{'espaço':<9}{'unid':<8}{'MAE':>12}{'RMSE':>12}"
          f"{'viés':>12}{'med|err|':>12}{'R²':>9}{'r':>8}")
    print(f"  {'-'*96}")
    for L in linhas:
        print(f"  {L['var']:<6}{L['espaco']:<9}{L['unidade']:<8}"
              f"{L['MAE']:>12.4g}{L['RMSE']:>12.4g}{L['bias']:>+12.4g}"
              f"{L['median_abs_err']:>12.4g}" + _rr(L['R2'], 9)
              + f"{L['corr']:>+8.3f}")
    print(f"  {'='*96}")

    print(f"\n  {'='*111}")
    print(f"  (B) ERRO RELATIVO [{_rot}] — o MESMO erro dividido por quatro "
          f"denominadores diferentes")
    print(f"  {'='*111}")
    print(f"  {'var':<6}{'espaço':<8}{'MAE/méd|y|':>12}"
          f"{'média':>11}{'σ':>10}{'|méd|/σ':>9}{'MAE/méd':>10}"
          f"{'MAE/σ':>9}{'RMSE/σ':>9}{'MAE/esc':>9}{'Acc10':>7}{'Acc20':>7}")
    print(f"  {'-'*111}")
    for L in linhas:
        marca = "*" if (np.isfinite(L['abs_media_sobre_sigma'])
                        and L['abs_media_sobre_sigma'] < 1.0) else " "
        print(f"  {L['var']:<5}{marca}{L['espaco']:<8}"
              + _pc(L['MAE_media_abs_pct'], 12)
              + f"{L['media_CFD']:>+11.4g}{L['sigma_CFD']:>10.4g}"
                f"{L['abs_media_sobre_sigma']:>9.2f}"
              + _pc(L['MAE_media_pct'], 10)
              + _pc(L['MAE_sigma_pct'], 9)
              + _pc(L['RMSE_sigma_pct'], 9)
              + _pc(L['MAE_escala_pct'], 9)
              + _pc(L['acc_10pct'], 7)
              + _pc(L['acc_20pct'], 7))
    print(f"  {'-'*111}")
    print(f"  MAE/méd|y| é a MÉTRICA PRINCIPAL deste relatório: MAE "
          f"dividido pela média do MÓDULO da variável — o erro")
    print(f"    percentual médio contra a magnitude típica do campo. "
          f"Não degenera em campo simétrico, e nas linhas")
    print(f"    sem '*' dá o mesmo número que o MAE/méd ao lado.")
    print(f"  MAE/méd: MAE da variável dividido pela MÉDIA daquela "
          f"variável no CFD — o erro percentual médio.")
    print(f"    '*' marca a linha onde |média| < σ: ali a média é pequena "
          f"perante a dispersão do campo e a")
    print(f"    porcentagem fica instável (u e w têm média ~0 por simetria; "
          f"a P depende de onde está o zero —")
    print(f"    aqui a P já está manométrica). Nessas linhas o número a "
          f"usar é o da PRIMEIRA coluna.")
    print(f"  Nas linhas 'log10' a média é a média de log10, então o "
          f"MAE/méd ali não é um erro percentual do")
    print(f"    campo físico: para k, ω e ν_t o número com essa leitura "
          f"está na linha 'linear' da mesma variável.")
    print(f"  MAE/σ e RMSE/σ: erro dividido pelo desvio padrão do campo no "
          f"CFD — o quanto havia para prever.")
    print(f"    RMSE/σ = 100,0% é EXATAMENTE o escore de quem chutasse a "
          f"média em todo ponto (R² = 0).")
    print(f"    Para o MAE essa mesma referência é 79,8% (num campo "
          f"gaussiano). R² = 1 − (RMSE/σ)².")
    print(f"  MAE/esc: erro dividido por uma escala FÍSICA fixa, igual em "
          f"toda rodada — comparável entre runs.")
    print(f"    V_INF={V_INF} m/s | ½ρV²={P_SCALE:.0f} Pa | "
          f"k_inlet={K_INLET:.4g} m²/s² | ω_inlet={OMEGA_INLET:.4g} 1/s | "
          f"ν_t,max={NU_T_MAX:.4g} m²/s | 1 dex nos campos log.")
    print(f"  Acc10/Acc20: FRAÇÃO DE PONTOS com |erro| < 10% / 20% da escala "
          f"física acima — não é um erro médio,")
    print(f"    é taxa de acerto dentro de uma faixa de tolerância "
          f"(ex.: 10% de V_INF = {0.10*V_INF:.2g} m/s; "
          f"10% de ½ρV² = {0.10*P_SCALE:.1f} Pa).")
    print(f"    Nos campos em log a escala é 1 dex, então Acc10% = dentro "
          f"de um fator 1,26 e Acc20% = fator 1,58.")
    print(f"  {'='*111}")

    for L in linhas:
        if L["n_excluido"] > 0:
            print(f"  >>> {L['var']} ({L['espaco']}): {L['n_excluido']:,} "
                  f"pontos ({L['pct_excluido']:.1f}%) ficaram DE FORA da "
                  f"métrica — em log10 só entram valores > 0.")
            print(f"      de onde vêm: {L['n_zero_cfd']:,} com valor ≤ 0 no "
                  f"CFD, {L['n_zero_pinn']:,} na PINN.")
            if L["var"] == "k" and L["n_zero_pinn"] > 0:
                print(f"      Os da PINN são o piso K_EPS = "
                      f"{CONSTANTS['K_EPS']:.4g}: k = exp(...) − K_EPS dá zero "
                      f"exato quando a exponencial cai abaixo dele.")

    out = os.path.join(SUMMARY_DIR, "final_metrics_snapshot.csv")
    salvar_csv(pd.DataFrame(linhas), out)
    print(f"\n  resumo: {out}")

    # --- fechamento da relação constitutiva ---
    if mt is not None:
        relatorio_nu_t(x, y, z, k_c, om_c, nut_c, f, nut_p, mt)
    # ema1: o instante, o n e a rede vão junto, para a comparação conferir
    # que as duas redes foram medidas no mesmo snapshot (cabeçalho)
    df_out = pd.DataFrame(linhas)
    df_out.attrs.update(t=float(actual_t), n=int(n), rotulo=_rot,
                        csv=os.path.basename(csv_path))
    return df_out


def plot_metric_comparison(df_siren, df_ema, output_dir, aviso_par=""):
    """
    [02/11/2026] MAE/méd|y| da rede de treino × EMA, no mesmo snapshot (ver
    o cabeçalho desta seção). df_siren e df_ema são o retorno de
    plot_scatter_and_histograms de cada rede. aviso_par é o aviso do
    check_pair (seção 10) — p.ex. best × best de épocas diferentes — e vai
    para o título e o CSV.

    Grava comparacao_MAE_med_abs.png e comparacao_MAE_med_abs.csv. A
    coluna delta_pp é EMA − rede de treino, em pontos percentuais:
    negativo = EMA melhor.
    """
    a_s, a_e = df_siren.attrs, df_ema.attrs
    if (a_s.get("t"), a_s.get("n"), a_s.get("csv")) != \
       (a_e.get("t"), a_e.get("n"), a_e.get("csv")):
        raise SystemExit(
            f"ERRO: as duas redes não foram medidas no mesmo snapshot: "
            f"rede de treino t={a_s.get('t')} n={a_s.get('n')} "
            f"({a_s.get('csv')}) × EMA t={a_e.get('t')} n={a_e.get('n')} "
            f"({a_e.get('csv')}). A comparação não mediria o EMA.")

    chave = ["var", "espaco"]
    m = df_siren[chave + ["unidade", "MAE_media_abs_pct"]].merge(
        df_ema[chave + ["MAE_media_abs_pct"]], on=chave,
        suffixes=("_siren", "_ema"))
    m["delta_pp"] = m["MAE_media_abs_pct_ema"] - m["MAE_media_abs_pct_siren"]
    m["t_s"] = a_s["t"]
    m["n_pontos"] = a_s["n"]
    m["rede_siren"] = a_s.get("rotulo", "rede de treino")
    m["rede_ema"] = a_e.get("rotulo", "EMA")
    if aviso_par:
        m["aviso"] = aviso_par

    # [23/11/2026] ν_t sai da FIGURA (PLOT_NU_T); o CSV e o console abaixo
    # continuam com todas as linhas
    m_fig = m if PLOT_NU_T else m[m["var"] != "ν_t"]
    paineis = [("linear", "escala física (linear) — erro percentual do campo"),
               ("log10", "log10 — MAE em dex / média de |log10 y| "
                         "(não é % do campo físico)")]
    paineis = [(e, t) for e, t in paineis if (m_fig["espaco"] == e).any()]
    fig, axes = plt.subplots(len(paineis), 1,
                             figsize=(13, 4.8 * len(paineis)), squeeze=False)
    C_S, C_E = "tab:blue", "tab:orange"      # mesmas cores da seção 13
    for ax, (esp, tit) in zip(axes[:, 0], paineis):
        sub = m_fig[m_fig["espaco"] == esp].reset_index(drop=True)
        xs = np.arange(len(sub)); w = 0.38
        b1 = ax.bar(xs - w/2, sub["MAE_media_abs_pct_siren"], w, color=C_S,
                    label=a_s.get("rotulo", "rede de treino"))
        b2 = ax.bar(xs + w/2, sub["MAE_media_abs_pct_ema"], w, color=C_E,
                    label=a_e.get("rotulo", "EMA"))
        for rect, v in list(zip(b1, sub["MAE_media_abs_pct_siren"])) + \
                       list(zip(b2, sub["MAE_media_abs_pct_ema"])):
            ax.annotate(f"{v:.1f}%", (rect.get_x() + rect.get_width()/2,
                                      rect.get_height()),
                        ha="center", va="bottom", fontsize=7.5,
                        xytext=(0, 2), textcoords="offset points")
        topo = float(np.nanmax(np.r_[sub["MAE_media_abs_pct_siren"],
                                     sub["MAE_media_abs_pct_ema"]]))
        ax.set_ylim(0, 1.22 * topo if np.isfinite(topo) and topo > 0 else 1)
        rot_x = [f"{v}\nΔ = {d:+.1f} pp" for v, d in
                 zip(sub["var"], sub["delta_pp"])]
        ax.set_xticks(xs); ax.set_xticklabels(rot_x, fontsize=9)
        ax.set_ylabel("MAE/méd|y| (%)")
        ax.set_title(tit, fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    sup = (f"MAE/méd|y| — EMA × rede de treino — t={a_s['t']:.1f}s | "
           f"n={a_s['n']:,} pontos | Δ = EMA − rede de treino "
           f"(negativo: EMA melhor)")
    if aviso_par:
        sup += f"\n>>> {aviso_par}"
    plt.suptitle(sup, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94 if aviso_par else 0.96])
    out = os.path.join(output_dir, "comparacao_MAE_med_abs.png")
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"  salvo: {out}")

    out_csv = os.path.join(output_dir, "comparacao_MAE_med_abs.csv")
    salvar_csv(m, out_csv)
    print(f"  tabela: {out_csv}")

    print(f"\n  MAE/méd|y| — t={a_s['t']:.1f}s"
          + (f"   >>> {aviso_par}" if aviso_par else ""))
    print(f"  {'var':<6}{'espaço':<8}{'treino':>10}{'EMA':>10}{'Δ (pp)':>10}")
    for _, r in m.iterrows():
        print(f"  {r['var']:<6}{r['espaco']:<8}"
              f"{r['MAE_media_abs_pct_siren']:>9.1f}%"
              f"{r['MAE_media_abs_pct_ema']:>9.1f}%"
              f"{r['delta_pp']:>+10.1f}")
    return m


def _ramos_constitutivos(nu_t, k, om):
    """
    Classifica cada ponto nos três ramos do SST, com o MESMO critério que a
    verificação (b) da Cell 5 usa no CSV:

        teto      : ν_t a menos de 1e-4 relativo de NU_T_MAX
        limitado  : ν_t < k/ω por mais de 1e-4 relativo
        livre     : ν_t = k/ω

    Usar um limiar aqui e outro lá era comparar coisas diferentes: a versão
    anterior deste script chamava de "limitado" qualquer denom > 1+1e-6,
    enquanto a Cell 5 exige 1e-4 no CSV. Com o mesmo critério nos dois, a
    comparação PINN × CFD passa a significar alguma coisa.
    """
    base = k / np.maximum(om, OMEGA_FLOOR)
    razao = nu_t / np.maximum(base, 1e-30)
    teto = nu_t >= NU_T_MAX * (1.0 - 1e-4)
    limitado = (~teto) & (razao < 1.0 - 1e-4)
    return teto, limitado


def relatorio_nu_t(x, y, z, k_c, om_c, nut_c, f, nut_p, mt):
    """
    ν_t é o campo que fecha a conta: ele realimenta o momento, e é o único
    que depende de S — que a rede obtém derivando o próprio campo de
    velocidade e que NÃO tem supervisão nenhuma (as 9 derivadas e o
    strain-rate-mag não foram exportados; é a pendência [CFD] da Cell 2).

    Por isso vale separar o erro de ν_t por região e, em cada uma, dizer
    qual RAMO do modelo está ativo na PINN e no CFD. Se k e ω estão certos
    e ν_t não, o que sobra é S ou o ramo.
    """
    dw = wall_distance(x, y, z)
    r_cil = np.hypot(x - XC_CYL, y - YC_CYL)
    bandas = [
        ("inlet y < 3 m",        y < 3.0),
        ("parede d_wall<1,3 m",  dw < 1.3),
        ("cilindro r < 2R",      (r_cil < 2*R_CYL) & (z < 1.5*H_CYL)),
        ("esteira y 220-600",    (y > 220.0) & (y < 600.0)),
        ("resto",                (y >= 3.0) & (dw >= 1.3) & (y <= 220.0)),
        ("TODOS",                np.ones_like(y, dtype=bool)),
    ]

    # k efetivo: o resíduo e a fórmula de μ_t usam exp(...) SEM o −K_EPS
    k_p_eff = f["k"] + CONSTANTS["K_EPS"]
    teto_p, lim_p = _ramos_constitutivos(nut_p, k_p_eff, f["omega"])
    teto_c, lim_c = _ramos_constitutivos(nut_c, k_c, om_c)

    print(f"\n  {'='*98}")
    print(f"  ν_t POR REGIÃO — ramos classificados com o MESMO critério "
          f"nos dois (Cell 5, 1e-4)")
    print(f"  {'='*98}")
    print(f"  {'região':<22s}{'n':>8s}{'dex':>7s}{'viés':>8s}"
          f"{'limitado% P/C':>18s}{'teto% P/C':>16s}"
          f"{'S p50':>10s}{'k/ω dex':>9s}")
    print(f"  {'-'*98}")
    for nome, m in bandas:
        if m.sum() < 10:
            continue
        mm = m & (nut_c > 0) & (nut_p > 0)
        if mm.sum() < 10:
            continue
        e = np.log10(nut_p[mm]) - np.log10(nut_c[mm])
        # quanto do erro de ν_t já vem de k e ω, antes de qualquer ramo
        base_p = k_p_eff[mm] / np.maximum(f["omega"][mm], OMEGA_FLOOR)
        base_c = k_c[mm] / np.maximum(om_c[mm], OMEGA_FLOOR)
        mb = (base_p > 0) & (base_c > 0)
        e_base = (np.mean(np.abs(np.log10(base_p[mb]) - np.log10(base_c[mb])))
                  if mb.sum() else float("nan"))
        print(f"  {nome:<22s}{int(m.sum()):>8,}{np.mean(np.abs(e)):>7.2f}"
              f"{np.mean(e):>+8.2f}"
              f"{100*lim_p[m].mean():>10.1f} /{100*lim_c[m].mean():>6.1f}"
              f"{100*teto_p[m].mean():>9.1f} /{100*teto_c[m].mean():>6.1f}"
              f"{np.median(mt['S'][m]):>10.3g}{e_base:>9.2f}")
    print(f"  {'-'*98}")
    print(f"  'dex' = erro de ν_t; 'k/ω dex' = erro de k/ω ANTES do "
          f"limitador e do teto.")
    print(f"  Se 'dex' >> 'k/ω dex', o erro NÃO vem de k nem de ω: vem do "
          f"ramo, ou seja, de S.")
    print(f"  S não tem supervisão nenhuma — strain-rate-mag não está no "
          f"CSV (pendência [CFD] da Cell 2).")
    print(f"  {'='*98}")


# ==============================================================================
# 16. MAIN
# ------------------------------------------------------------------------------
# [02/11/2026] ema1. O main roda as DUAS redes e a comparação num comando
# só (decisão de 02/11/2026):
#
#   python inferencia_ema.py                    par mais recente, 3 pastas
#   python inferencia_ema.py --checkpoint pinn_ema1_ema_005000.weights.h5
#                                               época 5000 das duas redes
#   python inferencia_ema.py --checkpoint pinn_ema1_ema_best.weights.h5
#                                               best × best (épocas
#                                               diferentes, com aviso)
#   python inferencia_ema.py --rede ema         só o EMA, sem comparação
#   python inferencia_ema.py --diff-slices      + slices da diferença
#   python inferencia_ema.py --list             tabela época × rede
#
# --checkpoint aceita o arquivo de QUALQUER uma das duas redes: ele diz a
# época (ou best/final), e o par é achado trocando o sufixo no nome (seção
# 10). --rede escolhe quais redes rodam. --run-tag saiu: com dois perfis da
# mesma rodada, ele só repetia o --rede.
#
# ORDEM:
#   0. GPU e snapshots, uma vez.
#   1. Com as duas redes: carrega as duas e confere que NÃO são a mesma
#      (|Δ saída| > 0 em 2.048 pontos do domínio). É a segunda metade do
#      check_pair (seção 10): a primeira confere os NOMES, esta os PESOS.
#      Se der zero, alguém copiou um arquivo por cima do outro, ou o EMA
#      nunca andou — e as figuras do "EMA" seriam as da rede de treino.
#   2. Para cada rede: perfil, metadata, self-test, loss history, slices e
#      scatter, na pasta dela — o fluxo de antes, uma vez por rede.
#   3. Comparação: loss (seção 13), MAE/méd|y| (seção 15) e, com
#      --diff-slices, os slices da diferença (seção 14).
#
# --output-dir passa a ser a PASTA-MÃE (inferencia_ema1/): as três pastas
# vão dentro dela, com os nomes de sempre.
# --metadata só vale com --rede siren ou --rede ema: cada rede tem o seu.
# --no-nut vale para tudo (figuras de cada rede e slices da diferença).
# ==============================================================================
def print_config():
    C = CONSTANTS
    print("=" * 76)
    print("  PINN k-ω SST — INFERÊNCIA")
    print("=" * 76)
    _p = RUNS.get(RUN_TAG, {})
    print(f"  Perfil: {RUN_TAG} — {_p.get('descricao', '?')}")
    print(f"    rede: {_p.get('rede', '?')}   par: {_p.get('par', '?')}   "
          f"EMA: β={EMA_CFG['beta']:g}, {EMA_CFG['steps_per_epoch']} passos/época")
    print(f"    CFD (job {_p.get('cfd_job_id', '?')}): {SNAPSHOT_DIR}")
    print(f"    checkpoints: {CHECKPOINT_DIR}")
    print(f"    log:         {CSV_LOG_PATH}")
    print(f"    saída:       {OUTPUT_DIR}")
    print(f"    comparação:  {COMPARACAO_DIR}")
    print(f"  Rede: {_arch_str(ARCH)}")
    print(f"  Entrada (z-score isotrópico no espaço, z-score no tempo):")
    print(f"    X_MEAN={C['X_MEAN']:.4f}  Y_MEAN={C['Y_MEAN']:.4f}  "
          f"Z_MEAN={C['Z_MEAN']:.4f}  DP_ISO={C['DP_ISO']:.4f} m")
    print(f"    T_MEAN={C['T_MEAN']:.4f} s  T_DP={C['T_DP']:.6f} s")
    print(f"  Saídas:")
    print(f"    u, w = saída × V_REF({C['V_REF']})")
    print(f"    v    = saída × {C['V_DP']:.4f} + {C['V_MEAN']:.4f}")
    print(f"    P    = saída × {C['P_DP']:.4f} + ({C['P_MEAN']:.4f})")
    print(f"    k    = exp(saída × {C['LNK_DP']:.4f} + ({C['LNK_MEAN']:.4f})) "
          f"− {C['K_EPS']:.6f}")
    print(f"    ω    = exp(saída × {C['LNW_DP']:.4f} + {C['LNW_MEAN']:.4f})")
    print(f"  BC dura de turbulência no inlet:")
    print(f"    k = {K_INLET:.5f} m²/s² (kn={KN_INLET:+.4f})  "
          f"{'ATIVA' if HARD_BC_K_INLET else 'desligada'}")
    print(f"    ω = {OMEGA_INLET:.2f} 1/s (ln ωn={LNWN_INLET:+.4f})  "
          f"{'ATIVA' if HARD_BC_OMEGA_INLET else 'desligada'}")
    print(f"  Ansatz: HC_MT={HC_MT}  HC_RATE={HC_RATE}  HC_K_R2={HC_K_R2}  "
          f"CYL_RIM_FIX={CYL_RIM_FIX}")
    print(f"  Cilindro normalizado: centro=({XC_N:+.4f}, {YC_N:+.4f})  "
          f"R_N={R_N:.4f}  topo z_n={H_N:+.4f}")
    print(f"  ν_t: fórmula SST completa, teto ν_t ≤ {NU_T_MAX:.4f} m²/s "
          f"(μ_t/μ ≤ {VISC_RATIO_MAX:.0e})")
    print("=" * 76)


def _carregar_rede(ckpt):
    """Monta a rede do perfil em uso e carrega os pesos. Sai com erro claro."""
    if not wait_until_stable(ckpt):
        print(f"  >>> o arquivo ainda está mudando de tamanho. O treino "
              f"provavelmente está reescrevendo este checkpoint agora.")
        print(f"      Use um checkpoint numerado (--list) em vez do 'best'.")
    model = build_pinn(ARCH["n_layers"], ARCH["n_neurons"], arch=ARCH)
    try:
        model.load_weights(ckpt)
    except ValueError as e:
        print(f"ERRO: os pesos não cabem na rede do perfil '{RUN_TAG}' "
              f"({_arch_str(ARCH)}).")
        print(f"  Mensagem do Keras:\n    {str(e)[:300]}")
        sys.exit(1)
    return model


def _ativar_rede(tag, ckpt, args):
    """
    Perfil, caminhos, BC e metadata da rede `tag` — tudo o que o main de
    antes fazia entre achar o checkpoint e montar a rede. Devolve
    (perfil, metadata ou None, rótulo para os títulos).
    """
    global CHECKPOINT_DIR, SNAPSHOT_DIR, CSV_LOG_PATH, OUTPUT_DIR
    global COMPARACAO_DIR
    global SLICES_DIR, SCATTER_DIR, HIST_DIR, SUMMARY_DIR, LOSSHIST_DIR
    global HARD_BC_K_INLET, HARD_BC_OMEGA_INLET
    prof = apply_profile(tag)

    # --- caminhos: os do perfil, salvo o que a linha de comando mudou ---
    if args.checkpoint_dir is None:
        CHECKPOINT_DIR = prof["checkpoint_dir"]
    SNAPSHOT_DIR = args.snapshot_dir or prof["snapshot_dir"]
    CSV_LOG_PATH = args.csv_log      or prof["csv_log"]
    if args.output_dir:
        # --output-dir é a pasta-MÃE; as três pastas mantêm o nome
        OUTPUT_DIR = os.path.join(args.output_dir,
                                  os.path.basename(prof["output_dir"]))
        COMPARACAO_DIR = os.path.join(args.output_dir,
                                      os.path.basename(prof["comparacao_dir"]))
    else:
        OUTPUT_DIR     = prof["output_dir"]
        COMPARACAO_DIR = prof["comparacao_dir"]
    SLICES_DIR   = os.path.join(OUTPUT_DIR, "slices")
    SCATTER_DIR  = os.path.join(OUTPUT_DIR, "scatter")
    HIST_DIR     = os.path.join(OUTPUT_DIR, "histograms")
    SUMMARY_DIR  = os.path.join(OUTPUT_DIR, "summary")
    LOSSHIST_DIR = os.path.join(OUTPUT_DIR, "loss_history")

    # A saída da inferência nunca vai para dentro de saida/: lá é só o que
    # o TREINO escreve, e dá para arquivar uma rodada sem misturar os dois.
    _abs_saida = os.path.abspath(SAIDA_DIR)
    for _d in (OUTPUT_DIR, COMPARACAO_DIR):
        if os.path.commonpath([os.path.abspath(_d), _abs_saida]) == _abs_saida:
            print(f"ERRO: --output-dir dentro de {SAIDA_DIR}/, que é só do treino.")
            sys.exit(1)

    # --- BC dura do inlet: o perfil manda, a flag sobrescreve ---
    if args.hard_bc_inlet != "auto":
        HARD_BC_K_INLET = HARD_BC_OMEGA_INLET = (args.hard_bc_inlet == "on")
    print(f"[config] perfil '{RUN_TAG}' ({prof['descricao']})  BC dura de "
          f"k/ω no inlet: {'LIGADA' if HARD_BC_K_INLET else 'desligada'}")

    # --- metadata: fonte de precisão total, quando existe ---
    meta = None
    meta_path = args.metadata
    if meta_path is None and not args.no_metadata:
        _cand = os.path.join(os.path.dirname(os.path.abspath(ckpt)),
                             f"pinn_{RUN_TAG}.metadata.json")
        if os.path.exists(_cand):
            meta_path = _cand
    if meta_path and not args.no_metadata:
        meta = apply_metadata(meta_path, ckpt=ckpt)
        _snap_meta = meta.get("procedencia", {}).get("snapshot_dir")
        if _snap_meta and (os.path.normpath(_snap_meta)
                           != os.path.normpath(SNAPSHOT_DIR)):
            print(f"  [metadata] >>> o treino leu      {_snap_meta}")
            print(f"                 a inferência lerá {SNAPSHOT_DIR}")
            print(f"                 Comparar a PINN com outro CFD não mede a PINN.")
    else:
        print(f"  [metadata] nenhum (treino em andamento?) — constantes do "
              f"perfil '{RUN_TAG}'")

    # rótulo dos títulos: rede + checkpoint; o best ganha a época, se o
    # metadata souber qual é (ver o bloco ema da Cell 16)
    lab = checkpoint_label(ckpt)
    if lab == "best" and meta and "ema" in meta:
        _ep = meta["ema"].get(f"best_epoch_{prof['rede']}")
        if _ep is not None:
            lab = f"best (época {_ep})"
    return prof, meta, f"{_rotulo_rede()}, {lab}"


def _conferir_par_carregado(model_a, model_b):
    """
    As duas redes carregadas TÊM que dar saídas diferentes (ver o
    cabeçalho desta seção, passo 1). Pontos sorteados no domínio
    normalizado, gerador próprio.
    """
    r = np.random.default_rng(SEED_PAR)
    pts = tf.constant(np.column_stack([
        r.uniform(XN_MIN, XN_MAX, 2048), r.uniform(YN_MIN, YN_MAX, 2048),
        r.uniform(ZN_MIN, ZN_MAX, 2048),
        (r.uniform(0.2, 100.0, 2048) - CONSTANTS["T_MEAN"]) / CONSTANTS["T_DP"],
    ]).astype(np.float32))
    d = float(np.max(np.abs(model_a(pts, training=False).numpy()
                            - model_b(pts, training=False).numpy())))
    print(f"  par carregado: |Δ saída| máx EMA × rede de treino = {d:.3e}  "
          f"(tem que ser > 0)")
    if d == 0.0:
        raise SystemExit(
            "ERRO: o EMA e a rede de treino dão a MESMA saída. Os dois "
            "arquivos têm os mesmos pesos (cópia por cima?) ou o EMA nunca "
            "foi atualizado. A comparação não mediria nada.")
    return d


SEED_PAR = 42


def main():
    global CHECKPOINT_DIR, SAIDA_DIR

    ap = argparse.ArgumentParser(
        description="Inferência e plots da PINN k-ω SST — rodada ema1 "
                    "(rede de treino, EMA e comparação)")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="nome do .h5 ou caminho, de QUALQUER uma das duas "
                         "redes (diz a época, best ou final); padrão: a "
                         "época mais alta em que as duas existem")
    ap.add_argument("--rede", type=str, default="ambas",
                    choices=["ambas", "siren", "ema"],
                    help="quais redes rodar; a comparação só sai com 'ambas'")
    ap.add_argument("--diff-slices", action="store_true",
                    help="gera também os slices da diferença EMA − rede de "
                         "treino (caro; só com --rede ambas)")
    ap.add_argument("--metadata", type=str, default=None,
                    help="metadata.json da Cell 16 (só com --rede siren/ema); "
                         "padrão: pinn_<tag>.metadata.json ao lado do "
                         "checkpoint, se existir")
    ap.add_argument("--no-metadata", action="store_true",
                    help="não lê o metadata nem quando ele existe")
    # Os caminhos vêm do PERFIL (seção 1); passar um deles sobrescreve só
    # aquele. Padrão None = "o do perfil".
    ap.add_argument("--checkpoint-dir", type=str, default=None)
    ap.add_argument("--snapshot-dir", type=str, default=None)
    ap.add_argument("--csv-log", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None,
                    help="pasta-MÃE das três pastas (padrão: "
                         "10-29_ema/inferencia_ema1/)")
    ap.add_argument("--list", action="store_true", help="lista checkpoints e sai")
    ap.add_argument("--selftest-only", action="store_true",
                    help="só confere as hard constraints e sai")
    ap.add_argument("--no-slices", action="store_true")
    ap.add_argument("--no-scatter", action="store_true")
    ap.add_argument("--no-loss", action="store_true")
    ap.add_argument("--no-nut", action="store_true",
                    help="não calcula ν_t pela fórmula SST (usa k/ω); "
                         "poupa o autodiff, que é o passo caro")
    ap.add_argument("--hard-bc-inlet", type=str, default="auto",
                    choices=["auto", "on", "off"],
                    help="auto: o que o perfil da rodada diz. O self-test "
                         "acusa se estiver errado.")
    ap.add_argument("--plane", type=str, default="xz", choices=["xz", "xy"])
    ap.add_argument("--times", type=str, default="10,30,58,100",
                    help="tempos físicos (s) para os slices")
    args = ap.parse_args()

    if args.metadata and args.rede == "ambas":
        print("ERRO: --metadata aponta UM arquivo, e cada rede tem o seu. "
              "Use com --rede siren ou --rede ema.")
        sys.exit(1)
    if args.diff_slices and args.rede != "ambas":
        print("ERRO: --diff-slices compara as duas redes; use com --rede ambas.")
        sys.exit(1)

    CHECKPOINT_DIR = args.checkpoint_dir or RUNS[RUN_TAG_PADRAO]["checkpoint_dir"]
    SAIDA_DIR = RUNS[RUN_TAG_PADRAO]["saida_dir"]
    TAG_DE = {RUNS[t]["rede"]: t for t in RUNS}      # "siren"/"ema" -> tag

    # --- --list: uma linha por época, uma coluna por rede ---
    if args.list:
        tab = checkpoint_table(CHECKPOINT_DIR)
        if not tab:
            print(f"nenhum checkpoint em {CHECKPOINT_DIR}")
            return
        print(f"checkpoints em {CHECKPOINT_DIR}:")
        print(f"  {'':<8}{'rede de treino (siren)':<46}{'EMA':<46}")
        chaves = sorted([k for k in tab if isinstance(k, int)]) + \
                 [k for k in ("best", "final") if k in tab]
        for k in chaves:
            lin = tab[k]
            def _c(rede):
                f = lin.get(rede)
                return (f"{os.path.basename(f)} "
                        f"({os.path.getsize(f)/1024:.0f} KB)") if f else "—"
            print(f"  {str(k):<8}{_c('siren'):<46}{_c('ema'):<46}"
                  + ("" if ("siren" in lin and "ema" in lin) else "   <- sem par"))
        return

    # --- checkpoints: o arquivo pedido diz a época; o par sai do nome ---
    redes = ["siren", "ema"] if args.rede == "ambas" else [args.rede]
    ckpts = {}
    if args.checkpoint is None:
        if args.rede == "ambas":
            a_, b_ = find_latest_pair(CHECKPOINT_DIR, TAG_DE["ema"])
            if a_ is None:
                print(f"ERRO: nenhuma época com as DUAS redes em "
                      f"{CHECKPOINT_DIR}. Rode com --list.")
                sys.exit(1)
            ckpts = {"ema": a_, "siren": b_}
        else:
            c_ = find_latest_checkpoint(CHECKPOINT_DIR, tag=TAG_DE[args.rede])
            if c_ is None:
                print(f"ERRO: nenhum checkpoint da rede '{args.rede}' em "
                      f"{CHECKPOINT_DIR}. Rode com --list.")
                sys.exit(1)
            ckpts = {args.rede: c_}
    else:
        c_, _tentou = resolve_checkpoint(args.checkpoint, CHECKPOINT_DIR)
        if c_ is None:
            print("ERRO: checkpoint não encontrado. Procurei em:")
            for t in (_tentou or [CHECKPOINT_DIR]):
                print(f"    {t}")
            print("Rode com --list para ver o que existe.")
            sys.exit(1)
        tag_c = tag_from_checkpoint(c_)
        if tag_c not in RUNS:
            print(f"ERRO: {os.path.basename(c_)} não é de nenhuma das duas "
                  f"redes da ema1 (perfis: {', '.join(RUNS)}). Para outras "
                  f"rodadas, usar o script de inferência delas.")
            sys.exit(1)
        ckpts[RUNS[tag_c]["rede"]] = c_
        par_, tag_par = partner_checkpoint(c_)
        ckpts[RUNS[tag_par]["rede"]] = par_
        ckpts = {r: ckpts[r] for r in redes}
        for r, f in ckpts.items():
            if not os.path.exists(f):
                print(f"ERRO: falta o checkpoint da rede '{r}': {f}")
                print("  (a outra rede desta época ainda não foi gravada? "
                      "Rode com --list, ou use --rede para rodar uma só)")
                sys.exit(1)

    aviso_par = ""
    if len(ckpts) == 2:
        aviso_par = check_pair(ckpts["ema"], ckpts["siren"])
    print(f"\n[checkpoints]")
    for r in redes:
        print(f"  {r:<6} {ckpts[r]}")
    if aviso_par:
        print(f"  >>> {aviso_par}")

    # --- 0. GPU e snapshots, uma vez ---
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError as e:
                print(e)
        print(f"\nGPU(s): {[g.name for g in gpus]}")
    else:
        print("\nAVISO: sem GPU — inferência em CPU")
    print(f"TensorFlow: {tf.__version__}")

    # --- 1. as duas redes carregadas não podem ser a mesma ---
    models = {}
    if len(ckpts) == 2:
        print(f"\n[par] conferindo que as duas redes são diferentes")
        for r in redes:
            apply_profile(TAG_DE[r])
            models[r] = _carregar_rede(ckpts[r])
        _conferir_par_carregado(models["ema"], models["siren"])

    # --- 2. cada rede, na sua pasta ---
    dfs, rotulos, snaps = {}, {}, None
    com_nut = not args.no_nut
    for i_r, r in enumerate(redes, 1):
        print(f"\n{'#'*76}\n#  REDE {i_r}/{len(redes)}: "
              f"{'rede de treino (siren)' if r == 'siren' else 'EMA'}\n{'#'*76}")
        prof, meta, rotulos[r] = _ativar_rede(TAG_DE[r], ckpts[r], args)

        if not os.path.isdir(SNAPSHOT_DIR) and not args.selftest_only:
            print(f"ERRO: pasta do CFD não existe: {SNAPSHOT_DIR}")
            sys.exit(1)
        for d in [SLICES_DIR, SCATTER_DIR, HIST_DIR, SUMMARY_DIR, LOSSHIST_DIR]:
            os.makedirs(d, exist_ok=True)
        print(f"saída em: {OUTPUT_DIR}/")

        print_config()
        _roundtrip_check()
        print("  round-trip normalize/denormalize de k e ω: OK")

        if not args.no_loss:
            print(f"\n[{r} 1/4] loss history")
            plot_loss_history(CSV_LOG_PATH, LOSSHIST_DIR, rede=r)

        print(f"\n[{r} 2/4] modelo — {rotulos[r]}")
        print(f"  checkpoint: {ckpts[r]}")
        print(f"  rede: {_arch_str(ARCH)}")
        if r not in models:
            models[r] = _carregar_rede(ckpts[r])
        model = models[r]
        print(f"  parâmetros: {model.count_params():,}  (esperado: 1.318.918)")
        if model.count_params() != 1_318_918:
            print(f"  >>> número de parâmetros DIFERENTE do treino. A "
                  f"arquitetura não é a mesma.")

        selftest_hard_constraints(model)
        if args.selftest_only:
            continue

        if snaps is None:
            snaps = discover_snapshots(SNAPSHOT_DIR)
            if not snaps:
                print(f"ERRO: nenhum snapshot em {SNAPSHOT_DIR}")
                sys.exit(1)
            print(f"\n  snapshots: {len(snaps)} "
                  f"(t={snaps[0][0]:.1f}s … {snaps[-1][0]:.1f}s)")
            print(f"  ATENÇÃO: t_00001 (t=0,1 s) é a inicialização híbrida, não "
                  f"escoamento — o treino pulou esse arquivo (SNAPSHOT_START=1).")

        if not args.no_slices:
            print(f"\n[{r} 3/4] slices CFD × PINN")
            t_targets = [float(t) for t in args.times.split(",")]
            plot_slices(model, snaps, t_targets, plano=args.plane,
                        com_nut=com_nut, rotulo=rotulos[r])

        if not args.no_scatter:
            print(f"\n[{r} 4/4] scatter, histogramas e tabela")
            dfs[r] = plot_scatter_and_histograms(model, snaps, com_nut=com_nut,
                                                 rotulo=rotulos[r])

    if args.selftest_only:
        return

    # --- 3. comparação ---
    if len(redes) == 2:
        print(f"\n{'#'*76}\n#  COMPARAÇÃO EMA × rede de treino\n{'#'*76}")
        os.makedirs(COMPARACAO_DIR, exist_ok=True)
        print(f"saída em: {COMPARACAO_DIR}/")
        if aviso_par:
            print(f"  >>> {aviso_par}")
        if not args.no_loss:
            print(f"\n[comparação] loss history (validação)")
            plot_loss_comparison(CSV_LOG_PATH, COMPARACAO_DIR)
        if "siren" in dfs and "ema" in dfs:
            print(f"\n[comparação] MAE/méd|y|")
            plot_metric_comparison(dfs["siren"], dfs["ema"], COMPARACAO_DIR,
                                   aviso_par=aviso_par)
        else:
            print(f"\n[comparação] MAE/méd|y| pulado (--no-scatter)")
        if args.diff_slices:
            print(f"\n[comparação] slices da diferença EMA − rede de treino")
            plot_diff_slices(models["siren"], models["ema"], snaps,
                             [float(t) for t in args.times.split(",")],
                             os.path.join(COMPARACAO_DIR, "slices_diferenca"),
                             plano=args.plane, com_nut=com_nut,
                             rotulo_siren=rotulos["siren"],
                             rotulo_ema=rotulos["ema"], aviso_par=aviso_par)

    print(f"\n{'='*60}\n  CONCLUÍDO\n{'='*60}")
    print(f"figuras em: {os.path.dirname(OUTPUT_DIR)}/")

if __name__ == "__main__":
    main()
