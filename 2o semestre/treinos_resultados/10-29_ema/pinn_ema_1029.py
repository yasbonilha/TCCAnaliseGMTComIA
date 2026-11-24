# ============================================================
# CELL 1 — BIBLIOTECAS E CONFIGURAÇÃO AMBIENTE
# ============================================================

import os
import re
import glob
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
rng = np.random.default_rng(SEED)

USE_MIXED_PRECISION = False

if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("WARNING: no GPU detected — training will be very slow on CPU.")

print(f"TensorFlow version: {tf.__version__}")


# ------------------------------------------------------------
# CONFIGURAÇÃO DO MODELO DE TURBULÊNCIA
# ------------------------------------------------------------
# O dado de CFD agora vem do caso URANS com k-omega SST (mesma malha e
# mesma geometria do caso laminar). A equação-alvo deixa de ser NS laminar:
# passa a ser continuidade + momento URANS com nu_t + transporte de k + de omega.
#
# Estes flags são lidos pelas células seguintes (rede, resíduos, loss).
# Mantê-los aqui em cima evita espalhar decisões estruturais pelo arquivo.

TURB_MODEL = "sst"      # "sst" (k-omega SST) | "laminar" (fallback p/ o código antigo)

# A rede prevê ln(omega) em vez de omega.
# Motivo: omega varia várias ordens de grandeza (O(1) no interior,
# O(1e4..1e6) junto às paredes). Prever o log mantém a saída em escala
# tratável e garante omega > 0 por construção (omega = exp(saída)).
LOG_OMEGA = True

# A rede prevê ln(k), com Z-SCORE por cima — mesmo tratamento de ω.
#
# Por que não z-score em k cru: k é positivo e fortemente assimétrico
# (lognormal, como toda escala de energia turbulenta). Medido numa
# distribuição representativa de esteira:
#     z-score em k    -> 77% dos alvos NEGATIVOS (a rede não consegue
#                        reproduzir: k ≥ 0), 95% espremidos em [-1,1],
#                        cauda indo até +135
#     z-score em ln k -> 68% em [-1,1], que é o valor de uma gaussiana;
#                        p99 = 2,3 e máximo 4,7
# O segundo é o que z-score deveria entregar. O primeiro é z-score no nome
# e nada mais — a média vira 0 mas a distribuição continua impossível para
# uma saída tanh.
#
# Ou seja: a normalização z-score continua sendo a espinha do treino, como
# foi no laminar. O que muda é a VARIÁVEL em que ela é aplicada.
#
# Bônus nada pequeno: com k e ω ambos em log, a relação constitutiva
#     ν_t = k/ω    vira    ln ν_t = ln k − ln ω
# ou seja, uma COMBINAÇÃO LINEAR das saídas da rede. O resíduo do SST fica
# muito melhor condicionado do que com uma divisão entre duas saídas que
# podem ambas se aproximar de zero.
#
# Positividade sai de graça: k = exp(...) > 0 sempre, sem softplus, sem
# clamp, sem gradiente morto.
K_NORM_MODE = "logz"    # "logz" (z-score em ln k)  — recomendado
                        # "scale" (k/K_SCALE, saída softplus)
                        # "z"     (z-score em k cru — ver acima, não use)

# F1 e F2 do SST dependem de distância à parede. Nesta geometria (piso plano,
# cilindro, paredes laterais e topo) a distância é ANALÍTICA — não precisa ser
# aprendida nem tabelada. A função entra na célula de geometria.
#
# E ainda existe o static_mesh.csv, exportado com cell-wall-distance DENTRO
# da mesma sessão do solve — ou seja, alinhado célula a célula com os t_*.csv.
# Se WALL_DISTANCE_SOURCE = "csv", a coluna do arquivo é usada nos pontos de
# dado; nos pontos de colocação (que não existem na malha) a fórmula analítica
# é obrigatória de qualquer jeito. Vale comparar as duas: é o teste mais barato
# de que o entendimento da geometria bate com o que o Fluent viu.
USE_WALL_DISTANCE    = True
WALL_DISTANCE_SOURCE = "analytic"   # "analytic" | "csv"

# Limitador de produção de k (prática padrão do Fluent):
#   P_k <- min(P_k, PROD_LIMITER_COEF * beta_star * rho * k * omega)
SST_PRODUCTION_LIMITER = True

# Limitador de tensão cisalhante do SST (eq. 4.118 do Theory Guide):
#   True  -> nu_t = (k/omega) / max(1, S*F2/(a1*omega))   (SST completo)
#   False -> nu_t = k / omega                             (k-omega padrão)
#
# ERA False. Virou True em 27/09/2026, e a mudança não é de opinião: o job
# 6027 reconstruiu viscosity-turb célula a célula sobre as 211.579 células
# da malha de produção e a fórmula completa acerta 100% delas (erro mediano
# e p90 < 1e-4 em |ln|).
#
# O que sustentava o False era uma verificação sobre UM snapshot, que mediu
# nu_t/(k/omega) = 1,000000. Aquilo não estava errado — estava incompleto.
# Onde o limitador não atua, nu_t = k/omega de fato, até 1e-10; a mediana
# de uma amostra dominada por essas células dá 1 mesmo com o limitador
# ativo em dezenas de milhares de outras.
#
# Detalhe que custou um job para resolver: o limitador usa S (strain rate),
# NÃO Omega (vorticidade). Nas 24.848 células em que as duas hipóteses
# divergem, S acerta 100% e Omega acerta 0%.
#
# PREÇO: o resíduo precisa de F2 e do tensor de deformação inteiro, e
# d(nu_t)/dx deixa de ter forma analítica curta. Ver Cell 10.
SST_SHEAR_LIMITER = True

# Impor as duas equações de transporte (k e omega) na loss física.
# Deixar False treina só com URANS + nu_t vindo do dado (etapa intermediária
# útil para depurar antes de ligar o SST completo).
ENFORCE_K_EQUATION     = True
ENFORCE_OMEGA_EQUATION = True

print(f"\n=== Configuração de turbulência ===")
print(f"  TURB_MODEL              = {TURB_MODEL}")
print(f"  LOG_OMEGA               = {LOG_OMEGA}")
print(f"  K_NORM_MODE             = {K_NORM_MODE}"
      f"  ->  k = {'exp(LNK_MEAN + LNK_DP·saída) − K_EPS' if K_NORM_MODE == 'logz' else 'ver Cell 4'}")
print(f"  USE_WALL_DISTANCE       = {USE_WALL_DISTANCE} (fonte: {WALL_DISTANCE_SOURCE})")
print(f"  SST_PRODUCTION_LIMITER  = {SST_PRODUCTION_LIMITER}")
print(f"  SST_SHEAR_LIMITER       = {SST_SHEAR_LIMITER}"
      f"  ->  nu_t = {'(k/ω)/max(1, S·F2/(a1·ω))' if SST_SHEAR_LIMITER else 'k/ω'}")
print(f"  ENFORCE_K_EQUATION      = {ENFORCE_K_EQUATION}")
print(f"  ENFORCE_OMEGA_EQUATION  = {ENFORCE_OMEGA_EQUATION}")



# ============================================================
# CELL 2 — CONSTANTES FÍSICAS E DIMENSÕES (CONFORME CFD)
# ============================================================
# GEOMETRIA INALTERADA em relação ao caso laminar: mesmo domínio,
# mesmo cilindro, mesmas BOIs, mesmas tolerâncias.
# O que muda nesta célula é o bloco de turbulência, reescrito em 27/09/2026
# contra o Theory Guide 2025 R2 §4.4 e contra três jobs de medição novos.
#
# [18/10/2026] DADO NOVO, rodada "sst3". O CFD foi refeito com UMA mudança
# no solver: teto de viscosidade turbulenta (μ_t/μ) de 1e5 para 1e7. Malha,
# dt, esquemas, modelo, contornos e a cadeia de escrita do CSV são os mesmos
# do job 5518. Nesta célula isso mexe em DOIS números — VISC_RATIO_MAX e
# NU_T_REF — e em todo texto que ainda descrevia o teto antigo como ativo.
#
# ------------------------------------------------------------
# PROCEDÊNCIA DO DADO — job 6125 (produção, 32 ranks)
#   teste   : job 6124, 20 passos — existe para provar que o teto de 1e7 foi
#             aceito pelo /solve/set/limits antes de gastar a produção
#   malha   : malha_simples_boi.msh.h5 (md5 bd26ce58…), 211.579 células,
#             a MESMA do job 5518
#   física  : idêntica à do 5518 (SST k-omega, BDF2, SIMPLE, momento em
#             Third-Order MUSCL, k e ω em 2nd Order Upwind)
#   tempo   : 1000 passos × 0,1 s = 100 s físicos, um CSV por passo
#   custo   : média de 13,2 iterações por passo, nenhum passo bateu no teto
#             de 50; 2490 s de Fluent
#   teto    : μ_t/μ ≤ 1e7, conferido no eco do /solve/set/limits e no
#             report_summary.txt
#
# RODADA ANTERIOR — job 5518 (gn02, 32 ranks, Fluent 2025 R2), mesma
#   cadeia, teto em 1e5; 13,8 iterações por passo. As duas rodadas são
#   idênticas até t ≈ 40 s (até ali o teto antigo ainda não agia) e
#   divergem depois, fora de fase e com envoltória parecida. Comparar
#   sst2 × sst3 só ESTATISTICAMENTE, nunca instante a instante.
#
# ------------------------------------------------------------
# JOBS DE MEDIÇÃO — o que cada um decidiu
#
#   6023  caixa 10×60×10 m, escoamento uniforme, paredes acompanhando.
#         DECIDIU A PRESSÃO. Ver o bloco PRESSÃO abaixo.
#   6025  probe das opções do SST, por dois caminhos independentes (prompt
#         do TUI e variáveis Scheme do caso em formato legado).
#         DECIDIU QUAIS TERMOS EXISTEM e as 18 constantes.
#   6027  reconstrução de ν_t célula a célula sobre a malha de produção.
#         DECIDIU A RELAÇÃO CONSTITUTIVA. Ver o bloco VISCOSIDADE.
#   6125  rodada de produção com teto 1e7. DECIDIU QUE O TETO NÃO AGE MAIS:
#         0% das células no teto em t = 100 s (também ponderado por volume).
#
# A caixa do job 6023 é uma MALHA DE TESTE, separada da malha de produção.
# O raio de 26,5 m que aparece no handoff veio de lá e NÃO vale aqui: o
# cilindro deste caso tem D = 54 m, como sempre teve.
#
#   GEOMETRIA CONFERIDA no check_field.py do último passo:
#     x: 0,47 .. 279,52   (assumido 0 .. 280)   OK
#     y: 0,46 .. 699,59   (assumido 0 .. 700)   OK
#     z: 0,17 .. 189,33   (assumido 0 .. 190)   OK
#     cilindro em x=140, y=200, raio ~27, altura ~65   OK
#     células dentro de r<27 e z<65: 0 (o furo da malha existe)
#
#   COLUNAS DO CSV — a ordem sai INVERTIDA em relação ao pedido no journal:
#     cellnumber, x-coordinate, y-coordinate, z-coordinate,
#     viscosity-turb, specific-diss-rate, turb-kinetic-energy,
#     pressure, z-velocity, y-velocity, x-velocity
#   O resolve_column_names da Cell 5 casa por NOME, então a ordem não importa.
#
#   DERIVADAS. O job 6125 exportou as 9 derivadas de velocidade e o
#   strain-rate-mag em deriv/d_NNNNN.csv, alinhadas linha a linha com os
#   t_NNNNN.csv (strain-rate-mag = √(2 S_ij S_ij) exatamente, conferido).
#   ESTE TREINO NÃO AS LÊ, por decisão (Cell 3): S e S² continuam saindo de
#   derivadas da REDE. Elas são a próxima variável — comparar o S da PINN
#   com o do CFD e medir o piso da continuidade — depois que a sst3 estiver
#   andando. Uma variável de cada vez.
# ------------------------------------------------------------

# --- Domínio ---
X_MIN, X_MAX = 0.0, 280.0     # Perpendicular ao escoamento (~ 5x D_cil.)
Y_MIN, Y_MAX = 0.0, 700.0     # Sentido do escoamento (~ 13x D_cil.)
Z_MIN, Z_MAX = 0.0, 190.0     # Altura (~ 3x H_cil.)

# --- Cilindro ---
D_CYL = 54.0
H_CYL = 65.0
R_CYL = D_CYL / 2.0
XC_CYL = 140.0                 # Centralizado em X
YC_CYL = 200.0                 # ~3H e ~4D
ZC_BASE = 0.0

# --- Escoamento ---
V_INF = 17.0                   # velocidade de inlet [m/s] direção +Y
V_MAX = 28.8                   # velocidade máxima registrada no domínio
P_SCALE = 15                   # igual escala de magnitude da pressão com velocidade

# --- Body Of Influence Cylinder (caixa de refinamento ao redor do cilindro) ---
BOI_C_X = (100.0, 180.0)
BOI_C_Y = (160.0, 240.0)
BOI_C_Z = (0.0,   100.0)

# --- Body Of Influence (prisma triangular da esteira) ---
BOI_T_V1 = (100.0, 240.0)
BOI_T_V2 = (180.0, 240.0)
BOI_T_V3 = (140.0, 600.0)
BOI_T_Z  = (0.0,   100.0)

# --- Tolerâncias para classificação de cell centers ---
BC_TOL  = 3.0     # distância às paredes externas [m]
CYL_TOL = 2.0     # distância à parede/topo do cilindro [m]


# --- Constantes de Referência ---
V_REF = 5 # [m/s] velocidade média na região do Atacama
          # utilizada para normalização de Vx (u) e Vz (w)

# --- Constantes Termodinâmicas e do Ar ---
P_OP = 101325.0                # pressão de operação
R_SP = 287.058                 # constante específica do ar [J/(kg·K)]
MU = 1.7894e-5                 # viscosidade dinâmica [Pa·s]
G_ACC = 9.81                   # gravidade em -Z
RHO_INF = 1.225
NU = MU / RHO_INF              # viscosidade cinemática MOLECULAR [m²/s] ≈ 1.4607e-5
#AVISO: CONFIGURADOS DE ACORDO COM A SIMULAÇÃO CFD!!!


# ============================================================
# TURBULÊNCIA — k-omega SST, NA FORMA QUE O FLUENT RESOLVE
# ============================================================
# Referência primária: Ansys Fluent Theory Guide 2025 R2, §4.4.2 e §4.4.3,
# equações 4.101 a 4.120. Cada bloco abaixo cita a equação.
#
# As equações efetivamente resolvidas, em forma cinemática (divididas por ρ,
# que é constante):
#
#   ∂u_j/∂x_j = 0
#
#   ∂u_i/∂t + u_j ∂u_i/∂x_j = −∂(p̃/ρ)/∂x_i
#                             + ∂/∂x_j[(ν + ν_t)(∂u_i/∂x_j + ∂u_j/∂x_i)]
#
#   ∂k/∂t + u_j ∂k/∂x_j = ∂/∂x_j[(ν + σ_k ν_t) ∂k/∂x_j] + P_k − β* k ω
#
#   ∂ω/∂t + u_j ∂ω/∂x_j = ∂/∂x_j[(ν + σ_ω ν_t) ∂ω/∂x_j] + α_∞ S² − β_i ω²
#                         + 2(1 − F1) σ_ω2 (1/ω) ∂k/∂x_j ∂ω/∂x_j
#
# Três coisas nessas linhas NÃO são o que estava escrito aqui antes, e as
# três foram MEDIDAS. Cada uma tem seu bloco: PRESSÃO, VISCOSIDADE e
# PRODUÇÃO DE ω.
# ------------------------------------------------------------


# ------------------------------------------------------------
# NUMÉRICA DO SOLVER — o que ela implica (e o que NÃO implica)
# ------------------------------------------------------------
# O job 5518 rodou com: SIMPLE para acoplamento pressão-velocidade,
# transiente de 2ª ordem implícito (BDF2), momento em Third-Order MUSCL,
# k e ω em 2nd Order Upwind.
#
# NENHUM DESSES ITENS MUDA UM TERMO DO RESÍDUO. É a distinção que importa:
# eles são ALGORITMO DE SOLUÇÃO e DISCRETIZAÇÃO. Decidem quão bem a solução
# discreta aproxima a EDP; não decidem qual EDP está sendo resolvida. A PINN
# impõe a EDP contínua, então nada disso entra na Cell 10.
#
# Só que eles importam de três maneiras indiretas, e nenhuma é desprezível.
#
# (a) ELES FIXAM UM PISO PARA A LOSS DE FÍSICA.
#     O dado é a solução DISCRETA. Ela satisfaz exatamente as equações
#     discretas, e a EDP contínua só até o erro de truncamento. Avaliar o
#     resíduo contínuo sobre o campo exato do Fluent NÃO dá zero: dá
#     O(Δx^p) + O(Δt^q). Com MUSCL de 3ª ordem no momento (que em malha
#     poliédrica não estruturada degrada para ~2ª), 2nd-order upwind em k e
#     ω, e BDF2 no tempo, existe um nível abaixo do qual empurrar a physics
#     loss deixa de significar "aprendi a física" e passa a significar
#     "inventei suavidade que o dado não tem".
#
#     Consequência prática na Cell 12: não adianta pesar o resíduo para
#     arrancar mais uma ordem de grandeza se o dado só é consistente até
#     certo ponto.
#
#     ESSE PISO AGORA É MENSURÁVEL, EM PARTE. O job 6125 exportou as 9
#     derivadas de velocidade (deriv/), e com elas:
#       - CONTINUIDADE: ∇·u = tr(∂u_i/∂x_j) sai inteira. É o piso exato do
#         resíduo de massa;
#       - MOMENTO: só parcial. Faltam o gradiente de pressão (dp-dx/dy/dz
#         existem no Fluent mas não foram exportados) e as segundas
#         derivadas do termo viscoso, que o Fluent não exporta. O termo
#         transiente sai de snapshots consecutivos.
#     Seria o número mais útil disponível antes de mexer nos pesos da
#     Cell 12. Fica para a investigação seguinte: este treino não lê
#     deriv/, então dentro dele o piso continua desconhecido.
#
# (b) ESQUEMAS UPWIND ADICIONAM DIFUSÃO NUMÉRICA, que é uma viscosidade que
#     NÃO está nas equações que a PINN impõe. MUSCL de 3ª ordem mistura
#     diferença central com upwind de 2ª ordem, então dissipa bem menos que
#     upwind puro, mas não zera. Onde a malha é grossa e o escoamento cruza
#     as células em diagonal — o campo afastado, aqui — a viscosidade
#     numérica efetiva pode competir com μ_t. É erro de forma de modelo
#     entre "a EDP" e "o que o dado satisfaz". Não há o que corrigir; há que
#     desconfiar do resíduo justamente na região grossa.
#
# (c) A COLUNA DO CSV É UMA MÉDIA DE CÉLULA, não um valor pontual. Em
#     volumes finitos o valor armazenado é a média sobre a célula, e ele
#     coincide com o valor no centroide só até O(Δx²). A Cell 5 trata cada
#     linha como amostra pontual. Mesmo erro de ordem que (a).
#
# DUAS COISAS QUE TRANQUILIZAM:
#
#   TEMPO BEM RESOLVIDO. Com St ≈ 0,2, a frequência de desprendimento é
#   f = St·V_INF/D = 0,063 Hz e o período 15,9 s — cerca de 159 passos de
#   0,1 s por ciclo. O erro de truncamento temporal do BDF2 aí é minúsculo:
#   o ∂/∂t contínuo da PINN e o do Fluent descrevem quase a mesma coisa.
#
#   RESSALVA sobre esse St (18/10/2026): ele é de cilindro 2D e serve aqui
#   só como ORDEM DE GRANDEZA. No dado do job 6125 o St NÃO é mensurável —
#   com 100 s a FFT tem resolução de 0,017 Hz, e os estimadores deram de
#   0,16 a 0,47 para o mesmo sinal. Com H/D ≈ 1,2 (cilindro de extremidade
#   livre) o desprendimento alternado clássico pode estar enfraquecido.
#   Conferir na literatura (p.ex. Sumner, extremidade livre) antes de citar.
#   Para o argumento acima basta a ordem de grandeza: mesmo com St = 0,47 o
#   ciclo tem ~68 passos.
#
#   SIMPLE CONVERGIU. 13,2 iterações internas por passo (job 6125; eram
#   13,8 no 5518) contra um teto de 50 significa que o acoplamento
#   pressão-velocidade fechou em todo passo. ∇·V no dado é pequeno, não
#   exatamente zero — mais um piso, não um viés.
#
# Resumo operacional: não mexer em nada do resíduo por causa da numérica.
# As derivadas exportadas no job 6125 permitem medir o piso de (a); isso
# fica para a investigação seguinte, fora deste treino.


# ------------------------------------------------------------
# PRESSÃO — a coluna `pressure` é a pressão MODIFICADA
# ------------------------------------------------------------
# MEDIDO, job 6023. Caixa 10×60×10 m, escoamento uniforme a 17 m/s, paredes
# movendo-se junto (sem cisalhamento), k decaindo da entrada à saída, mesmo
# modelo e mesma numérica da produção.
#
# Se a coluna fosse a pressão estática p, ela teria que variar como
# −(2/3)ρk ≈ −0,8167·k, o que dá ~0,65 Pa ao longo da caixa. O que se mediu:
#
#     variação de `pressure` no domínio inteiro : 1e-12 Pa  (ou seja, zero)
#     resíduo de momento desde a 1ª iteração    : 1e-13
#
# Doze ordens de grandeza de separação. A conclusão é que a equação que o
# solver resolve NÃO TEM o termo −∇(2k/3); ele está absorvido na variável.
# Para escoamento incompressível "absorvido" e "desprezado" dão na mesma:
# o que importa é que o termo não aparece.
#
#     pressure = p̃ = p + (2/3)ρk
#
# CONSEQUÊNCIA DIRETA NO RESÍDUO: o termo (2/3)ρ∇k NÃO entra no momento.
# Escrevê-lo o conta duas vezes. É o flag MOMENTUM_INCLUDE_K_GRADIENT da
# Cell 10, que já é DERIVADO deste (= not PRESSURE_IS_MODIFIED) e sai False.
#
# Duas consequências secundárias, nenhuma delas problema:
#   - `pressure = 0` no outlet é BC na mesma variável p̃, não em p;
#   - na parede k = 0, logo p̃ = p, e as forças no cilindro não mudam.
#
# HISTÓRICO, para não repetir: dois testes anteriores deram resposta errada
# ou nenhuma. O check_pressure.py olhou o outlet — mas o pressure-outlet
# zera A VARIÁVEL QUE O SOLVER RESOLVE, então inclinação nula cabe nas duas
# hipóteses, e o veredito "estática" que ele imprimiu estava errado. O
# check_pressure_bl.py olhou a camada limite no chão — mas ali k ≈ 0 (sinal
# ≤ 0,2 Pa) enquanto a pressão varia 6 Pa em z por bloqueio do cilindro:
# confusão sistemática, não ruído. Deu INCONCLUSIVO, corretamente.
PRESSURE_IS_MODIFIED = True     # p̃ = p + (2/3)ρk   (MEDIDO, job 6023)


# ------------------------------------------------------------
# CONSTANTES DE FECHAMENTO — e a armadilha das duas convenções
# ------------------------------------------------------------
# PROBE (job 6025) + MANUAL (§4.4.3.3): os 18 valores batem nos dois.
#
# ARMADILHA: os σ do Fluent são NÚMEROS DE PRANDTL. Eles DIVIDEM a
# viscosidade turbulenta (eq. 4.103-4.104):
#
#     Γ_k = μ + μ_t/σ_k^Fluent          Γ_ω = μ + μ_t/σ_ω^Fluent
#
# Nos artigos do Menter os σ MULTIPLICAM:  Γ_k = μ + σ_k^Menter μ_t.
# Os dois conjuntos são recíprocos um do outro, e os números são parecidos
# o bastante para trocar sem sintoma nenhum — erraria a difusão em 15 a 50%
# e o treino só ficaria "um pouco pior".
#
#     σ_k1: Fluent 1,176  ->  Menter 0,850340
#     σ_ω1: Fluent 2,0    ->  Menter 0,5
#     σ_k2: Fluent 1,0    ->  Menter 1,0
#     σ_ω2: Fluent 1,168  ->  Menter 0,856164
#
# E A MISTURA TAMBÉM MUDA DE FORMA. O Fluent mistura o INVERSO (4.105-4.106):
#
#     1/σ_k = F1/σ_k1^F + (1−F1)/σ_k2^F
#
# Chamando a = 1/σ^F = σ^M, isso é exatamente
#
#     σ_k^M = F1·σ_k1^M + (1−F1)·σ_k2^M
#
# ou seja: MISTURA LINEAR NA CONVENÇÃO DO MENTER É IDÊNTICA À MISTURA
# HARMÔNICA NA CONVENÇÃO DO FLUENT. Não são duas escolhas, são a mesma
# conta escrita de dois jeitos. O código usa a convenção do Menter (σ
# multiplica, mistura linear), que é a que deixa o resíduo mais simples,
# mas DERIVA os valores dos números do Fluent para não herdar arredondamento
# de artigo (0,85 contra 0,850340 é 0,04% — irrelevante, mas de graça).

# --- números de Prandtl, como o Fluent os declara (§4.4.3.3) ---
SST_PR_K1 = 1.176
SST_PR_W1 = 2.000
SST_PR_K2 = 1.000
SST_PR_W2 = 1.168

# --- os mesmos, na convenção do Menter: σ que MULTIPLICA μ_t ---
SST_SIGMA_K1  = 1.0 / SST_PR_K1      # 0,850340  (artigo: 0,85)
SST_SIGMA_W1  = 1.0 / SST_PR_W1      # 0,5
SST_SIGMA_K2  = 1.0 / SST_PR_K2      # 1,0
SST_SIGMA_W2  = 1.0 / SST_PR_W2      # 0,856164  (artigo: 0,856)

# --- β_i, misturados linearmente (eq. 4.116) ---
SST_BETA_1    = 0.075          # β_i1
SST_BETA_2    = 0.0828         # β_i2

# --- Constantes comuns ---
SST_BETA_STAR = 0.09           # β*_∞ (= C_mu). Com low-Re desligado, β* = β*_∞
SST_KAPPA     = 0.41           # constante de von Kármán
SST_A1        = 0.31           # limitador de Bradshaw (eq. 4.118)

# --- α_∞ (o "γ" do Menter), derivado, não arbitrado (eq. 4.112-4.113) ---
#   α_∞i = β_i/β*_∞ − κ²/(σ_ωi^Fluent · sqrt(β*_∞))
# Escrito com o σ do FLUENT no denominador, que é a forma literal do manual.
SST_GAMMA_1 = (SST_BETA_1 / SST_BETA_STAR
               - SST_KAPPA**2 / (SST_PR_W1 * np.sqrt(SST_BETA_STAR)))
SST_GAMMA_2 = (SST_BETA_2 / SST_BETA_STAR
               - SST_KAPPA**2 / (SST_PR_W2 * np.sqrt(SST_BETA_STAR)))

assert abs(SST_GAMMA_1 - 0.553167) < 1e-5, f"α_∞1 = {SST_GAMMA_1}"
assert abs(SST_GAMMA_2 - 0.440263) < 1e-5, f"α_∞2 = {SST_GAMMA_2}"
assert abs(SST_SIGMA_K1 - 0.850340) < 1e-6
assert abs(SST_SIGMA_W2 - 0.856164) < 1e-6


# ------------------------------------------------------------
# OPÇÕES DO MODELO — o que está ligado e o que não está
# ------------------------------------------------------------
# PROBE (job 6025), confirmado por dois caminhos independentes:
#
#   SST k-ω                                     LIGADO   kw-sst-on? #t
#   production limiter, fator 10                LIGADO   sst-pk-factor 10.
#   Kato-Launder / produção por vorticidade     desligado
#   low-Re, curvatura, correção de canto        desligado
#   compressibilidade, damping, SST v6.1        desligado
#   SAS, DES, Transition SST                    desligado
#   teto de viscosidade                         μ_t/μ ≤ 1e5 no probe;
#                                               1e7 no job 6125 (ver VISCOSIDADE)
#   tratamento de ω na parede                   "correlation" (y+-insensível)
#
# O que as opções DESLIGADAS garantem, e que simplifica muito o resíduo:
#   α* = 1          (sem correção de baixo Re, eq. 4.75)
#   β* = β*_∞ = 0,09 CONSTANTE  (f_β* = 1, F(M_t) = 0)
#   β  = β_i        (f_β = 1)
#   α  = α_∞        (sem o fator α_∞/α* da eq. 4.83)
# Ou seja: nada de Re_t, nada de χ_k, χ_ω, M_t. As equações ficam as de
# §4.4.2.5 com os coeficientes de §4.4.3.3.

# Limitador de produção de k (eq. 4.81 + opção):
#   P_k <- min(P_k, SST_PROD_LIMIT_COEF · β* · ρ · k · ω)
SST_PROD_LIMIT_COEF = 10.0     # = sst-pk-factor do probe

# Piso do cross-diffusion positivo no argumento de F1 (eq. 4.109).
# Está em unidades de ρ (kg/m³·s⁻²), por isso D_ω+ leva ρ explícito.
SST_CD_KW_FLOOR     = 1.0e-10

K_FLOOR             = 1.0e-10  # piso de k     [m²/s²] — evita divisão por zero
OMEGA_FLOOR         = 1.0e-3   # piso de ω     [1/s]


# ------------------------------------------------------------
# VISCOSIDADE TURBULENTA — a fórmula inteira, com limitador e teto
# ------------------------------------------------------------
# ESTE BLOCO SUBSTITUI a conclusão anterior de que ν_t = k/ω. Aquela leitura
# vinha de uma verificação sobre UM snapshot, e o que ela mediu foi a região
# onde o limitador não atua — que é a maioria do domínio, mas não ele todo.
#
# MEDIDO, job 6027, sobre as 211.579 células da malha de produção. É o
# resultado mais bem verificado de todos: reproduz o `viscosity-turb` do
# Fluent em 100% das células, com erro mediano e p90 abaixo de 1e-4 em |ln|.
#
#     ν_t = min(  (k/ω) / max(1, S·F2/(a1·ω))  ,  VISC_RATIO_MAX · ν  )
#
#     F2 = tanh(Φ2²)                                          (eq. 4.119)
#     Φ2 = max( 2√k/(β*·ω·y) , 500ν/(y²·ω) )                  (eq. 4.120)
#
# Três coisas que o job 6027 resolveu e que valem estar escritas:
#
#  1. O LIMITADOR USA S, NÃO Ω. Nas 24.848 células em que as duas hipóteses
#     dão respostas diferentes, S acerta 100% e Ω acerta 0%. A variável
#     `kw-sst-strain? #f` do probe NÃO quer dizer "usa Ω" — é outra coisa.
#     S = sqrt(2 S_ij S_ij) é exatamente o `strain-rate-mag` do Fluent
#     (razão 1,0000 célula a célula), e Ω = |rot u| o `vorticity-mag`.
#
#  2. ONDE O LIMITADOR NÃO ATUA, ν_t = k/ω até 1e-10. Ou seja, a leitura
#     antiga não estava errada — estava incompleta.
#
#  3. O TETO NÃO ERA DETALHE — com 1e5. No job 6027 ele agia em 3 células
#     no passo 1 e em 9.091 (4,3%) no passo 300, ainda subindo. No job 5518
#     o log registra até 15.265 células (7,2%). Com 1e7 (job 6125) ele não
#     age em célula nenhuma: ver o fim deste bloco.
#
# MUDANÇA DE POSTURA SOBRE O TETO. A versão anterior desta célula tratava o
# teto como DESCASAMENTO DADO/EQUAÇÃO e propunha mascarar essas células na
# loss. Isso estava errado, e a diferença é conceitual:
#
#     o teto não é um artefato aplicado POR FORA do modelo — ele é parte da
#     relação constitutiva que o solver usou.
#
# Se o resíduo da PINN aplicar o MESMO min, o dado fecha a equação naquelas
# células como fecha em qualquer outra. Não há nada para mascarar.
#
# ............................................................
# POR QUE O TETO EXISTE, E POR QUE 1e5 ERA BAIXO DEMAIS AQUI
# ............................................................
# (Análise de 29/09/2026, mantida porque é ela que justifica o job 6125.
#  O desfecho medido está no bloco seguinte.)
#
# ELE É SALVAGUARDA NUMÉRICA, NÃO MODELO. Durante a inicialização e nos
# primeiros passos o campo ainda não é físico: k dispara num ponto de
# estagnação, ou ω colapsa numa região morta, e ν_t = k/ω vai a valores
# absurdos. Um ν_t enorme deixa a equação de momento rigidíssima; o solver
# diverge, ou pior, não diverge e produz um campo errado que realimenta a
# turbulência. O default de 1e5 foi escolhido para NUNCA morder numa
# simulação convergida e bem-posta — o Fluent imprime a mensagem justamente
# para você notar quando morde.
#
# ISSO É FISICAMENTE ERRADO ONDE MORDE? É. Não tem justificativa física
# nenhuma; é um corte arbitrário. A pergunta útil é se ele morde por
# acidente de transiente ou porque a escala do problema não cabe no default.
# Aqui é o segundo caso, e não é perto. Estimando ν_t = k/ω com
# k = 1,5(TI·V_INF)², ε = u'³/ℓ e ω = ε/(β*k):
#
#     TI     ℓ [m]      k       ω        ν_t [m²/s]    μ_t/μ
#     10%      5,4     4,3    2,33          1,9        1,3e5
#     15%     10,8     9,8    1,75          5,6        3,8e5
#     20%     27,0    17,3    0,93         18,6        1,3e6
#
# O teto é ν_t ≤ 1,4607 m²/s. Uma estimativa conservadora para o núcleo da
# esteira de um cilindro de 54 m já dá de 1,3 a 13 VEZES acima dele. O
# default de 1e5 é calibrado para escoamentos de engenharia na escala de
# centímetros a metros; num domínio de 700 m com obstáculo de 54 m o número
# de Reynolds turbulento natural é uma ou duas ordens maior.
#
# E bate com o observado: no job 6027 o teto agia em 3 células no passo 1 e
# em 9.091 no passo 300, AINDA SUBINDO. Transiente cairia. Subindo é o
# padrão de um teto baixo demais para o regime permanente.
#
# O QUE ISSO CAUSA NA SIMULAÇÃO: onde morde, falta difusão turbulenta. A
# esteira mistura menos do que deveria, as estruturas persistem mais para
# jusante, a bolha de recirculação fica mais longa, a carga no cilindro sai
# enviesada. É erro do CFD, e não some treinando a PINN em cima dele.
#
# O QUE ISSO CAUSA NA PINN: nada. Com o mesmo min no resíduo (Cell 10), as
# células recortadas fecham a equação como quaisquer outras. A PINN aprende
# fielmente o campo que o Fluent produziu, teto incluído. São dois problemas
# INDEPENDENTES:
#     a PINN reproduz o CFD?          -> sim, com o min no lugar
#     o CFD reproduz o vento no cerro? -> não, onde o teto morde
#
# A recomendação que saía daqui era subir o limite para 1e7 e conferir que
# a mensagem do Fluent sumia. FOI FEITO, e é o job 6125.
#
# ............................................................
# DESFECHO — job 6125, teto em 1e7 (MEDIDO, 18/10/2026)
# ............................................................
#     células no teto, t = 100 s : 0%  (também ponderado por volume)
#     μ_t/μ                      : p50 2,84e3 · p99 3,72e5 · máx 5,12e5
#     ν_t no domínio [m²/s]      : p50 0,0415 · p90 1,6 · máx 7,47
#     ν_t na esteira (boi_t)     : p50 1,06 · p90 4,74 · máx 7,47
#
# O máximo de 5,12e5 é o número-chave: está 5× ACIMA do teto antigo. Ou
# seja, o 1e5 cortava campo físico, e a estimativa da tabela acima (ν_t de
# 2 a 19 m²/s na esteira) acertou a ordem de grandeza. Com 1e7 sobra uma
# folga de ~20× sobre o máximo medido: o teto voltou a ser só salvaguarda.
#
# O QUE ISSO MUDA NA PINN. O min(..., teto) da Cell 10 fica no código,
# como recomendado acima — é a relação constitutiva que o solver usou, e
# continua sendo. Só que agora ele não morde no dado. Duas consequências:
#
#   - A verificação (b) da Cell 5 deve mostrar ~0% no ramo do teto, e as
#     tabelas da 5b que contam "ν_t recortado" devem dar zero. Zero é o
#     resultado CERTO, não sinal de coluna trocada.
#
#   - O teto também deixou de segurar a REDE. Com 1e5 ele limitava ν_t da
#     PINN a 1,46 m²/s, o que funcionava como regularização implícita numa
#     rede que já superestimava ν_t (~2×, sst2). Com 1e7 a rede pode ir a
#     146 m²/s antes do min agir, e um ν_t grande deixa o momento rígido.
#     Não é motivo para mudar o resíduo — o resíduo tem que ter o mesmo
#     teto do solver. É motivo para olhar mu_t_p50, frac_visc_cap e
#     'skipped' no log das primeiras épocas.
VISC_RATIO_MAX = 1.0e+7                     # = turb-viscosity/max-limit (job 6125)
NU_T_MAX       = VISC_RATIO_MAX * NU        # [m²/s] ≈ 146,07

# Campo MEDIDO no job 6125 — referência para os prints e para as escalas
# de referência abaixo. Substituem o VISC_RATIO_NEEDED_EST, que era a
# ESTIMATIVA de quanto a escala do problema pediria e perdeu o sentido
# agora que o teto aplicado é esse mesmo valor.
VISC_RATIO_MEASURED_MAX = 5.12e+5           # μ_t/μ máximo no domínio
NU_T_MEASURED_P50       = 0.0415            # [m²/s] domínio inteiro
NU_T_MEASURED_P90       = 1.6               # [m²/s] domínio inteiro
NU_T_MEASURED_MAX       = 7.47              # [m²/s]

# Se esta conferência falhar, o teto voltou a ser baixo demais para o dado
# — exatamente o problema que o job 6125 existe para resolver.
assert VISC_RATIO_MAX > VISC_RATIO_MEASURED_MAX, \
    "teto de μ_t/μ abaixo do máximo medido no dado: o min da Cell 10 cortaria campo físico"

# A coluna que marca as células no teto continua sendo calculada na Cell 5 —
# é diagnóstico barato e útil (dá para ver ONDE elas estão, e acompanhar se
# a fração cresce no tempo). O que mudou é que ela NÃO exclui mais ninguém
# da loss. Com o teto em 1e7 ela deve sair zerada.
MASK_VISC_LIMITED     = False   # era True; ver a discussão acima
VISC_LIMITED_REL_TOL  = 1.0e-3
VISC_LIMITED_PCT_EXPECTED = 0.0   # % de células no teto, job 6125 (era 7,2 no 5518)


# ------------------------------------------------------------
# PRODUÇÃO DE ω — α_∞·S², e a faixa onde ela é incerta
# ------------------------------------------------------------
# A eq. 4.110 dá G_ω = (α α*/ν_t)·G_k. Com α* = 1 e G_k = μ_t S² = ρ ν_t S²,
# o ν_t CANCELA:
#
#     P_ω = α_∞ · S²          (por unidade de massa)
#
# Inclusive o teto cancela, porque aparece em cima e embaixo. Esta é a forma
# a usar.
#
# NÃO usar a forma α(ω/k)G_k do k-ω padrão (eq. 4.82): ela só coincide com
# a 4.110 onde ν_t = k/ω, e o limitador do SST existe justamente para que
# isso não valha em toda parte.
#
# A INCERTEZA: onde o production limiter corta, P_k vira min(μ_t S², 10β*kω)
# e o manual não diz se G_ω usa o P_k limitado ou o bruto. Medido (job 6027,
# t = 30 s): o limiter corta em 1.132 células (0,54%) e remove 0,004% da
# produção total, todas no ponto de estagnação à frente do cilindro
# (x 120-159, y 168-181).
#
# Resolver a dúvida custa mais do que contorná-la: excluir esses pontos do
# resíduo de ω. São meio por cento do domínio, numa região que já é a mais
# difícil de tudo.
MASK_PROD_LIMITED_OMEGA = True


# ------------------------------------------------------------
# Condições de entrada de turbulência (inlet y = Y_MIN)
# ------------------------------------------------------------
# CONFIRMADO — PROBE (job 6025) + MEDIDO na caixa (job 6023). Eram mesmo os
# defaults do Fluent, como esta célula já supunha; agora não é mais suposição.
#
# O JOURNAL_solve_csv define o inlet com uma linha só:
#     /define/boundary-conditions/set/velocity-inlet inlet () vmag no 17 quit
# Nenhum ke-spec, nenhuma intensidade, nenhuma razão de viscosidade. Valem
# os defaults para velocity-inlet com modelo de 2 equações:
# "Intensity and Viscosity Ratio", I = 5%, μ_t/μ = 10.
TI_INLET         = 0.05        # intensidade turbulenta [-] (5%)
VISC_RATIO_INLET = 10.0        # μ_t/μ no inlet [-]

# k = 3/2 (V·I)²
K_INLET = 1.5 * (V_INF * TI_INLET)**2                      # [m²/s²] ≈ 1,0838

# μ_t = (μ_t/μ)·μ  →  ν_t = (μ_t/μ)·ν  →  ω = k/ν_t
NU_T_INLET    = VISC_RATIO_INLET * NU                      # [m²/s]
OMEGA_INLET   = K_INLET / NU_T_INLET                       # [1/s]  ≈ 7419

# Comprimento de escala turbulenta equivalente (só para leitura/diagnóstico)
L_TURB_INLET  = np.sqrt(K_INLET) / (SST_BETA_STAR**0.25 * OMEGA_INLET)   # [m]


# ------------------------------------------------------------
# Escalas de referência para k, ω e ν_t
# ------------------------------------------------------------
# Papel análogo ao de V_REF: dar à rede saídas em O(1) e adimensionalizar
# resíduos. As MÉDIAS E DESVIOS de fato (z-score) saem do script de
# estatística sobre os CSVs e entram na Cell 4 — aqui ficam só as escalas
# físicas, que não dependem do dataset.
#
# NÃO usar K_REF = V_INF²: superestima k em ~1,5 ordem de grandeza.
#
# As três escalas NÃO são independentes. Fora da faixa do limitador (que é a
# maior parte do domínio) vale ν_t = k/ω, então fixar duas determina a
# terceira. Fixo K_REF e NU_T_REF e DERIVO ω.
#
# [18/10/2026] NU_T_REF DEIXOU DE SER 1/4 DO TETO. Era NU_T_REF =
# 0.25·NU_T_MAX, com o argumento "acima do teto não existe ν_t no dado".
# Isso fazia sentido com 1e5, porque ali o teto MORDIA (4,3 a 7,2% das
# células): ele era, na prática, a escala do ν_t. Com 1e7 e 0% no teto ele
# virou guarda numérica inerte, e a premissa da fórmula morreu junto.
# Mantida, ela daria NU_T_REF = 36,5 m²/s (5× o máximo medido) e
# OMEGA_REF = 0,267 1/s — cem vezes abaixo do ω típico do domínio.
#
# Hoje isso seria INERTE no treino, por acidente: o resíduo de ω usa a
# escala "local" (Cell 10), e LNW_MEAN é remedido pela Cell 4. Mas basta
# virar OMEGA_RESIDUAL_SCALE para "global" ou AUTO_NORM_STATS para False
# para o fator 100 entrar no treino sem aviso nenhum.
#
# Então NU_T_REF passa a ser um VALOR ABSOLUTO, e o valor escolhido é o
# da sst2 (0,25 × 1e5 × ν = 0,36518, com 4 algarismos). Dois motivos:
#   - cai entre o p50 (0,0415) e o p90 (1,6) do ν_t MEDIDO no job 6125,
#     e dá OMEGA_REF = 26,7 1/s, coerente com o k/ν_t ≈ 25 do domínio;
#   - deixa OMEGA_REF e SCALE_OM exatamente onde estavam na sst2. Esta
#     rodada muda UMA variável (o teto); as escalas de referência não
#     entram na conta.
TI_REF     = 0.15                              # intensidade típica na esteira

K_REF     = 1.5 * (TI_REF * V_INF)**2          # [m²/s²] ≈ 9,75
NU_T_REF  = 0.3652                             # [m²/s]  valor absoluto (= sst2)
OMEGA_REF = K_REF / NU_T_REF                   # [1/s]   ≈ 26,7, derivado de ν_t = k/ω

assert NU_T_REF < NU_T_MAX, "NU_T_REF acima do teto imposto pelo solver"
# A escala tem que descrever o campo que existe, não o teto: tem que cair
# dentro da faixa central do ν_t medido.
assert NU_T_MEASURED_P50 < NU_T_REF < NU_T_MEASURED_P90, \
    "NU_T_REF fora da faixa p50–p90 do ν_t medido no job 6125"

# Piso/teto em unidades normalizadas (usados na desnormalização segura)
LN_OMEGA_FLOOR = float(np.log(OMEGA_FLOOR))


# ------------------------------------------------------------
# Distância à parede (necessária para F1 e F2)
# ------------------------------------------------------------
# Zonas da malha, conforme o /report/summary do job 5518. As 6 do tipo
# "wall" entram no cálculo de distância; inlet e outlet NÃO.
#
# Parede móvel continua sendo parede para efeito de distância — o Fluent
# calcula d_wall sobre todas as zonas do tipo wall, independente de a
# velocidade tangencial ser prescrita. O summary confirma que lateral_min,
# lateral_max e top são "Moving Wall" com "Shear Boundary Condition: No Slip"
# e translação (0, 1, 0) a 17 m/s; ground, cylinder_wall e cylinder_top são
# "Stationary Wall / No Slip".
#
# (O ansatz da Cell 9 já trata isso: o decay_freestream inclui decay_xmin,
#  decay_xmax e decay_ztop, então v = V_INF nas três paredes móveis e v = 0
#  no chão e no cilindro. Conferido, nada a mudar lá.)
#
# Mapeamento nome-no-Fluent -> nome-no-código (as BCs da Cell 5):
FLUENT_ZONE_TO_BC = {
    "inlet":         "bc_inlet",     # velocity-inlet, y = Y_MIN   (não é parede)
    "outlet":        "bc_outlet",    # pressure-outlet, y = Y_MAX  (não é parede)
    "lateral_min":   "bc_xmin",      # wall móvel, x = X_MIN
    "lateral_max":   "bc_xmax",      # wall móvel, x = X_MAX
    "top":           "bc_top",       # wall móvel, z = Z_MAX
    "ground":        "bc_ground",    # wall fixa,  z = Z_MIN
    "cylinder_wall": "cyl_lateral",  # wall fixa
    "cylinder_top":  "cyl_top",      # wall fixa
}

WALL_ZONES = ("bc_ground", "bc_xmin", "bc_xmax", "bc_top",
              "cyl_lateral", "cyl_top")

D_WALL_FLOOR = 1.0e-3          # [m] piso da distância à parede (F1/F2 dividem por d²)

# Menor meia-largura do domínio: teto natural da distância à parede
D_WALL_MAX_EST = min(0.5 * (X_MAX - X_MIN), 0.5 * (Z_MAX - Z_MIN))


# ------------------------------------------------------------
# CÉLULAS COLADAS NA PAREDE — o que o Fluent faz ali
# ------------------------------------------------------------
# PROBE (job 6025): turbulence/wall-omega-treatment = "correlation".
# MANUAL: Theory Guide §4.18.3, eqs. 4.394-4.407. Esta seção foi lida em
# 29/09/2026 e corrige duas coisas que estavam escritas aqui.
#
# ............................................................
# O QUE "correlation" É, EXATAMENTE (eqs. 4.404-4.407)
# ............................................................
#     ω_wall   = ρ(u*)²/μ · ω⁺_wall
#     ω⁺_wall  = ω⁺_lam·[1 + (ω⁺_turb/ω⁺_lam)^C_exp]^(1/C_exp)   C_exp = 1,3
#     ω⁺_lam   = C_calib·6/(β_i (y⁺)²)                           C_calib = 1/3
#     ω⁺_turb  = (1/√β*_∞)·du⁺_turb/dy⁺ = 1/(√β* κ_w y⁺)
#     u*       = [(μ|ΔU_tan/Δy| + C_μ^{1/2} ρ k)/ρ]^{1/2}
#
# Em unidades dimensionais as assintotas são ω_turb = u*/(√β* κ_w y) e
# ω_lam = 2ν/(β_i y²) — note o C_calib = 1/3, que divide por três a forma
# clássica 6ν/(β_i y²). Calibrado em Couette plano a Re = 1e6.
#
# E AQUI ISSO NÃO IMPORTA. Com u* = C_μ^{1/4}√k e k ~ 2,4 m²/s²:
#
#     y [m]     ω_lam      ω_turb    razão     ω_blend/ω_turb
#      0,02     0,974       377,6      388        1,00033
#      0,15     0,0173       45,0     2601        1,00003
#      2,90     4,6e-05      1,84    39760        1,0000008
#
# A correlação É a lei logarítmica pura nesta malha, até a quarta casa. O
# argumento que sustentava WALL_RESIDUAL_MODE='mask' como default —
# "correlação não é lei log" — está formalmente certo e praticamente vazio
# a y⁺ ~ 1e5.
#
# ............................................................
# k E ω NÃO RECEBEM O MESMO TRATAMENTO (§4.18.3.2)
# ............................................................
# Esta célula dizia que "o solver não resolve o transporte" dos dois. Errado:
#
#   ω  É PRESCRITO. A equação de transporte não determina ω na célula
#      adjacente.
#
#   k  CONTINUA SENDO TRANSPORTADO, com ∂k/∂n = 0 na parede. O que muda é
#      só a produção, que vira a média harmônica (eqs. 4.400-4.403)
#          P_k,wall = (P_k,lam⁻¹ + P_k,turb⁻¹)⁻¹
#      em vez de μ_t S².
#
# Ainda assim, MASCARAR k CONTINUA CERTO — por outro motivo. P_k,wall
# depende de τ_w, u* e y⁺, todos definidos a partir de Δy, a ALTURA DA
# CÉLULA. Num ponto de colocação arbitrário dentro daquela célula, Δy não
# existe. O balanço que o solver fechou ali é de volume finito com fonte
# modificada, não uma EDP avaliada ponto a ponto. Não há resíduo pontual a
# impor.
#
# (Indício sobre ω, não prova: o §4.11.9, das BCs de parede do RSM, diz que
#  o Fluent "fixa" os valores na célula adjacente e resolve o transporte
#  "only in the bulk flow region". Se o SST segue o mesmo padrão, ω está
#  congelado na célula e mascarar é o certo. Decidível no dado — ver a
#  pendência da Cell 5.)
#
# ............................................................
# QUANDO A RELAÇÃO DA CAMADA LOG VALE
# ............................................................
# Do equilíbrio local com as fórmulas do Fluent:
#     P_k,turb = ρ u_τ² u*/(κ_w y)      ε = β* k ω = √β* k u*/(κ_w y)
#     P = ε  ⟹  k = u_τ²/√β*
# e com ω = u*/(√β* κ_w y):
#
#     k = √β* · κ_w² · y² · ω² · (u_τ/u*)²
#
# O FATOR (u_τ/u*)² é o que faltava aqui. O manual diz que u* foi
# introduzido "to improve the behavior at separation": em recirculação
# u_τ → 0 mas u* continua finito, porque depende de k. Logo:
#
#   PREVISÃO para a verificação (d) da Cell 5: a razão k/(C·y²ω²) deve dar
#   ≈ 1 no chão a montante e nas paredes laterais/topo, e cair bem abaixo
#   de 1 na parede do cilindro e no chão dentro da esteira. Se for isso,
#   'loglaw' é defensável POR PAREDE, não globalmente — e a Cell 5 já mede
#   por parede.
#
# Note que κ_w CANCELA na identidade acima (√β*κ²·u*²/(β*κ²) = u*²/√β*).
# Ele não cancela no CÓDIGO: o ω do CSV foi produzido com o κ das funções
# de parede do Fluent, então WALL_LAW_C tem que usar o MESMO valor. Usar
# 0,41 dá um viés sistemático de (0,4187/0,41)² = +4,3%.
#
# ............................................................
# OS TRÊS MODOS
# ............................................................
#   "mask"    — zera os resíduos de k e de ω na faixa. Não depende de
#               hipótese nenhuma sobre o que o tratamento de parede faz.
#   "loglaw"  — troca o resíduo de k pela relação acima, que em espaço log
#               fica LINEAR nas saídas da rede e já sai adimensional:
#                   r_wall = ln k − ln(√β*κ_w²) − 2 ln y − 2 ln ω
#               Ligar só onde a verificação (d) mostrar razão perto de 1.
#   "off"     — impõe o transporte em todo lugar (para comparação).
WALL_RESIDUAL_MODE = "mask"        # "mask" | "loglaw" | "off"

# κ DAS FUNÇÕES DE PAREDE — 0,4187, não 0,41.
# O Fluent usa dois valores em lugares diferentes: 0,41 na derivação de α_∞
# (eqs. 4.112-4.113, que dizem "where κ is 0.41") e 0,4187 nas funções de
# parede (eqs. 4.345, 4.397). Como o ω do dado saiu das funções de parede,
# a relação da camada log tem que usar 0,4187.
KAPPA_WALL = 0.4187

# Constante da relação da camada log, usada só no modo "loglaw"
WALL_LAW_C = float(np.sqrt(SST_BETA_STAR) * KAPPA_WALL**2)   # ≈ 0,052594

# Compatibilidade com a Cell 10, que lê este nome
MASK_WALL_FUNCTION_RESIDUAL = (WALL_RESIDUAL_MODE != "off")

# Até onde vai a faixa. O valor abaixo é só um chute inicial: a Cell 5 MEDE
# a partir do cell-wall-distance do static_mesh.csv, POR PAREDE, e
# sobrescreve (ver Y_BAND_BY_WALL e wall_band_by_point, Cell 5).
Y_WALL_FUNCTION = 5.0          # [m]


# ------------------------------------------------------------
# Fatos do dataset (job 6125) — usados para sanity check nas Cells 3 a 5
# ------------------------------------------------------------
# Os valores são os mesmos do job 5518: a malha, o dt e o nº de passos não
# mudaram. Só o teto de ν_t mudou.
CFD_JOB_ID    = 6125           # produção; 6124 foi o teste de 20 passos
N_CELLS_CFD   = 211_579        # células da malha_simples_boi
N_STEPS_CFD   = 1_000          # passos resolvidos e exportados
DT_CFD        = 0.1            # [s] passo de tempo
T_END_CFD     = N_STEPS_CFD * DT_CFD     # 100,0 s de tempo físico

# Uma travessia do domínio no sentido do escoamento
T_FLOWTHROUGH = (Y_MAX - Y_MIN) / V_INF  # = 41,2 s

# Campos exportados por passo, na ordem EM QUE APARECEM no CSV
CSV_COLUMNS_CFD = (
    "cellnumber", "x-coordinate", "y-coordinate", "z-coordinate",
    "viscosity-turb", "specific-diss-rate", "turb-kinetic-energy",
    "pressure", "z-velocity", "y-velocity", "x-velocity",
)

# "viscosity-turb" no Fluent é μ_t DINÂMICA [kg/(m·s)], não ν_t. Para
# comparar com a fórmula é preciso dividir por ρ. Confirmado pelo job 6027,
# que reconstruiu a coluna inteira em unidades cinemáticas.
VISC_TURB_IS_DYNAMIC = True

# Arquivo de malha estática, exportado DENTRO da sessão do solve
# (portanto alinhado célula a célula com os t_*.csv)
#
# No job 6125 entrou cell-volume, ANTES de cell-wall-distance. Nada no
# código lê esta tupla nem a 5ª coluna por posição: load_static_wall_distance
# (Cell 5) e validate_wall_distance (Cell 7) leem por NOME. A tupla é
# documentação, e está aqui na ordem em que o arquivo de fato vem.
STATIC_MESH_CSV    = "static_mesh.csv"
STATIC_MESH_FIELDS = ("cellnumber", "x-coordinate", "y-coordinate",
                      "z-coordinate", "cell-volume", "cell-wall-distance")


# ------------------------------------------------------------
# Números adimensionais de referência
# ------------------------------------------------------------
RE_D = RHO_INF * V_INF * D_CYL / MU        # Reynolds pelo diâmetro do cilindro
RE_H = RHO_INF * V_INF * H_CYL / MU        # Reynolds pela altura do cilindro


# ============================================================
# LOG
# ============================================================
print(f"Domain:    X[{X_MIN},{X_MAX}] Y[{Y_MIN},{Y_MAX}] Z[{Z_MIN},{Z_MAX}]  (m)")
print(f"Cylinder:  D={D_CYL}m H={H_CYL}m centered at ({XC_CYL},{YC_CYL},{ZC_BASE})")
print(f"BOI_C:     X{BOI_C_X} Y{BOI_C_Y} Z{BOI_C_Z}")
print(f"BOI_T:     V1={BOI_T_V1} V2={BOI_T_V2} V3={BOI_T_V3} Z{BOI_T_Z}")
print(f"Tolerâncias: BC_TOL={BC_TOL}m  CYL_TOL={CYL_TOL}m")
print(f"Flow:      V_inf={V_INF} m/s (+Y)")
print(f"Air:       rho_inf={RHO_INF:.4f} kg/m³, mu={MU:.2e}, nu={NU:.4e} m²/s")

print(f"\n--------- Numérica do solver (não entra no resíduo) ---------")
print(f"  SIMPLE · BDF2 · momento Third-Order MUSCL · k,ω 2nd Order Upwind")
print(f"  nada disso muda um termo da EDP, mas:")
print(f"    (a) o dado satisfaz a EDP contínua só até O(Δx^p)+O(Δt^q) — há")
print(f"        um PISO para a physics loss; mensurável em parte com deriv/")
print(f"        (job {CFD_JOB_ID}), mas este treino não lê deriv/")
print(f"    (b) upwind adiciona difusão numérica, que não está na EDP")
print(f"    (c) o valor do CSV é média de célula, não pontual (erro O(Δx²))")
_f_shed = 0.2 * V_INF / D_CYL
print(f"  tempo bem resolvido: St≈0,2 -> f={_f_shed:.4f} Hz, T={1/_f_shed:.1f} s")
print(f"      = {1/_f_shed/DT_CFD:.0f} passos de {DT_CFD} s por ciclo de desprendimento")
print(f"      (St de cilindro 2D, só ordem de grandeza: no dado não é mensurável)")
print(f"  SIMPLE convergiu: 13,2 iter/passo contra teto de 50 (job {CFD_JOB_ID})")

print(f"\n--------- Pressão (MEDIDO, job 6023) ---------")
print(f"  coluna `pressure` = p̃ = p + (2/3)ρk   (modificada)")
print(f"  -> o termo (2/3)ρ∇k NÃO entra no momento (seria contagem dupla)")

print(f"\n--------- Constantes do k-omega SST ---------")
print(f"  Prandtl (Fluent):  σ_k1={SST_PR_K1}  σ_ω1={SST_PR_W1}  "
      f"σ_k2={SST_PR_K2}  σ_ω2={SST_PR_W2}")
print(f"  Menter (=1/Pr):    σ_k1={SST_SIGMA_K1:.6f}  σ_ω1={SST_SIGMA_W1:.6f}  "
      f"σ_k2={SST_SIGMA_K2:.6f}  σ_ω2={SST_SIGMA_W2:.6f}")
print(f"  mistura linear nesta convenção == mistura harmônica na do Fluent")
print(f"  β_i1={SST_BETA_1}  β_i2={SST_BETA_2}  β*={SST_BETA_STAR}  "
      f"κ={SST_KAPPA}  a1={SST_A1}")
print(f"  α_∞1={SST_GAMMA_1:.6f} (manual 0.553167)   "
      f"α_∞2={SST_GAMMA_2:.6f} (manual 0.440263)")
print(f"  α*=1, β*=β*_∞, f_β=f_β*=1, F(M_t)=0  (low-Re e compressibilidade off)")
print(f"  limitador de produção: P_k ≤ {SST_PROD_LIMIT_COEF:.0f}·β*·ρ·k·ω  "
      f"({'ativo' if SST_PRODUCTION_LIMITER else 'inativo'})")
print(f"  pisos: k≥{K_FLOOR:.1e}  ω≥{OMEGA_FLOOR:.1e}  CD_ω≥{SST_CD_KW_FLOOR:.1e}")

print(f"\n--------- Viscosidade turbulenta (MEDIDO, job 6027) ---------")
print(f"  ν_t = min( (k/ω) / max(1, S·F2/(a1·ω)) , {NU_T_MAX:.4f} )   [m²/s]")
print(f"  F2 = tanh(Φ2²),  Φ2 = max(2√k/(β*ωy), 500ν/(y²ω))")
print(f"  limitador usa S = sqrt(2 S_ij S_ij), NÃO Ω "
      f"(24.848 células decidem: S 100%, Ω 0%)")
print(f"  teto μ_t/μ ≤ {VISC_RATIO_MAX:.0e}  ->  ν_t ≤ {NU_T_MAX:.2f} m²/s "
      f"(job {CFD_JOB_ID}; era 1e5 / 1,4607 no 5518)")
print(f"  campo medido no job {CFD_JOB_ID}:")
print(f"       μ_t/μ máx = {VISC_RATIO_MEASURED_MAX:.2e}  "
      f"(folga de {VISC_RATIO_MAX/VISC_RATIO_MEASURED_MAX:.0f}× até o teto)")
print(f"       ν_t p50 = {NU_T_MEASURED_P50:.4f}  p90 = {NU_T_MEASURED_P90:.2f}  "
      f"máx = {NU_T_MEASURED_MAX:.2f} m²/s")
print(f"       células no teto em t=100 s: {VISC_LIMITED_PCT_EXPECTED:.1f}%  "
      f"(no 5518 chegavam a {100*15265/N_CELLS_CFD:.1f}%)")
print(f"  >>> o teto continua no resíduo (é a relação constitutiva do solver),")
print(f"      mas não morde mais no dado — MASK_VISC_LIMITED = {MASK_VISC_LIMITED}")
print(f"  >>> zero no ramo do teto da verificação (b) e nas tabelas da 5b é")
print(f"      o resultado ESPERADO, não sinal de coluna trocada")
print(f"  >>> o teto também não segura mais a rede (1,46 -> {NU_T_MAX:.0f} m²/s):")
print(f"      acompanhar mu_t_p50, teto ν_t e 'skipped' nas primeiras épocas")

print(f"\n--------- Produção de ω ---------")
print(f"  P_ω = α_∞·S²   (o ν_t cancela, inclusive o teto)")
print(f"  onde o production limiter corta (0,54% das células), P_ω é incerto")
print(f"  MASK_PROD_LIMITED_OMEGA = {MASK_PROD_LIMITED_OMEGA}")

print(f"\n--------- Turbulência no inlet (defaults, PROBE job 6025) ---------")
print(f"  TI       = {TI_INLET*100:.1f} %")
print(f"  μ_t/μ    = {VISC_RATIO_INLET:.1f}")
print(f"  k_inlet  = {K_INLET:.6f} m²/s²")
print(f"  ν_t_in   = {NU_T_INLET:.6e} m²/s")
print(f"  ω_inlet  = {OMEGA_INLET:.2f} 1/s   (ln ω = {np.log(OMEGA_INLET):.4f})")
print(f"  l_turb   = {L_TURB_INLET:.4e} m")

print(f"\n--------- Escalas de referência (PLACEHOLDER até a Cell 4) ---------")
print(f"  TI_REF     = {TI_REF*100:.0f} %")
print(f"  K_REF      = {K_REF:.3f} m²/s²")
print(f"  NU_T_REF   = {NU_T_REF:.4f} m²/s   (valor absoluto, = sst2; entre o p50 "
      f"{NU_T_MEASURED_P50} e o p90 {NU_T_MEASURED_P90} medidos)")
print(f"  OMEGA_REF  = {OMEGA_REF:.3f} 1/s    (derivado de ν_t = k/ω; esperado ≈ 26,7)")
print(f"  ln ω esperado ∈ [{np.log(OMEGA_REF):+.2f} (núcleo), "
      f"{np.log(OMEGA_INLET):+.2f} (entrada)]  -> faixa de {np.log(OMEGA_INLET)-np.log(OMEGA_REF):.1f} "
      f"em ln, {OMEGA_INLET/OMEGA_REF:.0f}× em ω")

print(f"\n--------- Distância à parede ---------")
print(f"  fonte: {WALL_DISTANCE_SOURCE}   (o CFD exportou cell-wall-distance "
      f"em {STATIC_MESH_CSV}, alinhado com os t_*.csv)")
print(f"  zonas parede ({len(WALL_ZONES)}): {', '.join(WALL_ZONES)}")
print(f"  d_wall ∈ [{D_WALL_FLOOR:.1e}, ~{D_WALL_MAX_EST:.1f}] m")
print(f"  (inlet e outlet NÃO contam como parede)")

print(f"\n--------- Faixa junto à parede (Theory Guide §4.18.3) ---------")
print(f"  wall-omega-treatment = 'correlation' (y+-insensível):")
print(f"    ω  é PRESCRITO na célula adjacente (eqs. 4.404-4.407)")
print(f"    k  continua transportado, com ∂k/∂n=0 e produção P_k,wall")
print(f"       blendada (eqs. 4.400-4.403) em vez de μ_t S²")
print(f"  a y+ ~ 1e5 o blend devolve a lei log pura (erro < 4e-4), mas")
print(f"  P_k,wall depende de Δy (altura de CÉLULA), que não existe num")
print(f"  ponto de colocação -> não há resíduo pontual a impor em k também")
print(f"  WALL_RESIDUAL_MODE = '{WALL_RESIDUAL_MODE}'"
      f"   -> {'zera k e ω na faixa' if WALL_RESIDUAL_MODE=='mask' else ('troca k por r_wall' if WALL_RESIDUAL_MODE=='loglaw' else 'impõe transporte em todo lugar')}")
print(f"  Y_WALL_FUNCTION = {Y_WALL_FUNCTION:.2f} m (chute; a Cell 5 mede)")
print(f"  C = √β*·κ_w² = {WALL_LAW_C:.6f}  (κ_w={KAPPA_WALL}, não {SST_KAPPA};")
print(f"      só usado no modo 'loglaw'. Com κ=0,41 o viés seria +4,3%)")
print(f"  k = C·y²·ω²·(u_τ/u*)²: o último fator vale 1 em equilíbrio e cai")
print(f"      na região separada -> esperar razão ≈1 no chão a montante e")
print(f"      <1 atrás do cilindro (verificação (d) da Cell 5)")

print(f"\n--------- Dataset (job {CFD_JOB_ID}) ---------")
print(f"  células      = {N_CELLS_CFD:,}")
print(f"  passos       = {N_STEPS_CFD:,} × {DT_CFD} s = {T_END_CFD:.0f} s físicos")
print(f"  travessia    = {T_FLOWTHROUGH:.1f} s  -> {T_END_CFD/T_FLOWTHROUGH:.1f} travessias")
print(f"  colunas/CSV  = {len(CSV_COLUMNS_CFD)} (ordem invertida vs. o journal)")
print(f"  static_mesh  = {', '.join(STATIC_MESH_FIELDS[4:])} (lidos por NOME)")
print(f"  derivadas de velocidade e strain-rate-mag: exportadas em deriv/,")
print(f"      NÃO lidas por este treino (próxima variável, ver Cell 3)")
print(f"  comparar com a sst2 só estatisticamente: as rodadas de CFD divergem")
print(f"      depois de t ≈ 40 s")

print(f"\n--------- Reynolds ---------")
print(f"  Re_D (D={D_CYL} m) = {RE_D:.3e}")
print(f"  Re_H (H={H_CYL} m) = {RE_H:.3e}")

# ============================================================
# PENDÊNCIAS QUE ESTA CÉLULA CRIA PARA AS SEGUINTES
# ============================================================
# Esta célula mudou decisões que outras células ainda não leem. Enquanto
# elas não forem refeitas, o arquivo está num estado intermediário — e é
# melhor gritar do que deixar passar.
#
# [18/10/2026] Saíram os dois itens [CFD] — subir o teto e exportar as
# derivadas —, ambos feitos no job 6125. Entraram os textos das Cells 4 e
# 5b que ainda esperam ~7% de células no teto.
_PENDENCIAS = [
    ("Cells 3/9", "comentários dizem que a turbulência do inlet decai com "
                "'escala de 0,26 mm'. 0,26 mm é o comprimento turbulento "
                "(L_TURB_INLET); a distância em que ω cai à metade é "
                "V_INF/(β·ω0) ≈ 2,8 cm. Só texto — os números do código "
                "não dependem disso"),
    ("medir",   "o teto não segura mais a REDE (ν_t até 146 m²/s antes do "
                "min agir). Acompanhar mu_t_p50, frac_visc_cap e passos "
                "descartados nas primeiras épocas da sst3"),
    ("medir",   "confirmar K_RESIDUAL_FLOOR_XI (Cell 10) com a varredura "
                "de ξ do diagnostico_residuos.py no checkpoint em uso — "
                "critério: top 1% do MSE abaixo de ~30% sem 'free' "
                "disparar. O valor atual é um ponto de partida."),
    ("medir",   "ν_t é o pior campo da sst2 (0,457 dex, R²=0,679, viés "
                "+0,336 dex = ×2,17: a rede superestima). Com deriv/ dá "
                "para comparar o S da PINN com o strain-rate-mag do CFD e "
                "classificar o ramo do limitador com o MESMO limiar nos "
                "dois lados — próxima variável depois da sst3 andar"),
    ("medir",   "piso da physics loss: continuidade inteira a partir de "
                "deriv/; momento só parcial (falta ∇p, que não foi "
                "exportado)"),
]
# Feitas em 18/10/2026 (dado novo, job 6125, rodada "sst3"):
#   [Cell 3]  CFD_DIR apontando para new_sim_turb/results/; INLET_FACES_CSV
#             e HAS_INLET_FACES; RUN_TAG "sst3"; WARM_START_FROM = None
#             (cold start obrigatório: a normalização é remedida no dado
#             novo e os pesos da sst2 foram escritos com outra).
#   [Cell 4]  print das células recortadas compara com
#             VISC_LIMITED_PCT_EXPECTED (0% com 1e7) em vez de "~7,2% do
#             5518"; nota de que AUTO_NORM_STATS tem que ficar True na sst3
#             e de que o fallback ainda é o do laminar.
#   [Cell 4]  fallback = constantes MEDIDAS na calibração (job 6389), com
#             as do laminar comentadas. A tabela vira "fallback × medido" e
#             passa a ser teste: Δ ≈ 0 confirma o mesmo dado.
#   [Cell 3]  saiu o aviso velho de que a Cell 5 "tem que passar a usar
#             CSV_DELIMITER" — ela já usa.
#   [Cell 5]  verificação (d): razão ≈ 1 não marca mais 'mask' como
#             "REVER" — remete à (f), que distingue lei log de identidade
#             (ω congelado). Na calibração (job 6389) a (f) deu congelado.
#   [Cell 16] métrica log de k em ln(k + K_EPS), a variável do treino. Em
#             ln k cru a calibração (job 6389) deu "×243" — efeito de 34%
#             das células abaixo de K_EPS e de k levemente negativo virando
#             ln(1e-30) = −69, não erro da rede.
#   [Cell 16] metadata com bloco "procedencia" (job do CFD, CFD_DIR, faces
#             do inlet, pastas de saída), campo de ν_t medido e escalas de
#             referência em "turbulencia", e INLET_CHECK em "verificacoes".
#             _jsonable converte escalares do NumPy antes do json.dump.
#   [Cell 15] não redefine mais CHECKPOINT_DIR (usa o absoluto da Cell 3);
#             CSV e monitoramento em HISTORICO_DIR. Rodada nova (epoch_offset
#             = 0) zera também o monitoramento, não só o CSV. Cabeçalho com
#             a entrada da sst3 (cold start).
#   [slurm]   pinn_sst3_job.slurm: logs em 10-18_simples/logs/, script em
#             10-18_simples/, e o exit code real (o antigo imprimia sempre 0,
#             porque o $(date) da mesma linha zerava o $?).
#   [Cell 9c] sanity_check_hardconstraint.png em DIAG_DIR.
#   [Cell 8]  collocation_distribution.png e wall_distance_and_turbulence.png
#             em DIAG_DIR.
#   [Cell 5b] §2 compara a fração recortada com VISC_LIMITED_PCT_EXPECTED
#             (zero agora é o OK; recorte é que é suspeito). §3(3) trata o
#             caso de conjuntos vazios ANTES do Jaccard (que dava 1 por
#             convenção e "confirmava" alinhamento sem olhar célula) e
#             aponta o teste (b). §3(4) virou diagnóstico de decaimento e
#             remete ao veredito da (c). Três CSVs em DIAG_DIR.
#   [Cell 5]  verificação (c) reescrita: lê INLET_FACES_CSV e compara o
#             valor de FACE imposto pelo solver com K_INLET/OMEGA_INLET e
#             com KN_INLET/LNWN_INLET (o que a hard constraint impõe).
#             Aborta se divergirem (INLET_MISMATCH_ABORTS). Sem o arquivo,
#             roda o modo antigo rotulado como medida de decaimento, sem
#             veredito. Resultado em INLET_CHECK. Textos de S/deriv e
#             "cinco verificações" -> seis.
#   [Cell 3]  destino da saída com caminho absoluto: TRAIN_DIR e, dentro de
#             saida/, HISTORICO_DIR, DIAG_DIR e CHECKPOINT_DIR, criados na
#             hora e conferidos quanto à permissão de escrita. Aviso se a
#             pasta de checkpoints já tiver pesos (RUN_TAG repetido).
#   [Cell 2]  VISC_RATIO_MAX 1e5 -> 1e7 (NU_T_MAX 1,4607 -> 146,07).
#             NU_T_REF deixou de ser 0,25·NU_T_MAX e virou valor absoluto
#             (0,3652, o da sst2), mantendo OMEGA_REF ≈ 26,7 e SCALE_OM
#             ≈ 10,30. VISC_RATIO_NEEDED_EST substituído pelo campo medido
#             (VISC_RATIO_MEASURED_MAX, NU_T_MEASURED_*). STATIC_MESH_FIELDS
#             com cell-volume. CFD_JOB_ID = 6125. Textos do teto, da
#             numérica e do dataset atualizados.
# Feita em 29/09/2026 (XLA):
#   [Cell 9]  HardConstraintLayer: tf.concat -> tf.stack + reshape. O
#             gradiente do Concat vira Slice, e dentro de um contexto XLA o
#             gradiente do Slice vira XlaDynamicUpdateSlice, que não tem
#             gradiente registrado — o treino precisa de 3ª derivada e
#             morria no traçado. Mesmo valor, mesma ordem de colunas.
#   [Cell 14] optimizer.build(...) antes do traçado: com o XLA passando, o
#             apply_gradients passou a ser alcançado e os slots do Adam
#             nasciam dentro do grafo. E o reset do Adam pós-probe agora é
#             no lugar, não um optimizer novo (que o grafo já traçado não
#             veria). Medido depois das duas: train_step compila com
#             jit_compile=True, USE_XLA = True.
# Feitas em 01/10/2026 (diagnóstico do resíduo de k, checkpoint 400):
#   [Cell 9]  HARD_BC_K_INLET / HARD_BC_OMEGA_INLET: foram para False com
#             base no ckpt 400 e VOLTARAM para True com base no ckpt 2200,
#             onde o erro no inlet já era 0,13 dex. Ver o cabeçalho da
#             célula: o argumento caiu, e com a escala local o inlet
#             responde por 0,6% do MSE de r_k.
#   [Cell 10] K_RESIDUAL_SCALE = "local" com ξ = 1e-4: r_k passa a ser
#             dividido pelo termo de destruição, como o ω já era. Medido
#             no ckpt 2200: 45 pontos de 44.535 faziam 82,5% do MSE de
#             r_k e adj_inlet fazia 91,7%; com a escala local o top 1% cai
#             de 97,8% para 33,2% e a faixa dinâmica de 2490× para 14×.
#             ξ escolhido por varredura, não por chute — tabela na célula.
#   [Cell 12] BALANCE_N_COLL 1.024 -> 16.384. Com a cauda do r_k, 1.024
#             pontos mediam k = 1,15 quando o treino media 18-25.
#   [Cell 3]  RUN_TAG "sst2" e WARM_START_FROM apontando para o ckpt da
#             rodada "sst". Ficam na Cell 3, e não na 15 onde são usados,
#             para um caminho errado derrubar o job antes dos 30 min de
#             leitura dos CSVs. (Valores substituídos em 18/10/2026 pela
#             sst3 — ver acima. O motivo de morarem na Cell 3 continua.)
#   [Cell 15] warm start: carrega os pesos, confere que mudaram e zera o
#             Adam. CSV novo de propósito: r_k mudou de régua.
# Feita em 30/09/2026 (job 6072, modo calibration):
#   [Cell 14] a prova de XLA virou get_concrete_function. Executar um passo
#             de verdade pediu 39,06 GiB, o OOM corrompeu o BFCAllocator e o
#             processo abortou na op seguinte (core dumped) — a rede de
#             segurança derrubava o job. Ver ponto 6 do cabeçalho.
#   [Cell 7]  COLL_STEP_SCALE = 0.5 (24.000 pontos/passo em vez de 48.000).
#             O pool e o layout de regiões ficam idênticos; muda só quantos
#             pontos entram em cada passo. Memória ~linear no nº de pontos.
#   [Cell 15] pico de memória de GPU vai para o log nas 3 primeiras épocas.
# Feitas em 27/09/2026 (Cell 5 refeita):
#   [Cell 5] verificação de ν_t agora testa a fórmula inteira e separa os
#            três ramos (teto / livre / limitador), recuperando S por inversão
#   [Cell 5] faixa de parede medida POR PAREDE, pelo salto entre a 1ª e a 2ª
#            camada de células; Y_BAND_BY_WALL e wall_band_by_point
#   [Cell 5] coluna COL_DWALL no data_np, e verificação (e) da pressão
#   [Cell 7] sample_collocation devolve coll_ywall (limiar por ponto);
#            wall_distance_components vira ponto único de verdade e a
#            célula confere o nearest_wall_index da Cell 5 contra ela
# Feitas em 29/09/2026 (verificação (f) acrescentada à Cell 5):
#   [Cell 5] verify_omega_wall: reconstrói ω_wall pelas eqs. 4.404-4.407 e
#            compara com o ω da 1ª camada. Além de decidir a máscara de ω,
#            diz se a razão da verificação (d) é física ou identidade
# Feitas em 29/09/2026 (Cell 16 escrita):
#   [Cell 16] metadata com TODAS as constantes de desnormalização (a do
#             laminar referenciava L_REF/P_REF/TIME_REF, que não existem);
#             conferência do checkpoint recarregado; tabela de erro físico
#             na validação no lugar das tabelas de variáveis inexistentes
# Feitas em 29/09/2026 (Cell 15 escrita):
#   [Cell 15] validação sem réplica da fórmula (acumula as mesmas somas do
#             mean_zone e confere contra data_loss_grouped); val_mse4 para
#             comparar com o laminar; log das máscaras e dos descartes
# Feitas em 29/09/2026 (Cell 14 escrita):
#   [Cell 14] gather nos três tensores de colocação; pesos como tf.Variable
#             (rampa sem retracing); guarda de NaN com contador; dict
# Feitas em 29/09/2026 (Cell 13 escrita):
#   [Cell 13] batch com as 13 colunas e assert contra N_COLS_DATA; três
#             tensores de colocação e resample_collocation_pool atômico
# Feitas em 29/09/2026 (Cell 12 escrita):
#   [Cell 12] W com k e omega; ENFORCE_K_EQUATION/ENFORCE_OMEGA_EQUATION
#             da Cell 1 finalmente agem; balanço de contribuições medido
# Feitas em 29/09/2026 (Cell 11 escrita):
#   [Cell 10] devolve active_cont/active_mom/active_k/active_omega
#   [Cell 11] masked_mse divide pelos pontos ativos; data loss com 6
#             componentes e índices nomeados (a coluna 8 não é mais o gid)
# Feitas em 29/09/2026 (Theory Guide §4.18.3 + §4.11.9 lidos):
#   [Cell 2] bloco novo sobre a numérica do solver e o piso da physics loss
#   [Cell 2] análise de escala do teto de ν_t: baixo demais para 54 m
#   [Cell 2] κ das funções de parede (0,4187) separado do κ do modelo (0,41);
#            WALL_LAW_C corrigido de 0,050430 para 0,052594
#   [Cell 2/5/10] ω é prescrito na parede, k continua transportado — o
#            motivo de mascarar k mudou (Δy de célula não existe num ponto)
#   [Cell 5/10] fator (u_τ/u*)² na relação da camada log
# Feitas em 28/09/2026 (Cell 10 refeita):
#   [Cell 10] termo (2/3)ρ∇k fora (derivado de PRESSURE_IS_MODIFIED);
#             ν_t com F2, limitador de Bradshaw e teto; ∇μ_t por autodiff;
#             máscara de parede por ponto e por WALL_RESIDUAL_MODE;
#             máscaras de production limiter, limitador e teto no retorno
print(f"\n=== PENDÊNCIAS ({len(_PENDENCIAS)}) — células ainda não alinhadas "
      f"com esta ===")
for _c, _t in _PENDENCIAS:
    print(f"  [{_c}] {_t}")



# ============================================================
# CELL 3 — DISCOVER AND PARSE SNAPSHOT FILENAMES
# ============================================================
# Estrutura idêntica à versão laminar: parse_timestamp + discover_snapshots,
# mesmas assinaturas. As mudanças são todas consequência do dado novo:
#
#   1. Há um arquivo A MAIS na pasta — static_mesh.csv, com a distância à
#      parede. Ele NÃO é um snapshot. O regex antigo o descartaria por
#      acidente (não tem dígito no nome), mas acidente não é critério:
#      agora é excluído por nome e localizado explicitamente.
#
#      [18/10/2026] No dado novo o static_mesh.csv ganhou uma coluna,
#      cell-volume, ANTES de cell-wall-distance:
#          cellnumber, x-coordinate, y-coordinate, z-coordinate,
#          cell-volume, cell-wall-distance
#      Isso NÃO quebra nada, e a razão vale registrar: load_static_wall_
#      distance (Cell 5) e validate_wall_distance (Cell 7) leem
#      df["cell-wall-distance"] por NOME, nunca pela 5ª posição. Cabeçalho
#      conferido no arquivo em si, não no log do Fluent.
#
#   2. Os CSV são SEPARADOS POR VÍRGULA, não por espaço. O log do job 5518
#      mostra o prompt do Fluent sendo respondido com "yes":
#        "Delimiter/Comma?, if comma not chosen, space is assumed [no] yes"
#      A Cell 5 antiga lê com sep=r'\s+' e quebraria aqui. Em vez de
#      trocar a constante no escuro, o delimitador é DETECTADO do arquivo
#      e publicado em CSV_DELIMITER para a Cell 5 usar.
#
#   3. Existe a OPÇÃO de descartar o transiente inicial por tempo físico,
#      mas ela vem DESLIGADA — a versão laminar usa os 1000 arquivos e a
#      comparação entre as duas versões só é limpa se a janela for a mesma.
#      Ver a nota longa em DISCARD_TRANSIENT.
#
#   4. O que foi descoberto em disco é conferido contra as constantes da
#      Cell 2 (N_STEPS_CFD, DT_CFD, T_END_CFD). Se a pasta não for a do
#      job 6125, o log diz na hora — em vez de a inconsistência aparecer
#      como um treino ruim três horas depois.
#
# ==============================================================================
# [18/10/2026] RODADA NOVA DO CFD — TETO DE ν_t EM 1e7 (job 6125)
#
# O CFD foi refeito com UMA mudança no solver: turb-viscosity/max-limit de
# 1e5 para 1e7. Medido no dado novo: 0% das células no teto (também
# ponderado por volume), μ_t/μ máximo 5,12e5, ν_t até 7,47 m²/s. Ou seja: o
# teto antigo CORTAVA o campo físico, e o novo não corta mais nada.
#
# Quatro coisas mudam nesta célula — caminho do dado, identidade da rodada,
# warm start e destino da saída — e duas merecem justificativa.
#
# A SAÍDA PASSA A TER CAMINHO ABSOLUTO E A NASCER AQUI. Até a sst2 os
# checkpoints e o log eram definidos na Cell 15 com caminhos relativos
# ("./pinn_<tag>_checkpoints"), ou seja, caíam onde o sbatch tivesse sido
# chamado, e seis arquivos de diagnóstico das Cells 5b, 8 e 9c caíam soltos
# no mesmo lugar. Agora tudo o que o script de treino escreve vai para
# TRAIN_DIR/saida/, em três subpastas. Elas são definidas AQUI, e não na
# Cell 15, porque as Cells 5b, 8 e 9c rodam antes dela e já precisam do
# destino — e porque uma pasta sem permissão de escrita tem que derrubar o
# job em 2 segundos, não depois de 30 minutos de leitura de CSV.
#
# O WARM START VOLTA PARA None, e não é opcional. Os pesos do checkpoint
# 2200 foram treinados contra outro dicionário: a Cell 4 remede LNK/LNW no
# dado novo (AUTO_NORM_STATS = True) e a Cell 2 leva NU_T_MAX de 1,4607
# para 146,07 m²/s. Carregar aqueles pesos aqui seria lê-los com uma
# convenção diferente da que os escreveu — a saída 'kn' deixaria de
# significar o mesmo k. Cold start, treino inteiro de novo.
#
# E a comparação sst2 × sst3 é ESTATÍSTICA, nunca instante a instante: as
# duas rodadas de CFD são idênticas até t ≈ 40 s (até ali o teto ainda não
# agia) e divergem depois, fora de fase, com envoltória parecida.
#
# ==============================================================================
# [19/10/2026] RODADA "siren1" — MESMO DADO DA sst3, CAMADAS SIREN
#
# Esta rodada existe para medir UMA variável: a arquitetura (tanh -> SIREN,
# Cell 9). Dado, normalização, física, pesos e LR são os da sst3, e por
# isso aqui só mudam a identidade da rodada e a pasta de saída. CFD_DIR
# continua apontando para o job 6125, de propósito.
#
# TRAIN_DIR PRÓPRIO, e não a pasta da sst3, porque as duas rodam ao mesmo
# tempo e diagnosticos/ grava com nomes SEM a tag (CSVs da 5b, PNGs das
# Cells 8 e 9c). Na mesma pasta, a siren1 sobrescreveria os diagnósticos da
# sst3 em silêncio, no meio do treino dela.
#
# COLD START, de novo obrigatório, mas agora por outro motivo: os pesos da
# sst3 são de uma rede com outras camadas. Não é questão de convenção de
# normalização — eles simplesmente não têm onde entrar.
#
# A COMPARAÇÃO sst3 × siren1 PODE ser instante a instante: o CFD é o mesmo
# (job 6125), a normalização medida pela Cell 4 é a mesma e o split da Cell 6
# é o mesmo. Comparar na mesma época, com média móvel de 9 épocas (o dente
# de serra do TRAIN_SHUFFLE_MODE = "by_snapshot" aparece nas duas).
#
# ==============================================================================
# [29/10/2026] RODADA "ema1" — A siren1 COM UM MODELO EMA AO LADO
#
# Esta rodada mede UMA variável: a média móvel exponencial dos pesos
# (pedido do orientador). Uma segunda rede, model_ema, nasce como cópia
# exata da rede de treino (Cell 9) e, depois de cada passo do Adam, vai
# para β·θ_ema + (1 − β)·θ (Cell 14). A validação, o best e a inferência
# usam o EMA (Cells 15 e 16). Arquitetura, ω0, dado, normalização, física,
# pesos, LR e lote são os da siren1 — aqui só mudam, de novo, a identidade
# da rodada e a pasta de saída. CFD_DIR continua no job 6125.
#
# TRAIN_DIR PRÓPRIO (10-29_ema/) pelo mesmo motivo da siren1: a sst3 e a
# siren1 ainda podem estar rodando, e diagnosticos/ grava com nomes sem
# tag. Na pasta de uma delas, esta rodada sobrescreveria os diagnósticos
# dela no meio do treino.
#
# RUN_TAG NOVO, e as DUAS redes com sufixo próprio logo depois da tag:
#   rede de treino   pinn_ema1_siren_000100.weights.h5  -> tag "ema1_siren"
#   EMA              pinn_ema1_ema_000100.weights.h5    -> tag "ema1_ema"
# (idem _best e o final, Cells 15 e 16). Sufixo nas duas, e não só no EMA,
# para nenhum arquivo da rodada ter nome "sem rede": as duas têm formas
# idênticas, então carregar uma no lugar da outra passa em silêncio, e o
# nome do arquivo é a única coisa que as distingue. A inferência lê cada
# sufixo como um perfil próprio. NÃO pôr o sufixo depois de ".weights"
# (quebra a regex da época).
#
# COLD START, e aqui por um terceiro motivo: partir dos pesos da siren1
# misturaria duas variáveis (EMA e ponto de partida).
#
# A REDE DE TREINO NÃO É UMA RÉPLICA DA siren1, mesmo com a mesma SEED. Os
# inicializadores do Keras 3 sem semente própria (o RandomUniform da
# SirenLayer e o glorot_uniform da raw_fields) sorteiam a semente com o
# módulo `random` do Python, e a Cell 1 fixa só o NumPy e o TF. Medido em
# 29/10/2026 (TF 2.21, Keras 3): duas execuções com SEED = 42 dão pesos
# iniciais diferentes. A siren1 partiu de uma inicialização que não dá para
# recriar, e esta rodada parte de outra. Então:
#   - ema1 × siren1 (rede de treino × rede de treino) compara TENDÊNCIA,
#     com média móvel de 9 épocas, e não época a época. Uma rede de treino
#     muito pior ou muito melhor que a siren1 ainda é sinal de que algo
#     além do EMA mudou; uma diferença pequena pode ser só a inicialização;
#   - EMA × rede de treino, DENTRO desta rodada, é a comparação exata: as
#     duas partem dos mesmos pesos (set_weights na Cell 9, |Δ| = 0
#     conferido) e veem os mesmos passos.
#
# ==============================================================================
# [09/11/2026] RETOMADA DA ema1 NA ÉPOCA 18.100
#
# A manutenção do cluster (sexta, 06/11) derrubou o job do full depois do
# checkpoint da época 18.100. Isto CONTINUA a rodada ema1 — mesmo RUN_TAG,
# mesmo TRAIN_DIR, mesma pasta de checkpoints, mesmo training_log_ema1.csv
# — e não é uma rodada nova. Os checkpoints seguem numerados a partir de
# 18.100 e o CSV segue no mesmo arquivo, então a inferência vê o treino
# inteiro, de 1 a 50.000.
#
# O que a retomada recupera, e o que não:
#   - REDE DE TREINO: pesos da época 18.100 (WARM_START_FROM).
#   - EMA: pesos da época 18.100 (WARM_START_FROM_EMA). NÃO é recopiado da
#     rede de treino: a média acumulada em 1,81 M de passos é o que esta
#     rodada mede, e recopiar a jogaria fora.
#   - ESTADO DO ADAM: NÃO. Não é salvo no .h5 e recomeça zerado. Com o
#     Adam zerado, o primeiro passo é o MAIOR possível (±LR em cada peso,
#     m̂/√v̂ = ±1), não o menor. Com LR = 3e-5 isso deve dar um solavanco
#     pequeno e curto na loss da rede de treino logo depois da 18.100. O
#     EMA mal sente (cada passo novo entra com peso 1 − β = 0,1%).
#   - SEQUÊNCIA DE SORTEIOS (lotes, pool de colocação, ordem dos
#     snapshots): recomeça do início. Não muda o que a rede vê em média.
#   - BEST: a Cell 15 o retoma do CSV (melhor val_aligned_ema e val_aligned
#     até a época 18.100), para os arquivos _best só serem trocados por
#     algo MELHOR que o de antes da queda.
#   - CSV: a Cell 15 corta as linhas DEPOIS da época 18.100 (as épocas que
#     rodaram entre o checkpoint e a queda, que serão refeitas), guardando
#     o original numa cópia de segurança.
#
# A conferência abaixo exige que os dois arquivos sejam desta rodada, um
# de cada rede (_siren e _ema) e da época WARM_START_EPOCH. As duas redes
# têm as mesmas formas: o nome é a única coisa que impede carregar uma no
# lugar da outra.
# ==============================================================================


# -----  CONFIGURAÇÕES  -----
# Raiz do dado do CFD — é ENTRADA, não saída. Os resultados DO TREINO
# (checkpoints, training_log, monitoramento, diagnósticos) vão para
# TRAIN_DIR/saida/, definido mais abaixo nesta célula.
#
# Tudo o que o treino LÊ pendura daqui, para que trocar de rodada de CFD
# seja UMA linha e não um caça-caminhos pelo arquivo.
#   <CFD_DIR>/
#   ├── csv/    t_00001..t_01000.csv + static_mesh.csv   <- o treino
#   ├── bc/     inlet_faces_init.csv                     <- verificação (c)
#   ├── deriv/  d_00001..d_01000.csv                     <- ainda NÃO usado
#   ├── wall/   <zona>_NNNNN.csv                         <- ainda NÃO usado
#   └── meta/   transcript, report_summary, run_info, .cas
# deriv/ e wall/ ficam de fora DESTE treino de propósito: são a investigação
# do ν_t (S contra strain-rate-mag, piso da continuidade, y+ direto), que é
# a próxima variável. Uma de cada vez.
CFD_DIR           = "/home/tmoraes/treino_turb/new_sim_turb/results/"
SNAPSHOT_DIR      = os.path.join(CFD_DIR, "csv")
SNAPSHOT_EXT      = ".csv"

# Faces do inlet exportadas pelo solver (3.965 faces, todas em y = 0).
# A verificação (c) da Cell 5 comparava a BC da PINN com valores de CÉLULA
# já decaídos — com ω_inlet = 7419 1/s a escala de decaimento é 0,26 mm, e
# a 0,2 m o dado já caiu 8× — e concluía "TI ≈ 1,63%, μ_t/μ ≈ 24,8",
# parecendo acusar divergência onde não havia. Agora compara com o que o
# solver IMPÔS.
#
# Cabeçalho conferido no arquivo (10 colunas, NÃO as 6 que o handoff do CFD
# descreve — mais um motivo para ler por nome):
#   cellnumber, x-coordinate, y-coordinate, z-coordinate, viscosity-ratio,
#   turb-intensity, viscosity-turb, specific-diss-rate,
#   turb-kinetic-energy, y-velocity
#
# ARMADILHA: `turb-intensity` NÃO é a TI física. O Fluent escreve
# √(2k/3)/U_ref com U_ref = 1 m/s (Reference Values), e dá 0,850 onde a TI
# é 5%. A TI física é a coluna dividida por V_INF. A BC se lê de k e ω, que
# são o valor imposto. Conferido: k = 1,083750 e ω = 7419,2117 dão
# μ_t = ρk/ω = 1,78939e-4 (arquivo: 1,7894e-4) e μ_t/μ = 10,0003
# (arquivo: 10,000) — são exatamente K_INLET e OMEGA_INLET da Cell 2.
INLET_FACES_CSV   = os.path.join(CFD_DIR, "bc", "inlet_faces_init.csv")

# ------------------------------------------------------------------------------
# IDENTIDADE DA RODADA E WARM START
# ------------------------------------------------------------------------------
# Ficam AQUI, e não na Cell 15 onde são usados, por um motivo só: o
# carregamento dos 999 snapshots leva ~30 min. Um caminho errado descoberto
# depois disso custa meia hora de fila por digitação errada. A conferência
# está logo abaixo e roda em 2 segundos.
#
# Use CAMINHO ABSOLUTO. O SLURM não faz cd antes do python, então "./" é o
# diretório de onde o sbatch foi chamado — que pode não ser o que você acha.
# Tag nova, e não "sst3_siren" ou parecido: o padrão pinn_<tag>_checkpoints
# é o que o inferencia_sst.py usa para escolher o perfil da rodada, e a
# SIREN vai precisar de perfil próprio lá (outro build_pinn).
# ema1: os pesos saem como pinn_ema1_siren_* (rede de treino) e
# pinn_ema1_ema_* (EMA) — ver o cabeçalho.
RUN_TAG = "ema1"                 # siren1 + modelo EMA (β = 0,999, Cell 9)

# None treina do zero. Caminho de um .h5 continua de lá.
# NÃO religar sem reler a nota do cold start no cabeçalho desta célula: na
# siren1 nenhum checkpoint anterior serve (a arquitetura é outra).
# ema1: cold start obrigatório — partir da siren1 misturaria o ponto de
# partida com o EMA. E um warm start precisaria recopiar o model_ema DEPOIS
# do load_weights (Cell 15), senão as duas redes deixam de começar iguais.
#
# [09/11/2026] RETOMADA da ema1 depois da manutenção do cluster (sexta,
# 06/11): o job do full parou depois da época 18.100. Ver o bloco RETOMADA
# no cabeçalho desta célula. São TRÊS valores, e não um:
#   WARM_START_FROM      pesos da REDE DE TREINO (pinn_ema1_siren_NNNNNN)
#   WARM_START_FROM_EMA  pesos do EMA (pinn_ema1_ema_NNNNNN), da MESMA época.
#                        Com ele, a Cell 15 carrega o EMA deste arquivo em
#                        vez de recopiá-lo da rede de treino — é o que faz
#                        isto ser uma RETOMADA e não um warm start novo.
#                        None = warm start antigo (EMA recopiado).
#   WARM_START_EPOCH     a época desses arquivos. Numera o log e os
#                        checkpoints, e a Cell 15 usa para: cortar do CSV as
#                        épocas depois dela, achar o best já registrado e
#                        rodar só as épocas que faltam até 50.000.
# Caminhos ABSOLUTOS: o TRAIN_DIR só é definido mais abaixo.
WARM_START_FROM     = ("/home/tmoraes/treino_turb/10-29_ema/saida/"
                       "pinn_ema1_checkpoints/pinn_ema1_siren_018100.weights.h5")
WARM_START_FROM_EMA = ("/home/tmoraes/treino_turb/10-29_ema/saida/"
                       "pinn_ema1_checkpoints/pinn_ema1_ema_018100.weights.h5")
WARM_START_EPOCH    = 18100

if WARM_START_FROM:
    for _rotulo, _p in (("WARM_START_FROM", WARM_START_FROM),
                        ("WARM_START_FROM_EMA", WARM_START_FROM_EMA)):
        if _p and not os.path.exists(_p):
            raise FileNotFoundError(
                f"{_rotulo} não existe:\n    {_p}\n"
                f"Use caminho ABSOLUTO, ou ponha None para treinar do zero.")
    print(f"[warm start] arquivo encontrado: {WARM_START_FROM}")
    if WARM_START_FROM_EMA:
        # As duas redes têm as MESMAS formas: trocar um arquivo pelo outro
        # carregaria sem erro e o "EMA" passaria a ser a rede de treino.
        # O nome é a única proteção, e a época dos dois tem que ser a
        # mesma — senão o EMA retomado seria de outro ponto do treino.
        _re_ck = re.compile(r"pinn_(.+)_(siren|ema)_(\d+)\.weights\.h5$")
        _m_s = _re_ck.search(os.path.basename(WARM_START_FROM))
        _m_e = _re_ck.search(os.path.basename(WARM_START_FROM_EMA))
        _erros = []
        if not _m_s or _m_s.group(2) != "siren":
            _erros.append("WARM_START_FROM tem que ser pinn_<tag>_siren_NNNNNN"
                          ".weights.h5 (rede de treino)")
        if not _m_e or _m_e.group(2) != "ema":
            _erros.append("WARM_START_FROM_EMA tem que ser pinn_<tag>_ema_"
                          "NNNNNN.weights.h5 (EMA)")
        if _m_s and _m_e:
            if _m_s.group(1) != RUN_TAG or _m_e.group(1) != RUN_TAG:
                _erros.append(f"os arquivos são da rodada "
                              f"{_m_s.group(1)!r}/{_m_e.group(1)!r}, não de "
                              f"{RUN_TAG!r}: uma retomada continua a MESMA "
                              f"rodada")
            _ep_s, _ep_e = int(_m_s.group(3)), int(_m_e.group(3))
            if not (_ep_s == _ep_e == int(WARM_START_EPOCH)):
                _erros.append(f"épocas diferentes: rede de treino {_ep_s}, "
                              f"EMA {_ep_e}, WARM_START_EPOCH "
                              f"{WARM_START_EPOCH}")
        if _erros:
            raise ValueError("retomada inconsistente:\n  - "
                             + "\n  - ".join(_erros))
        print(f"[warm start] RETOMADA da rodada {RUN_TAG!r} na época "
              f"{WARM_START_EPOCH}")
        print(f"[warm start] EMA: {WARM_START_FROM_EMA}")
    print(f"[warm start] numerando as épocas a partir de {WARM_START_EPOCH}")
else:
    WARM_START_FROM_EMA = None
    WARM_START_EPOCH = 0
    print(f"[warm start] desligado — treino do zero")

# ------------------------------------------------------------------------------
# DESTINO DA SAÍDA DO TREINO
# ------------------------------------------------------------------------------
#   <TRAIN_DIR>/
#   ├── logs/                       .out e .err do SLURM — só isso. Quem
#   │                               escreve ali é o SLURM, não este script.
#   ├── saida/                      tudo o que ESTE script escreve
#   │   ├── pinn_<tag>_checkpoints/ pesos, metadata, histórico, métricas de
#   │   │                           validação (Cells 15 e 16)
#   │   ├── historico/              training_log e monitoramento: crescem
#   │   │                           durante o treino, uma linha por época
#   │   └── diagnosticos/           CSVs da 5b e PNGs das Cells 8 e 9c:
#   │                               escritos uma vez, antes do treino, e
#   │                               sobrescritos a cada rodada
#   └── (pastas dos códigos auxiliares — inferência etc. — FORA de saida/)
#
# A separação entre saida/ e as pastas auxiliares é de propósito: saida/
# contém só o que o treino produziu, e dá para apagar ou arquivar a rodada
# inteira sem tocar no que a inferência gerou a partir dela.
#
# O nome da pasta de checkpoints segue o padrão pinn_<tag>_checkpoints
# porque é assim que o inferencia_sst.py a reconhece. Não renomear.
#
# CAMINHO ABSOLUTO, pelo mesmo motivo de CFD_DIR: o slurm faz
# `cd $SLURM_SUBMIT_DIR`, então um caminho relativo depende de onde o
# sbatch foi chamado — foi assim que os arquivos da sst2 caíram soltos.
#
# siren1: pasta própria, NÃO a 10-18_simples/ da sst3 — ver o cabeçalho
# desta célula (diagnosticos/ sem tag seria sobrescrito).
# ema1: pasta própria também, NÃO a 10-19_siren/ — mesmo motivo.
TRAIN_DIR      = "/home/tmoraes/treino_turb/10-29_ema/"
SAIDA_DIR      = os.path.join(TRAIN_DIR, "saida")
HISTORICO_DIR  = os.path.join(SAIDA_DIR, "historico")
DIAG_DIR       = os.path.join(SAIDA_DIR, "diagnosticos")
CHECKPOINT_DIR = os.path.join(SAIDA_DIR, f"pinn_{RUN_TAG}_checkpoints")

# TRAIN_DIR tem que existir: é a pasta da rodada, criada à mão (é onde
# moram o .slurm e logs/). As subpastas de saida/ o script cria. Criar
# TRAIN_DIR sozinho esconderia um erro de digitação — o job "funcionaria"
# e escreveria numa pasta que ninguém vai procurar.
if not os.path.isdir(TRAIN_DIR):
    raise NotADirectoryError(
        f"TRAIN_DIR não existe:\n    {TRAIN_DIR}\n"
        f"É a pasta da rodada; tem que ser criada antes do sbatch.")
for _d in (SAIDA_DIR, HISTORICO_DIR, DIAG_DIR, CHECKPOINT_DIR):
    os.makedirs(_d, exist_ok=True)
    # makedirs com exist_ok não reclama de pasta existente SEM permissão
    # de escrita — e aí o primeiro save_weights falharia horas depois.
    if not os.access(_d, os.W_OK):
        raise PermissionError(f"sem permissão de escrita em:\n    {_d}")

print(f"[saída] TRAIN_DIR      = {TRAIN_DIR}")
print(f"[saída] historico/     = {HISTORICO_DIR}")
print(f"[saída] diagnosticos/  = {DIAG_DIR}")
print(f"[saída] checkpoints    = {CHECKPOINT_DIR}")
if os.path.abspath(os.getcwd()) != os.path.abspath(TRAIN_DIR):
    # Não é erro: com caminhos absolutos, de onde o sbatch é chamado não
    # importa mais. Só fica registrado, porque o .out do SLURM continua
    # indo para onde o #SBATCH --output mandar.
    print(f"[saída] (cwd = {os.getcwd()} — diferente de TRAIN_DIR, sem efeito "
          f"nos arquivos deste script)")

# Se a pasta de checkpoints já tiver pesos, isto é uma segunda rodada com o
# mesmo RUN_TAG: os checkpoints novos sobrescrevem os antigos de mesmo nome
# e o CSV recomeça (Cell 15). Avisa, não aborta — reexecutar depois de um
# job que caiu no começo é o caso comum.
_ckpt_old = glob.glob(os.path.join(CHECKPOINT_DIR, "*.weights.h5"))
if _ckpt_old and WARM_START_FROM_EMA:
    # retomada (09/11/2026): é esperado a pasta ter os checkpoints de antes
    print(f"[saída] retomada: {CHECKPOINT_DIR} já tem {len(_ckpt_old)} "
          f"arquivo(s) .weights.h5. Os novos são das épocas depois de "
          f"{WARM_START_EPOCH} e entram ao lado deles; os _best só são "
          f"trocados por um melhor (Cell 15).")
elif _ckpt_old:
    print(f"[saída] >>> {CHECKPOINT_DIR} já tem {len(_ckpt_old)} arquivo(s) "
          f".weights.h5.")
    print(f"        Esta rodada vai sobrescrever os de mesmo nome. Se for uma")
    print(f"        rodada NOVA, troque RUN_TAG; se for retomada, use WARM_START_FROM.")

# --- Conferência dos caminhos: 2 segundos agora, 30 minutos depois ---
if not os.path.isdir(SNAPSHOT_DIR):
    raise NotADirectoryError(
        f"SNAPSHOT_DIR não é uma pasta:\n    {SNAPSHOT_DIR}\n"
        f"CFD_DIR = {CFD_DIR}")
print(f"[dado] CFD_DIR       = {CFD_DIR}")
print(f"[dado] SNAPSHOT_DIR  = {SNAPSHOT_DIR}")

# O inlet NÃO aborta: sem ele a verificação (c) cai no modo antigo e o
# treino segue igual. É verificação, não insumo.
HAS_INLET_FACES = os.path.isfile(INLET_FACES_CSV)
if HAS_INLET_FACES:
    print(f"[inlet faces] {INLET_FACES_CSV}")
else:
    print(f"[inlet faces] NÃO encontrado:\n              {INLET_FACES_CSV}")
    print(f"              a verificação (c) da Cell 5 roda no modo antigo.")

DT_PHYSICAL       = DT_CFD       # 0.1 s — vem da Cell 2, não redigitado
TIMESTEP_IS_INDEX = True

# Arquivos na pasta que NÃO são snapshots
STATIC_MESH_BASENAME  = STATIC_MESH_CSV          # "static_mesh.csv"
NON_SNAPSHOT_BASENAMES = {STATIC_MESH_BASENAME}

# --- Descarte do transiente de inicialização — DESLIGADO POR PADRÃO ---
#
# A versão laminar usa TODA a janela temporal (sem descartar transiente), e
# isso está provado pelas próprias constantes da Cell 4 dela:
#     T_MEAN = 50.05                 T_DP = 28.867499025720953
# Para t = k·0,1 com k = 1..1000:  média = 50,0500000000
#                                  desvio = 28,867499025721
# Bate nas 12 casas. Com 999 arquivos (pulando o primeiro) daria
# 50,10 e 28,8386 — não é o que está escrito lá.
#
# (Sobre o SNAPSHOT_START = 1 do código laminar: a DECISÃO estava certa —
#  o primeiro arquivo é a inicialização. O que era acidental é que ele
#  carregava 999 arquivos e normalizava com a estatística dos 1000, então
#  T_MEAN/T_DP descreviam uma janela que não era a lida. Deslocava t_norm
#  em +0,0017σ, irrelevante, mas errado. Aqui a Cell 4 recalcula T_MEAN e
#  T_DP sobre a janela DE FATO carregada, então o mesmo descarte acontece
#  sem a inconsistência.)
#
# O ARGUMENTO A FAVOR DE DESCARTAR continua de pé e está no handoff: o
# campo inicial vem de hyb-initialization, que não é física — é um chute do
# solver. Os primeiros passos ensinam à rede o relaxamento numérico do
# Fluent. Uma travessia aqui é (Y_MAX-Y_MIN)/V_INF = 41,2 s = 411 passos.
#
# O ARGUMENTO CONTRA, que venceu: esta etapa existe para isolar o efeito de
# ACRESCENTAR TURBULÊNCIA. Mudar a janela temporal ao mesmo tempo mistura
# duas variáveis — se o resultado piorar, não dá para saber se foi o SST ou
# o dataset menor. Descartar 41% do dado é uma decisão separada, que merece
# ser testada sozinha depois, com o resto já estável.
#
# Para ligar: DISCARD_TRANSIENT = True. Aí T_MEAN e T_DP da Cell 4 TÊM que
# ser recalculados sobre a janela nova — a Cell 3 imprime os valores certos.
DISCARD_TRANSIENT     = False
N_FLOWTHROUGH_DISCARD = 1.0                      # quantas travessias jogar fora
T_DISCARD             = N_FLOWTHROUGH_DISCARD * T_FLOWTHROUGH   # [s]

# Stride temporal: pega 1 arquivo a cada SNAPSHOT_STRIDE (= 1 -> lê todos)
SNAPSHOT_STRIDE   = 1

# PULA O PRIMEIRO ARQUIVO. Não é detalhe de contagem: o t_00001 sai logo
# depois da hyb-initialization e os campos dele são muito diferentes dos
# demais — é o chute inicial do solver, não escoamento. A PINN laminar já
# fazia isso, e pelo mesmo motivo.
#
# Diferente do DISCARD_TRANSIENT abaixo, que joga fora uma travessia
# inteira (41,2 s, 41% do dado) por causa do relaxamento numérico. Aqui é
# UM arquivo, e o que se descarta não é transiente físico — é o campo que
# ainda não é solução de nada.
SNAPSHOT_START    = 1
SNAPSHOT_MAX      = None   # int para limitar total; None = sem limite

TEST_MODE   = False
TEST_STRIDE = 25

# Pre-compiled pattern: o último número não-sinalizado no basename
_TIME_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)(?=\D*$)")


def parse_timestamp(filename: str,
                    dt_physical: float = DT_PHYSICAL,
                    timestep_is_index: bool = TIMESTEP_IS_INDEX):

    # Extrai o tempo físico [s] do nome do arquivo para futura ordenação temporal

    stem = os.path.splitext(os.path.basename(filename))[0]
    matches = _TIME_TOKEN_RE.findall(stem)
    if not matches:
        return None, None
    raw = float(matches[-1])
    t_phys = raw * dt_physical if timestep_is_index else raw

    #T_phys: tempo físico
    #T_raw : número timestep
    return float(t_phys), raw


def detect_delimiter(path, n_probe=2048):
    """
    Descobre o separador lendo a primeira linha do arquivo.

    O Fluent pergunta "Delimiter/Comma?" no export; no job 5518 a resposta
    foi 'yes' (vírgula), mas rodadas antigas responderam 'no' (espaço).
    Detectar é mais barato que manter uma constante que pode mentir.

    Retorna algo que o pandas.read_csv aceita como `sep`.
    """
    with open(path, "r") as fh:
        head = fh.read(n_probe).splitlines()
    if not head:
        raise RuntimeError(f"Arquivo vazio: {path}")
    first = head[0]
    n_comma = first.count(",")
    n_space = len(re.findall(r"\s+", first.strip()))
    if n_comma >= 1 and n_comma >= n_space:
        return ",", n_comma + 1
    return r"\s+", n_space + 1


def verify_csv_header(path, sep, expected_cols=CSV_COLUMNS_CFD, verbose=True):
    """
    Confere o cabeçalho contra CSV_COLUMNS_CFD (Cell 2).

    Não falha se a ORDEM diferir — a Cell 5 casa por nome. Falha (avisa) se
    faltar coluna, que é o erro que importa: um campo ausente só apareceria
    muito depois, como KeyError no meio da leitura dos 1000 arquivos.
    """
    with open(path, "r") as fh:
        raw_header = fh.readline()
    seps = "," if sep == "," else None
    cols = [c.strip().strip('"\'﻿').lower()
            for c in (raw_header.split(seps) if seps else raw_header.split())]
    expected = [c.lower() for c in expected_cols]

    missing = [c for c in expected if c not in cols]
    extra   = [c for c in cols if c not in expected]
    same_order = cols == expected

    if verbose:
        print(f"  [header] {len(cols)} colunas: {cols}")
        if missing:
            print(f"  [header] >>> FALTANDO: {missing}")
        if extra:
            print(f"  [header] extras (ok, serão ignoradas): {extra}")
        if not missing and not same_order:
            print(f"  [header] ordem difere de CSV_COLUMNS_CFD "
                  f"(esperado — o Fluent exporta invertido). Casamento é por nome.")
        if not missing and same_order:
            print(f"  [header] ordem idêntica a CSV_COLUMNS_CFD.")
    return cols, missing


def discover_snapshots(snapshot_dir: str,
                       ext: str = SNAPSHOT_EXT,
                       dt_physical: float = DT_PHYSICAL,
                       timestep_is_index: bool = TIMESTEP_IS_INDEX,
                       stride: int = 1,
                       start_idx: int = 0,
                       max_snapshots: int = None,
                       t_discard: float = 0.0,
                       exclude_basenames: set = NON_SNAPSHOT_BASENAMES,
                       verbose: bool = True):

    # Lê o diretório e retorna uma lista do tempo físico e caminho em ordem cronológica

    if not os.path.isdir(snapshot_dir):
        raise NotADirectoryError(f"Snapshot directory not found: {snapshot_dir}")

    pattern = os.path.join(snapshot_dir, f"*{ext}")
    candidates = sorted(glob.glob(pattern))
    if verbose:
        print(f"Scanning '{snapshot_dir}' for *{ext}: "
              f"{len(candidates)} candidate(s)")

    # Separar timesteps e filtrar
    parsed = []
    skipped = []
    excluded = []
    for path in candidates:
        base = os.path.basename(path)
        # Exclusão EXPLÍCITA (static_mesh.csv e afins) — antes do regex
        if base in exclude_basenames:
            excluded.append(base)
            continue
        t_phys, raw = parse_timestamp(path, dt_physical, timestep_is_index)
        if t_phys is None:
            skipped.append(base)
            continue
        parsed.append((t_phys, path))

    # Ordenação cronológica
    parsed.sort(key=lambda s: s[0])
    n_total = len(parsed)

    # Descarte do transiente de inicialização — POR TEMPO FÍSICO
    n_before_discard = n_total
    if t_discard > 0.0:
        parsed = [(t, p) for (t, p) in parsed if t > t_discard]
    n_discarded = n_before_discard - len(parsed)

    # Aplicando intervalo (se houver)
    selected = parsed[start_idx::stride]
    if max_snapshots is not None:
        selected = selected[:max_snapshots]

    if verbose:
        if excluded:
            print(f"  [excl] {len(excluded)} arquivo(s) não-snapshot: {excluded}")
        if skipped:
            print(f"  [skip] {len(skipped)} unparseable file(s), "
                  f"e.g. {skipped[:3]}")
        if n_discarded:
            print(f"  [transiente] descartados {n_discarded} snaps com "
                  f"t <= {t_discard:.2f}s "
                  f"({N_FLOWTHROUGH_DISCARD:.1f} travessia(s))")
        print(f"  [filter] {n_total} parseable → {len(parsed)} após descarte → "
              f"{len(selected)} after stride={stride}, start={start_idx}, "
              f"max={max_snapshots}")

    return selected


# ----- localizar a malha estática (distância à parede) -----
# Exportada DENTRO da sessão do solve, logo alinhada célula a célula com os
# t_*.csv. É a referência para validar a fórmula analítica de d_wall.
static_mesh_path = os.path.join(SNAPSHOT_DIR, STATIC_MESH_BASENAME)
HAS_STATIC_MESH  = os.path.isfile(static_mesh_path)

if HAS_STATIC_MESH:
    print(f"[static mesh] encontrado: {static_mesh_path}")
else:
    static_mesh_path = None
    print(f"[static mesh] NÃO encontrado em '{SNAPSHOT_DIR}'.")
    print(f"              d_wall terá que vir da fórmula analítica sem conferência")
    print(f"              cruzada. WALL_DISTANCE_SOURCE='csv' fica indisponível.")
    if WALL_DISTANCE_SOURCE == "csv":
        raise RuntimeError(
            "WALL_DISTANCE_SOURCE='csv' mas static_mesh.csv não está na pasta."
        )


# ----- leitura arquivos na pasta -----
snaps = discover_snapshots(
    SNAPSHOT_DIR,
    stride=SNAPSHOT_STRIDE,
    start_idx=SNAPSHOT_START,
    max_snapshots=SNAPSHOT_MAX,
    t_discard=(T_DISCARD if DISCARD_TRANSIENT else 0.0),
)
if TEST_MODE:
    snaps = snaps[::TEST_STRIDE]
    print(f"[TEST_MODE] Subamostrando: {len(snaps)} snapshots "
        f"(stride={TEST_STRIDE}, dt_efetivo={DT_PHYSICAL*TEST_STRIDE:.2f}s)")

if len(snaps) == 0:
    raise RuntimeError(
        f"No usable '*{SNAPSHOT_EXT}' files in '{SNAPSHOT_DIR}' "
        f"(stride={SNAPSHOT_STRIDE}, start={SNAPSHOT_START}, "
        f"t_discard={T_DISCARD if DISCARD_TRANSIENT else 0.0:.2f}s)."
    )

times_all = np.array([t for t, _ in snaps], dtype=np.float32)

print(f"\nFinal snapshot selection: {len(snaps)} frames")
print(f"  Time range:    {times_all[0]:.4f} s → {times_all[-1]:.4f} s")

if len(times_all) > 1:
    dts = np.diff(times_all)

    print(f"  dt (between selected snaps): "
          f"mean={dts.mean():.4f}s  min={dts.min():.4f}s  max={dts.max():.4f}s")

    expected_dt = DT_PHYSICAL * SNAPSHOT_STRIDE

    print(f"  Expected dt (DT_PHYSICAL * stride): {expected_dt:.4f}s")

    # Buraco na série temporal = arquivo faltando. Numa PINN transiente isso
    # não é cosmético: u_t é imposto na loss e uma lacuna vira um salto.
    n_gaps = int((dts > 1.5 * expected_dt).sum())
    if n_gaps:
        idx_gap = np.where(dts > 1.5 * expected_dt)[0][:5]
        print(f"  >>> {n_gaps} LACUNA(S) na série temporal, ex. em t = "
              f"{[f'{times_all[i]:.2f}' for i in idx_gap]} s")
        print(f"      Arquivos faltando na pasta. Conferir antes de treinar.")

print(f"  First 3 selected files: {[os.path.basename(p) for _, p in snaps[:3]]}")
print(f"  Last  3 selected files: {[os.path.basename(p) for _, p in snaps[-3:]]}")


# ----- delimitador e cabeçalho (usados pela Cell 5) -----
print(f"\n--------- Formato dos CSV ---------")
_probe_path = snaps[0][1]
CSV_DELIMITER, _n_cols_probe = detect_delimiter(_probe_path)
print(f"  probe: {os.path.basename(_probe_path)}")
print(f"  delimitador detectado: "
      f"{'VÍRGULA' if CSV_DELIMITER == ',' else 'ESPAÇO'}  (sep={CSV_DELIMITER!r})")
# (A Cell 5 lê com sep=CSV_DELIMITER. O aviso que estava aqui — "a Cell 5
#  da versão laminar usa sep fixo, tem que passar a usar CSV_DELIMITER" —
#  era de antes dessa troca e ficou velho; saiu em 18/10/2026.)
print(f"  usado por load_snapshot (Cell 5) e pela leitura das faces do inlet")
_cols_found, _cols_missing = verify_csv_header(_probe_path, CSV_DELIMITER)
if _cols_missing:
    raise RuntimeError(
        f"CSV sem as colunas {_cols_missing}. A pasta não é a do job {CFD_JOB_ID}, "
        f"ou o journal de export mudou."
    )


# ----- conferência contra as constantes da Cell 2 -----
print(f"\n--------- Conferência disco × Cell 2 ---------")
_n_disk_total = len(glob.glob(os.path.join(SNAPSHOT_DIR, f"*{SNAPSHOT_EXT}")))
_n_snaps_disk = _n_disk_total - (1 if HAS_STATIC_MESH else 0)

def _chk(label, got, exp, tol=0):
    ok = abs(got - exp) <= tol
    print(f"  {'OK ' if ok else '>>>'} {label:<28} disco={got:<12} Cell 2={exp}")
    return ok

_ok = True
_ok &= _chk("snapshots na pasta", _n_snaps_disk, N_STEPS_CFD)
_ok &= _chk("dt entre snaps [s]", float(f"{dts.mean():.4f}") if len(times_all) > 1 else DT_CFD,
            DT_CFD * SNAPSHOT_STRIDE, tol=1e-6)
_ok &= _chk("t final [s]", round(float(times_all[-1]), 2), round(T_END_CFD, 2), tol=0.11)

if not _ok:
    print(f"  >>> Divergência. Ou SNAPSHOT_DIR aponta para outra rodada, ou as")
    print(f"      constantes da Cell 2 descrevem outro job. Resolver ANTES da Cell 4:")
    print(f"      T_MEAN e T_DP dependem dessa janela temporal.")

# ----- janela temporal e as constantes T_MEAN / T_DP da Cell 4 -----
# Estas duas constantes descrevem a JANELA DE FATO CARREGADA. Se a janela
# mudar (descarte de transiente, stride, TEST_MODE) e elas não mudarem
# junto, o tempo normalizado deixa de ter média 0 e desvio 1 — e a rede
# recebe t fora da faixa em que as outras entradas foram calibradas.
# Por isso o valor é calculado aqui e conferido, não copiado no escuro.
# Calculado em float64 a partir da lista original. NÃO usar times_all, que é
# float32: sobre 1000 amostras o erro de arredondamento chega a ~1e-6 s e
# faria a comparação abaixo acusar diferença onde não existe nenhuma.
_times64        = np.array([t for t, _ in snaps], dtype=np.float64)
T_MEAN_EXPECTED = float(_times64.mean())
T_DP_EXPECTED   = float(_times64.std())

# Valores herdados da versão laminar (Cell 4), para comparação
T_MEAN_LAMINAR = 50.05
T_DP_LAMINAR   = 28.867499025720953

# Tolerância: 1e-4 s é 1/1000 do passo de tempo — abaixo disso a diferença
# não tem significado físico nenhum.
T_CONST_TOL = 1.0e-4

print(f"\n--------- Janela de treino ---------")
if DISCARD_TRANSIENT:
    print(f"  descartado : t ∈ (0, {T_DISCARD:.1f}] s  "
          f"= {int(T_DISCARD/DT_CFD)} passos  ({100*T_DISCARD/T_END_CFD:.0f}% do dado)")
else:
    print(f"  descartado : transiente NÃO — DISCARD_TRANSIENT = False")
    print(f"               (mesma janela da versão laminar, de propósito:")
    print(f"                a variável em teste é a turbulência, não o dataset)")
if SNAPSHOT_START > 0:
    print(f"  pulados    : os {SNAPSHOT_START} primeiro(s) arquivo(s) "
          f"(SNAPSHOT_START={SNAPSHOT_START})")
    print(f"               o t_00001 é a hyb-initialization, não escoamento")
print(f"  usado      : t ∈ [{float(times_all[0]):.1f}, {float(times_all[-1]):.1f}] s  "
      f"= {len(snaps)} snapshots")
print(f"  travessias : {(float(times_all[-1]) - float(times_all[0]))/T_FLOWTHROUGH:.1f}")

print(f"\n  Constantes temporais para a Cell 4 (calculadas sobre esta janela):")
print(f"    T_MEAN = {T_MEAN_EXPECTED:.10f}")
print(f"    T_DP   = {T_DP_EXPECTED:.12f}")

_dm = abs(T_MEAN_EXPECTED - T_MEAN_LAMINAR)
_dd = abs(T_DP_EXPECTED   - T_DP_LAMINAR)
if _dm < T_CONST_TOL and _dd < T_CONST_TOL:
    print(f"    -> idênticas às da versão laminar. Pode manter a Cell 4 como está.")
else:
    print(f"    diferem das da versão laminar "
          f"(T_MEAN={T_MEAN_LAMINAR}, T_DP={T_DP_LAMINAR:.6f}): "
          f"Δ = {_dm:.4f} s e {_dd:.4f} s.")
    if SNAPSHOT_START > 0 and not DISCARD_TRANSIENT:
        print(f"        ERA PARA DIFERIR: com SNAPSHOT_START={SNAPSHOT_START} a")
        print(f"        janela tem {len(snaps)} arquivos, e as constantes do")
        print(f"        laminar foram calculadas sobre os {N_STEPS_CFD} — ele")
        print(f"        carregava {N_STEPS_CFD-1} e normalizava com {N_STEPS_CFD}.")
        print(f"        A Cell 4 sobrescreve com os valores acima, que descrevem")
        print(f"        a janela lida. t_norm sai com média 0; com os do laminar")
        print(f"        sairia com {(T_MEAN_EXPECTED-T_MEAN_LAMINAR)/T_DP_LAMINAR:+.4f}.")
    else:
        print(f"        A Cell 4 sobrescreve automaticamente pelos valores acima.")
        print(f"        Com os do laminar, t_norm sairia com média "
              f"{(T_MEAN_EXPECTED-T_MEAN_LAMINAR)/T_DP_LAMINAR:+.3f} em vez de 0.")


# ================================================================================
#              CELL 4 — NORMALIZATION CONSTANTS
#
#      NORMALIZAÇÃO COM BASE NO SCRIPT DE CÁLCULO DE MÉDIA E DP
#      Atingir média 0 e desvio padrão 1 para variáveis normalizadas
#      ficarem entre -1 e 1 = tanh(4) e para 99% dos pontos da
#      distribuição gaussiana serem contemplados
#
#      PARA COORDENADAS É DIFERENTE POR CONTA DA PROPORÇÂO DO CILINDRO
#      CASO FOSSE NORMALIZADO CONFORME ACIMA, VIRARIA UMA ELIPSE
#
# ================================================================================
#
# DUAS MUDANÇAS.
#
# ------------------------------------------------------------------------------
# 1. AS CONSTANTES ESPACIAIS SÃO DA MALHA, NÃO DA GEOMETRIA — CONFERIR.
#
#    X_MEAN, Y_MEAN, Z_MEAN e DP_ISO são estatísticas dos CENTROIDES DAS
#    CÉLULAS. Duas malhas da MESMA geometria dão constantes diferentes se
#    distribuírem as células de outro jeito — e o refino aqui não é uniforme:
#    BOI_C e BOI_T (ambos com z ≤ 100) concentram células embaixo e perto do
#    cilindro. Dá para ver o viés nos próprios valores herdados:
#    Z_MEAN = 66,39 num domínio [0, 190], quando uniforme daria 95.
#
#    A malha de treino do caso laminar tem contagem de células da mesma ordem
#    que esta, então a expectativa é que os valores herdados estejam PERTO.
#    Mas "da mesma ordem" não é "igual": basta a caixa de um BOI ter mudado
#    para as médias andarem.
#
#    Se estiverem errados, não há exceção nem aviso: o z-score sai com média
#    diferente de zero e a rede treina fora da faixa em que tanh é bem
#    condicionada. Aparece três horas depois como "não converge bem".
#
#    Por isso AUTO_NORM_STATS: as constantes são recalculadas lendo uma
#    amostra dos CSV, e a tabela mostra herdado × medido lado a lado. Com
#    malhas parecidas os Δ devem ser pequenos — e aí a tabela serve de
#    confirmação de que as duas malhas são de fato equivalentes, que é uma
#    informação útil por si só na hora de comparar laminar × turbulento.
#
# ------------------------------------------------------------------------------
# 2. DOIS CAMPOS NOVOS, com esquemas de normalização diferentes entre si.
#
#    O código laminar já usa um esquema HÍBRIDO, e a lógica dele vale aqui:
#      - u, w: divididos por V_REF. São centrados em zero por natureza
#              (média ~0) e trocam de sinal. Dividir preserva o zero.
#      - v, P: z-score completo. Têm offset (v ~ 17 m/s, P ~ -33 Pa),
#              então precisam subtrair a média.
#
#    Estendendo para k e ω:
#
#      - ln k  e  ln ω: Z-SCORE nos dois, IGUAL ao que já se faz com v e P.
#           A diferença é só a variável: o z-score é aplicado DEPOIS do log.
#
#           Aplicar z-score em k cru não funciona, e o motivo não é só o
#           sinal. Medido numa lognormal representativa de esteira:
#             k cru   -> 77% dos alvos negativos (irreproduzíveis: k ≥ 0)
#                        95% espremidos em [-1,1], cauda até +135
#             ln k    -> 68% em [-1,1] (valor exato de uma gaussiana)
#                        p99 = 2,3, máximo 4,7
#           Ou seja: em k cru o z-score zera a média e não faz mais nada —
#           a distribuição continua impossível para uma saída tanh. O log é
#           o que transforma a variável em algo que z-score sabe tratar.
#
#           A ordem importa: log PRIMEIRO, z-score DEPOIS.
#
#           Positividade sai de graça (k = exp(...) > 0), sem softplus e
#           sem clamp — logo sem região de gradiente morto.
#
#           E ν_t = k/ω vira ln ν_t = ln k − ln ω: uma combinação LINEAR
#           das saídas da rede, em vez de uma divisão entre dois números
#           que podem ambos tender a zero.
#
#    A distância à parede NÃO é normalizada: ela não é entrada da rede nem
#    saída, só aparece dentro de F1 (e F2, se ligado), onde entra em metros.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# [18/10/2026] RODADA sst3 — DADO NOVO (job 6125)
#
# AUTO_NORM_STATS TEM QUE FICAR True NESTA RODADA. As constantes abaixo são
# o fallback herdado do laminar, e as de turbulência (LNK_*, LNW_*) são só
# placeholders derivados de K_REF e OMEGA_REF da Cell 2. Nenhuma delas foi
# medida no dado do job 6125. É a medição que a Cell 4 faz aqui que define
# o dicionário com que a rede da sst3 é treinada — e é por isso que os
# pesos da sst2 não servem de warm start (Cell 3).
#
# FEITO (18/10/2026): o bloco de fallback abaixo agora traz os valores
# MEDIDOS no dado do job 6125, copiados da tabela que a calibração (job
# PINN 6389) imprimiu. Os do laminar ficam comentados ao lado, para
# histórico.
#
# AUTO_NORM_STATS CONTINUA True, e por um motivo novo: a amostragem é
# determinística (20 arquivos espaçados por linspace, SEED fixo), então a
# medição de cada rodada reproduz a do job 6389. Com o fallback agora
# medido, a tabela "fallback × medido" vira um TESTE: Δ ≈ 0,0% em tudo
# confirma que CFD_DIR é o mesmo dado; '<<<' em qualquer linha significa
# que o dado mudou e o dicionário da rede mudou junto — e aí os pesos da
# sst3 deixam de valer para ele. AUTO_NORM_STATS = False só para rodar sem
# reler disco, aceitando a precisão de 6 casas que o .out imprime.
#
# O que muda nesta célula é só o texto das células recortadas: com o teto
# em 1e7 a fração esperada é 0%, não os ~7,2% do job 5518, e a comparação
# agora usa VISC_LIMITED_PCT_EXPECTED da Cell 2.
# ------------------------------------------------------------------------------

# ---------- CONFIG ----------
# Recalcular as constantes lendo os CSV. Custa ~1 min e evita a falha
# silenciosa descrita acima. Pôr False só depois de colar os valores medidos
# aqui em cima, para a rodada ficar reproduzível sem reler disco.
AUTO_NORM_STATS      = True
NORM_STATS_N_FILES   = 20          # arquivos amostrados uniformemente no tempo
NORM_STATS_SUBSAMPLE = 200_000     # valores guardados p/ percentis (por campo)


#---- VALORES PARA NORMALIZAÇÃO ----#
# >>> MEDIDOS no dado do job 6125 (CFD, teto 1e7) pela calibração da sst3
# >>> (job PINN 6389, 18/10/2026), 6 casas como o .out imprime. Servem de
# >>> fallback e de termo de comparação; com AUTO_NORM_STATS=True são
# >>> sobrescritos — e a tabela abaixo confere que a medição bate com eles.
# >>> Os valores do laminar ficam comentados à direita, para histórico.
NORM_FALLBACK_SOURCE = "sst3, job 6389, dado CFD 6125"

# Coordenadas
X_MEAN = 140.003410       # laminar: 140.01174139636194
Y_MEAN = 290.804396       # laminar: 290.9150766181903
Z_MEAN = 66.353356        # laminar: 66.39128138069277

# dp_iso = mean(dp_x, dp_y, dp_z)
# mantém proporção do domínio e impede cilindro de virar elipse
DP_ISO = 101.319715       # laminar: 101.234187329139

# Tempo — janela de 999 arquivos (t_00001 pulado); a Cell 3 recalcula e a
# linha "T_MEAN = T_MEAN_EXPECTED" logo abaixo sobrescreve de qualquer jeito
T_MEAN = 50.1                       # laminar: 50.05 (1000 arquivos)
T_DP   = 28.838631497813            # laminar: 28.867499025720953

# velocidade x
# ---- valores para média 0 e desvio padrão 1 ----
# U_MEAN = 0.014754446585653998
# U_DP   = 4.133050105381722

# velocidade y
V_MEAN = 15.047482        # laminar: 14.321999262773115
V_DP   = 7.070827         # laminar: 7.399817765728233

# velocidade z
# ---- valores para média 0 e desvio padrão 1 ----
# W_MEAN = 0.251954634259236
# W_DP   = 2.33567163666255

# pressao
P_MEAN = -35.608335       # laminar: -32.78900000449526
P_DP   = 95.521821        # laminar: 98.41228792498488

# --- NOVOS: turbulência ---
# MEDIDOS (sst3, job 6389). Até 18/10/2026 eram placeholders derivados das
# escalas da Cell 2 — ln(K_REF) = 2,278 e ln(OMEGA_REF) = 3,285, desvios
# 1,5 —, e a medição mostrou o quanto estavam longe: a mediana de k no
# domínio é 0,048 m²/s², não 9,75, e a de ω é 2,0 1/s, não 26,7.

# Piso aditivo de k, para o log não explodir onde k -> 0.
#
# Não é um número numérico arbitrário: K_INLET é o menor k IMPOSTO no
# problema (a turbulência de fundo que entra pelo inlet). Qualquer coisa
# uma ou duas ordens abaixo disso é "turbulência nenhuma", e não faz
# diferença física distinguir 1e-6 de 1e-9 — mas faz muita diferença para
# o desvio padrão de ln k, que a cauda inflaria sozinha.
# ln(k + K_EPS) comprime tudo abaixo de K_EPS numa faixa estreita, de forma
# SUAVE (sem clamp, então sem gradiente morto).
K_EPS_FRAC = 0.01                    # fração de K_INLET
K_EPS      = K_EPS_FRAC * K_INLET    # [m²/s²] ≈ 0,0108

LNK_MEAN = -2.123991                 # média de ln(k + K_EPS)   (era ln K_REF = 2,278)
LNK_DP   = 2.223361                  # desvio                   (era chute de 1,5)
LNW_MEAN = 1.089561                  # média de ln(ω)           (era ln OMEGA_REF = 3,285)
LNW_DP   = 1.856329                  # desvio                   (era chute de 1,5)

# Só usado se K_NORM_MODE == "scale" (modo antigo, mantido como fallback)
K_SCALE  = K_REF


# ---- Tempo: usar a janela DE FATO carregada (vem da Cell 3) ----
# Elimina a chance de T_MEAN/T_DP descreverem uma janela diferente da que
# a Cell 5 vai ler. Com os 1000 arquivos isto reproduz 50,05 / 28,8675.
if abs(T_MEAN_EXPECTED - T_MEAN) > T_CONST_TOL or abs(T_DP_EXPECTED - T_DP) > T_CONST_TOL:
    print(f"[Cell 4] T_MEAN/T_DP herdados não descrevem a janela carregada.")
    print(f"         {T_MEAN:.6f} / {T_DP:.6f}  ->  "
          f"{T_MEAN_EXPECTED:.6f} / {T_DP_EXPECTED:.6f}")
T_MEAN = T_MEAN_EXPECTED
T_DP   = T_DP_EXPECTED


# ================================================================================
# Cálculo das constantes a partir do dado
# ================================================================================
# A Cell 3 já garantiu que o cabeçalho tem exatamente os nomes de
# CSV_COLUMNS_CFD, então aqui a leitura pode ser por nome exato — sem repetir
# o resolvedor tolerante da Cell 5.

_STATS_FIELDS = {
    "x": "x-coordinate",
    "y": "y-coordinate",
    "z": "z-coordinate",
    "u": "x-velocity",
    "v": "y-velocity",
    "w": "z-velocity",
    "P": "pressure",
    "k": "turb-kinetic-energy",
    "omega": "specific-diss-rate",
    "mut": "viscosity-turb",
}


def _read_stats_frame(path, sep=None):
    """Lê um CSV de snapshot e devolve dict de arrays float64 por nome canônico."""
    sep = sep if sep is not None else CSV_DELIMITER
    df = pd.read_csv(path, sep=sep, engine="python", skipinitialspace=True)
    df.columns = [re.sub(r'["\'﻿]', '', c).strip().lower() for c in df.columns]
    return {k: df[v].to_numpy(dtype=np.float64) for k, v in _STATS_FIELDS.items()}


def compute_normalization_stats(snaps,
                                n_files=NORM_STATS_N_FILES,
                                subsample=NORM_STATS_SUBSAMPLE,
                                seed=SEED,
                                verbose=True):
    """
    Estatísticas de normalização a partir de uma amostra de snapshots.

    Acumula soma e soma de quadrados em STREAMING — nunca guarda os
    211.579 × n_files valores na memória. Para percentis (que precisam da
    distribuição, não só dos dois primeiros momentos) guarda uma
    subamostra aleatória por campo.

    Os arquivos são escolhidos espaçados UNIFORMEMENTE NO TEMPO, não os
    n primeiros: o começo da série ainda carrega o transiente de
    inicialização e daria estatística enviesada.

    Returns: dict com média, desvio, min, max e percentis por campo.
    """
    n_avail = len(snaps)
    n_files = min(n_files, n_avail)
    idx = np.unique(np.linspace(0, n_avail - 1, n_files).astype(int))
    picked = [snaps[i] for i in idx]

    rng_local = np.random.default_rng(seed)
    keys = list(_STATS_FIELDS.keys()) + ["lnk", "lnw"]

    n_tot = 0
    acc_sum = {k: 0.0 for k in keys}
    acc_sq  = {k: 0.0 for k in keys}
    acc_min = {k: np.inf for k in keys}
    acc_max = {k: -np.inf for k in keys}
    pool    = {k: [] for k in keys}
    per_file_keep = max(1, subsample // len(picked))

    n_k_neg = 0
    n_w_neg = 0
    n_visc_lim = 0
    n_k_below_eps = 0

    if verbose:
        print(f"\n[norm stats] lendo {len(picked)} de {n_avail} snapshots "
              f"(espaçados uniformemente em t)")
    t0 = time.time()

    for j, (t_phys, path) in enumerate(picked):
        d = _read_stats_frame(path)

        # k e ω têm que ser positivos. Se o CSV trouxer zero ou negativo
        # (pode acontecer por under/overshoot do esquema convectivo), conta
        # e aplica o piso — em vez de deixar o log virar NaN silenciosamente.
        n_k_neg += int((d["k"] <= 0).sum())
        n_w_neg += int((d["omega"] <= 0).sum())
        d["k"]     = np.maximum(d["k"], K_FLOOR)
        d["omega"] = np.maximum(d["omega"], OMEGA_FLOOR)
        # ln(k + K_EPS): piso SUAVE. Ver a nota em K_EPS.
        n_k_below_eps += int((d["k"] < K_EPS).sum())
        d["lnk"]   = np.log(d["k"] + K_EPS)
        d["lnw"]   = np.log(d["omega"])

        # células onde o solver recortou ν_t (ver Cell 2)
        visc_ratio = d["mut"] / MU if VISC_TURB_IS_DYNAMIC else d["mut"] / NU
        n_visc_lim += int((visc_ratio >= VISC_RATIO_MAX * (1 - VISC_LIMITED_REL_TOL)).sum())

        n_here = len(d["x"])
        n_tot += n_here
        for k in keys:
            a = d[k]
            acc_sum[k] += float(a.sum())
            acc_sq[k]  += float(np.square(a).sum())
            acc_min[k]  = min(acc_min[k], float(a.min()))
            acc_max[k]  = max(acc_max[k], float(a.max()))
            take = min(per_file_keep, n_here)
            pool[k].append(rng_local.choice(a, size=take, replace=False))

        if verbose and ((j + 1) % 5 == 0 or j == 0 or j == len(picked) - 1):
            print(f"  [{j+1:3d}/{len(picked)}] t={t_phys:7.2f}s  "
                  f"{n_here:,} células  (acum {n_tot:,})")

    stats = {}
    for k in keys:
        mean = acc_sum[k] / n_tot
        var  = max(acc_sq[k] / n_tot - mean * mean, 0.0)
        p    = np.concatenate(pool[k])
        stats[k] = dict(
            mean=mean, std=float(np.sqrt(var)),
            min=acc_min[k], max=acc_max[k],
            p01=float(np.percentile(p, 1)),
            p50=float(np.percentile(p, 50)),
            p99=float(np.percentile(p, 99)),
            _pool=p,        # subamostra, para diagnósticos de distribuição
        )

    stats["_meta"] = dict(
        n_files=len(picked), n_points=n_tot,
        n_k_nonpos=n_k_neg, n_w_nonpos=n_w_neg,
        n_k_below_eps=n_k_below_eps,
        pct_k_below_eps=100.0 * n_k_below_eps / n_tot,
        n_visc_limited=n_visc_lim,
        pct_visc_limited=100.0 * n_visc_lim / n_tot,
        wall_time=time.time() - t0,
    )
    if verbose:
        print(f"[norm stats] {n_tot:,} pontos em {stats['_meta']['wall_time']:.1f}s")
    return stats


# ---- execução ----
NORM_STATS = None
if AUTO_NORM_STATS:
    NORM_STATS = compute_normalization_stats(snaps)

    _herdado = dict(X_MEAN=X_MEAN, Y_MEAN=Y_MEAN, Z_MEAN=Z_MEAN, DP_ISO=DP_ISO,
                    V_MEAN=V_MEAN, V_DP=V_DP, P_MEAN=P_MEAN, P_DP=P_DP,
                    LNK_MEAN=LNK_MEAN, LNK_DP=LNK_DP,
                    LNW_MEAN=LNW_MEAN, LNW_DP=LNW_DP)

    X_MEAN = NORM_STATS["x"]["mean"]
    Y_MEAN = NORM_STATS["y"]["mean"]
    Z_MEAN = NORM_STATS["z"]["mean"]
    # dp_iso = mean(dp_x, dp_y, dp_z) — mesma definição do script original
    DP_ISO = (NORM_STATS["x"]["std"]
              + NORM_STATS["y"]["std"]
              + NORM_STATS["z"]["std"]) / 3.0

    V_MEAN = NORM_STATS["v"]["mean"]
    V_DP   = NORM_STATS["v"]["std"]
    P_MEAN = NORM_STATS["P"]["mean"]
    P_DP   = NORM_STATS["P"]["std"]

    # ln k e ln ω: z-score, o mesmo esquema de v e P
    LNK_MEAN = NORM_STATS["lnk"]["mean"]
    LNK_DP   = NORM_STATS["lnk"]["std"]
    LNW_MEAN = NORM_STATS["lnw"]["mean"]
    LNW_DP   = NORM_STATS["lnw"]["std"]
    K_SCALE  = NORM_STATS["k"]["std"]     # só usado no modo "scale"

    _medido = dict(X_MEAN=X_MEAN, Y_MEAN=Y_MEAN, Z_MEAN=Z_MEAN, DP_ISO=DP_ISO,
                   V_MEAN=V_MEAN, V_DP=V_DP, P_MEAN=P_MEAN, P_DP=P_DP,
                   LNK_MEAN=LNK_MEAN, LNK_DP=LNK_DP,
                   LNW_MEAN=LNW_MEAN, LNW_DP=LNW_DP)

    print(f"\n{'='*76}")
    print(f"  CONSTANTES DE NORMALIZAÇÃO — fallback ({NORM_FALLBACK_SOURCE}) × medido")
    print(f"{'='*76}")
    print(f"  {'constante':<10}{'fallback':>18}{'medido':>18}{'Δ relativo':>16}")
    print(f"  {'-'*72}")
    for _k in _herdado:
        _a, _b = _herdado[_k], _medido[_k]
        _rel = (_b - _a) / abs(_a) if _a != 0 else float("nan")
        _flag = "  <<<" if abs(_rel) > 0.05 else ""
        print(f"  {_k:<10}{_a:>18.6f}{_b:>18.6f}{100*_rel:>15.1f}%{_flag}")
    print(f"  {'-'*72}")
    print(f"  '<<<' marca desvio acima de 5%. Para congelar estes valores:")
    print(f"  cole-os acima e ponha AUTO_NORM_STATS = False.")
    _n_flag = sum(1 for _k in _herdado
                  if _herdado[_k] != 0
                  and abs((_medido[_k] - _herdado[_k]) / abs(_herdado[_k])) > 0.05
                  and _k in ("X_MEAN", "Y_MEAN", "Z_MEAN", "DP_ISO"))
    # Com o fallback MEDIDO no mesmo dado e a amostragem determinística,
    # o esperado é Δ ≈ 0 em TODAS as linhas, não só nas espaciais.
    _n_any = sum(1 for _k in _herdado
                 if _herdado[_k] != 0
                 and abs((_medido[_k] - _herdado[_k]) / abs(_herdado[_k])) > 1e-4)
    if _n_any == 0:
        print(f"")
        print(f"  Todas as constantes batem com o fallback (Δ < 0,01%): o dado")
        print(f"  lido é o mesmo de {NORM_FALLBACK_SOURCE}. O dicionário da rede")
        print(f"  é o mesmo da calibração.")
    else:
        print(f"")
        print(f"  >>> {_n_any} constante(s) diferem do fallback em mais de 0,01%")
        print(f"      ({_n_flag} das 4 espaciais acima de 5%). O fallback foi")
        print(f"      medido em {NORM_FALLBACK_SOURCE}; se CFD_DIR aponta para")
        print(f"      outro dado, o dicionário da rede mudou e pesos treinados no")
        print(f"      dado antigo NÃO valem para este (ver o cold start, Cell 3).")

    _m = NORM_STATS["_meta"]
    if _m["n_k_nonpos"] or _m["n_w_nonpos"]:
        print(f"\n  >>> valores não-positivos na amostra: "
              f"k em {_m['n_k_nonpos']:,} pontos, ω em {_m['n_w_nonpos']:,}")
        print(f"      (pisos K_FLOOR/OMEGA_FLOOR aplicados antes do log)")
    # O esperado vem da Cell 2 (0% com teto 1e7, job 6125; era ~7,2% com
    # 1e5, job 5518). Avisar só quando a amostra contradiz o esperado — a
    # leitura completa, por zona, fica na Cell 5b §2.
    print(f"\n  células com ν_t recortado na amostra: {_m['n_visc_limited']:,} "
          f"({_m['pct_visc_limited']:.2f}%)  — esperado pela Cell 2: "
          f"{VISC_LIMITED_PCT_EXPECTED:g}% com teto {VISC_RATIO_MAX:.0e}")
    if VISC_LIMITED_PCT_EXPECTED < 0.1 and _m["pct_visc_limited"] > 0.1:
        print(f"      >>> com o teto em {VISC_RATIO_MAX:.0e} não devia haver recorte.")
        print(f"          Conferir se CFD_DIR (Cell 3) é mesmo o job {CFD_JOB_ID} e")
        print(f"          se VISC_RATIO_MAX (Cell 2) é o teto que o solver usou.")
    print(f"  células com k < K_EPS ({K_EPS:.4g} m²/s²): {_m['n_k_below_eps']:,} "
          f"({_m['pct_k_below_eps']:.2f}%)")
    if _m["pct_k_below_eps"] > 5.0:
        print(f"      >>> mais de 5% do domínio está abaixo do piso suave, ou seja")
        print(f"          comprimido numa faixa estreita de ln k. Se essa região")
        print(f"          importa para o treino, baixar K_EPS_FRAC.")


#---- DOMÍNIO NORMALIZADO ----#
# coordenada x
XN_MIN = (X_MIN - X_MEAN) / DP_ISO
XN_MAX = (X_MAX - X_MEAN) / DP_ISO

# coordenada y
YN_MIN = (Y_MIN - Y_MEAN) / DP_ISO
YN_MAX = (Y_MAX - Y_MEAN) / DP_ISO

# coordenada z
ZN_MIN = (Z_MIN - Z_MEAN) / DP_ISO
ZN_MAX = (Z_MAX - Z_MEAN) / DP_ISO

# cilindro
XC_N = (XC_CYL - X_MEAN) / DP_ISO
YC_N = (YC_CYL - Y_MEAN) / DP_ISO
ZC_N = (ZC_BASE - Z_MEAN) / DP_ISO
H_N  = (H_CYL  - Z_MEAN) / DP_ISO

# mantém escala do domínio normalizado
R_N  = R_CYL  / DP_ISO

# tempo
TN_MIN = (float(times_all[0])  - T_MEAN) / T_DP
TN_MAX = (float(times_all[-1]) - T_MEAN) / T_DP

# velocidade inlet normalizada
VN_INLET = (V_INF - V_MEAN) / V_DP

# --- NOVO: turbulência no inlet, em unidades normalizadas ---
# Usados pela HardConstraintLayer, do mesmo jeito que VN_INLET.
if K_NORM_MODE == "logz":
    KN_INLET = (float(np.log(K_INLET + K_EPS)) - LNK_MEAN) / LNK_DP
elif K_NORM_MODE == "scale":
    KN_INLET = K_INLET / K_SCALE
else:                                    # "z" — desaconselhado, ver Cell 1
    KN_INLET = (K_INLET - NORM_STATS["k"]["mean"]) / NORM_STATS["k"]["std"]

LNWN_INLET = (float(np.log(OMEGA_INLET)) - LNW_MEAN) / LNW_DP

# para velocidades x, y, z, pressão, k e ω:
# são normalizadas diretamente na Célula 5
# durante a leitura da variável crua ("raw")


# ---- VERIFICAÇÃO E LOG ----
print(f"\n--------- Escalas Normalizadas (z-score) ---------")
print(f"  médias (mi) espacial: ({X_MEAN:.3f}, {Y_MEAN:.3f}, {Z_MEAN:.3f}) m")
print(f"  devio padrão isotrópico espacial: {DP_ISO:.3f} m")
print(f"  média temporal: {T_MEAN:.3f} s, devio padrão temporal: {T_DP:.3f} s")
print(f"")
print(f"  x_norm  = [{XN_MIN:+.3f}, {XN_MAX:+.3f}]")
print(f"  y_norm  = [{YN_MIN:+.3f}, {YN_MAX:+.3f}]")
print(f"  z_norm  = [{ZN_MIN:+.3f}, {ZN_MAX:+.3f}]")
print(f"  t_norm  = [{TN_MIN:+.3f}, {TN_MAX:+.3f}]")
print(f"")
print(f"  Cilindro normalizado:")
print(f"    centro = ({XC_N:+.3f}, {YC_N:+.3f})")
print(f"    base z = {ZC_N:+.3f},  topo z = {H_N:+.3f}")
print(f"    R_n    = {R_N:.4f}  (circular, mesmo desvio padrão nos 3 eixos)")
print(f"")
print(f"  Saídas (u,v,w,P,k,ω) - aplicar no carregamento dos snapshots:")
print(f"    u: raw / V_REF          V_REF={V_REF}")
print(f"    v: (raw - μ)/σ          μ={V_MEAN:+.4f}, σ={V_DP:.4f}")
print(f"    w: raw / V_REF          V_REF={V_REF}")
print(f"    P: (raw - μ)/σ          μ={P_MEAN:+.3f}, σ={P_DP:.3f}")
if K_NORM_MODE == "logz":
    print(f"    k: (ln(raw+ε) - μ)/σ    μ={LNK_MEAN:+.4f}, σ={LNK_DP:.4f}, "
          f"ε={K_EPS:.4g}  [log ANTES do z-score]")
else:
    print(f"    k: modo '{K_NORM_MODE}'   K_SCALE={K_SCALE:.4f}")
print(f"    ω: (ln raw - μ)/σ       μ={LNW_MEAN:+.4f}, σ={LNW_DP:.4f}  [log ANTES do z-score]")
print(f"")
print(f"  Desnormalização (o que a Cell 9 tem que fazer na saída da rede):")
if K_NORM_MODE == "logz":
    print(f"    k = exp({LNK_MEAN:+.4f} + {LNK_DP:.4f}·k_n) - {K_EPS:.4g}      -> k > 0 sempre")
print(f"    ω = exp({LNW_MEAN:+.4f} + {LNW_DP:.4f}·ω_n)             -> ω > 0 sempre")
if K_NORM_MODE == "logz":
    print(f"    ln ν_t = ln k - ln ω = "
          f"({LNK_MEAN-LNW_MEAN:+.4f}) + {LNK_DP:.4f}·k_n - {LNW_DP:.4f}·ω_n")
    print(f"             ^ afim nas saídas da rede: nada de divisão 0/0 no resíduo")
print(f"")
print(f"  VN_INLET   (v_baseline da HardConstraint) = {VN_INLET:.4f}")
print(f"  KN_INLET   (k no inlet, normalizado)      = {KN_INLET:.4f}")
print(f"  LNWN_INLET (ln ω no inlet, normalizado)   = {LNWN_INLET:.4f}")

# O inlet de turbulência é um OUTLIER em relação ao interior do domínio, e
# isso não é um defeito da normalização — é física. O Fluent impõe ali
# I=5% e μ_t/μ=10, que é turbulência quase nenhuma; poucos metros adiante o
# escoamento já desenvolveu e ω cai ordens de grandeza. A entrada fica na
# ponta da distribuição.
#
# Por que importa: a hard constraint do inlet vai empurrar a saída da rede
# para LNWN_INLET. Se esse valor estiver muito fora de ±4, ele cai na região
# em que tanh satura, o gradiente some, e a restrição do inlet compete com
# o resto do domínio em vez de somar.
_BC_OUTLIER_TOL = 4.0
if abs(LNWN_INLET) > _BC_OUTLIER_TOL or abs(KN_INLET) > _BC_OUTLIER_TOL:
    print(f"")
    print(f"  >>> A BC de turbulência do inlet está FORA de ±{_BC_OUTLIER_TOL:.0f} "
          f"na escala normalizada.")
    print(f"      ω_inlet = {OMEGA_INLET:.0f} 1/s contra mediana ~"
          f"{np.exp(LNW_MEAN):.1f} 1/s no domínio "
          f"({OMEGA_INLET/np.exp(LNW_MEAN):.0f}× maior).")
    print(f"      Três saídas, em ordem de preferência:")
    print(f"        (a) não impor ω por hard constraint — deixar a data loss")
    print(f"            das células do inlet cuidar disso (elas existem no CSV);")
    print(f"        (b) impor a hard constraint em ln ω SEM z-score, ou seja")
    print(f"            direto no valor físico, sem passar pela escala saturada;")
    print(f"        (c) normalizar ω com estatística só do interior, tratando o")
    print(f"            inlet como caso à parte.")
    print(f"      Decidir na Cell 9, ao montar a HardConstraintLayer.")

if AUTO_NORM_STATS:
    print(f"\n--------- Sanity check das faixas normalizadas ---------")
    _s = NORM_STATS
    def _rng(label, lo, hi, alerta=4.0):
        _f = "  <<< fora de [-4, 4]" if (abs(lo) > alerta or abs(hi) > alerta) else ""
        print(f"    {label:<8} [{lo:+8.3f}, {hi:+8.3f}]{_f}")
    print(f"  p01 e p99 de cada saída, já normalizados:")
    _rng("u", _s["u"]["p01"]/V_REF, _s["u"]["p99"]/V_REF)
    _rng("v", (_s["v"]["p01"]-V_MEAN)/V_DP, (_s["v"]["p99"]-V_MEAN)/V_DP)
    _rng("w", _s["w"]["p01"]/V_REF, _s["w"]["p99"]/V_REF)
    _rng("P", (_s["P"]["p01"]-P_MEAN)/P_DP, (_s["P"]["p99"]-P_MEAN)/P_DP)
    if K_NORM_MODE == "logz":
        _rng("ln k", (_s["lnk"]["p01"]-LNK_MEAN)/LNK_DP,
                     (_s["lnk"]["p99"]-LNK_MEAN)/LNK_DP)
    else:
        _rng("k", _s["k"]["p01"]/K_SCALE, _s["k"]["p99"]/K_SCALE)
    _rng("ln ω", (_s["lnw"]["p01"]-LNW_MEAN)/LNW_DP,
                 (_s["lnw"]["p99"]-LNW_MEAN)/LNW_DP)
    print(f"  (99% dos pontos dentro de ±4 é o alvo; fora disso tanh satura)")

    # A prova de que o z-score está fazendo efeito, e não só zerando a média:
    # numa gaussiana, 68,3% dos pontos caem em [-1, 1]. Quanto mais longe
    # disso, mais assimétrica é a variável e menos o z-score está ajudando.
    print(f"\n  Fração em [-1,1] — numa gaussiana daria 68,3%:")
    for _lab, _p, _mu, _sd in [
            ("v",    _s["v"]["_pool"],   V_MEAN,   V_DP),
            ("P",    _s["P"]["_pool"],   P_MEAN,   P_DP),
            ("ln k", _s["lnk"]["_pool"], LNK_MEAN, LNK_DP),
            ("ln ω", _s["lnw"]["_pool"], LNW_MEAN, LNW_DP)]:
        _f = 100.0 * float(np.mean(np.abs((_p - _mu) / _sd) < 1.0))
        print(f"    {_lab:<6} {_f:5.1f}%")
    _fk_raw = 100.0 * float(np.mean(
        np.abs((_s["k"]["_pool"] - _s["k"]["mean"]) / _s["k"]["std"]) < 1.0))
    print(f"    {'k cru':<6} {_fk_raw:5.1f}%   <- comparação: z-score em k sem log")

    print(f"\n  Faixas FÍSICAS medidas (para comparar com o CFD):")
    for _k, _lab, _u in [("u","x-velocity","m/s"), ("v","y-velocity","m/s"),
                         ("w","z-velocity","m/s"), ("P","pressure","Pa"),
                         ("k","turb-kin-energy","m²/s²"),
                         ("omega","specific-diss","1/s"),
                         ("mut","viscosity-turb","Pa·s")]:
        _d = _s[_k]
        print(f"    {_lab:<16} min={_d['min']:>12.4g}  p50={_d['p50']:>10.4g}  "
              f"max={_d['max']:>12.4g}  [{_u}]")


# ================================================================================
# CELL 5 — LOAD AND STRATIFIED-SAMPLE SNAPSHOTS (BCs + DETALHAMENTO + INTERIOR)
# ================================================================================
#
#   TABELA DE PONTOS POR CATEGORIA *PARA* CADA ARQUIVO
#  -------------------------------------------------------------------------
#   Subtotal BCs (16k):
#     1. cyl_lateral  (z ≤ H_CYL, |r - R_CYL| < CYL_TOL)         3500 pts
#     2. cyl_top      (z > H_CYL, r ≤ R_CYL, z - H_CYL < CYL_TOL) 500 pts
#     3. bc_inlet     (dist ao plano y=Y_MIN < BC_TOL)           2000 pts
#     4. bc_outlet    (dist ao plano y=Y_MAX < BC_TOL)           2000 pts
#     5. bc_xmin      (dist ao plano x=X_MIN < BC_TOL)           2000 pts
#     6. bc_xmax      (dist ao plano x=X_MAX < BC_TOL)           2000 pts
#     7. bc_ground    (dist ao plano z=Z_MIN < BC_TOL)           2000 pts
#     8. bc_top       (dist ao plano z=Z_MAX < BC_TOL)           2000 pts
#
#   Subtotal Detalhamento (16k):
#     9. boi_c        (interior do bbox BOI_Cylinder)           10000 pts
#    10. boi_t        (interior do prisma BOI_Triangle)          6000 pts
#
#   Subtotal Interior (16k):
#    11. freestream   (resto do domínio)                        16000 pts
#                                                          -----------------
#                                                  TOTAL:    48000 pts/snap
#
# HIERARQUIA (mais específico vence, sobrescrevendo):
#   freestream → boi_t → boi_c → bc_* → cyl_top → cyl_lateral
#
# Para BCs, a atribuição é EXCLUSIVA (cada ponto vai pra BC mais próxima).
# Pontos perto do cilindro têm PRIORIDADE — vão pra cyl_* mesmo se também
# estão perto de uma BC ou dentro de algum BOI.
#
# A geometria, a hierarquia e as quotas ficam IDÊNTICAS ao laminar.
# ================================================================================
# O QUE MUDA NESTA CÉLULA
#
# 1. DELIMITADOR. A versão laminar lê com sep=r'\s+' fixo. Os CSV do job 5518
#    são separados por VÍRGULA (confirmado no log do Fluent). Agora usa
#    CSV_DELIMITER, detectado na Cell 3 — funciona nos dois formatos.
#
# 2. TRÊS COLUNAS NOVAS lidas do CSV: turb-kinetic-energy, specific-diss-rate
#    e viscosity-turb. As duas primeiras viram saídas da rede; a terceira NÃO
#    é saída — serve só para marcar onde o solver recortou ν_t.
#
# 3. data_np passa de 9 para 13 colunas: entram kn e ωn como alvos, um
#    marcador binário de célula com ν_t recortado e a DISTÂNCIA À PAREDE.
#    A distância não é entrada da rede nem alvo — é coeficiente. Está aqui
#    porque é o que permite dizer, ponto a ponto, se aquele dado veio do
#    transporte resolvido ou do tratamento de parede.
#
# 4. SEIS VERIFICAÇÕES contra o dado, feitas no primeiro snapshot. Cada uma
#    testa uma hipótese que as células anteriores assumiram sem poder provar:
#      (a) viscosity-turb é μ_t dinâmica ou ν_t cinemática?
#      (b) a relação constitutiva INTEIRA fecha — teto, limitador e k/ω?
#      (c) a BC de turbulência que a PINN impõe no inlet é a que o solver
#          impôs? (lida das FACES do inlet, não das células — ver abaixo)
#      (d) até onde vai a região de tratamento de parede, EM CADA PAREDE, e
#          a relação da camada log vale ali?
#      (e) a coluna `pressure` é p ou p̃ = p + (2/3)ρk?
#      (f) o ω da 1ª camada é o ω_wall imposto, ou saiu do transporte?
#          Decide se mascarar ω é obrigatório — e, de quebra, diz se a
#          razão medida em (d) é física ou tautologia.
#    Se qualquer uma falhar, o diagnóstico aparece aqui e não como um treino
#    que não converge.
#
# ------------------------------------------------------------------------------
# REVISÃO DE 27/09/2026 — o que esta célula deixou de afirmar
#
# Três conclusões desta célula eram consequência de testes incompletos, e os
# jobs 6023/6025/6027 as derrubaram. Ficam registradas porque o raciocínio
# que levou a cada uma é plausível e voltaria sozinho:
#
#   1. "ν_t = k/ω, logo SST_SHEAR_LIMITER=False."  A mediana de 1,000000 era
#      real, mas medida numa amostra dominada pelas células onde o limitador
#      não atua. A verificação (b) agora separa os TRÊS ramos (teto, livre,
#      limitador) e mede a fração de cada um, em vez de olhar uma mediana.
#
#   2. "Y_WALL_FUNCTION = d_wall.min()."  Isso é a célula mais fina da malha
#      INTEIRA. As paredes têm refinos diferentes, então a primeira camada
#      das mais grosseiras ficava fora da faixa e a máscara não pegava
#      praticamente nenhum ponto de colocação. Agora a faixa é medida POR
#      PAREDE, achando o salto entre a 1ª e a 2ª camada de células.
#
#   3. "Onde o teto de ν_t agiu, o resíduo não fecha — mascarar."  O teto é
#      parte da relação constitutiva que o solver usou, não um artefato
#      aplicado por fora. Com o mesmo min no resíduo (Cell 10), aquelas
#      células fecham a equação como quaisquer outras. MASK_VISC_LIMITED
#      virou False na Cell 2; a coluna continua sendo calculada, agora só
#      como diagnóstico.
#
# ------------------------------------------------------------------------------
# REVISÃO DE 18/10/2026 — dado novo (job 6125, teto de ν_t em 1e7)
#
#   4. A VERIFICAÇÃO (c) COMPARAVA A BC COM O QUE A BC JÁ TINHA VIRADO.
#      Ela tomava a mediana de k e ω nas CÉLULAS a menos de 5 m do inlet e
#      comparava com K_INLET/OMEGA_INLET. Só que ω_inlet = 7419 1/s decai
#      muito depressa: sem produção, dω/dt = −βω² dá ω(x) = ω0/(1 + x/L) com
#      L = V_INF/(β·ω0) ≈ 2,8 cm — a 0,2 m o dado já caiu 8×, e o centro da
#      primeira célula está muito além disso. (Os 0,26 mm citados em outros
#      comentários são o COMPRIMENTO TURBULENTO √k/(β*^¼ω), L_TURB_INLET da
#      Cell 2, não a distância de decaimento.) O teste
#      concluía "TI ≈ 1,63%, μ_t/μ ≈ 24,8" e mandava corrigir a Cell 2 —
#      acusando divergência onde não havia nenhuma.
#
#      Agora a (c) lê bc/inlet_faces_init.csv (INLET_FACES_CSV, Cell 3), o
#      valor de FACE que o solver impôs, e compara com o que a PINN impõe:
#      K_INLET/OMEGA_INLET em unidades físicas e KN_INLET/LNWN_INLET em
#      unidades da rede. Medido no arquivo: k = 1,08375 e ω = 7419,21 —
#      exatamente os valores da Cell 2. A PINN sempre teve a BC certa; o que
#      estava errado era a verificação.
#
#      Sem o arquivo (HAS_INLET_FACES = False), a (c) cai no modo antigo,
#      mas agora rotulado como o que ele é: medida de decaimento, que NÃO
#      decide nada sobre a BC.
#
#   5. As outras cinco verificações não mudaram de código. Rodam de novo e
#      dão números diferentes — a (b) deve mostrar ~0% no ramo do teto, que
#      com 1e7 não age mais. Isso é o esperado, não regressão.
#
# NOTA SOBRE AS QUOTAS: 48.000 pontos de 211.579 células é ~23% do domínio
# por snapshot. A malha do caso laminar tem contagem da mesma ordem, então
# esse regime não mudou e as quotas seguem iguais. Os déficits que aparecerem
# na tabela da Cell 5b devem ser os mesmos de antes — tipicamente nas cascas
# finas (cyl_top, BCs), onde o volume disponível é que é pequeno, não a malha.
# ================================================================================


# ---------- USER CONFIG ----------
QUOTAS = {
    # Parede do cilindro (no-slip)
    "cyl_lateral": 3500,
    "cyl_top":      500,
    # BCs externas
    "bc_inlet":    2000,
    "bc_outlet":   2000,
    "bc_xmin":     2000,
    "bc_xmax":     2000,
    "bc_ground":   2000,
    "bc_top":      2000,
    # Detalhamento volumétrico (BOIs)
    "boi_c":      10_000,
    "boi_t":       6_000,
    # Interior livre
    "freestream": 16_000,
}

N_POINTS_PER_SNAP = sum(QUOTAS.values())   # 48000
MAX_SNAPSHOTS     = None                   # para usar todos

CATEGORY_NAMES = list(QUOTAS.keys())


# ---------- LAYOUT DE data_np ----------
# Índices nomeados em vez de números soltos. A versão laminar usava
# batch[:, 4:8] e batch[:, 8] direto; com 6 saídas em vez de 4 esses números
# mudam, e um erro de índice aqui é silencioso — treinaria trocando ω por gid.
COL_XN, COL_YN, COL_ZN, COL_TN = 0, 1, 2, 3     # entradas
COL_UN, COL_VN, COL_WN, COL_PN = 4, 5, 6, 7     # alvos: campo médio
COL_KN, COL_OMN                = 8, 9           # alvos: turbulência
COL_GID                        = 10             # id do grupo (zona)
COL_VLIM                       = 11             # 1.0 se ν_t bateu no teto
COL_DWALL                      = 12             # distância à parede [m]

N_COLS_DATA = 13
SLICE_INPUT  = slice(COL_XN, COL_TN + 1)        # [:, 0:4]
SLICE_TARGET = slice(COL_UN, COL_OMN + 1)       # [:, 4:10]
N_TARGETS    = COL_OMN - COL_UN + 1             # 6

TARGET_NAMES = ["u", "v", "w", "P", "k", "omega"]


NEEDED_COLS_PATTERNS = {
    "x": [r"x.coordinate", r"^x$"],
    "y": [r"y.coordinate", r"^y$"],
    "z": [r"z.coordinate", r"^z$"],
    "P": [r"^pressure$", r"static.pressure", r"^p$"],
    "u": [r"x.velocity", r"^u$"],
    "v": [r"y.velocity", r"^v$"],
    "w": [r"z.velocity", r"^w$"],
    # --- NOVOS ---
    "k":     [r"turb.kinetic.energy", r"^k$", r"^tke$"],
    "omega": [r"specific.diss.rate", r"^omega$", r"^w_sst$"],
    "mut":   [r"viscosity.turb", r"turb.*viscosity", r"^mut$"],
}

def _norm_col(s):
    """Normaliza: strip, lower, remove aspas e caracteres invisíveis."""
    return re.sub(r'["\'﻿]', '', s).strip().lower()

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
            raise KeyError(f"No match for '{key}'. Patterns: {patterns}. "
                            f"Available normalized: {list(norm_map.keys())}")
        mapping[key] = found
    return mapping

def load_snapshot(path, col_map=None):
    df = pd.read_csv(
        path,
        sep=CSV_DELIMITER,      # detectado na Cell 3 (vírgula no job 5518)
        engine='python',        # regex sep precisa do engine python
        skipinitialspace=True,  # "a, b, c" -> nomes sem espaço à esquerda
        dtype=np.float32,
    )

    # Normalizar os nomes ANTES de descartar cellnumber. Na versão laminar o
    # drop vinha primeiro e só casava se o nome já estivesse limpo; com o
    # cabeçalho do Fluent (" cellnumber") ele não casava e a coluna seguia
    # adiante em silêncio.
    df.columns = [_norm_col(c) for c in df.columns]
    df = df.drop(columns=["cellnumber"], errors="ignore")
    if col_map is None:
        col_map = resolve_column_names(df.columns)
    out = {k: df[col_map[k]].to_numpy(dtype=np.float32, copy=False)
            for k in NEEDED_COLS_PATTERNS}
    return out, col_map

# ================================================================================
# Funções modulares de classificação geométrica
#
# Todas operam em arrays NumPy.
# seção "4. MÁSCARAS GEOMÉTRICAS" e seção "6b. Atribuição exclusiva".
# INALTERADAS em relação ao laminar — a geometria é a mesma.
# ================================================================================

def is_inside_cylinder(x, y, z):
    """Dentro do cilindro sólido (deve ser excluído antes de classificar)."""
    r2 = (x - XC_CYL)**2 + (y - YC_CYL)**2
    return (r2 <= R_CYL**2) & (z <= H_CYL)


def is_adj_cyl_lateral(x, y, z):
    """Camada lateral do cilindro: z ≤ H_CYL e |r - R_CYL| < CYL_TOL."""
    r = np.sqrt((x - XC_CYL)**2 + (y - YC_CYL)**2)
    return (z <= H_CYL) & (np.abs(r - R_CYL) < CYL_TOL)


def is_adj_cyl_top(x, y, z):
    """Camada do topo: z > H_CYL, r ≤ R_CYL, (z - H_CYL) < CYL_TOL."""
    r = np.sqrt((x - XC_CYL)**2 + (y - YC_CYL)**2)
    return (z > H_CYL) & (r <= R_CYL) & ((z - H_CYL) < CYL_TOL)


def is_inside_boi_cyl(x, y, z):
    """Dentro da caixa BOI_Cylinder."""
    return ((x >= BOI_C_X[0]) & (x <= BOI_C_X[1]) &
            (y >= BOI_C_Y[0]) & (y <= BOI_C_Y[1]) &
            (z >= BOI_C_Z[0]) & (z <= BOI_C_Z[1]))


def is_inside_boi_tri(x, y, z):
    """Dentro do prisma triangular BOI_Triangle (teste baricêntrico em XY)."""
    v1, v2, v3 = BOI_T_V1, BOI_T_V2, BOI_T_V3
    denom = (v2[1]-v3[1])*(v1[0]-v3[0]) + (v3[0]-v2[0])*(v1[1]-v3[1])
    a = ((v2[1]-v3[1])*(x-v3[0]) + (v3[0]-v2[0])*(y-v3[1])) / denom
    b = ((v3[1]-v1[1])*(x-v3[0]) + (v1[0]-v3[0])*(y-v3[1])) / denom
    c = 1.0 - a - b
    in_xy = (a >= 0) & (b >= 0) & (c >= 0)
    in_z  = (z >= BOI_T_Z[0]) & (z <= BOI_T_Z[1])
    return in_xy & in_z


def classify_external_bc(x, y, z):
    """
    Para cada ponto, retorna:
      is_adj_bc : True onde dist < BC_TOL a alguma BC externa
      bc_idx    : índice 0..5 da BC mais próxima (atribuição exclusiva)

    Convenção:
      0 = bc_inlet (y=Y_MIN)
      1 = bc_outlet (y=Y_MAX)
      2 = bc_xmin (x=X_MIN)
      3 = bc_xmax (x=X_MAX)
      4 = bc_ground (z=Z_MIN)
      5 = bc_top (z=Z_MAX)
    """
    d_inlet  = np.abs(y - Y_MIN)
    d_outlet = np.abs(y - Y_MAX)
    d_xmin   = np.abs(x - X_MIN)
    d_xmax   = np.abs(x - X_MAX)
    d_ground = np.abs(z - Z_MIN)
    d_top    = np.abs(z - Z_MAX)
    dists = np.stack([d_inlet, d_outlet, d_xmin, d_xmax, d_ground, d_top], axis=1)
    d_min = dists.min(axis=1)
    bc_idx = dists.argmin(axis=1).astype(np.int8)
    is_adj_bc = d_min < BC_TOL
    return is_adj_bc, bc_idx


BC_IDX_TO_NAME = {
    0: "bc_inlet",
    1: "bc_outlet",
    2: "bc_xmin",
    3: "bc_xmax",
    4: "bc_ground",
    5: "bc_top",
}


def classify_points(x, y, z):
    """
    Classifica pontos em CATEGORY_NAMES seguindo hierarquia (mais específico vence):
      freestream → boi_t → boi_c → bc_* → cyl_top → cyl_lateral

    Pontos no interior sólido do cilindro são marcados como None (devem ter
    sido removidos antes; mantido por segurança).
    """
    n = len(x)
    category = np.full(n, None, dtype=object)

    # Step 0: pontos no interior do cilindro ficam None
    inside_cyl = is_inside_cylinder(x, y, z)

    # Step 1: freestream cobre tudo que não é interior do cilindro
    category[~inside_cyl] = "freestream"

    # Step 2: BOI_T sobrescreve freestream
    mask = is_inside_boi_tri(x, y, z) & ~inside_cyl
    category[mask] = "boi_t"

    # Step 3: BOI_C sobrescreve BOI_T (e freestream) — onde há overlap
    mask = is_inside_boi_cyl(x, y, z) & ~inside_cyl
    category[mask] = "boi_c"

    # Step 4: BCs externas sobrescrevem BOIs e freestream
    is_bc, bc_idx = classify_external_bc(x, y, z)
    for k, name in BC_IDX_TO_NAME.items():
        mask = is_bc & (bc_idx == k) & ~inside_cyl
        category[mask] = name

    # Step 5: cilindro topo sobrescreve BCs
    category[is_adj_cyl_top(x, y, z)] = "cyl_top"

    # Step 6: cilindro lateral (prioridade máxima)
    category[is_adj_cyl_lateral(x, y, z)] = "cyl_lateral"

    return category


def _stratified_sample(category, quotas, rng):
    """
    Sampling sem reposição respeitando as quotas. Se uma categoria tiver menos
    pontos disponíveis que a quota, usa todos os disponíveis.

    Returns: picked_idx, avail_counts (dict), taken_counts (dict)
    """
    avail_counts = {}
    taken_counts = {}
    picked = []
    for cat_name, quota in quotas.items():
        idx_cat = np.where(category == cat_name)[0]
        n_avail = len(idx_cat)
        avail_counts[cat_name] = n_avail
        if n_avail == 0:
            taken_counts[cat_name] = 0
            continue
        taken = idx_cat if n_avail <= quota else rng.choice(idx_cat, size=quota, replace=False)
        taken_counts[cat_name] = len(taken)
        picked.append(taken)
    picked_idx = np.concatenate(picked) if picked else np.array([], dtype=np.int64)
    return picked_idx, avail_counts, taken_counts


# ============================================================
# Mapeamento de categorias para grupos (RUN 6 — agrupamento)
# ============================================================
GROUP_TO_ID = {
    "cyl":   0,
    "bc":    1,
    "boi_c": 2,
    "boi_t": 3,
    "free":  4,
}

CAT_TO_GROUP_ID = {
    "cyl_lateral": GROUP_TO_ID["cyl"],
    "cyl_top":     GROUP_TO_ID["cyl"],
    "bc_inlet":    GROUP_TO_ID["bc"],
    "bc_outlet":   GROUP_TO_ID["bc"],
    "bc_xmin":     GROUP_TO_ID["bc"],
    "bc_xmax":     GROUP_TO_ID["bc"],
    "bc_ground":   GROUP_TO_ID["bc"],
    "bc_top":      GROUP_TO_ID["bc"],
    "boi_c":       GROUP_TO_ID["boi_c"],
    "boi_t":       GROUP_TO_ID["boi_t"],
    "freestream":  GROUP_TO_ID["free"],
}

# IDs como float32 pra serem usados em tensor
GID_CYL   = float(GROUP_TO_ID["cyl"])
GID_BC    = float(GROUP_TO_ID["bc"])
GID_BOI_C = float(GROUP_TO_ID["boi_c"])
GID_BOI_T = float(GROUP_TO_ID["boi_t"])
GID_FREE  = float(GROUP_TO_ID["free"])

print(f"Group IDs: cyl={GID_CYL}, bc={GID_BC}, boi_c={GID_BOI_C}, "
      f"boi_t={GID_BOI_T}, free={GID_FREE}")


# ================================================================================
# Normalização das saídas de turbulência
# ================================================================================
# Uma função só, para a Cell 5 e a Cell 9 não divergirem: se a
# desnormalização da rede não for o inverso exato disto, o treino compara
# coisas diferentes e nada avisa.

def normalize_k(k_raw):
    """k físico -> k normalizado, conforme K_NORM_MODE (Cell 1)."""
    if K_NORM_MODE == "logz":
        return (np.log(np.maximum(k_raw, 0.0) + K_EPS) - LNK_MEAN) / LNK_DP
    elif K_NORM_MODE == "scale":
        return np.maximum(k_raw, 0.0) / K_SCALE
    else:  # "z"
        return (k_raw - NORM_STATS["k"]["mean"]) / NORM_STATS["k"]["std"]


def denormalize_k(k_n):
    """Inverso exato de normalize_k. Usado na Cell 9/10."""
    if K_NORM_MODE == "logz":
        return np.exp(k_n * LNK_DP + LNK_MEAN) - K_EPS
    elif K_NORM_MODE == "scale":
        return k_n * K_SCALE
    else:
        return k_n * NORM_STATS["k"]["std"] + NORM_STATS["k"]["mean"]


def normalize_omega(om_raw):
    """ω físico -> ln ω normalizado (z-score)."""
    return (np.log(np.maximum(om_raw, OMEGA_FLOOR)) - LNW_MEAN) / LNW_DP


def denormalize_omega(om_n):
    """Inverso exato de normalize_omega."""
    return np.exp(om_n * LNW_DP + LNW_MEAN)


# round-trip: se isto falhar, a Cell 9 vai desnormalizar errado
_kt = np.array([1e-4, K_INLET, 1.0, 50.0, 500.0], dtype=np.float64)
_ot = np.array([1e-2, 1.0, OMEGA_INLET, 1e5], dtype=np.float64)
assert np.allclose(denormalize_k(normalize_k(_kt)), _kt, rtol=1e-6, atol=1e-8), \
    "normalize_k / denormalize_k não são inversas"
assert np.allclose(denormalize_omega(normalize_omega(_ot)), _ot, rtol=1e-6), \
    "normalize_omega / denormalize_omega não são inversas"
print(f"Round-trip normalize/denormalize (k, ω): OK")


# ================================================================================
# Verificações contra o dado — rodam no primeiro snapshot
# ================================================================================

def sst_F2_np(k, om, y):
    """
    F2 do SST em NumPy (eq. 4.119-4.120). Mesma fórmula que a Cell 10 usa em
    TF — aqui serve para reconstruir ν_t a partir do CSV.

        Φ2 = max( 2√k/(β*·ω·y) , 500ν/(y²·ω) )      F2 = tanh(Φ2²)
    """
    y  = np.maximum(np.asarray(y, np.float64), D_WALL_FLOOR)
    om = np.maximum(np.asarray(om, np.float64), OMEGA_FLOOR)
    kk = np.maximum(np.asarray(k, np.float64), 0.0)
    phi2 = np.maximum(2.0 * np.sqrt(kk) / (SST_BETA_STAR * om * y),
                      500.0 * NU / (y * y * om))
    return np.tanh(phi2 ** 2)


def verify_nu_t_formula(k, om, mut, d_wall, verbose=True):
    """
    (a) `viscosity-turb` é μ_t [kg/(m·s)] ou ν_t [m²/s]?
    (b) A RELAÇÃO CONSTITUTIVA INTEIRA fecha?

        ν_t = min( (k/ω) / max(1, S·F2/(a1·ω)) ,  VISC_RATIO_MAX·ν )

    ESTA FUNÇÃO SUBSTITUI verify_viscosity_convention. A antiga testava
    ν_t = k/ω e concluía, de uma mediana igual a 1, que o limitador de
    cisalhamento estava inativo. A conclusão era incompleta: onde o
    limitador não atua ν_t = k/ω vale mesmo, e como essa é a maioria das
    células, a mediana dá 1 com o limitador ativo em dezenas de milhares
    de outras. O job 6027 mediu a fórmula completa acertando 100% das
    211.579 células.

    O PROBLEMA: S não está no CSV de estado. Desde o job 6125 ele existe em
    deriv/ (strain-rate-mag), mas este treino não lê deriv/ — é a próxima
    variável (Cell 3). Sem S não dá para calcular ν_t direto. Dá para
    INVERTER, que responde a mesma pergunta:

      - o teto é verificável sozinho    (não depende de S);
      - k/ω é verificável sozinho       (não depende de S);
      - onde ν_t < k/ω sem teto, o limitador é a única explicação possível
        dentro do modelo, e a razão devolve S·F2 = a1·ω·(k/ω)/ν_t.
        Como F2 SÓ depende de k, ω e y — todos disponíveis — dá para
        recuperar o S implícito e perguntar se ele é plausível.

    O teste que isso permite, e que não é tautológico: o limitador do SST
    é um dispositivo de camada limite. Ele não pode disparar onde F2 ≈ 0.
    Se aparecerem células "limitadas" longe de qualquer parede, a fórmula
    (ou o alinhamento das colunas) está errada.

    Devolve dict com as frações de cada ramo — a Cell 10 precisa delas para
    saber o que esperar.
    """
    good = (k > K_FLOOR) & (om > OMEGA_FLOOR) & np.isfinite(mut)
    if good.sum() == 0:
        print("  [visc] nenhum ponto válido para o teste")
        return None
    kk = k[good].astype(np.float64)
    oo = om[good].astype(np.float64)
    mm = mut[good].astype(np.float64)

    # ---- (a) dinâmica ou cinemática? Os dois casos diferem por ρ = 1,225 ----
    r_dyn = np.median((mm / RHO_INF) * oo / kk)
    r_kin = np.median(mm * oo / kk)
    is_dyn = abs(r_dyn - 1.0) < abs(r_kin - 1.0)
    nu_t = (mm / RHO_INF) if is_dyn else mm

    if verbose:
        print(f"\n  [verificação a] convenção de viscosity-turb")
        print(f"    supondo DINÂMICA  (μ_t): mediana de ν_t/(k/ω) = {r_dyn:.6f}")
        print(f"    supondo CINEMÁTICA(ν_t): mediana de ν_t/(k/ω) = {r_kin:.6f}")
        print(f"    -> a coluna é {'DINÂMICA (μ_t)' if is_dyn else 'CINEMÁTICA (ν_t)'}"
              f"   (Cell 2 assumiu "
              f"{'DINÂMICA' if VISC_TURB_IS_DYNAMIC else 'CINEMÁTICA'})")
        if is_dyn != VISC_TURB_IS_DYNAMIC:
            print(f"    >>> DIVERGE de VISC_TURB_IS_DYNAMIC na Cell 2. Corrigir lá.")

    base  = kk / oo                       # k/ω, o ν_t sem limitador nem teto
    ratio = nu_t / base
    n = len(ratio)

    # ---- ν_t > k/ω é IMPOSSÍVEL: o max no denominador só REDUZ ----
    # Se aparecer em quantidade, a explicação não é física — é μ_t vindo de
    # uma célula diferente de k e ω (o defeito do job 5479, que o journal do
    # 5518 existe para evitar). Este teste vem primeiro: um dado desalinhado
    # também produz "ν_t ≠ k/ω" em toda parte, e concluir dali que o
    # limitador está ativo seria explicar um bug de leitura com física.
    n_viol = int((ratio > 1.0 + 1e-6).sum())
    if verbose:
        print(f"\n  [verificação b] relação constitutiva completa")
        print(f"    ν_t > k/ω (impossível no SST): {n_viol:,} de {n:,} "
              f"({100*n_viol/n:.3f}%)")
    if 100.0 * n_viol / n > 1.0:
        print(f"    >>> {100*n_viol/n:.1f}% dos pontos violam um limite que o")
        print(f"        modelo NÃO PODE violar. Suspeitar do desalinhamento de")
        print(f"        colunas do job 5479. NÃO interpretar o resto enquanto")
        print(f"        isso não estiver resolvido (a Cell 5b tem o teste cruzado).")

    # ---- os três ramos ----
    cap_val  = NU_T_MAX
    capped   = nu_t >= cap_val * (1.0 - VISC_LIMITED_REL_TOL)
    free     = (~capped) & (np.abs(ratio - 1.0) < 1e-4)
    limited  = (~capped) & (ratio < 1.0 - 1e-4)
    other    = ~(capped | free | limited)

    f_cap, f_free, f_lim = capped.mean(), free.mean(), limited.mean()
    if verbose:
        print(f"\n    {'ramo':<34}{'células':>11}{'%':>9}")
        print(f"    {'-'*54}")
        for _lab, _m in [(f"teto (ν_t = {cap_val:.4f} m²/s)", capped),
                         ("livre (ν_t = k/ω, ±1e-4)", free),
                         ("limitador de cisalhamento", limited),
                         ("não classificado", other)]:
            print(f"    {_lab:<34}{int(_m.sum()):>11,}{100*_m.mean():>9.3f}")

    # ---- o ramo do teto é verificável sem S ----
    if capped.any() and verbose:
        err_cap  = np.abs(nu_t[capped] / cap_val - 1.0)
        binds    = base[capped] > cap_val
        print(f"\n    RAMO DO TETO")
        print(f"      |ν_t/teto − 1| : p50={np.median(err_cap):.2e}  "
              f"max={err_cap.max():.2e}")
        print(f"      k/ω > teto (o teto realmente morde): "
              f"{100*binds.mean():.2f}% das células recortadas")
        if binds.mean() < 0.99:
            print(f"      >>> em {100*(1-binds.mean()):.1f}% delas k/ω NÃO passa do")
            print(f"          teto. Ou o teto não é {VISC_RATIO_MAX:.0e}, ou as")
            print(f"          colunas não são da mesma célula.")

    # ---- o ramo do limitador: recuperar S e testar plausibilidade ----
    out_S = None
    if limited.any():
        y_ok = d_wall is not None and len(d_wall) == len(k)
        if not y_ok:
            if verbose:
                print(f"\n    RAMO DO LIMITADOR: sem d_wall alinhado, F2 não é")
                print(f"      calculável — teste de plausibilidade pulado.")
        else:
            yy = np.asarray(d_wall, np.float64)[good]
            F2 = sst_F2_np(kk, oo, yy)
            # ν_t = (k/ω)·(a1·ω)/(S·F2)  ->  S = a1·ω/(ratio·F2)
            with np.errstate(divide="ignore", invalid="ignore"):
                S_imp = SST_A1 * oo[limited] / (ratio[limited]
                                                * np.maximum(F2[limited], 1e-30))
            F2_lim = F2[limited]
            y_lim  = yy[limited]
            out_S = S_imp
            if verbose:
                print(f"\n    RAMO DO LIMITADOR — S recuperado por inversão")
                print(f"      S = a1·ω/(ratio·F2), com F2 calculado de k, ω e y")
                print(f"      S implícito [1/s]: p10={np.percentile(S_imp,10):.4g}  "
                      f"p50={np.median(S_imp):.4g}  p90={np.percentile(S_imp,90):.4g}")
                print(f"      escala de referência V_inf/D = "
                      f"{V_INF/D_CYL:.4g} 1/s;  V_inf/y_p50 = "
                      f"{V_INF/max(np.median(y_lim),1e-6):.4g} 1/s")
                print(f"      F2 nas células limitadas: p10={np.percentile(F2_lim,10):.4f}"
                      f"  p50={np.median(F2_lim):.4f}")
                print(f"      d_wall nas células limitadas [m]: "
                      f"p50={np.median(y_lim):.3g}  p90={np.percentile(y_lim,90):.3g}"
                      f"   (domínio: p50={np.median(yy):.3g})")
                # O TESTE QUE NÃO É TAUTOLÓGICO
                n_far = int((F2_lim < 0.01).sum())
                print(f"      limitador ativo com F2 < 0,01 (longe de parede): "
                      f"{n_far:,} ({100*n_far/len(F2_lim):.2f}%)")
                if n_far > 0.05 * len(F2_lim):
                    print(f"      >>> o limitador do SST é um dispositivo de camada")
                    print(f"          limite: com F2 ≈ 0 ele não pode disparar. Mais de")
                    print(f"          5% assim significa que a fórmula, o d_wall ou o")
                    print(f"          alinhamento das colunas está errado.")
                else:
                    print(f"      -> coerente: o limitador só dispara onde F2 é")
                    print(f"         apreciável, como o modelo exige.")

    # ---- veredito sobre a configuração da Cell 1 ----
    if verbose:
        print(f"\n    VEREDITO")
        if f_lim < 1e-4 and f_cap < 1e-4:
            print(f"      ν_t = k/ω em praticamente todo o domínio.")
            print(f"      SST_SHEAR_LIMITER poderia ser False sem prejuízo —")
            print(f"      mas o job 6027 mostra o contrário na malha de")
            print(f"      produção. Desconfiar da amostra antes do modelo.")
        else:
            print(f"      {100*(f_lim+f_cap):.2f}% das células NÃO seguem ν_t = k/ω.")
            print(f"      SST_SHEAR_LIMITER = {SST_SHEAR_LIMITER} (Cell 1) e o teto")
            print(f"      têm que estar no resíduo da Cell 10, os dois.")
        print(f"\n    (as células onde o PRODUCTION LIMITER corta não são")
        print(f"     identificáveis daqui: P_k = μ_t·S² precisa de S, que não")
        print(f"     está no CSV. A Cell 10 as detecta no próprio resíduo, com")
        print(f"     o S da rede — é o que MASK_PROD_LIMITED_OMEGA liga.)")

    return dict(is_dynamic=bool(is_dyn),
                frac_capped=float(f_cap), frac_free=float(f_free),
                frac_limited=float(f_lim), frac_other=float(other.mean()),
                n_violations=n_viol, S_implied=out_S)


def verify_pressure_convention(x, y, z, P, k, band=None, verbose=True):
    """
    (e) A coluna `pressure` é p ou p̃ = p + (2/3)ρk?

    O job 6023 decidiu isso na malha da CAIXA, e decidiu bem. Este teste
    repete o argumento na malha de PRODUÇÃO — de graça, mas MUITO mais
    fraco, e é importante entender por quê antes de olhar o número.

    O RACIOCÍNIO. O pressure-outlet zera A VARIÁVEL QUE O SOLVER RESOLVE.
    Então, na faixa junto à saída:

        se a coluna for p̃  ->  p̃ ≈ 0, sem depender de k
        se a coluna for p   ->  p = p̃ − (2/3)ρk ≈ −0,8167·k

    O CONFUNDIMENTO, que é o motivo de a caixa ter existido: numa esteira,
    p e k são fisicamente correlacionados — os dois respondem ao déficit de
    velocidade. Regredir pressure contra k devolve inclinação não nula
    mesmo quando a coluna é p̃, e a inclinação espúria pode calhar de ficar
    perto de −0,8167. A caixa eliminou isso usando escoamento uniforme, em
    que a ÚNICA fonte de variação em k era o decaimento da turbulência.

    Por isso aqui não se AJUSTA uma inclinação: comparam-se duas hipóteses
    FIXAS, e só se declara vencedor quando uma explica a variância muito
    melhor que a outra (fator EXPLAIN_FACTOR). No resto dos casos, o
    veredito honesto é INCONCLUSIVO — que é o que este teste costuma dar
    numa esteira, e não é defeito dele.
    """
    EXPLAIN_FACTOR = 4.0
    band = band if band is not None else 10.0 * BC_TOL
    near = (Y_MAX - y) < band
    nn = int(near.sum())
    if nn < 100:
        if verbose:
            print(f"\n  [verificação e] só {nn} células a menos de {band:.0f} m "
                  f"da saída — teste pulado.")
        return None
    kk = k[near].astype(np.float64)
    pp = P[near].astype(np.float64)
    spread_k = float(kk.max() - kk.min())
    slope_static = -(2.0 / 3.0) * RHO_INF          # ≈ −0,8167 Pa/(m²/s²)

    # H0 (modificada): pressure não depende de k  ->  resíduo = pressure − média
    r_mod = pp - pp.mean()
    # H1 (estática):   pressure = −0,8167·k + const
    pred  = slope_static * (kk - kk.mean())
    r_sta = (pp - pp.mean()) - pred

    v_mod, v_sta = float(r_mod.var()), float(r_sta.var())
    # inclinação ajustada, só como informação
    A = np.stack([kk, np.ones_like(kk)], 1)
    coef, *_ = np.linalg.lstsq(A, pp, rcond=None)
    slope_fit = float(coef[0])
    r2_fit = 1.0 - (pp - A @ coef).var() / max(pp.var(), 1e-30)

    if v_sta * EXPLAIN_FACTOR < v_mod:
        verdict = "static"
    elif v_mod * EXPLAIN_FACTOR < v_sta:
        verdict = "modified"
    else:
        verdict = "inconclusive"

    if verbose:
        print(f"\n  [verificação e] a coluna `pressure` é p ou p̃?")
        print(f"    faixa: {nn:,} células a menos de {band:.0f} m de y={Y_MAX}")
        print(f"    k na faixa: p50={np.median(kk):.4g}  amplitude={spread_k:.4g}")
        print(f"    sinal que a hipótese ESTÁTICA prevê: "
              f"σ[0,8167·k] = {abs(slope_static)*kk.std():.4g} Pa")
        print(f"    variação observada de pressure:      "
              f"σ[p] = {pp.std():.4g} Pa")
        print(f"    variância residual de cada hipótese FIXA:")
        print(f"      p̃ (pressure ⟂ k) : {v_mod:.4g}")
        print(f"      p  (a = {slope_static:+.4f}) : {v_sta:.4g}")
        print(f"      razão = {v_sta/max(v_mod,1e-30):.3f}  "
              f"(precisa passar de {EXPLAIN_FACTOR:g}× para decidir)")
        print(f"    [informativo] inclinação ajustada a = {slope_fit:+.4f}, "
              f"R² = {r2_fit:.4f}")
        if spread_k < 1e-3:
            print(f"    -> k quase constante: o teste não separa nada. INCONCLUSIVO.")
            verdict = "inconclusive"
        elif verdict == "inconclusive":
            print(f"    -> INCONCLUSIVO, que é o esperado numa esteira: p e k são")
            print(f"       correlacionados por física, não pela definição de p̃.")
            print(f"       Quem decidiu isto foi a caixa do job 6023 "
                  f"(PRESSURE_IS_MODIFIED = {PRESSURE_IS_MODIFIED}).")
        else:
            _is_mod = (verdict == "modified")
            print(f"    -> a coluna é "
                  f"{'MODIFICADA (p̃)' if _is_mod else 'ESTÁTICA (p)'}, com "
                  f"margem de {max(v_sta/max(v_mod,1e-30), v_mod/max(v_sta,1e-30)):.1f}×")
            if _is_mod != PRESSURE_IS_MODIFIED:
                print(f"    >>> DIVERGE de PRESSURE_IS_MODIFIED na Cell 2, fixado")
                print(f"        pelo job 6023 com 12 ordens de grandeza de margem.")
                print(f"        Antes de mudar a Cell 2, desconfiar DESTE teste: o")
                print(f"        confundimento p–k na esteira é justamente o que a")
                print(f"        caixa foi feita para eliminar.")
            else:
                print(f"    -> concorda com o job 6023.")
    return dict(verdict=verdict, var_modified=v_mod, var_static=v_sta,
                slope_fit=slope_fit, r2_fit=float(r2_fit),
                n=nn, spread_k=spread_k)


# Tolerância RELATIVA para "a BC da PINN é a BC do solver". O arquivo traz
# k e ω com 6 a 8 algarismos e a Cell 2 os recalcula das mesmas fórmulas do
# Fluent, então a concordância esperada é na 5ª casa. 1e-3 deixa folga para
# arredondamento de escrita e ainda pega qualquer diferença de modelo (uma
# TI de 5,1% em vez de 5% já dá 4% em k).
INLET_BC_REL_TOL = 1.0e-3

# Se a BC do solver e a da PINN DIVERGEM, abortar. A hard constraint da
# Cell 9 é multiplicativa e sempre ganha da data loss: uma BC de inlet
# errada não se corrige treinando, só produz horas de treino contra um
# alvo que a arquitetura não deixa alcançar. Diferente das outras
# verificações, que diagnosticam e seguem, esta decide algo que nada depois
# conserta. A falta do ARQUIVO não aborta (ver HAS_INLET_FACES, Cell 3).
INLET_MISMATCH_ABORTS = True


def _read_inlet_faces(path, verbose=True):
    """
    Lê bc/inlet_faces_init.csv por NOME de coluna.

    O cabeçalho conferido tem 10 colunas, com cellnumber, x, y, z na frente
    e a ordem interna invertida em relação ao journal — não as 6 que o
    handoff do CFD descrevia. Ler por posição teria lido viscosity-ratio
    achando que era k. Delimitador detectado como nos snapshots (Cell 3).

    Devolve dict de arrays float64, ou None se faltar coluna obrigatória.
    """
    sep, _ = detect_delimiter(path)
    df = pd.read_csv(path, sep=sep, engine="python", skipinitialspace=True)
    df.columns = [_norm_col(c) for c in df.columns]
    need = {"x": "x-coordinate", "y": "y-coordinate", "z": "z-coordinate",
            "k": "turb-kinetic-energy", "omega": "specific-diss-rate"}
    opt  = {"mut": "viscosity-turb", "vr": "viscosity-ratio",
            "ti_col": "turb-intensity", "v": "y-velocity"}
    miss = [c for c in need.values() if c not in df.columns]
    if miss:
        if verbose:
            print(f"  [verificação c] {os.path.basename(path)} sem {miss}.")
            print(f"      colunas presentes: {list(df.columns)}")
        return None
    out = {k_: df[c].to_numpy(np.float64) for k_, c in need.items()}
    for k_, c in opt.items():
        if c in df.columns:
            out[k_] = df[c].to_numpy(np.float64)
    return out


def verify_inlet_turbulence(x, y, z, k, om, band=5.0, verbose=True):
    """
    (c) A BC de turbulência que a PINN impõe no inlet é a que o solver impôs?

    MODO "faces" (HAS_INLET_FACES): lê o valor de FACE exportado pelo solver
    e compara com o que a PINN usa — K_INLET/OMEGA_INLET (Cell 2) e
    KN_INLET/LNWN_INLET (Cell 4), que são o que a HardConstraintLayer
    (Cell 9) de fato impõe. É a comparação certa: a PINN impõe a BC em
    y = Y_MIN, e a face é exatamente y = Y_MIN.

    O que é conferido, em ordem de importância:
      1. geometria: todas as faces em y = Y_MIN, cobrindo x e z do domínio;
      2. uniformidade: k e ω constantes nas faces (o Fluent impõe um valor);
      3. k e ω contra K_INLET e OMEGA_INLET              -> DECIDE
      4. k e ω normalizados contra KN_INLET e LNWN_INLET -> DECIDE
         (pega erro de normalização na Cell 4, não só de valor na Cell 2)
      5. μ_t = ρk/ω, μ_t/μ, TI = √(2k/3)/V_INF e y-velocity, contra as
         constantes da Cell 2 e contra as colunas do próprio arquivo.

    ARMADILHA, documentada na Cell 3: a coluna `turb-intensity` NÃO é a TI
    física — o Fluent escreve √(2k/3)/U_ref com U_ref = 1 m/s. Ela entra
    aqui só como conferência cruzada, dividida por V_INF. E nas faces,
    `turb-intensity` e `viscosity-ratio` podem vir da CÉLULA vizinha (no
    inlet_faces_end.csv elas dão valores já decaídos). Por isso o veredito
    se lê de k e ω, que são o valor imposto.

    MODO "células" (sem o arquivo): mediana de k e ω nas células a menos de
    `band` m do inlet. Fica como diagnóstico de DECAIMENTO e não decide nada
    — com ω0 = 7419 1/s, ω cai à metade em L = V_INF/(β·ω0) ≈ 2,8 cm, então
    o valor de célula é sempre bem menor que o imposto.

    Devolve dict (guardado em INLET_CHECK) com o modo, o veredito e os
    números, para a Cell 16 levar aos metadados.
    """
    # --- o que as células mostram: decaimento, nos dois modos ---
    # Distância em que ω cai à metade, sem produção: dω/dt = −βω² ao longo
    # de x = V_INF·t dá ω(x) = ω0/(1 + x/L), L = V_INF/(β·ω0). β = β_i2
    # porque longe das paredes F1 ≈ 0.
    L_dec = V_INF / (SST_BETA_2 * OMEGA_INLET)
    near = (y - Y_MIN) < band
    n_cells = int(near.sum())
    k_cell = float(np.median(k[near])) if n_cells else float("nan")
    o_cell = float(np.median(om[near])) if n_cells else float("nan")

    faces = None
    if HAS_INLET_FACES:
        try:
            faces = _read_inlet_faces(INLET_FACES_CSV, verbose=verbose)
        except Exception as _exc:
            print(f"  [verificação c] falha ao ler {INLET_FACES_CSV}: {_exc}")
            faces = None

    # ======================================================================
    # MODO "células" — sem o arquivo de faces
    # ======================================================================
    if faces is None:
        if verbose:
            print(f"\n  [verificação c] modo CÉLULAS — sem o valor de face do "
                  f"solver, a BC NÃO é verificável")
            if n_cells:
                print(f"    {n_cells:,} células a menos de {band} m de y={Y_MIN}:")
                print(f"      k mediana {k_cell:10.4f}   (BC da PINN: {K_INLET:.4f}, "
                      f"razão {k_cell/K_INLET:.3f})")
                print(f"      ω mediana {o_cell:10.2f}   (BC da PINN: {OMEGA_INLET:.2f}, "
                      f"razão {o_cell/OMEGA_INLET:.3f})")
            print(f"    Valores de CÉLULA ficam abaixo da BC por construção: com")
            print(f"    ω_inlet = {OMEGA_INLET:.0f} 1/s, ω cai à metade em "
                  f"{100*L_dec:.1f} cm. Isto mede o")
            print(f"    decaimento, não a BC. Para verificar a BC, exportar")
            print(f"    bc/inlet_faces_init.csv.")
        return dict(mode="cells", ok=None, n_cells=n_cells,
                    k_cell_med=k_cell, omega_cell_med=o_cell)

    # ======================================================================
    # MODO "faces"
    # ======================================================================
    kf, of = faces["k"], faces["omega"]
    nf = len(kf)

    # 1. geometria: as faces estão no plano do inlet e cobrem a seção?
    y_dev = float(np.abs(faces["y"] - Y_MIN).max())
    x_lo, x_hi = float(faces["x"].min()), float(faces["x"].max())
    z_lo, z_hi = float(faces["z"].min()), float(faces["z"].max())
    geo_ok = (y_dev < 1e-3) and (x_hi - x_lo > 0.9 * (X_MAX - X_MIN)) \
             and (z_hi - z_lo > 0.9 * (Z_MAX - Z_MIN))

    # 2. uniformidade: o Fluent impõe UM valor; dispersão denuncia arquivo
    #    errado (p.ex. um export de células em vez de faces)
    k_med, o_med = float(np.median(kf)), float(np.median(of))
    k_spr = float((kf.max() - kf.min()) / max(abs(k_med), 1e-30))
    o_spr = float((of.max() - of.min()) / max(abs(o_med), 1e-30))
    uni_ok = (k_spr < INLET_BC_REL_TOL) and (o_spr < INLET_BC_REL_TOL)

    # 3. valor físico: solver × PINN
    e_k = abs(k_med / K_INLET - 1.0)
    e_o = abs(o_med / OMEGA_INLET - 1.0)
    val_ok = (e_k < INLET_BC_REL_TOL) and (e_o < INLET_BC_REL_TOL)

    # 4. o mesmo valor nas unidades da REDE. normalize_k/normalize_omega
    #    (acima nesta célula) são as funções que geram os alvos; KN_INLET e
    #    LNWN_INLET (Cell 4) são o que a HardConstraintLayer impõe. Se a
    #    Cell 4 tiver normalizado a BC com outra conta, aparece aqui.
    kn_face = float(normalize_k(np.array([k_med]))[0])
    on_face = float(normalize_omega(np.array([o_med]))[0])
    d_kn = abs(kn_face - KN_INLET)
    d_on = abs(on_face - LNWN_INLET)
    # em unidades normalizadas, 1e-3 relativo em k é ~1e-3/LNK_DP em kn
    norm_ok = (d_kn < 1e-2) and (d_on < 1e-2)

    # 5. grandezas derivadas, contra a Cell 2
    mut_face = RHO_INF * k_med / o_med
    vr_face  = mut_face / MU
    ti_face  = np.sqrt(2.0 * k_med / 3.0) / V_INF

    ok = bool(geo_ok and uni_ok and val_ok and norm_ok)

    if verbose:
        print(f"\n  [verificação c] BC de turbulência no inlet — modo FACES")
        print(f"    arquivo : {INLET_FACES_CSV}")
        print(f"    {nf:,} faces   |y − Y_MIN| máx = {y_dev:.2e} m   "
              f"x ∈ [{x_lo:.1f}, {x_hi:.1f}]   z ∈ [{z_lo:.1f}, {z_hi:.1f}]"
              f"   {'OK' if geo_ok else '<<< não é o plano do inlet'}")
        print(f"    uniformidade (max−min)/mediana:  k {k_spr:.1e}   ω {o_spr:.1e}"
              f"   {'OK' if uni_ok else '<<< NÃO uniforme'}")
        print(f"\n    {'':<14}{'solver (face)':>16}{'PINN':>16}{'erro rel.':>12}")
        print(f"    {'-'*58}")
        print(f"    {'k [m²/s²]':<14}{k_med:>16.6f}{K_INLET:>16.6f}{e_k:>12.1e}")
        print(f"    {'ω [1/s]':<14}{o_med:>16.4f}{OMEGA_INLET:>16.4f}{e_o:>12.1e}")
        print(f"    {'kn (rede)':<14}{kn_face:>16.6f}{KN_INLET:>16.6f}"
              f"{d_kn:>12.1e}  (abs)")
        print(f"    {'ln ωn (rede)':<14}{on_face:>16.6f}{LNWN_INLET:>16.6f}"
              f"{d_on:>12.1e}  (abs)")
        print(f"    {'-'*58}")
        print(f"    derivados de k e ω da face, contra a Cell 2:")
        print(f"      μ_t = ρk/ω = {mut_face:.5e} Pa·s"
              + (f"   (coluna viscosity-turb: {np.median(faces['mut']):.5e})"
                 if "mut" in faces else ""))
        print(f"      μ_t/μ      = {vr_face:.4f}      (Cell 2: {VISC_RATIO_INLET:g})"
              + (f"   (coluna viscosity-ratio: {np.median(faces['vr']):.4f})"
                 if "vr" in faces else ""))
        print(f"      TI         = {100*ti_face:.3f} %     (Cell 2: {100*TI_INLET:g} %)"
              + (f"   (coluna turb-intensity/V_INF: "
                 f"{100*np.median(faces['ti_col'])/V_INF:.3f} %)"
                 if "ti_col" in faces else ""))
        if "v" in faces:
            _vm = float(np.median(faces["v"]))
            print(f"      y-velocity = {_vm:.4f} m/s  (V_INF: {V_INF:g})"
                  f"{'' if abs(_vm/V_INF - 1) < 1e-3 else '   <<<'}")
        print(f"      (turb-intensity é √(2k/3)/U_ref com U_ref = 1 m/s; dividida")
        print(f"       por V_INF vira a TI física. Só conferência — o veredito é k e ω.)")

        if n_cells:
            print(f"\n    Para contraste, as CÉLULAS a menos de {band} m do inlet:")
            print(f"      k mediana {k_cell:.4f} ({k_cell/k_med:.3f}× a face)   "
                  f"ω mediana {o_cell:.1f} ({o_cell/o_med:.3f}× a face)")
            print(f"      É o decaimento (ω cai à metade em {100*L_dec:.1f} cm), "
                  f"não divergência.")
            print(f"      Era isto que a versão antiga desta verificação comparava")
            print(f"      com a BC, e por isso ela concluía 'TI≈1,63%, μ_t/μ≈24,8'.")

        print(f"\n    VEREDITO")
        if ok:
            print(f"      A BC que a PINN impõe no inlet é a que o solver impôs,")
            print(f"      em unidades físicas e nas unidades da rede.")
        else:
            print(f"      >>> NÃO bate:"
                  + ("" if geo_ok else " [geometria]")
                  + ("" if uni_ok else " [uniformidade]")
                  + ("" if val_ok else " [k/ω contra a Cell 2]")
                  + ("" if norm_ok else " [kn/ln ωn contra a Cell 4]"))
            if not val_ok:
                _ti = np.sqrt(2.0 * k_med / 3.0) / V_INF
                print(f"          a face implica TI = {100*_ti:.3f} % e μ_t/μ = "
                      f"{vr_face:.3f}: corrigir TI_INLET/VISC_RATIO_INLET")
                print(f"          na Cell 2 (KN_INLET/LNWN_INLET da Cell 4 seguem).")
            elif not norm_ok:
                print(f"          k e ω batem, mas a normalização da BC na Cell 4")
                print(f"          não é a mesma de normalize_k/normalize_omega.")

    result = dict(mode="faces", ok=ok, n_faces=nf,
                  k_face=k_med, omega_face=o_med,
                  err_rel_k=e_k, err_rel_omega=e_o,
                  kn_face=kn_face, lnwn_face=on_face,
                  visc_ratio_face=vr_face, ti_face=float(ti_face),
                  geo_ok=bool(geo_ok), uniform_ok=bool(uni_ok),
                  n_cells=n_cells, k_cell_med=k_cell, omega_cell_med=o_cell)

    if not ok and INLET_MISMATCH_ABORTS:
        raise RuntimeError(
            "verificação (c): a BC de turbulência do inlet da PINN NÃO é a do "
            "solver. A hard constraint da Cell 9 imporia o valor errado e a "
            "data loss não consegue corrigir. Ver a tabela acima. "
            "(INLET_MISMATCH_ABORTS = False para seguir mesmo assim.)")
    return result


def load_static_wall_distance(path, sep=None, verbose=True):
    """
    Lê cell-wall-distance do static_mesh.csv.

    O arquivo foi exportado DENTRO da sessão do solve, antes do laço, então
    está alinhado linha a linha com os t_*.csv — mesma malha, mesma ordem de
    células. É isso que permite juntar d_wall com k e ω por índice de linha,
    sem interpolar nada.
    """
    sep = sep if sep is not None else CSV_DELIMITER
    df = pd.read_csv(path, sep=sep, engine="python", skipinitialspace=True)
    df.columns = [re.sub(r'["\'﻿]', '', c).strip().lower() for c in df.columns]
    if "cell-wall-distance" not in df.columns:
        if verbose:
            print(f"  [wall law] {os.path.basename(path)} sem cell-wall-distance "
                  f"— verificação (d) pulada. Colunas: {list(df.columns)}")
        return None
    d = df["cell-wall-distance"].to_numpy(np.float64)
    if verbose:
        print(f"  [wall law] {len(d):,} distâncias lidas de "
              f"{os.path.basename(path)}")
    return d


# Nomes das 5 candidatas a parede mais próxima, NA MESMA ORDEM em que a
# Cell 7 monta o np.minimum.reduce. Trocar a ordem aqui sem trocar lá
# desalinharia os limiares por parede — por isso a lista mora num lugar só.
WALL_CANDIDATE_NAMES = ("ground", "top", "x_min", "x_max", "cilindro")


def nearest_wall_index(x, y, z):
    """Índice (0..4) da parede mais próxima, na ordem de WALL_CANDIDATE_NAMES."""
    xf = np.asarray(x, np.float64); yf = np.asarray(y, np.float64)
    zf = np.asarray(z, np.float64)
    r  = np.sqrt((xf - XC_CYL)**2 + (yf - YC_CYL)**2)
    d_cyl = np.sqrt(np.maximum(r - R_CYL, 0.0)**2
                    + np.maximum(zf - H_CYL, 0.0)**2)
    dists = np.stack([zf - Z_MIN, Z_MAX - zf, xf - X_MIN, X_MAX - xf, d_cyl], 1)
    return dists.argmin(axis=1)


def measure_first_layer(d_sorted, max_frac=0.10, min_gap_ratio=1.30,
                        min_count_frac=0.002):
    """
    Acha o topo da PRIMEIRA CAMADA de células numa lista ordenada de
    distâncias à parede.

    Numa malha com prism layers os centros de célula se agrupam em camadas,
    e entre camadas há um salto. A função procura o primeiro salto relativo
    grande na cauda inferior e põe o limiar dentro dele (média geométrica
    dos dois vizinhos, que é o meio do salto em escala log).

    Devolve (limiar, n_na_camada) ou (None, None) se não houver salto claro —
    caso em que quem chama tem que cair num múltiplo de y1, e dizer que caiu.
    """
    d = np.asarray(d_sorted, np.float64)
    if len(d) < 50:
        return None, None
    n_look = max(20, int(max_frac * len(d)))
    n_look = min(n_look, len(d) - 1)
    seg = d[:n_look + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        jumps = seg[1:] / np.maximum(seg[:-1], 1e-300)
    i_min = max(1, int(min_count_frac * len(d)))
    cand = np.where(jumps[i_min:] >= min_gap_ratio)[0]
    if len(cand) == 0:
        return None, None
    i = int(cand[0]) + i_min
    thr = float(np.sqrt(seg[i] * seg[i + 1]))
    return thr, int(i + 1)


# Política de escolha de Y_WALL_FUNCTION a partir das medidas por parede.
#   "per_wall"  -> mede a faixa de CADA parede; o escalar global vira o maior
#                  dos limiares (cobre a 1ª camada de todas), e o dicionário
#                  Y_BAND_BY_WALL fica disponível para uso ponto a ponto
#   "global_min"-> comportamento antigo: d_wall.min() do domínio inteiro
#   "manual"    -> não mexe no valor da Cell 2
Y_WALL_POLICY   = "per_wall"
WALL_BAND_MULT  = 1.5      # usado só onde a detecção de salto falhar

Y_FIRST_BY_WALL = {}       # parede -> distância do 1º centro de célula [m]
Y_BAND_BY_WALL  = {}       # parede -> topo da 1ª camada [m]


def wall_band_by_point(x, y, z):
    """
    Limiar da faixa de tratamento de parede para CADA ponto: o da parede mais
    próxima dele. É isto que a Cell 7 deve passar adiante junto com d_wall —
    um escalar único trata a parede mais grossa e a mais fina como iguais.

    Antes de verify_wall_law rodar, Y_BAND_BY_WALL está vazio e a função
    devolve o escalar Y_WALL_FUNCTION em todos os pontos.
    """
    x = np.asarray(x, np.float64)
    if not Y_BAND_BY_WALL:
        return np.full(x.shape, float(Y_WALL_FUNCTION))
    idx = nearest_wall_index(x, y, z)
    lut = np.array([Y_BAND_BY_WALL.get(nm, Y_WALL_FUNCTION)
                    for nm in WALL_CANDIDATE_NAMES], np.float64)
    return lut[idx]


def verify_wall_law(x, y, z, k, om, d_wall, verbose=True):
    """
    (d) Onde acaba a região de tratamento de parede, e o que vale nela.

    O QUE MUDOU NESTA VERSÃO (27/09/2026):

      1. A FAIXA É MEDIDA POR PAREDE, não pelo mínimo global. A versão
         anterior fazia Y_WALL_FUNCTION = d_wall.min(), ou seja, a célula
         mais fina da MALHA INTEIRA. Como as paredes têm refinos diferentes,
         a primeira camada das paredes mais grosseiras ficava FORA da faixa,
         e como os pontos de colocação são sorteados no volume, quase nenhum
         caía dentro: a máscara ligava para ~0% deles. Agora cada parede tem
         seu limiar, achado pelo salto entre a 1ª e a 2ª camada de células.

      2. A LEI LOG VIROU HIPÓTESE A TESTAR, POR OUTRO MOTIVO. O manual
         (§4.18.3, lido em 29/09/2026) mostra que a correlação
         y+-insensível DEVOLVE a lei log pura a y+ ~ 1e5 — o ramo viscoso
         fica 3 a 4 ordens abaixo. Não é aí que mora a dúvida.

         A dúvida é o fator (u_τ/u*)²:

             k = √β* · κ_w² · y² · ω² · (u_τ/u*)²

         que vale 1 em equilíbrio e cai onde há separação: u_τ → 0 mas u*
         continua finito, porque depende de k. Foi exatamente para isso que
         o Fluent introduziu u*. Logo, esperar razão ≈ 1 no chão a montante
         e nas laterais, e < 1 na parede do cilindro e no chão dentro da
         esteira. Se for isso, 'loglaw' é defensável POR PAREDE, e esta
         função já mede por parede.

         E mascarar continua obrigatório de qualquer jeito, mas por um
         motivo diferente do que estava escrito aqui: ω é prescrito na
         célula adjacente, e k, embora CONTINUE sendo transportado
         (∂k/∂n = 0), tem a produção trocada por P_k,wall, que depende de
         Δy — a altura da CÉLULA, que não existe num ponto de colocação.

    Escreve Y_WALL_FUNCTION, Y_FIRST_BY_WALL e Y_BAND_BY_WALL (globais).
    """
    global Y_WALL_FUNCTION
    if d_wall is None:
        return None
    if len(d_wall) != len(x):
        print(f"\n  [verificação d] static_mesh tem {len(d_wall):,} linhas e o "
              f"snapshot {len(x):,} — NÃO estão alinhados.")
        print(f"      A junção por índice seria inválida; verificação pulada.")
        print(f"      Os dois arquivos têm que vir da MESMA sessão do solve.")
        return None

    dw_all  = np.asarray(d_wall, np.float64)
    nearest = nearest_wall_index(x, y, z)
    kk = np.maximum(k.astype(np.float64), K_FLOOR)
    oo = np.maximum(om.astype(np.float64), OMEGA_FLOOR)

    if verbose:
        print(f"\n  [verificação d] faixa de tratamento de parede, POR PAREDE")
        print(f"    {'parede':<11}{'y1 [m]':>9}{'topo 1a':>9}{'como':>8}"
              f"{'células':>9}{'k/(C·y²ω²)':>12}{'u_τ [m/s]':>10}{'y+':>10}")
        print(f"    {'-'*78}")

    ratios_all = []
    per_wall = {}
    _strays = {}
    for i, nm in enumerate(WALL_CANDIDATE_NAMES):
        m = (nearest == i)
        if not m.any():
            continue
        d_i = np.sort(dw_all[m])
        # y1 ROBUSTO. O mínimo cru é frágil: basta UMA célula da camada de
        # outra parede cair perto desta para y1 despencar e levar o limiar
        # junto. Pula os 0,2% mais baixos, e avisa quando isso muda o valor.
        n_skip = min(max(0, int(0.002 * len(d_i))), max(len(d_i) - 10, 0))
        y1_raw = float(d_i[0])
        y1     = float(d_i[n_skip])
        d_use  = d_i[n_skip:]
        thr, n_layer = measure_first_layer(d_use)
        how = "salto"
        if thr is None:
            thr, how = WALL_BAND_MULT * y1, f"{WALL_BAND_MULT:g}·y1"
            n_layer = int((d_use <= thr).sum())
        Y_FIRST_BY_WALL[nm] = y1
        Y_BAND_BY_WALL[nm]  = float(thr)

        mb = m & (dw_all <= thr)
        if mb.any():
            dwb   = dw_all[mb]
            ratio = kk[mb] / np.maximum(WALL_LAW_C * dwb**2 * oo[mb]**2, 1e-300)
            u_tau = oo[mb] * np.sqrt(SST_BETA_STAR) * SST_KAPPA * dwb
            yplus = dwb * u_tau / NU
            ratios_all.append(ratio)
            per_wall[nm] = dict(y1=y1, band=float(thr), n=int(mb.sum()),
                                ratio_med=float(np.median(ratio)))
            _flag = "*" if y1_raw < 0.9 * y1 else " "
            if verbose:
                print(f"    {nm:<11}{y1:>9.3f}{_flag}{thr:>8.3f}{how:>8}"
                      f"{int(mb.sum()):>9,}{np.median(ratio):>12.4f}"
                      f"{np.median(u_tau):>10.4f}{np.median(yplus):>10.2e}")
        elif verbose:
            print(f"    {nm:<11}{y1:>9.3f} {thr:>8.3f}{how:>8}{0:>9}"
                  f"{'—':>12}{'—':>10}{'—':>10}")
        if y1_raw < 0.9 * y1:
            _strays[nm] = (y1_raw, y1)

    if not Y_BAND_BY_WALL:
        if verbose:
            print(f"    nenhuma parede com células. Verificação inconclusiva.")
        return None

    # ---- o escalar global, para quem não puder usar o limiar por parede ----
    y_band_max = max(Y_BAND_BY_WALL.values())
    y_band_min = min(Y_BAND_BY_WALL.values())
    y_first_global = float(dw_all.min())
    if Y_WALL_POLICY == "per_wall":
        Y_WALL_FUNCTION = float(y_band_max)
    elif Y_WALL_POLICY == "global_min":
        Y_WALL_FUNCTION = y_first_global

    if verbose:
        print(f"    {'-'*78}")
        print(f"    'como' = o limiar veio do salto entre a 1ª e a 2ª camada,")
        print(f"             ou de {WALL_BAND_MULT:g}·y1 quando não há salto claro.")
        if _strays:
            print(f"    '*' = há célula mais perto que o y1 robusto (0,2% mais")
            print(f"          baixos descartados). Normal quando a camada de uma")
            print(f"          parede encosta na de outra, p.ex. no rodapé:")
            for _nm, (_a, _b) in _strays.items():
                print(f"            {_nm:<10} mínimo cru {_a:.3f} m, "
                      f"y1 robusto {_b:.3f} m")
        print(f"\n    Limiar por parede: de {y_band_min:.3f} m a {y_band_max:.3f} m "
              f"({y_band_max/max(y_band_min,1e-12):.1f}× de diferença)")
        print(f"    Y_WALL_POLICY = '{Y_WALL_POLICY}'  ->  "
              f"Y_WALL_FUNCTION = {Y_WALL_FUNCTION:.3f} m")
        print(f"    (o critério antigo, d_wall.min() global, daria "
              f"{y_first_global:.3f} m)")
        if Y_WALL_POLICY == "per_wall":
            _extra = int(((dw_all <= y_band_max) & (dw_all > wall_band_by_point(x, y, z))).sum())
            print(f"    Um limiar ESCALAR de {y_band_max:.3f} m cobriria "
                  f"{int((dw_all <= y_band_max).sum()):,} células, sendo")
            print(f"    {_extra:,} delas de 2ª camada ou além, em paredes mais")
            print(f"    finas. Use wall_band_by_point() para evitar isso — é a")
            print(f"    razão de Y_BAND_BY_WALL existir.")

    ratio_all = np.concatenate(ratios_all)
    med = float(np.median(ratio_all))

    if verbose:
        print(f"\n    A LEI LOG VALE? razão k/(√β*κ²·y²·ω²), alvo 1")
        print(f"      p10={np.percentile(ratio_all,10):.4f}  p50={med:.4f}  "
              f"p99={np.percentile(ratio_all,99):.4f}")
        _spread = per_wall and (max(v['ratio_med'] for v in per_wall.values())
                                / max(min(v['ratio_med'] for v in per_wall.values()), 1e-12))
        if _spread:
            print(f"      dispersão entre paredes: {_spread:.1f}×")
        if abs(med - 1.0) < 0.25:
            # [18/10/2026] Esta razão ≈ 1 NÃO decide nada sozinha. Se ω está
            # congelado em ω_wall na 1ª camada — o que a verificação (f),
            # logo abaixo, mede —, ela vale 1 POR CONSTRUÇÃO, e 'loglaw'
            # imporia a definição do tratamento de parede, não física. A
            # versão anterior concluía "A RELAÇÃO VALE" e marcava 'mask'
            # como "REVER"; na calibração da sst3 (job 6389) a (f) mostrou
            # ω congelado nas 5 paredes, ou seja, 'mask' era o certo.
            print(f"      -> razão ≈ 1. Isto só é evidência da lei log se ω NÃO")
            print(f"         estiver congelado em ω_wall na 1ª camada: se estiver,")
            print(f"         a razão vale 1 por construção. Quem decide é a")
            print(f"         verificação (f), logo abaixo. Com ω congelado, 'mask'")
            print(f"         é o certo; com ω transportado, 'loglaw' é defensável.")
        else:
            print(f"      >>> NÃO VALE NA MÉDIA (mediana {med:.3f}).")
            print(f"          O responsável é o fator (u_τ/u*)², que cai onde há")
            print(f"          separação. OLHAR A COLUNA POR PAREDE acima antes de")
            print(f"          desistir: se 'ground', 'x_min', 'x_max' e 'top'")
            print(f"          derem ≈1 e só 'cilindro' destoar, 'loglaw' ainda")
            print(f"          serve para as primeiras — o limiar já é por parede,")
            print(f"          o modo também poderia ser. Na dúvida, manter 'mask'.")
        # Veredito sobre o modo: com razão ≈ 1 ele depende da (f), que roda
        # depois; aqui só se aponta. Com razão longe de 1, 'loglaw' não tem
        # apoio no dado e só 'mask' (ou 'off') é coerente.
        if abs(med - 1.0) < 0.25:
            _wm = "decidido pela (f)"
        else:
            _wm = "COERENTE" if WALL_RESIDUAL_MODE != "loglaw" else "REVER"
        print(f"      Cell 2 está em WALL_RESIDUAL_MODE = '{WALL_RESIDUAL_MODE}'"
              f"  ->  {_wm}")

        # ---- as cascas de amostragem têm célula? ----
        print(f"\n    As cascas de amostragem contêm células?")
        for nm_slab, tol, cats in [("adj_* planas", BC_TOL,
                                    "bc_ground/bc_top/bc_xmin/bc_xmax"),
                                   ("adj_cyl_*", CYL_TOL, "cyl_lateral/cyl_top")]:
            n_in = int((dw_all < tol).sum())
            if n_in == 0:
                print(f"      {nm_slab:<14} {tol:.1f} m  ->  0 células  <<<")
                print(f"        VAZIA. As categorias {cats}")
                print(f"        não recebem NENHUM ponto de dado — a Cell 5b vai")
                print(f"        mostrar 100% de déficit e aquelas zonas ficam sem")
                print(f"        supervisão.")
            else:
                print(f"      {nm_slab:<14} {tol:.1f} m  ->  {n_in:,} células "
                      f"({100*n_in/len(dw_all):.2f}% do domínio)")

    return dict(y_first_global=y_first_global,
                y_first_by_wall=dict(Y_FIRST_BY_WALL),
                y_band_by_wall=dict(Y_BAND_BY_WALL),
                y_wall_function=float(Y_WALL_FUNCTION),
                ratio_med=med,
                ratio_p10=float(np.percentile(ratio_all, 10)),
                ratio_p99=float(np.percentile(ratio_all, 99)),
                per_wall=per_wall,
                n_cells_bc_tol=int((dw_all < BC_TOL).sum()),
                n_cells_cyl_tol=int((dw_all < CYL_TOL).sum()))


def verify_omega_wall(x, y, z, u, v, w, k, om, d_wall, verbose=True):
    """
    (f) Na célula colada à parede, o ω do CSV é o ω_wall imposto pelo
    tratamento de parede, ou saiu da equação de transporte?

    POR QUE ISSO IMPORTA, e não é só curiosidade de implementação:

      1. Decide se o resíduo de ω deve ser mascarado na faixa. Se o valor
         é imposto, a equação de transporte não vale ali e impor o resíduo
         é impor uma equação que o dado não satisfaz. Se é transportado (e
         ω_wall entra só como fluxo na face), o resíduo VALE e mascarar
         joga fora ~8% dos pontos de colocação — os de maior gradiente.

      2. MUDA COMO SE LÊ A VERIFICAÇÃO (d). Se ω é congelado em
         ω_wall = u*/(√β*·κ_w·y) com u* = C_μ^{1/4}√k, então

             k / (√β*·κ_w²·y²·ω²) = k·√β*/u*² = k·√β*/(√β*·k) = 1

         IDENTICAMENTE, por construção. Ou seja: a razão da verificação (d)
         dar ~1 na primeira camada não seria evidência de que a lei log
         vale — seria tautologia. E ligar WALL_RESIDUAL_MODE='loglaw' ali
         não imporia física, imporia a definição do tratamento de parede.
         (O que não é inútil — a rede tem que reproduzir o dado — mas é
         outra coisa, e a dissertação não pode chamar de validação.)

    COMO SE MEDE (eqs. 4.394-4.407 do Theory Guide):

        u*      = [ (μ|ΔU_tan/Δy| + C_μ^{1/2}·ρ·k) / ρ ]^{1/2}        (4.399)
        y⁺      = Δy·ρ·u*/μ                                            (4.398)
        ω⁺_lam  = C_calib·6/(β_i·(y⁺)²)          C_calib = 1/3         (4.406)
        ω⁺_turb = 1/(√β*_∞ · κ_w · y⁺)                                 (4.407)
        ω⁺      = ω⁺_lam·[1 + (ω⁺_turb/ω⁺_lam)^C_exp]^(1/C_exp)        (4.405)
        ω_wall  = ρ(u*)²/μ · ω⁺                                        (4.404)

    ΔU_tan é a velocidade TANGENTE à parede, relativa à parede: lateral_min,
    lateral_max e top se movem a V_INF em +y (Cell 2), então ali a
    diferença não é |V|. A y⁺ ~ 1e5 esse termo é ~400× menor que C_μ^{1/2}ρk
    e quase não muda u*, mas calcular certo custa dez linhas e tira uma
    ressalva do resultado.
    """
    if d_wall is None or Y_BAND_BY_WALL == {}:
        if verbose:
            print(f"\n  [verificação f] sem d_wall alinhado ou sem as faixas "
                  f"medidas em (d) — pulada.")
        return None
    if len(d_wall) != len(x):
        if verbose:
            print(f"\n  [verificação f] static_mesh desalinhado — pulada.")
        return None

    C_CALIB, C_EXP = 1.0 / 3.0, 1.3
    xf = x.astype(np.float64); yf = y.astype(np.float64); zf = z.astype(np.float64)
    uf = u.astype(np.float64); vf = v.astype(np.float64); wf = w.astype(np.float64)
    kk = np.maximum(k.astype(np.float64), K_FLOOR)
    oo = np.maximum(om.astype(np.float64), OMEGA_FLOOR)
    dw = np.asarray(d_wall, np.float64)
    idx = nearest_wall_index(xf, yf, zf)

    # --- normal de cada parede e velocidade DELA ---
    nrm = np.zeros((len(xf), 3))
    nrm[idx == 0] = (0.0, 0.0,  1.0)      # ground
    nrm[idx == 1] = (0.0, 0.0, -1.0)      # top
    nrm[idx == 2] = (1.0, 0.0,  0.0)      # x_min
    nrm[idx == 3] = (-1.0, 0.0, 0.0)      # x_max
    _m = (idx == 4)                        # cilindro: radial, ou vertical no topo
    if _m.any():
        _r = np.sqrt((xf[_m] - XC_CYL)**2 + (yf[_m] - YC_CYL)**2)
        _vert = zf[_m] > H_CYL
        _n = np.stack([(xf[_m] - XC_CYL) / np.maximum(_r, 1e-12),
                       (yf[_m] - YC_CYL) / np.maximum(_r, 1e-12),
                       np.zeros(_m.sum())], 1)
        _n[_vert] = (0.0, 0.0, 1.0)
        nrm[_m] = _n

    # paredes MÓVEIS: top, x_min, x_max andam a V_INF em +y (Cell 2)
    v_wall = np.zeros((len(xf), 3))
    v_wall[np.isin(idx, [1, 2, 3]), 1] = V_INF

    dV = np.stack([uf, vf, wf], 1) - v_wall
    dV_n = np.sum(dV * nrm, axis=1)
    dU_tan = np.sqrt(np.maximum(np.sum(dV * dV, axis=1) - dV_n**2, 0.0))

    # --- u*, y+, ω_wall ---
    y_saf = np.maximum(dw, D_WALL_FLOOR)
    term_visc = MU * dU_tan / y_saf
    term_turb = np.sqrt(SST_BETA_STAR) * RHO_INF * kk      # C_mu^{1/2}·ρ·k
    u_star = np.sqrt((term_visc + term_turb) / RHO_INF)
    y_plus = y_saf * RHO_INF * u_star / MU
    om_p_lam  = C_CALIB * 6.0 / (SST_BETA_1 * y_plus**2)
    om_p_turb = 1.0 / (np.sqrt(SST_BETA_STAR) * KAPPA_WALL * y_plus)
    om_p = om_p_lam * (1.0 + (om_p_turb / om_p_lam)**C_EXP)**(1.0 / C_EXP)
    om_wall = RHO_INF * u_star**2 / MU * om_p

    # --- só a PRIMEIRA CAMADA de cada parede (as faixas medidas em (d)) ---
    band = np.array([Y_BAND_BY_WALL.get(nm, Y_WALL_FUNCTION)
                     for nm in WALL_CANDIDATE_NAMES])[idx]
    first = dw <= band
    if not first.any():
        if verbose:
            print(f"\n  [verificação f] nenhuma célula na 1ª camada — pulada.")
        return None

    ratio = oo / np.maximum(om_wall, 1e-300)
    out = dict(frac_visc_in_ustar=float(np.median(
                   term_visc[first] / (term_visc[first] + term_turb[first]))),
               n_first=int(first.sum()))

    # VEREDITO POR PAREDE, não global. Uma mediana global mente quando as
    # paredes têm contagens muito diferentes: no teste de controle, 3 de 5
    # paredes batiam em 1,0000 e a mediana global deu 1,0000 — escondendo
    # que duas não batiam. E, quando UMA parede destoa, a causa mais
    # provável não é o solver: é a faixa daquela parede, medida em (d),
    # ter pegado 2ª camada junto.
    if verbose:
        print(f"\n  [verificação f] ω da 1ª camada × ω_wall das eqs. 4.404-4.407")
        print(f"    {'parede':<11}{'células':>9}{'y+ p50':>11}{'ω_CSV p50':>12}"
              f"{'ω_wall p50':>12}{'razão p50':>11}{'p10':>9}{'p90':>9}  veredito")
        print(f"    {'-'*88}")
    por_parede = {}
    for i, nm in enumerate(WALL_CANDIDATE_NAMES):
        m = first & (idx == i)
        if not m.any():
            continue
        r = ratio[m]
        md = float(np.median(r))
        sp = float(np.percentile(r, 90) - np.percentile(r, 10))
        bate = abs(md - 1.0) < 0.02 and sp < 0.10
        por_parede[nm] = dict(n=int(m.sum()), med=md, spread=sp, congelado=bate)
        out[nm] = md
        if verbose:
            print(f"    {nm:<11}{int(m.sum()):>9,}{np.median(y_plus[m]):>11.2e}"
                  f"{np.median(oo[m]):>12.4g}{np.median(om_wall[m]):>12.4g}"
                  f"{md:>11.4f}{np.percentile(r,10):>9.4f}"
                  f"{np.percentile(r,90):>9.4f}  "
                  f"{'CONGELADO' if bate else 'não bate'}")

    n_ok = sum(1 for v in por_parede.values() if v["congelado"])
    n_w  = len(por_parede)
    out["por_parede"] = por_parede
    out["n_paredes_congeladas"] = n_ok
    out["n_paredes"] = n_w
    out["congelado"] = bool(n_w > 0 and n_ok == n_w)

    if verbose:
        print(f"    {'-'*88}")
        print(f"    {n_ok} de {n_w} paredes com ω congelado em ω_wall")
        print(f"    a parte viscosa de u* é {100*out['frac_visc_in_ustar']:.3f}% "
              f"do total — por isso u* ≈ C_μ^(1/4)·√k a y+ ~ 1e5")
        if n_ok == n_w and n_w > 0:
            print(f"\n    -> ω ESTÁ CONGELADO em ω_wall na célula adjacente, em")
            print(f"       todas as paredes. Mascarar o resíduo de ω na faixa é")
            print(f"       obrigatório: WALL_RESIDUAL_MODE='mask' está certo.")
            print(f"       ATENÇÃO À VERIFICAÇÃO (d): com ω congelado, a razão")
            print(f"       k/(C·y²ω²) vale 1 POR CONSTRUÇÃO na 1ª camada")
            print(f"       (k·√β*/u*² = 1 com u* = C_μ^(1/4)√k). Se ela deu ~1")
            print(f"       lá em cima, NÃO é validação da lei log — é")
            print(f"       identidade. Não citar como evidência física.")
        elif n_ok == 0:
            print(f"\n    -> ω NÃO bate com ω_wall em parede nenhuma.")
            print(f"       Hipótese: ω_wall entra como valor de FACE e o ω da")
            print(f"       célula sai do transporte. Se for isso, mascarar o")
            print(f"       resíduo de ω joga fora ~8% dos pontos de colocação,")
            print(f"       justamente os de maior gradiente — vale testar")
            print(f"       WALL_RESIDUAL_MODE com ω ativo.")
        else:
            print(f"\n    -> MISTO: {n_ok} parede(s) batem e "
                  f"{n_w - n_ok} não.")
            print(f"       O Fluent aplica o MESMO tratamento em todas as")
            print(f"       paredes, então isto quase certamente não é o solver:")
            print(f"       é a faixa de (d) das paredes que não bateram ter")
            print(f"       pegado 2ª camada junto. Da 2ª camada em diante ω é")
            print(f"       transportado e a razão sobe. Conferir a coluna")
            print(f"       'topo 1a' daquelas paredes na verificação (d) antes")
            print(f"       de concluir qualquer coisa sobre o solver.")
            print(f"       Paredes que não bateram: "
                  f"{[k_ for k_, v in por_parede.items() if not v['congelado']]}")
    return out


# Lido uma vez: o static_mesh é estático por definição.
D_WALL_STATIC = None
if HAS_STATIC_MESH:
    try:
        D_WALL_STATIC = load_static_wall_distance(static_mesh_path)
    except Exception as _exc:
        print(f"  [wall law] falha ao ler {static_mesh_path}: {_exc}")


def load_all_snapshots(snaps,
                       quotas=QUOTAS,
                       max_snapshots=MAX_SNAPSHOTS,
                       seed=SEED,
                       keep_first_raw=True,
                       verbose_every=20):
    """
    Stream-load com sampling estratificado por categoria.

    Returns:
      data_np        : (N_total, 13) float32
                       [xn,yn,zn,tn,un,vn,wn,Pn,kn,omn,gid,vlim,d_wall]
                       As 10 primeiras normalizadas; gid e vlim são
                       marcadores; d_wall está em METROS.
      snap_offsets   : list[(t_phys, start, end)]
      first_snap_raw : dict com arrays brutos do primeiro snapshot, ou None
      stats_df       : DataFrame com (snapshot, t_phys, category, available, taken, quota)
                       — usado para a tabela agregada (Cell 5b).
    """
    if max_snapshots is not None:
        snaps = snaps[:max_snapshots]
    n_snaps = len(snaps)
    if n_snaps == 0:
        raise RuntimeError("No snapshots to load.")

    pts_per_snap = sum(quotas.values())
    cap = n_snaps * pts_per_snap
    data_np = np.empty((cap, N_COLS_DATA), dtype=np.float32)
    snap_offsets = []
    write_ptr = 0

    col_map = None
    P_offset = None
    first_snap_raw = None
    stats_records = []
    rng_local = np.random.default_rng(seed)

    n_vlim_total = 0
    n_k_nonpos   = 0
    n_w_nonpos   = 0

    print(f"Loading {n_snaps} snapshots with stratified sampling.")
    print(f"Quotas per snap (total = {pts_per_snap:,}):")
    for k, v in quotas.items():
        print(f"  {k:<14} {v:>6,}")
    ram_mb = cap * N_COLS_DATA * 4 / 1e6
    print(f"Upper-bound RAM ~{ram_mb:.0f} MB ({ram_mb/1024:.2f} GB)"
          f"   [{N_COLS_DATA} colunas; era 9 no laminar]")
    t0 = time.time()

    for i, (t_phys, path) in enumerate(snaps):
        try:
            data, col_map = load_snapshot(path, col_map=col_map)
        except Exception as exc:
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue

        x, y, z = data["x"], data["y"], data["z"]
        u, v, w = data["u"], data["v"], data["w"]
        P       = data["P"]
        kt      = data["k"]
        om      = data["omega"]
        mut     = data["mut"]

        if i == 0:  # só no primeiro snapshot
            print(f"\n[DEBUG] Primeiro snapshot bruto:")
            print(f"  P range: [{P.min():.2f}, {P.max():.2f}]")
            print(f"  P percentis: 1%={np.percentile(P,1):.2f}, "
                f"50%={np.percentile(P,50):.2f}, 99%={np.percentile(P,99):.2f}")
            print(f"  Pontos com |P| > 1000 Pa: {(np.abs(P) > 1000).sum()}")
            print(f"  Pontos com |P| > 3000 Pa: {(np.abs(P) > 3000).sum()}")
            print(f"  Pontos com |P| > 5000 Pa: {(np.abs(P) > 5000).sum()}")
            print(f"  k range: [{kt.min():.4g}, {kt.max():.4g}] m²/s²")
            print(f"  ω range: [{om.min():.4g}, {om.max():.4g}] 1/s")
            print(f"  μ_t range: [{mut.min():.4g}, {mut.max():.4g}]")

            # --- as seis verificações ---
            # A ordem importa: (d) é a que mede a faixa por parede e escreve
            # Y_BAND_BY_WALL, e (b) usa d_wall para calcular F2. As duas leem
            # o mesmo D_WALL_STATIC, então (d) pode vir depois sem problema,
            # mas (b) sem d_wall alinhado perde o teste de plausibilidade.
            globals()["NU_T_CHECK"] = verify_nu_t_formula(
                kt, om, mut, D_WALL_STATIC)
            # (c) lê o arquivo de faces do inlet; com BC divergente ABORTA
            # (INLET_MISMATCH_ABORTS), porque a hard constraint não deixaria
            # o treino corrigir. É barata e roda antes de todo o laço.
            globals()["INLET_CHECK"] = verify_inlet_turbulence(
                x, y, z, kt, om)
            globals()["WALL_LAW_CHECK"] = verify_wall_law(
                x, y, z, kt, om, D_WALL_STATIC)
            globals()["PRESSURE_CHECK"] = verify_pressure_convention(
                x, y, z, P, kt)
            # (f) depois de (d): usa Y_BAND_BY_WALL, que (d) escreve
            globals()["OMEGA_WALL_CHECK"] = verify_omega_wall(
                x, y, z, u, v, w, kt, om, D_WALL_STATIC)

        # Auto-detect pressure convention (absolute vs gauge) uma vez
        if P_offset is None:
            if P.mean() > 1e4:
                P_offset = P_OP
                print(f"  Pressure mean={P.mean():.1f} Pa -> absolute "
                      f"(subtract P_op={P_OP})")
            else:
                P_offset = 0.0
                print(f"  Pressure mean={P.mean():.3f} Pa -> gauge")
        P = P - P_offset

        # k e ω têm que ser positivos; contar violações e aplicar piso
        n_k_nonpos += int((kt <= 0).sum())
        n_w_nonpos += int((om <= 0).sum())

        # Marcar células onde ν_t bateu no teto (ver Cell 2).
        #
        # ERA um critério de exclusão; virou diagnóstico. O teto faz parte da
        # relação constitutiva que o solver usou, então, com o mesmo min no
        # resíduo da Cell 10, essas células fecham a equação normalmente.
        # A coluna continua porque é barata e responde a perguntas que
        # importam: ONDE elas estão, e se a fração cresce com o tempo (no
        # job 6027 ia de 3 células no passo 1 a 9.091 no passo 300).
        visc_ratio = mut / MU if VISC_TURB_IS_DYNAMIC else mut / NU
        vlim = (visc_ratio >= VISC_RATIO_MAX * (1.0 - VISC_LIMITED_REL_TOL))
        n_vlim_total += int(vlim.sum())

        # Distância à parede POR CÉLULA. Preferência pelo static_mesh.csv,
        # que é o que o Fluent de fato usou em F1 e F2; a fórmula analítica
        # entra só se o arquivo faltar ou não estiver alinhado. Tem que ser
        # montada ANTES do corte do cilindro, para o índice ainda casar com
        # as linhas do CSV.
        if (D_WALL_STATIC is not None) and (len(D_WALL_STATIC) == len(x)):
            dwall = D_WALL_STATIC.astype(np.float32)
        else:
            _r = np.sqrt((x - XC_CYL)**2 + (y - YC_CYL)**2)
            _dc = np.sqrt(np.maximum(_r - R_CYL, 0.0)**2
                          + np.maximum(z - H_CYL, 0.0)**2)
            dwall = np.minimum.reduce([z - Z_MIN, Z_MAX - z,
                                       x - X_MIN, X_MAX - x, _dc]).astype(np.float32)
            if i == 0:
                print(f"  [d_wall] static_mesh indisponível ou desalinhado — "
                      f"usando a fórmula analítica")

        # Excluir interior sólido do cilindro
        inside = is_inside_cylinder(x, y, z)
        if inside.any():
            keep = ~inside
            x, y, z = x[keep], y[keep], z[keep]
            u, v, w = u[keep], v[keep], w[keep]
            P     = P[keep]
            kt    = kt[keep]
            om    = om[keep]
            vlim  = vlim[keep]
            dwall = dwall[keep]

        # Classificar e amostrar
        category = classify_points(x, y, z)
        picked, avail_counts, taken_counts = _stratified_sample(category, quotas, rng_local)

        for cat_name in quotas:
            stats_records.append({
                "snapshot": i,
                "t_phys": t_phys,
                "category": cat_name,
                "available": avail_counts.get(cat_name, 0),
                "taken": taken_counts.get(cat_name, 0),
                "quota": quotas[cat_name],
            })

        n_take = len(picked)
        end = write_ptr + n_take
        sl = slice(write_ptr, end)

        data_np[sl, COL_XN] = (x[picked] - X_MEAN) / DP_ISO
        data_np[sl, COL_YN] = (y[picked] - Y_MEAN) / DP_ISO
        data_np[sl, COL_ZN] = (z[picked] - Z_MEAN) / DP_ISO
        data_np[sl, COL_TN] = (t_phys - T_MEAN) / T_DP
        data_np[sl, COL_UN] = (u[picked]) / V_REF
        data_np[sl, COL_VN] = (v[picked] - V_MEAN) / V_DP
        data_np[sl, COL_WN] = (w[picked]) / V_REF
        data_np[sl, COL_PN] = (P[picked] - P_MEAN) / P_DP
        # turbulência: log ANTES do z-score (ver Cell 4)
        data_np[sl, COL_KN]  = normalize_k(kt[picked])
        data_np[sl, COL_OMN] = normalize_omega(om[picked])

        # Coluna 10: group_id (RUN 6)
        cat_picked = category[picked]
        group_ids = np.array(
            [CAT_TO_GROUP_ID[c] for c in cat_picked],
            dtype=np.float32
        )
        data_np[sl, COL_GID] = group_ids

        # Coluna 11: marcador de ν_t no teto (diagnóstico)
        data_np[sl, COL_VLIM] = vlim[picked].astype(np.float32)

        # Coluna 12: distância à parede em METROS, não normalizada.
        # Em metros de propósito: é assim que entra em F1, F2 e no critério
        # da faixa de parede. Normalizá-la só criaria uma chance a mais de
        # desnormalizar errado.
        data_np[sl, COL_DWALL] = dwall[picked]

        snap_offsets.append((t_phys, write_ptr, end))
        write_ptr = end

        if keep_first_raw and first_snap_raw is None:
            first_snap_raw = dict(
                x=x.copy(), y=y.copy(), z=z.copy(),
                u=u.copy(), v=v.copy(), w=w.copy(),
                P=P.copy(), k=kt.copy(), omega=om.copy(),
                t=float(t_phys),
            )

        del data, x, y, z, u, v, w, P, kt, om, mut, vlim, dwall, category, picked

        if (i + 1) % verbose_every == 0 or i == 0 or i == n_snaps - 1:
            print(f"  [{i+1:4d}/{n_snaps}] t={t_phys:.4f}s  "
                  f"+{n_take} pts (total={write_ptr:,})")

    data_np = data_np[:write_ptr]
    stats_df = pd.DataFrame(stats_records)

    print(f"\nDone. Supervised points: {len(data_np):,} "
          f"across {len(snap_offsets)} snapshots")
    print(f"Wall time: {time.time() - t0:.1f}s    "
          f"Memory: {data_np.nbytes / 1e6:.0f} MB ({data_np.nbytes/1e9:.2f} GB)")

    if n_k_nonpos or n_w_nonpos:
        print(f"  [aviso] valores não-positivos no CSV: k em {n_k_nonpos:,} "
              f"células, ω em {n_w_nonpos:,} (pisos aplicados)")
    print(f"  células com ν_t recortado (todas as leituras): {n_vlim_total:,}")
    return data_np, snap_offsets, first_snap_raw, stats_df

# ----- Run loader -----
data_np, snap_offsets, first_snap_raw, stats_df = load_all_snapshots(snaps)
n_snaps = len(snap_offsets)

print(f"\n--- Sanity check do data_np após z-score ---")
labels = ['xn', 'yn', 'zn', 'tn', 'un', 'vn', 'wn', 'Pn', 'kn', 'omn']
for i, name in enumerate(labels):
    col = data_np[:, i]
    print(f"  {name}: μ={col.mean():+.4f}, σ={col.std():.4f}, "
          f"range=[{col.min():+.3f}, {col.max():+.3f}]")

# As colunas de alvo deveriam sair com μ≈0 e σ≈1 (exceto un/wn, que são
# min-max por V_REF, e kn no modo "scale"). Desvio grande aqui significa que
# as constantes da Cell 4 não descrevem o dado que acabou de ser lido — o
# sampling estratificado reamostra o domínio e pode deslocar as médias.
print(f"\n--- Conferência das colunas z-score ---")
for _i, _n in [(COL_VN, 'vn'), (COL_PN, 'Pn'), (COL_OMN, 'omn')] + \
              ([(COL_KN, 'kn')] if K_NORM_MODE == 'logz' else []):
    _c = data_np[:, _i]
    _bad = abs(_c.mean()) > 0.25 or abs(_c.std() - 1.0) > 0.25
    print(f"  {_n}: μ={_c.mean():+.4f} (alvo 0), σ={_c.std():.4f} (alvo 1)"
          f"{'   <<< amostragem deslocou a distribuição' if _bad else ''}")
print(f"  (a amostragem estratificada NÃO é uniforme no domínio: ela")
print(f"   superrepresenta paredes e BOIs. Um desvio moderado é esperado.)")

_vlim_frac = float(data_np[:, COL_VLIM].mean())
print(f"\n--- Células com ν_t no teto, no conjunto amostrado ---")
print(f"  {int(data_np[:, COL_VLIM].sum()):,} de {len(data_np):,} "
      f"({100*_vlim_frac:.2f}%)")
print(f"  MASK_VISC_LIMITED = {MASK_VISC_LIMITED}"
      f"  ->  {'a Cell 11 exclui esses pontos' if MASK_VISC_LIMITED else 'coluna só diagnóstica; ninguém é excluído'}")
if not MASK_VISC_LIMITED:
    print(f"  (o teto entra no resíduo da Cell 10 como min(...), então essas")
    print(f"   células fecham a equação igual às outras — ver Cell 2)")

# --- distância à parede por ponto de dado ---
_dw = data_np[:, COL_DWALL].astype(np.float64)
print(f"\n--- Distância à parede dos pontos de dado (coluna {COL_DWALL}) ---")
print(f"  [m] min={_dw.min():.4g}  p01={np.percentile(_dw,1):.4g}  "
      f"p50={np.percentile(_dw,50):.4g}  max={_dw.max():.4g}")
if Y_BAND_BY_WALL:
    _thr_pt = wall_band_by_point(
        data_np[:, COL_XN] * DP_ISO + X_MEAN,
        data_np[:, COL_YN] * DP_ISO + Y_MEAN,
        data_np[:, COL_ZN] * DP_ISO + Z_MEAN)
    _in_band = _dw <= _thr_pt
    print(f"  dentro da faixa de tratamento de parede (limiar POR parede):")
    print(f"    {int(_in_band.sum()):,} de {len(_dw):,} "
          f"({100*_in_band.mean():.2f}%)")
    print(f"  com o limiar ESCALAR Y_WALL_FUNCTION = {Y_WALL_FUNCTION:.3f} m:")
    print(f"    {int((_dw <= Y_WALL_FUNCTION).sum()):,} "
          f"({100*(_dw <= Y_WALL_FUNCTION).mean():.2f}%)")
    print(f"  (nesses pontos k e ω são saída do tratamento de parede, não")
    print(f"   transporte resolvido. Continuam servindo de DADO — é a resposta")
    print(f"   do solver — mas não de resíduo. Quem decide é a Cell 11.)")
else:
    print(f"  (verificação (d) não rodou: sem static_mesh alinhado, não há")
    print(f"   limiar por parede e wall_band_by_point cai no escalar)")

print("\n--- Investigação de outliers em P ---")
Pn = data_np[:, COL_PN]

# Reconstrói P físico
P_phys = Pn * P_DP + P_MEAN

# Quantos outliers e onde
thresholds = [500, 1000, 2000, 3000, 5000]
for thr in thresholds:
    n = (np.abs(P_phys) > thr).sum()
    pct = 100 * n / len(P_phys)
    print(f"  |P_phys| > {thr:5d} Pa: {n:>8,} pontos ({pct:.4f}%)")

# Distribuição temporal dos outliers grandes
mask_extreme = np.abs(P_phys) > 3000
if mask_extreme.any():
    tn_outliers = data_np[mask_extreme, COL_TN]
    t_phys_outliers = tn_outliers * T_DP + T_MEAN
    print(f"\n  Outliers extremos (|P|>3000 Pa):")
    print(f"    {mask_extreme.sum()} pontos")
    print(f"    t físico: min={t_phys_outliers.min():.2f}s, "
          f"max={t_phys_outliers.max():.2f}s, "
          f"média={t_phys_outliers.mean():.2f}s")
    # Histograma temporal grosseiro
    bins = [0, 5, 10, 20, 50, 100]
    hist, _ = np.histogram(t_phys_outliers, bins=bins)
    for i, h in enumerate(hist):
        print(f"    t ∈ [{bins[i]:3d}, {bins[i+1]:3d}]s: {h:>6,} outliers")

print("\n--- Faixas físicas reconstruídas de k e ω ---")
_k_phys  = denormalize_k(data_np[:, COL_KN].astype(np.float64))
_om_phys = denormalize_omega(data_np[:, COL_OMN].astype(np.float64))
print(f"  k : min={_k_phys.min():.4g}  p50={np.percentile(_k_phys,50):.4g}  "
      f"p99={np.percentile(_k_phys,99):.4g}  max={_k_phys.max():.4g}  [m²/s²]")
print(f"  ω : min={_om_phys.min():.4g}  p50={np.percentile(_om_phys,50):.4g}  "
      f"p99={np.percentile(_om_phys,99):.4g}  max={_om_phys.max():.4g}  [1/s]")
_nut_phys = _k_phys / np.maximum(_om_phys, OMEGA_FLOOR)
print(f"  ν_t = k/ω : p50={np.percentile(_nut_phys,50):.4g}  "
      f"max={_nut_phys.max():.4g}  [m²/s]   (teto do solver: {NU_T_MAX:.4f})")
print(f"  k > 0 em todos os pontos: {bool((_k_phys > 0).all())}")
print(f"  ω > 0 em todos os pontos: {bool((_om_phys > 0).all())}")
del _k_phys, _om_phys, _nut_phys


# ================================================================================
# CELL 5b — TABELA AGREGADA: AVAILABLE vs TAKEN por categoria
# ================================================================================
# Replica a tabela do MATLAB top.m mas para o conjunto inteiro de snapshots,
# mostrando média ± std das contagens disponíveis no CSV vs efetivamente
# amostradas para o treino. Identifica regiões com déficit (avail < quota).
#
# A TABELA DE QUOTAS É IDÊNTICA À DO LAMINAR — a amostragem não mudou.
# O que entra aqui são duas leituras novas, ambas POR ZONA, porque as duas
# perguntas que importam agora são geográficas:
#
#   §2 ONDE estão as células com ν_t recortado? Com o teto em 1e5 eram ~7%
#      do domínio (job 5518), e 7% espalhados no freestream e 7%
#      concentrados na esteira têm consequências opostas. Se o recorte
#      estivesse justamente em boi_c/boi_t — as zonas que existem porque é
#      ali que a física interessa — a loss de ω estaria contaminada onde
#      mais dói.
#
#   §3 k e ω batem com a física esperada de esteira? Num corpo rombudo,
#      k tem que ser MAIOR na esteira (boi_t) e na região do cilindro (boi_c)
#      do que no freestream. Se sair invertido, o problema não é da rede —
#      é dado trocado, coluna trocada, ou normalização errada. Custa uma
#      mediana por zona e fecha essa porta antes de treinar.
#
# ------------------------------------------------------------------------------
# REVISÃO DE 18/10/2026 — dado novo (job 6125, teto de ν_t em 1e7)
#
# Com o teto novo, ZERO células recortadas é o resultado esperado (medido
# pela conversa do CFD: 0% em t = 100 s, também ponderado por volume). Três
# leituras desta célula tinham sido escritas para o teto antigo e passariam
# a dizer coisas erradas:
#
#   - §2 tratava "zero recortadas" como motivo de DESCONFIANÇA ("o log do
#     5518 indicava ~7,2%") e sugeria subir o teto. Agora a leitura compara
#     com VISC_LIMITED_PCT_EXPECTED (Cell 2) e avisa no sentido CONTRÁRIO:
#     o suspeito passa a ser haver recorte.
#
#   - §3(3) declarava o alinhamento μ_t/k/ω "confirmado" quando os dois
#     conjuntos eram VAZIOS: com a união vazia, o Jaccard vale 1 por
#     convenção, e o teste OK vinha antes do teste de conjunto vazio. Um
#     teste sem nenhum caso não confirma nada. Agora o caso vazio vem
#     primeiro e diz isso, e aponta o teste de alinhamento que continua
#     tendo poder: ν_t > k/ω na verificação (b) da Cell 5.
#
#   - §3(4) comparava o ω das CÉLULAS a menos de 3 m do inlet com OMEGA_INLET,
#     com tolerância de 20%. Mesmo defeito da antiga verificação (c): o ω
#     imposto cai à metade em ~2,8 cm, então as células estão sempre
#     ~8× abaixo e o teste acusava ">>>" sem haver erro. Virou diagnóstico
#     de decaimento; o veredito sobre a BC é o da (c), em INLET_CHECK.
#
# E os três CSVs que esta célula grava vão para DIAG_DIR (Cell 3), em vez
# de caírem no diretório de onde o sbatch foi chamado.


print(f"\n{'='*82}")
print(f"  TABELA AGREGADA por categoria  (n_snapshots = {n_snaps})")
print(f"  Tolerâncias: BC_TOL={BC_TOL}m  CYL_TOL={CYL_TOL}m")
print(f"{'='*82}")
print(f"{'Categoria':<14}{'Quota':>8}{'Avail (μ±σ)':>20}"
      f"{'Taken (μ±σ)':>20}{'% Quota':>10}{'Status':>10}")
print(f"{'-'*82}")

summary_records = []
for cat_name in CATEGORY_NAMES:
    sub = stats_df[stats_df["category"] == cat_name]
    avail_mean = sub["available"].mean()
    avail_std  = sub["available"].std()
    taken_mean = sub["taken"].mean()
    taken_std  = sub["taken"].std()
    quota      = QUOTAS[cat_name]
    pct_quota  = 100.0 * taken_mean / quota if quota > 0 else 0.0
    status = "OK" if pct_quota >= 99.9 else ("DÉFICIT" if pct_quota < 95.0 else "near")
    summary_records.append({
        "category": cat_name, "quota": quota,
        "avail_mean": avail_mean, "avail_std": avail_std,
        "taken_mean": taken_mean, "taken_std": taken_std,
        "pct_quota": pct_quota, "status": status,
    })
    print(f"{cat_name:<14}{quota:>8d}"
          f"{avail_mean:>11.1f} ± {avail_std:<5.1f}"
          f"{taken_mean:>11.1f} ± {taken_std:<5.1f}"
          f"{pct_quota:>9.1f}%{status:>10}")

print(f"{'-'*82}")
total_quota = sum(QUOTAS.values())
total_taken_mean = sum(r["taken_mean"] for r in summary_records)
print(f"{'TOTAL':<14}{total_quota:>8d}{'—':>20}"
      f"{total_taken_mean:>11.1f}{'':<8}{100*total_taken_mean/total_quota:>9.1f}%")
print(f"{'='*82}\n")

# Save per-snapshot stats e summary — em DIAG_DIR (Cell 3), não no cwd
_p_stats   = os.path.join(DIAG_DIR, "snapshot_sampling_stats.csv")
_p_summary = os.path.join(DIAG_DIR, "snapshot_sampling_summary.csv")
stats_df.to_csv(_p_stats, index=False)
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(_p_summary, index=False)
print(f"Per-snapshot stats salvos em: {_p_stats}")
print(f"Summary agregado salvos em:  {_p_summary}\n")
print("Summary:")
print(summary_df.to_string(index=False, float_format='%.1f'))


# ================================================================================
# §2 — ONDE o solver recortou ν_t
# ================================================================================
ID_TO_GROUP = {v: k for k, v in GROUP_TO_ID.items()}
_GROUP_ORDER = ["cyl", "bc", "boi_c", "boi_t", "free"]

_gid_all  = data_np[:, COL_GID]
_vlim_all = data_np[:, COL_VLIM]
_n_vlim_total = float(_vlim_all.sum())

print(f"\n{'='*82}")
print(f"  CÉLULAS COM ν_t RECORTADO (μ_t/μ = {VISC_RATIO_MAX:.0e}), POR ZONA")
print(f"{'='*82}")
print(f"{'Zona':<10}{'pontos':>14}{'recortados':>14}"
      f"{'% da zona':>12}{'% do total':>13}{'':>8}")
print(f"{'-'*82}")

zone_records = []
for gname in _GROUP_ORDER:
    gid = float(GROUP_TO_ID[gname])
    m = (_gid_all == gid)
    n_pts = int(m.sum())
    if n_pts == 0:
        continue
    n_vl = float(_vlim_all[m].sum())
    pct_zona  = 100.0 * n_vl / n_pts
    pct_total = 100.0 * n_vl / _n_vlim_total if _n_vlim_total > 0 else 0.0
    # zonas onde a física interessa: é ali que contaminação incomoda
    _core = gname in ("boi_c", "boi_t")
    flag = "  <<<" if (_core and pct_zona > 5.0) else ""
    zone_records.append({
        "zone": gname, "n_points": n_pts, "n_visc_limited": int(n_vl),
        "pct_of_zone": pct_zona, "pct_of_all_limited": pct_total,
    })
    print(f"{gname:<10}{n_pts:>14,}{int(n_vl):>14,}"
          f"{pct_zona:>11.2f}%{pct_total:>12.1f}%{flag:>8}")

print(f"{'-'*82}")
print(f"{'TOTAL':<10}{len(data_np):>14,}{int(_n_vlim_total):>14,}"
      f"{100*_n_vlim_total/len(data_np):>11.2f}%"
      f"{(f'{100.0:>12.1f}%') if _n_vlim_total > 0 else f'{chr(8212):>13}'}")
print(f"{'='*82}")

_zone_df = pd.DataFrame(zone_records)
_core_pct = _zone_df.loc[_zone_df["zone"].isin(["boi_c", "boi_t"]),
                         "pct_of_all_limited"].sum()
# A fração amostrada de cada zona é FIXADA pelas quotas, não pelo volume que
# a zona ocupa. Então "% do total recortado" diz onde o recorte está DENTRO
# DO CONJUNTO DE TREINO — que é o que importa para a loss — e não onde ele
# está no domínio físico. As duas coisas não coincidem, e é a primeira que
# decide o efeito da máscara.
_core_zone_pct = _zone_df.loc[_zone_df["zone"].isin(["boi_c", "boi_t"]),
                              "pct_of_zone"]
# O que se espera depende do teto, e a expectativa mora na Cell 2
# (VISC_LIMITED_PCT_EXPECTED): 7,2 com 1e5 (job 5518), 0 com 1e7 (job 6125).
# A leitura compara com ela, em vez de supor um dos dois casos.
_pct_vlim_obs = 100.0 * _n_vlim_total / max(len(data_np), 1)
_expect_zero  = VISC_LIMITED_PCT_EXPECTED < 0.1
print(f"\n  Leitura (esperado pela Cell 2: {VISC_LIMITED_PCT_EXPECTED:g}% "
      f"com teto {VISC_RATIO_MAX:.0e}; observado: {_pct_vlim_obs:.3f}%):")
if _n_vlim_total == 0 and _expect_zero:
    print(f"    OK  nenhum ponto recortado, como esperado: com o teto em "
          f"{VISC_RATIO_MAX:.0e},")
    print(f"        o μ_t/μ máximo medido ({VISC_RATIO_MEASURED_MAX:.2e}) fica "
          f"{VISC_RATIO_MAX/VISC_RATIO_MEASURED_MAX:.0f}× abaixo dele.")
    print(f"        A coluna {COL_VLIM} do data_np sai toda zero, e é isso mesmo.")
elif _n_vlim_total == 0:
    print(f"    Nenhum ponto recortado na amostra, mas a Cell 2 esperava")
    print(f"    ~{VISC_LIMITED_PCT_EXPECTED:g}%. Ou o teto não agiu nesta rodada, ou")
    print(f"    a coluna viscosity-turb não é a que VISC_TURB_IS_DYNAMIC supõe —")
    print(f"    zero aqui é motivo de desconfiança, não de alívio.")
elif _expect_zero and _pct_vlim_obs > 0.1:
    print(f"    >>> {_pct_vlim_obs:.2f}% dos pontos no teto, e com {VISC_RATIO_MAX:.0e} o")
    print(f"        esperado é ~0%. Suspeitar, nesta ordem: o dado não é o do")
    print(f"        job {CFD_JOB_ID} (conferir CFD_DIR na Cell 3); VISC_RATIO_MAX da")
    print(f"        Cell 2 não é o teto que o solver usou; VISC_TURB_IS_DYNAMIC")
    print(f"        errado (dividir μ_t por μ quando a coluna já é ν_t, ou o")
    print(f"        contrário, desloca a razão por ρ = {RHO_INF}).")
elif _core_pct > 60.0:
    print(f"    {_core_pct:.1f}% de todo o recorte cai em boi_c + boi_t — as zonas")
    print(f"    de refino, que existem porque é ali que a física interessa.")
    print(f"    >>> Isso põe a máscara (MASK_VISC_LIMITED) no centro do problema,")
    print(f"        e nenhuma das duas escolhas é boa: sem ela, o resíduo de ω é")
    print(f"        imposto onde a equação não fecha; com ela, some uma fatia")
    print(f"        grande da supervisão de ω em plena esteira.")
    print(f"        Vale considerar subir o teto de μ_t/μ no Fluent e regravar.")
elif _core_pct > 30.0:
    print(f"    {_core_pct:.1f}% do recorte cai em boi_c + boi_t. É uma fatia")
    print(f"    relevante das zonas de refino, mas não a maioria. Mascarar é")
    print(f"    viável; acompanhar quanto da supervisão de ω sobra na esteira")
    print(f"    (coluna '% da zona' acima) antes de decidir.")
else:
    print(f"    Só {_core_pct:.1f}% do recorte cai nas zonas de refino. Mascarar")
    print(f"    esses pontos custa pouca supervisão onde ela importa.")
print(f"    Contaminação das zonas de refino: "
      f"{_core_zone_pct.min():.1f}% a {_core_zone_pct.max():.1f}% dos pontos.")


# ================================================================================
# §3 — k, ω e ν_t por zona: a física está de pé?
# ================================================================================
# Subamostra: medianas exatas sobre dezenas de milhões de pontos não pagam o
# custo. 400k por zona já dá percentil estável em 3 casas.
_SUB_PER_ZONE = 400_000
_rng_5b = np.random.default_rng(SEED)

print(f"\n{'='*82}")
print(f"  TURBULÊNCIA POR ZONA (valores FÍSICOS, desnormalizados)")
print(f"{'='*82}")
print(f"{'Zona':<10}{'k p50':>11}{'k p99':>11}{'ω p50':>11}{'ω p99':>11}"
      f"{'ν_t p50':>11}{'ν_t max':>11}")
print(f"{'':<10}{'[m²/s²]':>11}{'[m²/s²]':>11}{'[1/s]':>11}{'[1/s]':>11}"
      f"{'[m²/s]':>11}{'[m²/s]':>11}")
print(f"{'-'*82}")

turb_zone_records = []
for gname in _GROUP_ORDER:
    gid = float(GROUP_TO_ID[gname])
    idx = np.where(_gid_all == gid)[0]
    if len(idx) == 0:
        continue
    if len(idx) > _SUB_PER_ZONE:
        idx = _rng_5b.choice(idx, size=_SUB_PER_ZONE, replace=False)
    kp  = denormalize_k(data_np[idx, COL_KN].astype(np.float64))
    omp = denormalize_omega(data_np[idx, COL_OMN].astype(np.float64))
    nut = kp / np.maximum(omp, OMEGA_FLOOR)
    rec = dict(zone=gname,
               k_p50=float(np.percentile(kp, 50)),  k_p99=float(np.percentile(kp, 99)),
               om_p50=float(np.percentile(omp, 50)), om_p99=float(np.percentile(omp, 99)),
               nut_p50=float(np.percentile(nut, 50)), nut_max=float(nut.max()))
    turb_zone_records.append(rec)
    print(f"{gname:<10}{rec['k_p50']:>11.4g}{rec['k_p99']:>11.4g}"
          f"{rec['om_p50']:>11.4g}{rec['om_p99']:>11.4g}"
          f"{rec['nut_p50']:>11.4g}{rec['nut_max']:>11.4g}")
print(f"{'='*82}")

turb_zone_df = pd.DataFrame(turb_zone_records)
turb_zone_df = turb_zone_df.merge(_zone_df, on="zone", how="outer")
_p_turb = os.path.join(DIAG_DIR, "turbulence_by_zone.csv")
turb_zone_df.to_csv(_p_turb, index=False)
print(f"\nTabela por zona salva em: {_p_turb}")

# --- checagens físicas ---
print(f"\n  Checagens físicas:")
_kz = {r["zone"]: r["k_p50"] for r in turb_zone_records}

# (1) esteira mais turbulenta que o freestream
if "boi_t" in _kz and "free" in _kz:
    _ratio = _kz["boi_t"] / max(_kz["free"], 1e-30)
    if _ratio > 1.0:
        print(f"    OK  k na esteira (boi_t) é {_ratio:.1f}× o do freestream.")
        print(f"        É o esperado: o cilindro gera turbulência que o")
        print(f"        escoamento carrega para jusante.")
    else:
        print(f"    >>> k na esteira é {_ratio:.2f}× o do freestream — ou seja,")
        print(f"        a esteira está MENOS turbulenta que o escoamento livre.")
        print(f"        Isso não acontece num corpo rombudo. Suspeitar, nesta ordem:")
        print(f"        coluna trocada na leitura (k ↔ ω saem adjacentes no CSV),")
        print(f"        normalização invertida, ou os BOIs não estão onde se pensa.")

# (2) entorno do cilindro — informativo, não é critério de aprovação
#
# boi_c é uma caixa CENTRADA no cilindro: y ∈ [160, 240] com o cilindro em
# y = 200. Metade dela está a montante, onde há região de estagnação e k
# baixo; metade a jusante, nas camadas cisalhantes, onde k é alto. A mediana
# das duas juntas pode dar perto do freestream sem que nada esteja errado.
# Só vale o alarme se boi_c ficar claramente ABAIXO do freestream.
if "boi_c" in _kz and "free" in _kz:
    _r2 = _kz["boi_c"] / max(_kz["free"], 1e-30)
    print(f"    ... k no entorno do cilindro (boi_c) é {_r2:.1f}× o do freestream")
    print(f"        (a caixa cobre montante e jusante do cilindro; razão perto")
    print(f"         de 1 é normal)")
    if _r2 < 0.8:
        print(f"    >>> mas ABAIXO do freestream. A região do cilindro não pode")
        print(f"        ser menos turbulenta que o escoamento livre.")

# (3) k/ω acima do teto tem que COINCIDIR com o marcador de recorte
#
# Atenção: k/ω PODE passar do teto, e isso não é erro. O limitador do Fluent
# recorta o campo de μ_t que ele grava; k e ω são as variáveis transportadas
# e vão para o CSV como foram resolvidas. Onde o limitador agiu, portanto:
#     μ_t gravado = teto      mas      k/ω > teto
# É justamente essa discrepância que define "o limitador agiu ali".
#
# Isso dá uma verificação cruzada de graça. O marcador COL_VLIM foi montado
# a partir da coluna μ_t; a razão k/ω vem de duas outras colunas. Se os dois
# conjuntos coincidirem, as três colunas estão alinhadas célula a célula —
# que é exatamente o que o job 5479 quebrou e o 5518 existe para garantir.
#
# COM O TETO EM 1e7 ESTA CONFERÊNCIA PERDE O PODER. Nenhuma célula chega ao
# teto, os dois conjuntos ficam vazios, e um teste sem nenhum caso não
# confirma alinhamento nenhum. A versão anterior testava "Jaccard > 0,95"
# ANTES de "conjunto vazio", e o Jaccard de dois conjuntos vazios é 1 por
# convenção — imprimia "OK, confirma que μ_t, k e ω vêm da mesma célula"
# sem ter olhado uma célula sequer. O caso vazio agora vem primeiro.
#
# O teste de alinhamento que CONTINUA com poder é o da verificação (b) da
# Cell 5: ν_t > k/ω é impossível no SST (o max do limitador só reduz), e
# acontece em massa se μ_t vier de outra célula que k e ω. Ele não depende
# do teto. O resultado está em NU_T_CHECK["n_violations"].
_kk_all  = denormalize_k(data_np[:, COL_KN].astype(np.float64))
_oo_all  = denormalize_omega(data_np[:, COL_OMN].astype(np.float64))
_nut_all = _kk_all / np.maximum(_oo_all, OMEGA_FLOOR)
_above   = _nut_all > NU_T_MAX
_flagged = data_np[:, COL_VLIM] > 0.5

_n_ab, _n_fl = int(_above.sum()), int(_flagged.sum())
_both  = int((_above & _flagged).sum())
_only_a = _n_ab - _both
_only_f = _n_fl - _both
_union  = _both + _only_a + _only_f
_jacc   = _both / _union if _union > 0 else 1.0

print(f"    ν_t = k/ω máximo = {_nut_all.max():.4g} m²/s "
      f"(teto do solver: {NU_T_MAX:.4f})")
print(f"        k/ω > teto: {_n_ab:,}    μ_t marcado como recortado: {_n_fl:,}")
print(f"        nos dois: {_both:,}   só k/ω: {_only_a:,}   só μ_t: {_only_f:,}")
if _union == 0:
    print(f"    ... nenhum ponto acima do teto por nenhum dos dois critérios —")
    print(f"        esperado com teto {VISC_RATIO_MAX:.0e}. Com os dois conjuntos")
    print(f"        vazios, este teste NÃO diz nada sobre alinhamento de colunas.")
    _nc = globals().get("NU_T_CHECK")
    if _nc:
        _nv = int(_nc.get("n_violations", -1))
        print(f"        O teste que vale é o da verificação (b): ν_t > k/ω (impossível")
        print(f"        no SST) em {_nv:,} células do 1º snapshot"
              f"{' — alinhamento OK' if _nv == 0 else ' — ver a (b) acima'}.")
elif _jacc > 0.95:
    print(f"    OK  os dois conjuntos coincidem ({100*_jacc:.1f}%).")
    print(f"        Confirma que μ_t, k e ω vêm da MESMA célula em cada linha —")
    print(f"        o desalinhamento do job 5479 não está presente.")
else:
    print(f"    >>> os conjuntos só coincidem em {100*_jacc:.1f}%.")
    print(f"        μ_t e (k, ω) estão contando histórias diferentes sobre as")
    print(f"        mesmas células. Suspeitar de desalinhamento entre colunas")
    print(f"        (o defeito do job 5479) ou de VISC_TURB_IS_DYNAMIC errado.")
del _kk_all, _oo_all, _nut_all, _above, _flagged

# (4) k e ω nos pontos de dado junto ao inlet — DIAGNÓSTICO, não veredito
#
# Isto comparava o ω das células a menos de BC_TOL do inlet com OMEGA_INLET,
# tolerância de 20%, e marcava ">>>" quando não batia. Não batia nunca: o ω
# imposto cai à metade em V_INF/(β·ω0) ≈ 2,8 cm, e o centro da 1ª célula
# está bem mais longe que isso. Era o mesmo defeito da antiga verificação
# (c). O valor de célula não diz se a BC está certa — diz quanto ela já
# decaiu. O veredito sobre a BC é o da (c) da Cell 5, que lê as FACES.
#
# O round-trip normalizar -> desnormalizar, que esta linha também dizia
# testar, já é garantido pelo assert logo depois de normalize_k (Cell 5).
_bc_idx = np.where(_gid_all == float(GROUP_TO_ID["bc"]))[0]
if len(_bc_idx) > 0:
    _yn_bc = data_np[_bc_idx, COL_YN] * DP_ISO + Y_MEAN
    _inl = _bc_idx[_yn_bc < (Y_MIN + BC_TOL)]
    if len(_inl) > 0:
        _om_inl = float(np.median(denormalize_omega(
            data_np[_inl, COL_OMN].astype(np.float64))))
        _k_inl  = float(np.median(denormalize_k(
            data_np[_inl, COL_KN].astype(np.float64))))
        print(f"    ... pontos de dado a menos de {BC_TOL:g} m do inlet "
              f"({len(_inl):,}):")
        print(f"        k = {_k_inl:.4f} ({_k_inl/K_INLET:.3f}× a BC)   "
              f"ω = {_om_inl:.0f} ({_om_inl/OMEGA_INLET:.3f}× a BC)")
        print(f"        Abaixo da BC por construção: decaimento, não erro.")
        _ic = globals().get("INLET_CHECK")
        if _ic:
            _v = {True: "OK", False: "FALHOU", None: "sem veredito"}[_ic.get("ok")]
            print(f"        Veredito sobre a BC: verificação (c), modo "
                  f"'{_ic.get('mode')}' -> {_v}.")

del _gid_all, _vlim_all


# ============================================================
# CELL 6 — TRAIN / VALIDATION SPLIT (HOLD-OUT TEMPORAL)
# ============================================================
# A LÓGICA DO SPLIT É A MESMA DO LAMINAR e continua sendo o padrão: ciclos de
# 20 arquivos, os 2 últimos de cada ciclo vão para validação. Ela não depende
# de quantos campos existem — separa LINHAS por snapshot, e as colunas novas
# (kn, omn, vlim) vão junto sem nada a mudar.
#
# O que entra aqui são dois diagnósticos, e o primeiro descobriu algo que já
# valia para o laminar.
#
# ------------------------------------------------------------------------------
# 1. O HOLD-OUT INTERCALADO QUASE NÃO É UM HOLD-OUT
#
#    Com 18 treino / 2 validação em ciclos de 20, cada snapshot de validação
#    fica a UM PASSO de um snapshot de treino:
#
#        distância ao treino mais próximo:  mediana 0,10 s   máx 0,20 s
#        período de desprendimento (St≈0,2): 15,9 s
#        ou seja: 2,3 GRAUS de fase
#
#    Dois campos separados por 2° de fase são praticamente o mesmo campo. A
#    rede não precisa aprender a dinâmica para acertar a validação — basta
#    interpolar entre os dois vizinhos que ela viu. A métrica de validação
#    mede suavidade em t, não capacidade de reproduzir o escoamento.
#
#    Isso não é consequência da turbulência; vale igual no laminar. Mas
#    piora aqui: k e ω têm estrutura espacial mais fina que u e P, e uma
#    validação que não distingue nada vai deixar passar justamente o que há
#    de novo.
#
#    NÃO MUDEI O PADRÃO. Trocar o split ao mesmo tempo que se acrescenta
#    turbulência estraga a comparação entre as duas versões — mesma razão
#    pela qual o transiente não é descartado. VAL_MODE = "blocks" está
#    implementado e desligado, para ser testado sozinho depois.
#
# ------------------------------------------------------------------------------
# 2. TREINO E VALIDAÇÃO PRECISAM SER COMPARÁVEIS
#
#    A loss de validação só significa alguma coisa se os dois conjuntos
#    tiverem a mesma composição. Com as colunas novas há duas formas novas de
#    isso falhar: proporção diferente de células com ν_t recortado, e
#    distribuição diferente de k e ω. O bloco no fim mede as duas.


# ---------- USER CONFIG ----------
VAL_MODE = "interleaved"    # "interleaved" (padrão, igual ao laminar)
                            # "blocks"      (hold-out em blocos contíguos)

# --- modo "interleaved" ---
TRAIN_FILES = 18
VAL_FILES   = 2
CYCLE_SIZE  = TRAIN_FILES + VAL_FILES  # Ciclo de 20 arquivos

# --- modo "blocks" ---
# Existe um conflito real aqui, e ele não tem solução boa: quanto menor a
# fração de validação, mais perto do treino ficam os pontos retidos. Não dá
# para ter hold-out barato E temporalmente distante. Os dois botões abaixo
# expõem esse conflito em vez de escondê-lo num default.
VAL_FRACTION  = 0.15        # fração de snapshots retidos
N_VAL_BLOCKS  = 3           # em quantos blocos contíguos

SKIP_FIRST  = False  # mantém o primeiro arquivo fora (apenas inicialização do dom.)

# Escala de referência para julgar as distâncias temporais. Strouhal de
# cilindro a Re alto; serve como ordem de grandeza, não como medida.
ST_CYL = 0.20
T_SHED = D_CYL / (ST_CYL * V_INF)      # ≈ 15,9 s

# Auto-relax se não tivermos snapshots suficientes para fechar um ciclo
if n_snaps < CYCLE_SIZE:
    VAL_FILES = max(1, n_snaps // 10)  # Garante pelo menos 1 arquivo de validação
    TRAIN_FILES = n_snaps - VAL_FILES
    CYCLE_SIZE = n_snaps
    print(f"[info] apenas {n_snaps} snapshots -> reduzindo para {TRAIN_FILES} treino e {VAL_FILES} val")

# Lógica de seleção: pega o resto da divisão por 20.
# Se o resto for >= 18 (ou seja, 18 ou 19), é validação.
val_snap_indices = set()

if VAL_MODE == "interleaved":
    for i in range(n_snaps):
        if (i % CYCLE_SIZE) >= TRAIN_FILES:
            val_snap_indices.add(i)

elif VAL_MODE == "blocks":
    # Blocos contíguos, espalhados uniformemente na série. O bloco inteiro
    # sai do treino, então o miolo dele fica a meio comprimento de bloco do
    # dado supervisionado mais próximo — que é o que dá sentido à métrica.
    _n_val_target = int(round(VAL_FRACTION * n_snaps))
    _block_len    = max(1, _n_val_target // max(1, N_VAL_BLOCKS))

    # Os blocos NÃO encostam nas pontas da série. Reter o primeiro ou o
    # último snapshot transforma o teste de interpolação em extrapolação —
    # sem dado de um dos lados, a rede tem que extrapolar em t, que é um
    # problema diferente e mais difícil. Medir os dois juntos confunde as
    # duas coisas na mesma métrica.
    _margin = _block_len
    _lo = _margin
    _hi = n_snaps - _block_len - _margin
    if _hi <= _lo:                       # série curta: cai para sem margem
        _lo, _hi = 0, max(0, n_snaps - _block_len)
        print(f"[blocks] série curta — blocos encostam nas pontas; a validação")
        print(f"         vai incluir extrapolação em t, não só interpolação.")
    _starts = np.unique(np.linspace(_lo, _hi, N_VAL_BLOCKS).astype(int))
    for _s in _starts:
        val_snap_indices.update(range(int(_s), int(_s) + _block_len))

    _bl_s = _block_len * DT_CFD * SNAPSHOT_STRIDE
    print(f"[blocks] {len(_starts)} blocos de {_block_len} snapshots "
          f"({_bl_s:.1f} s cada = {_bl_s / T_SHED:.2f} período de desprendimento)")
    if _bl_s / T_SHED < 0.15:
        print(f"         >>> blocos curtos demais para um hold-out significativo.")
        print(f"             Subir VAL_FRACTION ou baixar N_VAL_BLOCKS: o miolo do")
        print(f"             bloco fica a meio comprimento do treino, então blocos")
        print(f"             de {_bl_s:.1f} s afastam só {_bl_s/2:.1f} s.")

else:
    raise ValueError(f"VAL_MODE desconhecido: {VAL_MODE!r}")

if SKIP_FIRST:
    val_snap_indices.discard(0)

# Build a boolean mask over ALL rows from snap_offsets (no float-equality games)
val_mask = np.zeros(len(data_np), dtype=bool)
for i, (_t_phys, start, end) in enumerate(snap_offsets):
    if i in val_snap_indices:
        val_mask[start:end] = True

train_np = data_np[~val_mask]
val_np   = data_np[val_mask]

val_times_phys = [snap_offsets[i][0] for i in sorted(val_snap_indices)]
print(f"Train points: {len(train_np):,}    Val points: {len(val_np):,}")
print(f"Validation snapshots: {len(val_snap_indices)} held out  "
      f"({100*len(val_snap_indices)/n_snaps:.1f}% dos snapshots, modo '{VAL_MODE}')")
print(f"  Physical times (s): "
      f"{[f'{t:.3f}' for t in val_times_phys[:5]]}"
      f"{' ...' if len(val_times_phys) > 5 else ''}")
print(f"  Memória: train {train_np.nbytes/1e9:.2f} GB + val {val_np.nbytes/1e9:.2f} GB"
      f"  (data_np original: {data_np.nbytes/1e9:.2f} GB — são CÓPIAS)")


# ================================================================================
# Diagnóstico 1 — quão longe do treino está cada snapshot de validação?
# ================================================================================
_idx_train = np.array(sorted(set(range(n_snaps)) - val_snap_indices))
_idx_val   = np.array(sorted(val_snap_indices))

print(f"\n--- Distância temporal entre validação e treino ---")
if len(_idx_train) == 0 or len(_idx_val) == 0:
    print(f"  (um dos conjuntos está vazio)")
else:
    _t_arr = np.array([snap_offsets[i][0] for i in range(n_snaps)], dtype=np.float64)
    _gaps = np.array([np.min(np.abs(_t_arr[_idx_train] - _t_arr[i])) for i in _idx_val])
    _med, _mx = float(np.median(_gaps)), float(_gaps.max())
    print(f"  período de desprendimento estimado (St={ST_CYL}): {T_SHED:.1f} s")
    print(f"  distância ao snapshot de treino mais próximo:")
    print(f"    mediana {_med:.2f} s   máximo {_mx:.2f} s")
    print(f"    em fase: mediana {360*_med/T_SHED:.1f}°   máximo {360*_mx/T_SHED:.1f}°")

    _frac = _med / T_SHED
    if _frac >= 0.20:
        print(f"  OK  hold-out a {360*_frac:.0f}° de fase — a validação exige")
        print(f"      reconstruir o campo, não interpolar entre vizinhos.")
    else:
        print(f"  >>> A validação está a {360*_frac:.1f}° de fase do treino.")
        print(f"      Dois campos tão próximos são praticamente o mesmo campo: a")
        print(f"      rede acerta a validação interpolando os vizinhos, sem ter")
        print(f"      aprendido a dinâmica. A métrica mede suavidade em t.")
        if VAL_MODE == "interleaved":
            print(f"      É inerente ao modo intercalado: com {TRAIN_FILES}/"
                  f"{VAL_FILES} em ciclos de {CYCLE_SIZE}, cada snapshot retido")
            print(f"      fica a um passo de um snapshot de treino, sempre.")
            print(f"      Para um hold-out de verdade: VAL_MODE = 'blocks'.")
            print(f"      (Mantido assim de propósito — ver o cabeçalho da célula.)")
        else:
            print(f"      Modo 'blocks' já ativo, mas os blocos estão curtos:")
            print(f"      {len(_idx_val)} snapshots em {N_VAL_BLOCKS} blocos. Subir")
            print(f"      VAL_FRACTION (hoje {VAL_FRACTION:.0%}) ou baixar")
            print(f"      N_VAL_BLOCKS alonga cada bloco e afasta o miolo.")


# ================================================================================
# Diagnóstico 2 — treino e validação têm a mesma composição?
# ================================================================================
# Se não tiverem, a loss de validação não é comparável com a de treino e a
# curva das duas lado a lado engana.
print(f"\n--- Composição: treino × validação ---")

def _compare(label, a, b, tol, fmt="{:+.4f}"):
    d = abs(a - b)
    flag = "  <<<" if d > tol else ""
    print(f"  {label:<22}{fmt.format(a):>12}{fmt.format(b):>12}"
          f"{fmt.format(b-a):>12}{flag}")
    return d <= tol

print(f"  {'':<22}{'treino':>12}{'validação':>12}{'Δ':>12}")
print(f"  {'-'*58}")

_ok_comp = True
# fração de cada zona
for _gname in ["cyl", "bc", "boi_c", "boi_t", "free"]:
    _g = float(GROUP_TO_ID[_gname])
    _a = float((train_np[:, COL_GID] == _g).mean())
    _b = float((val_np[:, COL_GID] == _g).mean())
    _ok_comp &= _compare(f"zona {_gname}", _a, _b, tol=0.02, fmt="{:.4f}")

# fração de células com ν_t recortado
_a = float(train_np[:, COL_VLIM].mean())
_b = float(val_np[:, COL_VLIM].mean())
_ok_comp &= _compare("ν_t recortado", _a, _b, tol=0.02, fmt="{:.4f}")

print(f"  {'-'*58}")
# média e desvio de cada alvo
for _i, _nm in [(COL_UN,"u"), (COL_VN,"v"), (COL_WN,"w"),
                (COL_PN,"P"), (COL_KN,"k"), (COL_OMN,"ω")]:
    _a, _b = float(train_np[:, _i].mean()), float(val_np[:, _i].mean())
    _sa, _sb = float(train_np[:, _i].std()), float(val_np[:, _i].std())
    _d = abs(_a - _b)
    _ds = abs(_sa - _sb)
    _flag = "  <<<" if (_d > 0.10 or _ds > 0.10) else ""
    print(f"  {_nm+' (μ / σ)':<22}{_a:>6.3f} /{_sa:>5.3f}"
          f"{_b:>7.3f} /{_sb:>5.3f}{_b-_a:>12.3f}{_flag}")
    _ok_comp &= (_d <= 0.10 and _ds <= 0.10)

print(f"  {'-'*58}")
if _ok_comp:
    print(f"  OK  os dois conjuntos são estatisticamente equivalentes.")
    print(f"      As curvas de loss de treino e validação são comparáveis entre si.")
else:
    print(f"  >>> treino e validação têm composições diferentes nas linhas marcadas.")
    print(f"      A loss de validação NÃO é comparável com a de treino: parte da")
    print(f"      diferença entre as curvas vem do conjunto, não do ajuste.")
    print(f"      Com split temporal isso costuma significar que os snapshots")
    print(f"      retidos caem numa fase específica do desprendimento — em modo")
    print(f"      'blocks', deslocar ou aumentar o número de blocos resolve.")


# ================================================================================
# CELL 7 — COLLOCATION POOL COM 11 SUB-REGIÕES
# ================================================================================
# Layout determinístico do pool (sem shuffle):
#
#   adj_inlet     [0,      6000 ]       quota: 2000 per step × 3
#   adj_outlet    [6000,   12000]
#   adj_xmin      [12000,  18000]
#   adj_xmax      [18000,  24000]
#   adj_ground    [24000,  30000]
#   adj_top       [30000,  36000]
#   adj_cyl_lat   [36000,  46500]      quota: 3500 × 3 = 10500
#   adj_cyl_top   [46500,  48000]      quota: 500 × 3 = 1500
#   boi_c         [48000,  81000]      quota: 11000 × 3 = 33000
#   boi_t         [81000,  96000]      quota: 5000 × 3 = 15000
#   free          [96000,  144000]     quota: 16000 × 3 = 48000
#
# A hierarquia de exclusão evita duplicação geométrica entre regiões:
#   - adj_cyl_lat/top: cascas mais específicas (prioridade máxima)
#   - adj_BC (6 faces): cascas das paredes externas
#   - BOI_C/T: caixas de refinamento (excluem cilindro sólido + cascas adj_cyl)
#   - FREE: domínio menos todas as outras regiões
#
# A função is_adj_cyl_lateral (Cell 5) usa |r - R| < CYL_TOL (anel de espessura
# 2×CYL_TOL). Aqui amostramos APENAS o exterior r ∈ [R, R+CYL_TOL] pois é onde
# há fluido — interior é cilindro sólido.
#
# ================================================================================
# O QUE MUDA: DOIS CAMPOS A MAIS POR PONTO
#
# As 11 regiões, as quotas, o POOL_MULT e todos os samplers ficam IDÊNTICOS.
# O que entra são dois arrays paralelos ao pool:
#
#   coll_dwall  distância à parede [m]        — entra em F1 e F2
#   coll_ywall  limiar da faixa de tratamento de parede daquele ponto [m]
#               — decide, ponto a ponto, se o resíduo de k e ω vale ali
#
# O SEGUNDO É NOVO (27/09/2026) e é consequência direta da Cell 5. Ela mediu
# a faixa POR PAREDE, e os limiares saíram muito diferentes entre si — na
# malha de teste, 13× entre a parede mais fina e a mais grossa. Um limiar
# escalar único trata as duas como iguais: ou deixa a primeira camada das
# paredes grossas de fora, ou engole a segunda e a terceira camada das finas.
# Como cada ponto de colocação sabe qual é a sua parede mais próxima, o
# limiar certo é o dela. wall_band_by_point (Cell 5) faz esse mapeamento; a
# Cell 7 só o aplica no pool e passa o resultado adiante.
#
# A função de mistura F1 do SST depende de distância à parede — ela decide,
# ponto a ponto, se o modelo se comporta como k-ω (junto à parede) ou como
# k-ε transformado (longe). Sem F1 não há SST, só um dos dois ramos.
#
# E aqui está o problema que a Cell 1 chamou de estrutural: o CFD exportou
# cell-wall-distance para as CÉLULAS DA MALHA, mas os pontos de colocação não
# são células — são sorteados em posições arbitrárias do domínio, a cada
# rodada diferentes. Nenhum arquivo pode fornecer d_wall para eles.
#
# Nesta geometria, porém, a distância à parede é ANALÍTICA. O domínio é uma
# caixa com um cilindro dentro, e as paredes são:
#
#     ground (z=Z_MIN) · top (z=Z_MAX) · lateral_min (x=X_MIN) ·
#     lateral_max (x=X_MAX) · cylinder_wall · cylinder_top
#
# Note que Y NÃO APARECE: inlet e outlet são velocity-inlet e pressure-outlet,
# não wall. O Fluent calcula d_wall só sobre zonas do tipo wall, então a
# distância independe de y. Parede móvel continua sendo parede — o que a torna
# wall é o tipo da zona, não a velocidade prescrita nela.
#
# A fórmula fica com o dado disponível para conferência: o static_mesh.csv tem
# o cell-wall-distance calculado pelo próprio Fluent, nos centroides das
# células. Se a fórmula analítica reproduzir aquilo, o entendimento da
# geometria está confirmado — e aí ela pode ser usada com confiança nos
# pontos de colocação, onde não há alternativa.
#
# ------------------------------------------------------------------------------
# UMA IMPLEMENTAÇÃO SÓ DAS CINCO DISTÂNCIAS
#
# A Cell 5 precisa saber QUAL parede é a mais próxima (para medir a faixa de
# cada uma) e roda antes desta, então ela tem o seu nearest_wall_index. Esta
# célula precisa da MENOR das mesmas cinco distâncias. São a mesma conta.
#
# Duas cópias da mesma conta já deram problema neste arquivo: a réplica das
# funções de distância que a Cell 9c mantinha tinha divergido em três pontos
# antes de alguém notar. Então aqui a stack das cinco distâncias vira uma
# função (wall_distance_components) e, logo depois de defini-la, a célula
# CONFERE que o nearest_wall_index da Cell 5 concorda com ela em pontos
# sorteados. Se algum dia divergirem, quebra aqui e não no treino.
# ================================================================================


# ---------- USER CONFIG ----------
POOL_MULT = 3   # multiplicador do pool sobre per-step (maior = mais variedade)

# Rejeitar pontos de colocação MAIS PERTO DA PAREDE que o primeiro centro de
# célula daquela parede.
#
# O argumento a favor: abaixo do primeiro centro de célula o CFD não tem
# valor nenhum. O que existe ali é o perfil que o tratamento de parede
# ASSUME, e nem k, nem ω, nem a velocidade são resultado de equação
# resolvida. Impor qualquer resíduo lá é impor a hipótese do modelo de
# parede, não a física do escoamento.
#
# O argumento contra, que mantém o default em False: o resíduo de k e ω
# nessa faixa já é mascarado por WALL_RESIDUAL_MODE (Cell 2), e continuidade
# e momento ali são amarrados pelas hard constraints da Cell 9 (u=v=w=0 na
# parede), que é justamente a informação que se quer propagar para dentro do
# fluido. Tirar os pontos tira também esse mecanismo.
#
# Fica desligado, mas a contagem é impressa nos dois casos — é o número que
# a decisão precisa.
COLL_REJECT_BELOW_FIRST_CELL = False

# ------------------------------------------------------------------------------
# COLL_STEP_SCALE — quantos pontos de colocação entram em CADA passo
# ------------------------------------------------------------------------------
# MEDIDO (job 6072, A40 48 GB): com os 48.000 pontos da versão laminar, o
# grafo compilado pelo XLA pediu 39,06 GiB e não coube. O job morreu ali.
#
# Por que o turbulento não cabe e o laminar cabia, com o MESMO número de
# pontos: o passo precisa de terceira derivada, e o número de varreduras
# do grafo cresceu com o SST. No laminar são 4 saídas (u,v,w,P) e um
# laplaciano por componente de velocidade. Aqui são 6 saídas, mais os
# laplacianos de k e ω, mais ∇μ_t por autodiff, mais F1, F2, CD_kω e o
# limitador — cada um deles é mais uma passagem registrada na fita. A
# memória é aproximadamente linear no nº de pontos, então é ESTE o botão.
#
# O que a escala NÃO muda (de propósito, para manter a comparação com o
# laminar honesta):
#   - o pool continua com os mesmos 144.000 pontos e o mesmo layout de
#     regiões — todas as tabelas de diagnóstico continuam comparáveis;
#   - as PROPORÇÕES entre regiões são preservadas (a escala é a mesma
#     para todas);
#   - a cobertura por época é a mesma, porque o pool é o mesmo e o
#     sorteio é por passo.
# O que muda: cada passo vê menos pontos, então o gradiente da física é
# mais ruidoso por passo. Com POOL_MULT efetivo maior (o pool não
# encolheu), a variedade entre passos aumenta, o que compensa em parte.
#
# 1.0  = 48.000/passo, ~39 GiB  -> NÃO COUBE na A40
# 0.5  = 24.000/passo, ~20 GiB  -> default
# 0.33 = 16.000/passo, ~13 GiB  -> se 0.5 ainda estourar
COLL_STEP_SCALE = 0.5

_REGION_DEFS = [
    # (nome,    per-step com COLL_STEP_SCALE = 1)
    ("adj_inlet",    2_000),
    ("adj_outlet",   2_000),
    ("adj_xmin",     2_000),
    ("adj_xmax",     2_000),
    ("adj_ground",   2_000),
    ("adj_top",      2_000),
    ("adj_cyl_lat",  3_500),
    ("adj_cyl_top",    500),
    ("boi_c",       11_000),
    ("boi_t",        5_000),
    ("free",        16_000),
]
# ----------------------------------


# Compute layout do pool
COLLOCATION_REGIONS = []
_cursor = 0
for _name, _per_step_full in _REGION_DEFS:
    # O POOL usa o valor cheio: seu tamanho e o layout de regiões ficam
    # idênticos aos do laminar qualquer que seja COLL_STEP_SCALE.
    _pool_size = _per_step_full * POOL_MULT
    _per_step  = max(1, int(round(_per_step_full * COLL_STEP_SCALE)))
    COLLOCATION_REGIONS.append({
        "name": _name,
        "per_step": _per_step,
        "per_step_full": _per_step_full,
        "pool_size": _pool_size,
        "start": _cursor,
        "end": _cursor + _pool_size,
    })
    _cursor += _pool_size

N_COLL = _cursor    # 144.000 — independente de COLL_STEP_SCALE
N_COLL_PER_STEP = sum(r["per_step"] for r in COLLOCATION_REGIONS)
N_COLL_PER_STEP_FULL = sum(r["per_step_full"] for r in COLLOCATION_REGIONS)  # 48.000
print(f"\n[colocação] COLL_STEP_SCALE = {COLL_STEP_SCALE:g}  ->  "
      f"{N_COLL_PER_STEP:,} pontos/passo "
      f"(cheio: {N_COLL_PER_STEP_FULL:,}; pool: {N_COLL:,}, "
      f"POOL_MULT efetivo {N_COLL/N_COLL_PER_STEP:.1f}×)")


# ================================================================================
# DISTÂNCIA À PAREDE — analítica
# ================================================================================

def distance_to_cylinder(x, y, z):
    """
    Distância de um ponto EXTERIOR à superfície do cilindro sólido
    (lateral + topo). Base assente no chão, então a face de baixo não conta.

    Em coordenadas cilíndricas o cilindro é uma "caixa" em (r, z):
        dr = r - R_CYL     folga radial
        dz = z - H_CYL     folga vertical
    A distância a uma caixa, por fora, é a norma das folgas POSITIVAS:
        d = sqrt(max(dr,0)² + max(dz,0)²)
    Os três casos saem sozinhos:
        ao lado  (dr>0, dz≤0) -> dr          radial puro
        acima    (dr≤0, dz>0) -> dz          vertical puro
        na quina (dr>0, dz>0) -> hipotenusa  a aresta superior
        dentro   (ambos ≤0)   -> 0
    """
    r  = np.sqrt((x - XC_CYL)**2 + (y - YC_CYL)**2)
    dr = np.maximum(r - R_CYL, 0.0)
    dz = np.maximum(z - H_CYL, 0.0)
    return np.sqrt(dr*dr + dz*dz)


def wall_distance_components(x, y, z):
    """
    As CINCO distâncias candidatas, na ordem de WALL_CANDIDATE_NAMES (Cell 5):
        ground, top, x_min, x_max, cilindro

    Ponto único de verdade da geometria de parede. wall_distance tira o
    mínimo, nearest_wall tira o argmin, e o nearest_wall_index da Cell 5 —
    que existe porque aquela célula roda antes desta — é conferido contra
    esta função logo abaixo.

    Zonas do tipo wall (ver WALL_ZONES, Cell 2): ground, top, lateral_min,
    lateral_max, cylinder_wall, cylinder_top. inlet e outlet NÃO entram, são
    velocity-inlet e pressure-outlet — por isso y não aparece em nenhum
    termo de caixa.
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    return np.stack([
        z - Z_MIN,                      # ground      (fixa)
        Z_MAX - z,                      # top         (móvel, 17 m/s em +y)
        x - X_MIN,                      # lateral_min (móvel)
        X_MAX - x,                      # lateral_max (móvel)
        distance_to_cylinder(x, y, z),  # cylinder_wall + cylinder_top (fixas)
    ], axis=-1)


def wall_distance(x, y, z, apply_floor=True):
    """
    Distância à parede mais próxima [m].

    O piso D_WALL_FLOOR importa: F1 e F2 dividem por y², e os samplers das
    cascas adj_* sorteiam pontos encostados na parede. Sem piso, 1/y² estoura.
    """
    d = wall_distance_components(x, y, z).min(axis=-1)
    return np.maximum(d, D_WALL_FLOOR) if apply_floor else d


def nearest_wall(x, y, z):
    """Índice da parede mais próxima, na ordem de WALL_CANDIDATE_NAMES."""
    return wall_distance_components(x, y, z).argmin(axis=-1)


def _first_cell_by_point(x, y, z):
    """
    Distância do 1º centro de célula da parede mais próxima de cada ponto,
    medida pela Cell 5. Zero onde a Cell 5 não mediu (aí nada é rejeitado).
    """
    lut = np.array([Y_FIRST_BY_WALL.get(nm, 0.0)
                    for nm in WALL_CANDIDATE_NAMES], np.float64)
    return lut[nearest_wall(x, y, z)]


# ---- a Cell 5 e a Cell 7 têm que concordar sobre qual parede é a mais
#      próxima. São duas funções porque a ordem das células obriga; são a
#      mesma conta, e é isto que garante que continuem sendo.
_rng_chk = np.random.default_rng(SEED)
_xc = _rng_chk.uniform(X_MIN, X_MAX, 20_000)
_yc = _rng_chk.uniform(Y_MIN, Y_MAX, 20_000)
_zc = _rng_chk.uniform(Z_MIN, Z_MAX, 20_000)
_ok_idx = np.array_equal(nearest_wall(_xc, _yc, _zc),
                         nearest_wall_index(_xc, _yc, _zc))
_ok_dst = np.allclose(wall_distance(_xc, _yc, _zc, apply_floor=False),
                      wall_distance_components(_xc, _yc, _zc).min(axis=-1))
print(f"  [geometria] nearest_wall (Cell 7) == nearest_wall_index (Cell 5) "
      f"em 20.000 pontos: {_ok_idx}")
assert _ok_idx and _ok_dst, (
    "as duas implementações da distância à parede divergiram. "
    "Elas descrevem a MESMA geometria; corrigir antes de seguir.")
del _rng_chk, _xc, _yc, _zc


# ---- conferência contra o cell-wall-distance do Fluent ----
# Este é o teste que decide se a fórmula acima descreve a mesma geometria que
# o solver viu. Se passar, d_wall nos pontos de colocação é confiável.
def validate_wall_distance(path, sep=None, max_rows=None, verbose=True):
    sep = sep if sep is not None else CSV_DELIMITER
    df = pd.read_csv(path, sep=sep, engine="python", skipinitialspace=True)
    df.columns = [re.sub(r'["\'﻿]', '', c).strip().lower() for c in df.columns]
    need = ["x-coordinate", "y-coordinate", "z-coordinate", "cell-wall-distance"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"  [d_wall] {os.path.basename(path)} não tem {miss} — sem conferência.")
        print(f"           colunas presentes: {list(df.columns)}")
        return None
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=SEED)

    x = df["x-coordinate"].to_numpy(np.float64)
    y = df["y-coordinate"].to_numpy(np.float64)
    z = df["z-coordinate"].to_numpy(np.float64)
    d_fluent = df["cell-wall-distance"].to_numpy(np.float64)
    d_calc   = wall_distance(x, y, z, apply_floor=False)

    err  = d_calc - d_fluent
    rel  = err / np.maximum(d_fluent, 1e-9)
    corr = float(np.corrcoef(d_calc, d_fluent)[0, 1])

    if verbose:
        print(f"\n  [d_wall] fórmula analítica × cell-wall-distance do Fluent")
        print(f"    {len(df):,} células   correlação = {corr:.6f}")
        print(f"    erro absoluto: mediana {np.median(np.abs(err)):.4f} m   "
              f"p99 {np.percentile(np.abs(err), 99):.4f} m   "
              f"máx {np.abs(err).max():.4f} m")
        print(f"    erro relativo: mediana {100*np.median(np.abs(rel)):.3f}%   "
              f"p99 {100*np.percentile(np.abs(rel), 99):.3f}%")
        print(f"    viés (mediana do erro com sinal): {np.median(err):+.4f} m")

        # Tolerância: o Fluent mede distância às FACES da malha, que são uma
        # aproximação poliédrica do cilindro. Divergir na escala da célula
        # (~ alguns metros nesta malha) é esperado; divergir sistematicamente,
        # ou perder correlação, significa geometria diferente.
        ok_corr = corr > 0.999
        ok_bias = abs(float(np.median(err))) < 1.0
        if ok_corr and ok_bias:
            print(f"    -> CONFERE. A fórmula descreve a mesma geometria que o")
            print(f"       solver viu. Pode ser usada nos pontos de colocação.")
        else:
            print(f"    >>> NÃO CONFERE.")
            if not ok_corr:
                print(f"        Correlação {corr:.4f} — não é a mesma função.")
                print(f"        Hipótese mais provável: o conjunto de zonas-parede")
                print(f"        está errado. Testar tirar 'top' (se o Fluent o")
                print(f"        tratar como simetria) ou incluir inlet/outlet.")
            if not ok_bias:
                print(f"        Viés de {np.median(err):+.2f} m — há um deslocamento")
                print(f"        sistemático, não ruído de discretização.")
            print(f"        F1 do SST depende disto; não seguir sem resolver.")
    return dict(corr=corr, med_abs=float(np.median(np.abs(err))),
                bias=float(np.median(err)), n=len(df))


WALL_DIST_CHECK = None
if HAS_STATIC_MESH:
    WALL_DIST_CHECK = validate_wall_distance(static_mesh_path, max_rows=200_000)
else:
    print(f"\n  [d_wall] sem static_mesh.csv — a fórmula analítica NÃO foi")
    print(f"           conferida contra o solver. Ela é usada assim mesmo (não")
    print(f"           há alternativa nos pontos de colocação), mas um erro na")
    print(f"           lista de zonas-parede passaria despercebido.")


# ================================================================================
# Funções de sampling — uniforme dentro de cada região, com exclusões
# ================================================================================
# Todas as funções `is_inside_cylinder`, `is_inside_boi_cyl`, `is_inside_boi_tri`,
# `is_adj_cyl_lateral`, `is_adj_cyl_top` vêm da Cell 5 reformulada.
# INALTERADAS em relação ao laminar.


def _make_slab_sampler(axis, side):
    """
    Factory: retorna função que amostra n pts uniformemente em uma camada
    de espessura BC_TOL adjacente a uma face do domínio.

    axis: 'x' | 'y' | 'z'    side: 'min' | 'max'
    """
    AXIS_BOUNDS = {'x': (X_MIN, X_MAX), 'y': (Y_MIN, Y_MAX), 'z': (Z_MIN, Z_MAX)}
    a_min, a_max = AXIS_BOUNDS[axis]
    if side == 'min':
        slab_lo, slab_hi = a_min, a_min + BC_TOL
    else:
        slab_lo, slab_hi = a_max - BC_TOL, a_max

    bounds = dict(AXIS_BOUNDS)   # copia
    bounds[axis] = (slab_lo, slab_hi)

    def sampler(n, rng):
        out = np.empty((n, 3), dtype=np.float32)
        written = 0
        while written < n:
            n_try = (n - written) + 256
            x = rng.uniform(*bounds['x'], n_try).astype(np.float32)
            y = rng.uniform(*bounds['y'], n_try).astype(np.float32)
            z = rng.uniform(*bounds['z'], n_try).astype(np.float32)
            # Exclui interior do cilindro (necessário p/ ground; demais slabs não tocam)
            keep = ~is_inside_cylinder(x, y, z)
            n_keep = int(keep.sum())
            take = min(n_keep, n - written)
            out[written:written+take, 0] = x[keep][:take]
            out[written:written+take, 1] = y[keep][:take]
            out[written:written+take, 2] = z[keep][:take]
            written += take
        return out
    return sampler

# As 6 BCs externas
sample_adj_inlet_pts  = _make_slab_sampler('y', 'min')
sample_adj_outlet_pts = _make_slab_sampler('y', 'max')
sample_adj_xmin_pts   = _make_slab_sampler('x', 'min')
sample_adj_xmax_pts   = _make_slab_sampler('x', 'max')
sample_adj_ground_pts = _make_slab_sampler('z', 'min')
sample_adj_top_pts    = _make_slab_sampler('z', 'max')


def sample_adj_cyl_lateral_pts(n, rng):
    """
    Anel cilíndrico exterior à parede do cilindro: r ∈ [R, R+CYL_TOL], z ∈ [0, H].
    Usa r²-uniforme para garantir densidade uniforme em área (XY).
    """
    out = np.empty((n, 3), dtype=np.float32)
    written = 0
    while written < n:
        n_try = (n - written) + 256
        # r² uniforme → área uniforme
        r = np.sqrt(rng.uniform(R_CYL**2, (R_CYL + CYL_TOL)**2, n_try)).astype(np.float32)
        theta = rng.uniform(0.0, 2*np.pi, n_try).astype(np.float32)
        x = XC_CYL + r * np.cos(theta)
        y = YC_CYL + r * np.sin(theta)
        z = rng.uniform(Z_MIN, H_CYL, n_try).astype(np.float32)
        take = min(n_try, n - written)
        out[written:written+take, 0] = x[:take]
        out[written:written+take, 1] = y[:take]
        out[written:written+take, 2] = z[:take]
        written += take
    return out


def sample_adj_cyl_top_pts(n, rng):
    """
    Disco no topo do cilindro: r ∈ [0, R], z ∈ [H, H+CYL_TOL].
    r²-uniforme p/ densidade uniforme em área.
    """
    out = np.empty((n, 3), dtype=np.float32)
    written = 0
    while written < n:
        n_try = (n - written) + 256
        r = np.sqrt(rng.uniform(0.0, R_CYL**2, n_try)).astype(np.float32)
        theta = rng.uniform(0.0, 2*np.pi, n_try).astype(np.float32)
        x = XC_CYL + r * np.cos(theta)
        y = YC_CYL + r * np.sin(theta)
        z = rng.uniform(H_CYL, H_CYL + CYL_TOL, n_try).astype(np.float32)
        take = min(n_try, n - written)
        out[written:written+take, 0] = x[:take]
        out[written:written+take, 1] = y[:take]
        out[written:written+take, 2] = z[:take]
        written += take
    return out


def sample_boi_cyl_pts(n, rng):
    """
    BOI_C, EXCLUINDO cilindro sólido + cascas adj_cyl (estão em outras regiões).
    Rejection: cyl sólido ~23% + adj_cyl ~3% = ~26% rejeição → mult 1.6×.
    """
    out = np.empty((n, 3), dtype=np.float32)
    written = 0
    while written < n:
        n_try = int((n - written) * 1.6) + 256
        x = rng.uniform(BOI_C_X[0], BOI_C_X[1], n_try).astype(np.float32)
        y = rng.uniform(BOI_C_Y[0], BOI_C_Y[1], n_try).astype(np.float32)
        z = rng.uniform(BOI_C_Z[0], BOI_C_Z[1], n_try).astype(np.float32)
        keep = ~(is_inside_cylinder(x, y, z)
                 | is_adj_cyl_lateral(x, y, z)
                 | is_adj_cyl_top(x, y, z))
        n_keep = int(keep.sum())
        take = min(n_keep, n - written)
        out[written:written+take, 0] = x[keep][:take]
        out[written:written+take, 1] = y[keep][:take]
        out[written:written+take, 2] = z[keep][:take]
        written += take
    return out


def sample_boi_tri_pts(n, rng):
    """
    Prisma triangular BOI_T. Bounding box 80 x 360 m²; triângulo ocupa ~50%.
    Rejection rate ~50% → mult 2.5 x.
    BOI_T não overlap com cilindro nem com adj_cyl (longe), sem exclusões extra.
    """
    v1, v2, v3 = BOI_T_V1, BOI_T_V2, BOI_T_V3
    x_min = min(v1[0], v2[0], v3[0]); x_max = max(v1[0], v2[0], v3[0])
    y_min = min(v1[1], v2[1], v3[1]); y_max = max(v1[1], v2[1], v3[1])

    out = np.empty((n, 3), dtype=np.float32)
    written = 0
    while written < n:
        n_try = int((n - written) * 2.5) + 256
        x = rng.uniform(x_min, x_max, n_try).astype(np.float32)
        y = rng.uniform(y_min, y_max, n_try).astype(np.float32)
        z = rng.uniform(BOI_T_Z[0], BOI_T_Z[1], n_try).astype(np.float32)
        keep = is_inside_boi_tri(x, y, z)
        n_keep = int(keep.sum())
        take = min(n_keep, n - written)
        out[written:written+take, 0] = x[keep][:take]
        out[written:written+take, 1] = y[keep][:take]
        out[written:written+take, 2] = z[keep][:take]
        written += take
    return out


def sample_freestream_pts(n, rng):
    """
    Freestream = domínio menos: cilindro, BOI_C, BOI_T, todas as cascas adj_BC,
    cascas adj_cyl. Rejection rate ~12% → mult 1.4×.
    """
    out = np.empty((n, 3), dtype=np.float32)
    written = 0
    while written < n:
        n_try = int((n - written) * 1.4) + 256
        x = rng.uniform(X_MIN, X_MAX, n_try).astype(np.float32)
        y = rng.uniform(Y_MIN, Y_MAX, n_try).astype(np.float32)
        z = rng.uniform(Z_MIN, Z_MAX, n_try).astype(np.float32)
        in_cyl_solid = is_inside_cylinder(x, y, z)
        in_boi_c     = is_inside_boi_cyl(x, y, z)
        in_boi_t     = is_inside_boi_tri(x, y, z)
        in_adj_bc = (
            (y < Y_MIN + BC_TOL) | (y > Y_MAX - BC_TOL) |
            (x < X_MIN + BC_TOL) | (x > X_MAX - BC_TOL) |
            (z < Z_MIN + BC_TOL) | (z > Z_MAX - BC_TOL)
        )
        in_adj_cyl_lat = is_adj_cyl_lateral(x, y, z)
        in_adj_cyl_top = is_adj_cyl_top(x, y, z)
        keep = ~(in_cyl_solid | in_boi_c | in_boi_t | in_adj_bc
                 | in_adj_cyl_lat | in_adj_cyl_top)
        n_keep = int(keep.sum())
        take = min(n_keep, n - written)
        out[written:written+take, 0] = x[keep][:take]
        out[written:written+take, 1] = y[keep][:take]
        out[written:written+take, 2] = z[keep][:take]
        written += take
    return out


# Mapeamento name → sampler (para o loop de geração)
SAMPLERS = {
    "adj_inlet":    sample_adj_inlet_pts,
    "adj_outlet":   sample_adj_outlet_pts,
    "adj_xmin":     sample_adj_xmin_pts,
    "adj_xmax":     sample_adj_xmax_pts,
    "adj_ground":   sample_adj_ground_pts,
    "adj_top":      sample_adj_top_pts,
    "adj_cyl_lat":  sample_adj_cyl_lateral_pts,
    "adj_cyl_top":  sample_adj_cyl_top_pts,
    "boi_c":        sample_boi_cyl_pts,
    "boi_t":        sample_boi_tri_pts,
    "free":         sample_freestream_pts,
}


def sample_collocation():
    """
    Gera pool em layout determinístico de 11 regiões. Sem shuffle final.
    Tempo: uniforme em [TN_MIN, TN_MAX] (range real dos dados normalizados).

    Returns:
      coll_np    : (N_COLL, 4) float32 — [xn, yn, zn, tn]  (igual ao laminar)
      coll_dwall : (N_COLL,)   float32 — distância à parede em METROS
      coll_ywall : (N_COLL,)   float32 — limiar da faixa de tratamento de
                   parede DAQUELE ponto, em metros (a faixa da parede mais
                   próxima dele). A Cell 10 mascara onde d_wall <= y_wall.

      Os dois últimos são arrays SEPARADOS de propósito: são coeficientes do
      resíduo, não entradas da rede. Como colunas de coll_np entrariam no
      GradientTape e a rede acabaria recebendo 6 features em vez de 4.
    """
    rng_coll = np.random.default_rng(SEED)
    pieces = []
    n_redrawn = 0
    for r in COLLOCATION_REGIONS:
        pts = SAMPLERS[r["name"]](r["pool_size"], rng_coll)
        if COLL_REJECT_BELOW_FIRST_CELL and Y_FIRST_BY_WALL:
            # Rejeição POR REAMOSTRAGEM, não por descarte: o layout
            # determinístico de COLLOCATION_REGIONS (start/end) é usado pela
            # Cell 8 e pelo treino. Encurtar um bloco desalinharia todos os
            # seguintes em silêncio. Aqui cada ponto rejeitado é substituído
            # por outro da mesma região, e o tamanho não muda.
            for _ in range(40):
                d1 = _first_cell_by_point(pts[:, 0], pts[:, 1], pts[:, 2])
                bad = wall_distance(pts[:, 0].astype(np.float64),
                                    pts[:, 1].astype(np.float64),
                                    pts[:, 2].astype(np.float64)) < d1
                nb = int(bad.sum())
                if nb == 0:
                    break
                n_redrawn += nb
                pts[bad] = SAMPLERS[r["name"]](nb, rng_coll)
            else:
                print(f"  [aviso] {r['name']}: ainda restam {nb} pontos abaixo "
                      f"da 1ª célula após 40 reamostragens — a casca é mais "
                      f"fina que a primeira camada dessa parede.")
        pieces.append(pts)
    if n_redrawn:
        print(f"  [colocação] {n_redrawn:,} pontos reamostrados por caírem "
              f"abaixo do 1º centro de célula (COLL_REJECT_BELOW_FIRST_CELL)")
    xyz = np.concatenate(pieces, axis=0)
    assert len(xyz) == N_COLL, f"pool size mismatch: {len(xyz)} vs {N_COLL}"

    t = rng_coll.uniform(TN_MIN, TN_MAX, len(xyz)).astype(np.float32).reshape(-1, 1)

    # xyz_n = xyz / L_REF <-- Normalização antiga por L_REF (= y_max)
    x_n = (xyz[: ,0] - X_MEAN) / DP_ISO
    y_n = (xyz[: ,1] - Y_MEAN) / DP_ISO
    z_n = (xyz[: ,2] - Z_MEAN) / DP_ISO

    # d_wall e o limiar da faixa: nas coordenadas FÍSICAS, antes de normalizar
    xf = xyz[:, 0].astype(np.float64)
    yf = xyz[:, 1].astype(np.float64)
    zf = xyz[:, 2].astype(np.float64)
    d_w = wall_distance(xf, yf, zf).astype(np.float32)
    # wall_band_by_point vem da Cell 5, que MEDIU a faixa de cada parede no
    # static_mesh.csv. Sem static_mesh alinhado ela cai no escalar
    # Y_WALL_FUNCTION, e a Cell 5 avisa quando isso acontece.
    y_w = wall_band_by_point(xf, yf, zf).astype(np.float32)

    return (np.column_stack([x_n, y_n, z_n, t]).astype(np.float32), d_w, y_w)


# ----- Generate -----
print(f"Gerando pool de collocation...")
_t0 = time.time()
coll_np, coll_dwall, coll_ywall = sample_collocation()
print(f"Tempo de geração: {time.time()-_t0:.2f}s")

# Pontos abaixo do primeiro centro de célula: contados SEMPRE, removidos só
# se COLL_REJECT_BELOW_FIRST_CELL (por reamostragem, dentro do sampler — o
# layout de regiões não muda nos dois casos).
_y1_pt = _first_cell_by_point(coll_np[:, 0] * DP_ISO + X_MEAN,
                              coll_np[:, 1] * DP_ISO + Y_MEAN,
                              coll_np[:, 2] * DP_ISO + Z_MEAN)
_below = coll_dwall < _y1_pt
print(f"\nPontos mais perto da parede que o 1º centro de célula daquela")
print(f"parede: {int(_below.sum()):,} de {N_COLL:,} ({100*_below.mean():.2f}%)")
print(f"  Ali o CFD não tem valor nenhum — o que existe é o perfil que o")
print(f"  tratamento de parede assume (COLL_REJECT_BELOW_FIRST_CELL = "
      f"{COLL_REJECT_BELOW_FIRST_CELL}).")

print(f"\nCollocation pool: {N_COLL:,} pontos (POOL_MULT={POOL_MULT})")
print(f"Per step total: {N_COLL_PER_STEP:,}")
print(f"\nGrupos:")
n_bc   = sum(r["per_step"] for r in COLLOCATION_REGIONS if r["name"].startswith("adj"))
n_boi  = sum(r["per_step"] for r in COLLOCATION_REGIONS if r["name"].startswith("boi"))
n_free = next(r["per_step"] for r in COLLOCATION_REGIONS if r["name"] == "free")
print(f"  BCs       (8 sub-regiões): {n_bc:>6,}  ({100*n_bc/N_COLL_PER_STEP:>4.1f}%)")
print(f"  Detail BOI (2 sub-regiões): {n_boi:>6,}  ({100*n_boi/N_COLL_PER_STEP:>4.1f}%)")
print(f"  Freestream                : {n_free:>6,}  ({100*n_free/N_COLL_PER_STEP:>4.1f}%)")
print(f"\nLayout do pool:")
for r in COLLOCATION_REGIONS:
    print(f"  {r['name']:<14} [{r['start']:>6d}, {r['end']:>6d})  "
          f"per_step={r['per_step']:>5d}  pool={r['pool_size']:>6d}")
print(f"\nTempo normalizado sorteado em: [{TN_MIN:.4f}, {TN_MAX:.4f}]")


print(f"coll_np shape: {coll_np.shape}")  # Deve ser (N_COLL, 4)
print(f"\nRange das coords normalizadas:")
for i, name in enumerate(['xn', 'yn', 'zn', 'tn']):
    col = coll_np[:, i]
    print(f"  {name}: [{col.min():+.3f}, {col.max():+.3f}]  μ={col.mean():+.3f}")

# Desnormaliza pra verificar que volta ao domínio físico
x_phys = coll_np[:, 0] * DP_ISO + X_MEAN
y_phys = coll_np[:, 1] * DP_ISO + Y_MEAN
z_phys = coll_np[:, 2] * DP_ISO + Z_MEAN
print(f"\nRange físico (após desnormalizar):")
print(f"  x: [{x_phys.min():.1f}, {x_phys.max():.1f}] m  (esperado: [{X_MIN:.0f}, {X_MAX:.0f}])")
print(f"  y: [{y_phys.min():.1f}, {y_phys.max():.1f}] m  (esperado: [{Y_MIN:.0f}, {Y_MAX:.0f}])")
print(f"  z: [{z_phys.min():.1f}, {z_phys.max():.1f}] m  (esperado: [{Z_MIN:.0f}, {Z_MAX:.0f}])")

# Nenhum ponto pode ter caído dentro do cilindro sólido
_inside = is_inside_cylinder(x_phys, y_phys, z_phys)
print(f"\n  pontos dentro do cilindro sólido: {int(_inside.sum())} (tem que ser 0)")
assert _inside.sum() == 0, "pool de colocação tem pontos dentro do cilindro"


# ================================================================================
# d_wall por região — o que F1 vai ver
# ================================================================================
print(f"\n{'='*78}")
print(f"  DISTÂNCIA À PAREDE NO POOL DE COLOCAÇÃO")
print(f"{'='*78}")
print(f"{'Região':<14}{'d_wall p01':>12}{'p50':>11}{'p99':>11}"
      f"{'no piso':>10}{'na faixa':>10}{'F1→1':>8}")
print(f"{'-'*78}")

# 'na faixa' é MEDIDO: fração de pontos com d_wall <= o limiar da faixa de
# tratamento de parede daquele ponto (Cell 5, por parede). É exatamente o
# conjunto que a Cell 10 vai mascarar no resíduo de k e ω.
#
# 'F1→1' é uma ESTIMATIVA grosseira: o primeiro termo do argumento de F1 é
# √k/(β*·ω·y), que passa de 1 quando y < √k/(β*·ω). Com k e ω típicos do
# interior dá uma noção de onde F1 satura. O valor de verdade só sai ponto a
# ponto na Cell 10, onde k e ω são os da rede.
_k_typ  = float(np.exp(LNK_MEAN)) if K_NORM_MODE == "logz" else float(K_REF)
_om_typ = float(np.exp(LNW_MEAN))
_y_switch = np.sqrt(_k_typ) / (SST_BETA_STAR * _om_typ)

_in_band_all = coll_dwall <= coll_ywall
for r in COLLOCATION_REGIONS:
    sl = slice(r["start"], r["end"])
    d  = coll_dwall[sl]
    n_floor = int((d <= D_WALL_FLOOR * 1.0000001).sum())
    f_band  = float(_in_band_all[sl].mean())
    n_wall  = int((d < _y_switch).sum())
    print(f"{r['name']:<14}{np.percentile(d,1):>12.3f}{np.percentile(d,50):>11.3f}"
          f"{np.percentile(d,99):>11.3f}{n_floor:>10,}"
          f"{100*f_band:>9.1f}%{100*n_wall/len(d):>7.0f}%")
print(f"{'-'*78}")
print(f"{'POOL':<14}{np.percentile(coll_dwall,1):>12.3f}"
      f"{np.percentile(coll_dwall,50):>11.3f}{np.percentile(coll_dwall,99):>11.3f}"
      f"{int((coll_dwall <= D_WALL_FLOOR*1.0000001).sum()):>10,}"
      f"{100*_in_band_all.mean():>9.1f}%"
      f"{100*(coll_dwall < _y_switch).mean():>7.0f}%")
print(f"{'='*78}")
print(f"  d_wall ∈ [{coll_dwall.min():.4f}, {coll_dwall.max():.2f}] m   "
      f"piso D_WALL_FLOOR = {D_WALL_FLOOR:.1e} m")
print(f"  'F1→1' estima a fração em regime de parede, com y < "
      f"√k/(β*·ω) = {_y_switch:.2f} m")
print(f"  (k≈{_k_typ:.3f} m²/s², ω≈{_om_typ:.1f} 1/s, medianas da Cell 4)")


# ================================================================================
# O LIMIAR POR PONTO — quanto ele muda em relação a um escalar
# ================================================================================
print(f"\n{'='*78}")
print(f"  FAIXA DE TRATAMENTO DE PAREDE NO POOL")
print(f"{'='*78}")
if Y_BAND_BY_WALL:
    print(f"  limiares medidos pela Cell 5, por parede [m]:")
    for _nm in WALL_CANDIDATE_NAMES:
        if _nm in Y_BAND_BY_WALL:
            _n_here = int((nearest_wall(x_phys, y_phys, z_phys) ==
                           WALL_CANDIDATE_NAMES.index(_nm)).sum())
            print(f"    {_nm:<10} {Y_BAND_BY_WALL[_nm]:>7.3f}     "
                  f"({_n_here:>7,} pontos do pool têm esta como parede mais próxima)")
    _n_pt  = int(_in_band_all.sum())
    _n_sc  = int((coll_dwall <= Y_WALL_FUNCTION).sum())
    print(f"\n  pontos dentro da faixa:")
    print(f"    com limiar POR PONTO (coll_ywall) : {_n_pt:>8,}  "
          f"({100*_n_pt/len(coll_dwall):.2f}%)")
    print(f"    com o escalar Y_WALL_FUNCTION={Y_WALL_FUNCTION:.3f} m : "
          f"{_n_sc:>8,}  ({100*_n_sc/len(coll_dwall):.2f}%)")
    _delta = abs(_n_sc - _n_pt)
    print(f"    diferença: {_delta:,} pontos "
          f"({100*_delta/max(len(coll_dwall),1):.2f}% do pool)")
    print(f"  Esses são os pontos que o escalar classificaria errado — quase")
    print(f"  todos de 2ª camada ou além nas paredes mais finas. É por isso")
    print(f"  que coll_ywall existe.")
else:
    print(f"  A Cell 5 não mediu as faixas (sem static_mesh alinhado).")
    print(f"  coll_ywall = Y_WALL_FUNCTION = {Y_WALL_FUNCTION:.3f} m em todos os")
    print(f"  pontos, que é o comportamento antigo. O resíduo funciona, mas a")
    print(f"  máscara trata todas as paredes como se tivessem o mesmo refino.")
print(f"  WALL_RESIDUAL_MODE = '{WALL_RESIDUAL_MODE}' (Cell 2)")
print(f"{'='*78}")

_n_floor_tot = int((coll_dwall <= D_WALL_FLOOR * 1.0000001).sum())
if _n_floor_tot > 0:
    print(f"\n  {_n_floor_tot:,} pontos ({100*_n_floor_tot/N_COLL:.3f}%) estão NO PISO.")
    print(f"  São pontos sorteados praticamente encostados na parede. F1 e F2")
    print(f"  dividem por y², então sem o piso o resíduo iria a infinito ali.")
    print(f"  Com o piso, 1/y² = {1/D_WALL_FLOOR**2:.1e} — grande, mas finito em")
    print(f"  float32. Se a loss divergir logo no começo, é o primeiro suspeito:")
    print(f"  subir D_WALL_FLOOR. (Tirar as cascas adj_* do resíduo de k e ω")
    print(f"  já é o que WALL_RESIDUAL_MODE='{WALL_RESIDUAL_MODE}' faz, via coll_ywall.)")

if WALL_DIST_CHECK is not None and WALL_DIST_CHECK["corr"] > 0.999:
    print(f"\n  A fórmula foi conferida contra o Fluent "
          f"(correlação {WALL_DIST_CHECK['corr']:.6f}) — ver acima.")


# ================================================================================
# CELL 8 — VISUALIZATION + SANITY CHECKS
# ================================================================================
# Os checks geométricos ficam IDÊNTICOS — a geometria não mudou.
#
# Duas mudanças:
#
# 1. UMA COR. cor_boi_c era laranja (0.95,0.55,0.10) e cor_boi_t verde
#    (0.20,0.65,0.30). Esse par fica a ΔE 3,5 sob protanopia — indistinguível
#    para quem tem a deficiência de cor mais comum em homens (~1 em 12). E são
#    justamente as duas regiões que se sobrepõem na vista XY, onde distinguir
#    importa. Trocado para roxo (0.55,0.36,0.84): o pior par entre as quatro
#    cores "reais" sobe para ΔE 24,8. O cinza do freestream fica de fora do
#    critério de propósito — ele é o fundo, não uma série.
#
# 2. DOIS PAINÉIS NOVOS, para os campos que a versão laminar não tinha.
#    A conferência numérica de d_wall já foi feita na Cell 7 (correlação com
#    o Fluent); o que um gráfico mostra e um coeficiente não é a FORMA do
#    campo: a sombra do cilindro, e o fato de d_wall não depender de y.
#    Se a lista de zonas-parede estivesse errada, isso apareceria como um
#    gradiente em y que não deveria existir.
#
# [18/10/2026] As duas figuras vão para DIAG_DIR (Cell 3), em vez de
# caírem no diretório de onde o sbatch foi chamado. Nada mais muda: a
# geometria, a malha e o pool de colocação são os mesmos da sst2.

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.colors import LogNorm

# Desnormaliza para plotar em metros
x_phys = coll_np[:, 0] * DP_ISO + X_MEAN
y_phys = coll_np[:, 1] * DP_ISO + Y_MEAN
z_phys = coll_np[:, 2] * DP_ISO + Z_MEAN

# xyz_phys = coll_np[:, :3] * L_REF
# x_phys, y_phys, z_phys = xyz_phys[:, 0], xyz_phys[:, 1], xyz_phys[:, 2]


# ============================================================
# SANITY CHECKS — cada região está dentro do esperado
# ============================================================
print(f"\n{'='*78}")
print(f"  SANITY CHECKS por região")
print(f"{'='*78}")

# Funções de verificação por região
def _check_slab(name, axis, side, tol):
    r = next(r for r in COLLOCATION_REGIONS if r["name"] == name)
    sl = slice(r["start"], r["end"])
    coord = {'x': x_phys, 'y': y_phys, 'z': z_phys}[axis][sl]
    AXIS = {'x': (X_MIN, X_MAX), 'y': (Y_MIN, Y_MAX), 'z': (Z_MIN, Z_MAX)}
    if side == 'min':
        ok = (coord >= AXIS[axis][0]) & (coord <= AXIS[axis][0] + tol)
    else:
        ok = (coord >= AXIS[axis][1] - tol) & (coord <= AXIS[axis][1])
    n_bad = int((~ok).sum())
    print(f"  {name:<14}  pts fora do slab: {n_bad}  (esperado: 0)")
    assert n_bad == 0, f"{name}: {n_bad} pontos fora do slab"

_check_slab("adj_inlet",   'y', 'min', BC_TOL)
_check_slab("adj_outlet",  'y', 'max', BC_TOL)
_check_slab("adj_xmin",    'x', 'min', BC_TOL)
_check_slab("adj_xmax",    'x', 'max', BC_TOL)
_check_slab("adj_ground",  'z', 'min', BC_TOL)
_check_slab("adj_top",     'z', 'max', BC_TOL)

# adj_cyl_lateral: r ∈ [R, R+CYL_TOL], z ∈ [0, H]
r_lat = next(r for r in COLLOCATION_REGIONS if r["name"] == "adj_cyl_lat")
sl = slice(r_lat["start"], r_lat["end"])
r_vals = np.sqrt((x_phys[sl] - XC_CYL)**2 + (y_phys[sl] - YC_CYL)**2)
ok = (r_vals >= R_CYL - 1e-3) & (r_vals <= R_CYL + CYL_TOL + 1e-3) & \
     (z_phys[sl] >= Z_MIN) & (z_phys[sl] <= H_CYL)
n_bad = int((~ok).sum())
print(f"  {'adj_cyl_lat':<14}  pts fora do anel: {n_bad}  (esperado: 0)")
assert n_bad == 0

# adj_cyl_top: r ∈ [0, R], z ∈ [H, H+CYL_TOL]
r_top = next(r for r in COLLOCATION_REGIONS if r["name"] == "adj_cyl_top")
sl = slice(r_top["start"], r_top["end"])
r_vals = np.sqrt((x_phys[sl] - XC_CYL)**2 + (y_phys[sl] - YC_CYL)**2)
ok = (r_vals <= R_CYL + 1e-3) & \
     (z_phys[sl] >= H_CYL - 1e-3) & (z_phys[sl] <= H_CYL + CYL_TOL + 1e-3)
n_bad = int((~ok).sum())
print(f"  {'adj_cyl_top':<14}  pts fora do disco: {n_bad}  (esperado: 0)")
assert n_bad == 0

# BOI_C: pts no bloco devem estar dentro de BOI_C e fora de cilindro/adj_cyl
r_bc = next(r for r in COLLOCATION_REGIONS if r["name"] == "boi_c")
sl = slice(r_bc["start"], r_bc["end"])
in_box = is_inside_boi_cyl(x_phys[sl], y_phys[sl], z_phys[sl])
in_cyl = is_inside_cylinder(x_phys[sl], y_phys[sl], z_phys[sl])
in_adj_cyl = (is_adj_cyl_lateral(x_phys[sl], y_phys[sl], z_phys[sl])
              | is_adj_cyl_top(x_phys[sl], y_phys[sl], z_phys[sl]))
n_bad = int((~in_box | in_cyl | in_adj_cyl).sum())
print(f"  {'boi_c':<14}  pts inválidos: {n_bad}  (esperado: 0)")
assert n_bad == 0

# BOI_T: dentro do prisma
r_bt = next(r for r in COLLOCATION_REGIONS if r["name"] == "boi_t")
sl = slice(r_bt["start"], r_bt["end"])
ok = is_inside_boi_tri(x_phys[sl], y_phys[sl], z_phys[sl])
n_bad = int((~ok).sum())
print(f"  {'boi_t':<14}  pts fora do prisma: {n_bad}  (esperado: 0)")
assert n_bad == 0

# Freestream: fora de TUDO
r_free = next(r for r in COLLOCATION_REGIONS if r["name"] == "free")
sl = slice(r_free["start"], r_free["end"])
in_cyl = is_inside_cylinder(x_phys[sl], y_phys[sl], z_phys[sl])
in_boi_c = is_inside_boi_cyl(x_phys[sl], y_phys[sl], z_phys[sl])
in_boi_t = is_inside_boi_tri(x_phys[sl], y_phys[sl], z_phys[sl])
in_adj_bc = (
    (y_phys[sl] < Y_MIN + BC_TOL) | (y_phys[sl] > Y_MAX - BC_TOL) |
    (x_phys[sl] < X_MIN + BC_TOL) | (x_phys[sl] > X_MAX - BC_TOL) |
    (z_phys[sl] < Z_MIN + BC_TOL) | (z_phys[sl] > Z_MAX - BC_TOL)
)
in_adj_cyl = (is_adj_cyl_lateral(x_phys[sl], y_phys[sl], z_phys[sl])
              | is_adj_cyl_top(x_phys[sl], y_phys[sl], z_phys[sl]))
n_bad = int((in_cyl | in_boi_c | in_boi_t | in_adj_bc | in_adj_cyl).sum())
print(f"  {'free':<14}  pts em outras regiões: {n_bad}  (esperado: 0)")
assert n_bad == 0

# --- NOVO: d_wall coerente com a região de onde o ponto veio ---
# d_wall é o MÍNIMO sobre todas as paredes, então nunca pode exceder a
# distância a nenhuma delas em particular. Testar contra a parede que
# DEFINE cada região pega inversão de sinal, eixo trocado e zona faltando —
# tudo que a correlação global da Cell 7 poderia mascarar.
print(f"\n  d_wall por construção:")
_dw_cases = [
    ("adj_ground",  lambda s: z_phys[s] - Z_MIN),
    ("adj_top",     lambda s: Z_MAX - z_phys[s]),
    ("adj_xmin",    lambda s: x_phys[s] - X_MIN),
    ("adj_xmax",    lambda s: X_MAX - x_phys[s]),
    ("adj_cyl_lat", lambda s: np.sqrt((x_phys[s]-XC_CYL)**2
                                      + (y_phys[s]-YC_CYL)**2) - R_CYL),
    ("adj_cyl_top", lambda s: z_phys[s] - H_CYL),
]
for _nm, _expect in _dw_cases:
    _r = next(r for r in COLLOCATION_REGIONS if r["name"] == _nm)
    _s = slice(_r["start"], _r["end"])
    _d_esperado = _expect(_s)
    _d_real = coll_dwall[_s]
    _viol = int((_d_real > _d_esperado + 1e-3).sum())
    print(f"    {_nm:<14}  d_wall > dist. à própria parede: {_viol}  (esperado: 0)")
    assert _viol == 0, f"{_nm}: d_wall excede a distância à parede que o define"

# inlet e outlet NÃO são parede: d_wall ali tem que ser GRANDE, não ~0.
_r = next(r for r in COLLOCATION_REGIONS if r["name"] == "adj_inlet")
_d_inlet = coll_dwall[_r["start"]:_r["end"]]
print(f"    {'adj_inlet':<14}  d_wall mediana = {np.median(_d_inlet):.1f} m "
      f"(tem que ser >> BC_TOL={BC_TOL}: inlet não é parede)")
assert np.median(_d_inlet) > 5 * BC_TOL, \
    "d_wall pequena no inlet — inlet está entrando como parede"

print(f"  ✓ Todos os checks passaram.\n")


# ============================================================
# PLOT 1 — XY e XZ com 5 cores agrupadas
# ============================================================
# adj_BC externas (6 faces) → azul
# adj_cyl (lat + top)       → vermelho
# BOI_C                     → roxo    (era laranja; ver cabeçalho da célula)
# BOI_T                     → verde
# free                      → cinza   (neutro de propósito: é o fundo)

cor_bc      = (0.20, 0.45, 0.85)
cor_cyl     = (0.85, 0.20, 0.20)
cor_boi_c   = (0.55, 0.36, 0.84)
cor_boi_t   = (0.20, 0.65, 0.30)
cor_free    = (0.50, 0.50, 0.55)

# Slices agrupadas
def _region_slice(name):
    r = next(r for r in COLLOCATION_REGIONS if r["name"] == name)
    return slice(r["start"], r["end"])

# Indices das BCs externas concatenadas
_external_bc_names = ["adj_inlet", "adj_outlet", "adj_xmin", "adj_xmax",
                       "adj_ground", "adj_top"]
_ext_idx = np.concatenate([
    np.arange(*_region_slice(n).indices(N_COLL)) for n in _external_bc_names
])
_cyl_idx = np.concatenate([
    np.arange(*_region_slice("adj_cyl_lat").indices(N_COLL)),
    np.arange(*_region_slice("adj_cyl_top").indices(N_COLL)),
])
_boic_idx = np.arange(*_region_slice("boi_c").indices(N_COLL))
_boit_idx = np.arange(*_region_slice("boi_t").indices(N_COLL))
_free_idx = np.arange(*_region_slice("free").indices(N_COLL))

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- XY (top) ---
ax = axes[0]
ax.scatter(x_phys[_free_idx], y_phys[_free_idx], s=0.3, alpha=0.20,
            c=[cor_free], rasterized=True, label=f'Free ({len(_free_idx)//1000}k)')
ax.scatter(x_phys[_ext_idx],  y_phys[_ext_idx],  s=0.5, alpha=0.45,
            c=[cor_bc],   rasterized=True, label=f'adj_BC ext. ({len(_ext_idx)//1000}k)')
ax.scatter(x_phys[_boit_idx], y_phys[_boit_idx], s=0.5, alpha=0.50,
            c=[cor_boi_t],rasterized=True, label=f'BOI_T ({len(_boit_idx)//1000}k)')
ax.scatter(x_phys[_boic_idx], y_phys[_boic_idx], s=0.5, alpha=0.50,
            c=[cor_boi_c],rasterized=True, label=f'BOI_C ({len(_boic_idx)//1000}k)')
ax.scatter(x_phys[_cyl_idx],  y_phys[_cyl_idx],  s=1.0, alpha=0.70,
            c=[cor_cyl],  rasterized=True, label=f'adj_cyl ({len(_cyl_idx)//1000}k)')

# Contornos
ax.add_patch(Rectangle((BOI_C_X[0], BOI_C_Y[0]),
                         BOI_C_X[1]-BOI_C_X[0], BOI_C_Y[1]-BOI_C_Y[0],
                         fill=False, edgecolor=cor_boi_c, lw=2, ls='--', zorder=5))
ax.add_patch(Polygon([BOI_T_V1, BOI_T_V2, BOI_T_V3],
                       fill=False, edgecolor=cor_boi_t, lw=2, ls='--', zorder=5))
ax.add_patch(Circle((XC_CYL, YC_CYL), R_CYL, fill=False, color='black', lw=1.5, zorder=6))

ax.annotate('', xy=(20, 120), xytext=(20, 30),
             arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax.text(25, 75, 'flow', fontsize=10, color='gray')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title(f'Top view (XY) — pool total: {N_COLL//1000}k pts')
ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=8, markerscale=8)

# --- XZ (side) ---
ax = axes[1]
ax.scatter(x_phys[_free_idx], z_phys[_free_idx], s=0.3, alpha=0.20,
            c=[cor_free], rasterized=True)
ax.scatter(x_phys[_ext_idx],  z_phys[_ext_idx],  s=0.5, alpha=0.45,
            c=[cor_bc],   rasterized=True)
ax.scatter(x_phys[_boit_idx], z_phys[_boit_idx], s=0.5, alpha=0.50,
            c=[cor_boi_t],rasterized=True)
ax.scatter(x_phys[_boic_idx], z_phys[_boic_idx], s=0.5, alpha=0.50,
            c=[cor_boi_c],rasterized=True)
ax.scatter(x_phys[_cyl_idx],  z_phys[_cyl_idx],  s=1.0, alpha=0.70,
            c=[cor_cyl],  rasterized=True)
ax.add_patch(Rectangle((XC_CYL - R_CYL, 0), 2*R_CYL, H_CYL,
                         fill=False, color='black', lw=1.5, zorder=6))
ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
ax.set_title('Side view (XZ)')
ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Z_MIN, Z_MAX)

plt.tight_layout()
_p_coll_png = os.path.join(DIAG_DIR, 'collocation_distribution.png')
plt.savefig(_p_coll_png, dpi=120)
plt.close()
print(f"Plot salvo: {_p_coll_png}")


# ============================================================
# PLOT 2 — os campos novos: d_wall, k e ω
# ============================================================
# Cada painel é uma rampa sequencial de UM tom, clara → escura. A grandeza
# é magnitude, não identidade, então o tom não distingue categorias —
# distingue "quanto". Tons diferentes entre painéis só separam GRANDEZAS
# diferentes, cada uma com sua própria barra de escala.
#
# k e ω em escala LOG: as duas varrem ordens de grandeza (ω vai de ~1 no
# interior a ~7400 no inlet). Em escala linear, 99% do domínio vira uma cor
# só e a estrutura da esteira some.

_Z_SLAB = H_CYL / 2.0        # meia altura do cilindro
_Y_SLAB = YC_CYL             # plano do eixo do cilindro
_SLAB_HALF = 6.0             # [m] espessura da fatia

def _draw_cyl_xy(ax):
    ax.add_patch(Circle((XC_CYL, YC_CYL), R_CYL, fill=False,
                        color='black', lw=1.2, zorder=6))

fig2, ax2 = plt.subplots(2, 2, figsize=(15, 13))

_m_xy = np.abs(z_phys - _Z_SLAB) < _SLAB_HALF
_m_xz = np.abs(y_phys - _Y_SLAB) < _SLAB_HALF * 3

# UMA escala de cor para os dois painéis de d_wall. São a mesma grandeza:
# com escalas separadas, 30 m apareceria escuro num painel e médio no outro,
# e a comparação entre as duas vistas — que é o objetivo — ficaria falsa.
_dw_vmin = 0.0
_dw_vmax = float(np.percentile(
    np.concatenate([coll_dwall[_m_xy], coll_dwall[_m_xz]]), 99.5))

# --- (0,0) d_wall em XY ---
ax = ax2[0, 0]
sc = ax.scatter(x_phys[_m_xy], y_phys[_m_xy], c=coll_dwall[_m_xy], s=1.5,
                cmap='Blues', vmin=_dw_vmin, vmax=_dw_vmax, rasterized=True)
plt.colorbar(sc, ax=ax, label='d_wall [m]')
_draw_cyl_xy(ax)
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal')
ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
ax.set_title(f'Distância à parede — XY em z={_Z_SLAB:.0f}±{_SLAB_HALF:.0f} m\n'
             f'({int(_m_xy.sum()):,} pts do pool)', fontsize=10)

# --- (0,1) d_wall em XZ ---
ax = ax2[0, 1]
sc = ax.scatter(x_phys[_m_xz], z_phys[_m_xz], c=coll_dwall[_m_xz], s=1.5,
                cmap='Blues', vmin=_dw_vmin, vmax=_dw_vmax, rasterized=True)
plt.colorbar(sc, ax=ax, label='d_wall [m]')
ax.add_patch(Rectangle((XC_CYL - R_CYL, 0), 2*R_CYL, H_CYL,
                       fill=False, color='black', lw=1.2, zorder=6))
ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Z_MIN, Z_MAX)
ax.set_title(f'Distância à parede — XZ em y={_Y_SLAB:.0f}±{3*_SLAB_HALF:.0f} m\n'
             f'(mesma escala de cor do painel ao lado; y NÃO entra)', fontsize=10)

# --- (1,0) e (1,1): k e ω do primeiro snapshot ---
if first_snap_raw is not None:
    _fx, _fy, _fz = first_snap_raw["x"], first_snap_raw["y"], first_snap_raw["z"]
    _fk, _fom     = first_snap_raw["k"], first_snap_raw["omega"]
    mf = np.abs(_fz - _Z_SLAB) < _SLAB_HALF

    for _ax, _val, _lab, _cmap in [
            (ax2[1, 0], np.maximum(_fk[mf],  K_FLOOR),     'k [m²/s²]',      'Oranges'),
            (ax2[1, 1], np.maximum(_fom[mf], OMEGA_FLOOR), 'ω [1/s]',        'Purples')]:
        _lo = max(float(np.percentile(_val, 1)), 1e-8)
        _hi = float(np.percentile(_val, 99.5))
        sc = _ax.scatter(_fx[mf], _fy[mf], c=_val, s=1.5, cmap=_cmap,
                         norm=LogNorm(vmin=_lo, vmax=_hi), rasterized=True)
        plt.colorbar(sc, ax=_ax, label=_lab + '  (log)')
        _draw_cyl_xy(_ax)
        _ax.set_xlabel('x [m]'); _ax.set_ylabel('y [m]'); _ax.set_aspect('equal')
        _ax.set_xlim(X_MIN, X_MAX); _ax.set_ylim(Y_MIN, Y_MAX)
        _ax.set_title(f'{_lab.split(" ")[0]} do CFD — t={first_snap_raw["t"]:.2f} s, '
                      f'XY em z={_Z_SLAB:.0f}±{_SLAB_HALF:.0f} m\n'
                      f'({int(mf.sum()):,} células)', fontsize=10)
        _ax.annotate('', xy=(20, 120), xytext=(20, 30),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        _ax.text(25, 75, 'flow', fontsize=9, color='gray')
else:
    for _ax in (ax2[1, 0], ax2[1, 1]):
        _ax.text(0.5, 0.5, 'first_snap_raw indisponível',
                 ha='center', va='center', transform=_ax.transAxes)
        _ax.set_axis_off()

plt.tight_layout()
_p_wall_png = os.path.join(DIAG_DIR, 'wall_distance_and_turbulence.png')
plt.savefig(_p_wall_png, dpi=120)
plt.close()
print(f"Plot salvo: {_p_wall_png}")
print(f"""
  O que olhar nos dois painéis de cima:
    - a sombra do cilindro: d_wall cai para zero num anel em volta dele,
      e o campo em XZ mostra a quina superior;
    - AUSÊNCIA de gradiente em y no painel XZ. Se houver faixa escura no
      meio e clara nas bordas em y, inlet/outlet entraram como parede.
  E nos dois de baixo:
    - k alto na esteira, a jusante do cilindro (+y), e baixo no escoamento
      livre. Se a esteira não aparecer, o campo não é o que se pensa.
    - ω alto junto às paredes e MUITO alto na faixa da entrada (y≈0) —
      é o default do Fluent (μ_t/μ=10), o item 02 do checklist de CFD.
""")


# ============================================================
# CELL 9 — NEURAL NETWORK (SIREN + Hard Constraints)
# ============================================================
# ==============================================================================
# [19/10/2026] RODADA siren1 — AS CAMADAS OCULTAS VIRAM SIREN
#
# É a ÚNICA variável desta rodada em relação à sst3 (Cell 3). O que muda:
#
#   Dense(512, tanh, glorot)  ->  SirenLayer(512): sin(ω0·(W·x + b))
#
# com a inicialização de Sitzmann et al. (2020), que é o que faz uma rede
# de senos treinar:
#   1ª camada   W ~ U(−1/n, +1/n)              n = fan_in (= 4, as entradas)
#   demais      W ~ U(−√(6/n)/ω0, +√(6/n)/ω0)  n = 512
#   viés        zero
# Nas camadas ocultas o ω0 do argumento e o 1/ω0 do peso se cancelam: na
# inicialização elas não dependem de ω0, e durante o treino ω0 age como um
# multiplicador do gradiente delas. A frequência que a rede "enxerga" no
# espaço de entrada é decidida na PRIMEIRA camada.
#
# O QUE FICA IGUAL, de propósito (uma variável de cada vez):
#   - 6 camadas × 512: o número de parâmetros é EXATAMENTE o da sst3
#     (1.318.918), porque a SirenLayer tem os mesmos W e b de um Dense.
#   - a saída raw_fields: Dense linear com glorot_uniform (o default do
#     Keras), como na sst3 E como na SIREN laminar. O artigo inicializaria
#     a saída com a regra das ocultas; aqui isso seria uma segunda mudança.
#   - a HardConstraintLayer e as hc_distance_functions, linha por linha.
#     Ela recebe `inp` em z-score, como antes — a SIREN não toca nas BCs.
#
# ω0 = 10, SEM REESCALAR AS ENTRADAS (decisão do orientador, 19/10/2026).
# É o valor que estabilizou a SIREN laminar (lá ω0 = 30 fazia o gnorm
# explodir). As entradas continuam no z-score da Cell 4, então a SIREN NÃO
# as vê em [−1, 1] como o artigo supõe: y vai até +4,04. Por isso o log
# abaixo imprime a "frequência efetiva" por eixo, ω0·max|x_i|, que é o
# número a citar quando ω0 for testado.
#
# XLA: tf.sin e tf.matmul têm gradiente registrado em qualquer ordem, e a
# SirenLayer não usa concat nem fatiamento. O tf.stack + reshape da
# HardConstraintLayer (ver o fim dela) continua sendo o que mantém o XLA
# ligado — a SIREN laminar tinha tf.concat ali e NÃO serviria de base.
#
# ==============================================================================
# [29/10/2026] RODADA ema1 — UMA SEGUNDA REDE, model_ema
#
# A rede de treino (model) fica exatamente como na siren1. Ao lado dela
# nasce model_ema, do MESMO build_pinn, e recebe uma cópia dos pesos da
# rede de treino antes de qualquer passo de treino. Esse é o "inicializados
# iguais" do orientador. Daí em diante o Adam só mexe em model, e model_ema
# acompanha pela média móvel exponencial, θ_ema ← β·θ_ema + (1 − β)·θ, aplicada
# na Cell 14, depois de cada passo.
#
# POR QUE build_pinn + set_weights, E NÃO clone_model: o clone_model copia
# a arquitetura e REINICIALIZA os pesos. As duas redes começariam
# diferentes, e o EMA arrastaria uma inicialização que a rede de treino
# nunca teve. O set_weights copia valor por valor, e a conferência no fim
# desta célula exige |Δ| = 0.
#
# β = 0,999: horizonte de ~1/(1 − β) = 1.000 passos = 10 épocas, cerca de um
# período do dente de serra de 9 épocas (TRAIN_SHUFFLE_MODE = "by_snapshot").
# Atualização em todo passo, sem aquecimento (decisão de 29/10/2026). No
# começo o EMA ainda carrega a inicialização aleatória: β^n dela depois de n
# passos, ou seja, 37% na época 10, 13% na época 20 e 1% na época 46. Na
# calibração (20 épocas) a validação do EMA TEM que sair pior que a da rede
# de treino. Isso é esperado, não defeito.
#
# model_ema FICA trainable=True, embora nunca receba gradiente. Ele não
# entra na fita (a Cell 14 deriva só em relação a model.trainable_variables)
# nem no Adam (optimizer.build só recebe model). Marcar trainable=False
# mandaria os 14 tensores para non_trainable_weights, e o laço da Cell 14
# sobre model_ema.trainable_weights viraria um zip VAZIO: o EMA ficaria
# congelado na inicialização sem nenhum erro. A conferência abaixo pega isso.
#
# A CRIAÇÃO FICA NO FIM DESTA CÉLULA, depois de todos os diagnósticos da
# siren1, para eles rodarem idênticos. O model_ema é construído DEPOIS do
# model, então a inicialização da rede de treino não muda por causa dele. As
# variáveis do EMA também precisam existir antes do traçado do train_step
# (Cell 14), como os slots do Adam, e aqui elas já existem.
# ==============================================================================
#
# A rede passa de 4 para 6 saídas: (u, v, w, P, k, ω).
# O ansatz de u, v, w e P fica INALTERADO, linha por linha.
#
# ------------------------------------------------------------------------------
# A DECISÃO PRINCIPAL: k E ω NÃO RECEBEM HARD CONSTRAINT NAS PAREDES
#
# O reflexo é impor k = 0 na parede (as flutuações somem no no-slip) e
# ω → ∞ (ω ~ 6ν/(β₁y²) quando y→0). As duas coisas são corretas para um
# SST RESOLVIDO até a parede. Este CFD não é isso.
#
# A malha tem y+ da ordem de 10⁵ e o Fluent rodou com LEI DE PAREDE. O
# centro da primeira célula está a ~5 m da superfície, e o valor de k ali
# não é física resolvida — é o que o tratamento de parede devolve. Pela
# lei logarítmica, com u_τ ≈ 0,05·V_inf ≈ 0,85 m/s:
#
#     k_parede ≈ u_τ²/√β*  ≈ 0,72/0,3  ≈ 2,4 m²/s²        (não 0)
#     ω_parede ≈ u_τ/(√β*·κ·y) ≈ 0,85/(0,3·0,41·5) ≈ 1,4 1/s  (não ∞)
#
# Impor k = 0 ali contradiria o dado por um fator infinito. A rede ficaria
# presa entre uma restrição arquitetural que manda zero e uma data loss que
# mostra 2,4 — e a restrição dura sempre ganha, porque é multiplicativa.
#
# Então: nenhuma hard constraint de k ou ω em parede. Os valores junto às
# paredes vêm da data loss, com o peso baixo que a Cell 12 já dá às zonas
# cyl e bc (0,5), e com a ressalva do handoff de que aquela região não é
# defensável nesta etapa.
#
# ------------------------------------------------------------------------------
# A ÚNICA BC DURA DE TURBULÊNCIA É O INLET — e a forma importa
#
# No inlet o Fluent impôs k e ω uniformes (Dirichlet de verdade, não
# gradiente zero). Isso merece hard constraint, igual a v = V_INF.
#
# A Cell 4 avisou que LNWN_INLET ≈ 4,9 cai fora de ±4 e satura o tanh. Esse
# aviso vale para uma formulação em que a REDE tem que produzir 4,9. A forma
# usada aqui é aditiva:
#
#     ωn_hard = ωn_raw · D_inlet  +  LNWN_INLET · (1 − D_inlet)
#
# Perto do inlet D_inlet → 0: o termo da rede desaparece e o valor vem da
# CONSTANTE, não da saída da rede. A rede nunca precisa representar 4,9.
# É o mesmo mecanismo do v: `V_INF · decay_freestream` também é um termo
# aditivo, e é por isso que o offset de v nunca foi problema.
#
# Fica uma consequência real, que não é da rede e sim do CFD: ω cai de 7419
# para ~20 em cerca de 2 m de fetch. O resíduo de ω nessa faixa é rígido.
# Se o item 02 do checklist de CFD for adotado (μ_t/μ do inlet de 10 para
# 10³–10⁴), ω_inlet cai para 7–74 e a rigidez desaparece junto.
#
# ------------------------------------------------------------------------------
# POSITIVIDADE: automática, não imposta
#
# k e ω saem da rede em espaço LOG normalizado (Cell 4). A desnormalização é
# exponencial, então k > 0 e ω > 0 por construção — sem softplus, sem clamp,
# sem região de gradiente morto. E a mistura aditiva acima acontece no espaço
# log, o que significa interpolação GEOMÉTRICA em k e ω: o valor intermediário
# entre 20 e 7419 é ~385, não ~3700. Para grandezas que varreem ordens de
# grandeza, é a interpolação certa.
# ============================================================


# ---------- USER CONFIG ----------
# BCs duras de turbulência. Ver o cabeçalho para por que não há opção de
# parede: ela seria fisicamente errada com este dado, não uma alternativa.
#
# ------------------------------------------------------------------------------
# FICARAM LIGADAS — a evidência que mandava desligar não sobreviveu
# ------------------------------------------------------------------------------
# Em 01/10/2026 estas duas foram para False com base no checkpoint 400, e
# voltaram para True no mesmo dia com base no checkpoint 2200. Vale contar
# a história inteira, porque o raciocínio errado é instrutivo.
#
# O QUE O CHECKPOINT 400 MOSTRAVA:
#
#   (a) AJUSTE AO DADO, erro médio em ordens de grandeza (dex):
#
#         faixa                      k        ω
#         inlet y < 3 m           1.36     0.45     <<<
#         y 3-20 m                0.36     0.10
#         parede d_wall < 1,3 m   0.12     0.19
#         cilindro r < 2R         0.16     0.10
#         esteira                 0.28     0.08
#
#       1,36 dex é um fator de 23. A faixa do inlet é o ÚNICO lugar do
#       domínio onde a rede não consegue ajustar o dado — e é justamente
#       onde a restrição dura age. Ela não está ajudando a rede a acertar
#       o inlet; está impedindo.
#
#       A razão: a BC impõe k = 1,0838 e ω = 7419 em y = 0, enquanto a
#       mediana MEDIDA na faixa y < 3 m é k = 0,123 e ω = 900. Fator ~8
#       nos dois. Como D_inlet = tanh(135·Δy/101,3), a constante ainda
#       vale 74% do valor a 0,2 m e 13% a 1 m: a imposição contamina a
#       faixa inteira, não só o plano y = 0.
#
#   (b) RESÍDUO. Com 48.000 pontos de colocação, 96,2% do MSE de r_k vem
#       de adj_inlet (4,4% dos pontos), e 83% vem de 45 pontos. A causa
#       física está no parágrafo do cabeçalho que já previa isso: com
#       ω = 7419, β*ω dá comprimento de decaimento de 0,26 mm. A hard
#       constraint espalha o valor por ~2 m. Nenhuma combinação de pesos
#       faz a convecção equilibrar a destruição num perfil 7.000× mais
#       largo do que o físico. O resíduo ali NÃO É MINIMIZÁVEL.
#
# O QUE O CHECKPOINT 2200 MOSTROU, 1800 ÉPOCAS DEPOIS:
#
#         faixa                 k dex ep400 -> ep2200    ω dex
#         inlet y < 3 m            1.36  ->  0.13     0.45 -> 0.09
#         cilindro r < 2R          0.16  ->  0.12     0.10 -> 0.05
#         esteira                  0.28  ->  0.21     0.08 -> 0.05
#
# A rede APRENDEU o inlet. Demorou, mas chegou lá — e tinha como, porque a
# mistura é afim no espaço log normalizado:
#
#     ωn_hard = ωn_raw·D_inlet + LNWN_INLET·(1 − D_inlet)
#
# A rede compensa a constante produzindo ωn_raw do outro lado. O argumento
# "a restrição impede o ajuste" era falso; o certo seria "a restrição
# torna o ajuste lento naquela faixa".
#
# E o resíduo deixou de precisar disso: com a escala local de r_k (Cell
# 10), adj_inlet responde por 0,6% do MSE, contra 91,7% na escala global.
# O conserto do r_k veio do denominador, não daqui.
#
# POR QUE NÃO DESLIGAR MESMO ASSIM, num warm start: a compensação aprendida
# é enorme perto de y = 0. Tirando a restrição e mantendo os pesos, o que
# sobra da saída crua é (calculado com a lei de decaimento livre):
#
#      y [m]   D_inlet   ω alvo   ω sem a BC   fator de erro
#       0.05    0.0665     2643      0.0014      2,0e+06
#       0.10    0.1325     1608      0.072       2,2e+04
#       0.25    0.3213      739      5.7         130
#       0.50    0.5825      389      47          8,3
#       1.00    0.8698      200      116         1,7
#       2.00    0.9904      101      97          1,0
#
# Ou seja: desligar num warm start destrói justamente a faixa que custou
# 2000 épocas para ficar boa, e faz isso na mesma rodada em que o r_k muda
# de régua — duas perturbações de uma vez.
#
# QUANDO VOLTAR A ESTA DECISÃO. Desligar continua defensável, mas como
# experimento PRÓPRIO e com treino DO ZERO, para não ter compensação
# herdada a desfazer. O motivo que sobra não é mais o ajuste: é que o
# valor ω_inlet = 7419 1/s se apoia na verificação (c) da Cell 5, que
# compara a BC com valores já decaídos e não vale como está (ver
# pendências). Se o item 02 do checklist de CFD for adotado (μ_t/μ do
# inlet de 10 para 10³–10⁴), ω_inlet cai para 7–74, o decaimento passa a
# ter escala de metros e a questão desaparece sozinha.
# ------------------------------------------------------------------------------
HARD_BC_K_INLET     = True
HARD_BC_OMEGA_INLET = True

# ------------------------------------------------------------------------------
# CORREÇÃO DE UM DEFEITO HERDADO: no-slip falhava na quina do cilindro
#
# D_cyl deveria valer 0 em TODA a superfície do cilindro — é ele que zera a
# velocidade ali. Medido na versão laminar:
#
#     lateral z=32 m (meia altura)   D_cyl = 0.0000   ok
#     lateral z=60 m                 D_cyl = 0.0785
#     lateral z=64 m (1 m da quina)  D_cyl = 0.8503
#     QUINA  (r=R, z=H)              D_cyl = 0.9757   <- no-slip ausente
#     topo   r=0.9R                  D_cyl = 0.7542
#     topo   r=0.5R                  D_cyl = 0.0575
#
# A causa é o smooth_max. Ele é uma sobrestimativa suave do max, com excesso
# de até log(2)/k: onde as DUAS distâncias se anulam — exatamente a quina —
# smooth_max(0,0) = log(2)/75 = 0.00924 em vez de 0, e tanh(238·0.00924)
# = 0.976. A restrição simplesmente não age naquele canto.
#
# Não foi introduzido por mim, mas passa a importar mais: a quina é onde as
# camadas cisalhantes do topo e da lateral se encontram, e é onde a produção
# de k é máxima num corpo rombudo.
#
# A CORREÇÃO: trocar o smooth_max por max, sem suavizar.
#
#     d_cyl = max( d_radial_sq , 2·R_N · d_top )
#
# É a mesma estrutura do laminar — o máximo entre a folga radial e a folga
# vertical — só que sem o arredondamento que desloca o valor na quina.
# Vale 0 em TODA a superfície: na lateral d_radial_sq=0 e d_top<0, então o
# máximo é 0; no topo o inverso; na quina os dois são 0. Fora, é positivo.
# Dentro do cilindro os dois são negativos e o tf.maximum(·,0) que já existia
# corta para 0 — e o limite desse corte coincide EXATAMENTE com a parede,
# não com uma superfície deslocada.
#
# O fator 2·R_N põe as duas folgas na mesma escala. Para gap pequeno,
# d_radial_sq = r_n²−R_n² = 2·R_N·δ + δ², ou seja ≈ 2·R_N·δ, enquanto d_top
# é δ direto. Sem o fator, a camada de transição no topo seria 1,13 m contra
# 2,11 m na lateral; com ele as duas ficam em 2,11 m, batendo com CYL_TOL=2 m.
#
# Sobre a dobra: max(a,b) tem uma dobra onde a=b, aqui um cone saindo da
# quina. Não dá para eliminar. Uma função contínua e diferenciável que valha
# zero em TODO o interior do cilindro (o corte garante isso) tem gradiente
# nulo na parede, o que forçaria u ~ δ² junto a ela em vez do u ~ δ físico.
# Então alguma não-suavidade é inevitável; o que se escolhe é onde ela fica
# e quantas superfícies ela ocupa.
#
# Medido em 166 mil pontos de colocação reais (cascas adj_cyl_* + boi_c):
#
#                        D_cyl na quina   |∂²D/∂z²| p99   p99.9   pts com D≡0
#   original (laminar)        0.976            3.90        4.18       0.0%
#   subtrair log(2)/k         0.000            4.09        4.09      45.5%
#   somar partes positivas    0.000            1.21       60.58       0.0%
#   max sem suavizar          0.000            1.21        1.21       0.0%
#
# As linhas 2 e 3 foram tentativas minhas, descartadas. Subtrair log(2)/k
# desloca a superfície para fora e cria uma casca de ~1,5 m onde D_cyl ≡ 0:
# ali a velocidade fica travada em zero, e em 45% dos pontos perto do
# cilindro o resíduo de momento degenera para ∇P = 0 — uma restrição sobre a
# pressão que ninguém pediu. Somar as partes positivas não tem casca, mas
# põe DUAS superfícies de dobra dentro do fluido (o cilindro estendido acima
# do topo e o plano z=H fora dele), e a cauda de curvatura sobe 50×.
#
# O max tem UMA superfície de dobra e reproduz o laminar valor por valor na
# lateral: gap 0,25/0,5/1/2/3 m -> 0.305/0.560/0.856/0.989/0.999 nos dois.
CYL_RIM_FIX = True     # False reproduz o comportamento laminar, com o defeito


# ------------------------------------------------------------------------------
# FUNÇÕES DE DISTÂNCIA — uma implementação só
#
# Estavam dentro do call() da HardConstraintLayer, e a Cell 9c as REPLICAVA
# para poder inspecioná-las, com um comentário admitindo o risco: "se você
# mudar os parâmetros internos da layer, atualize aqui também".
#
# A réplica já tinha divergido em três pontos antes de qualquer mudança minha:
#   - o smooth_max dela não subtraía o máximo, então exp(75·a) chegava a 1e162
#     (INF em float32); o log salvava por acidente;
#   - o D_vel dela multiplicava por D_outlet, que a layer NÃO inclui — a
#     réplica mostrava velocidade zerada no outlet, coisa que não acontece;
#   - o decay_inlet dela era 1−exp(−rate·Δ), o da layer é exp(−rate·Δ): em
#     Δ=5,6 m um dá 0,98 e o outro 0,019.
#
# Ou seja: a Cell 9c validava uma função diferente da que a rede usa. Extrair
# para uma função única é o que impede isso de voltar a acontecer.
# ------------------------------------------------------------------------------

HC_MT       = 135.0    # inclinação das tanh nos planos
HC_RATE     = 72.0     # decaimento do baseline de v
HC_K_SMOOTH = 75.0     # suavização do max (só usada se CYL_RIM_FIX=False)
HC_K_R2     = 238.0    # inclinação da tanh do cilindro


def hc_smooth_max(a, b, k=HC_K_SMOOTH):
    """
    Aproximação suave de max(a, b).

    A subtração de m = max(a,b) é o que evita overflow: sem ela, exp(k·a) com
    k=75 estoura em float32 já para a ≈ 1,2. Com ela, os dois expoentes são
    ≤ 0 e o pior caso é underflow para 0, que é inofensivo.
    """
    m = tf.maximum(a, b)
    return m + (1.0 / k) * tf.math.log(
        tf.exp(k * (a - m)) + tf.exp(k * (b - m))
    )


def hc_distance_functions(xn, yn, zn):
    """
    Todas as funções de distância do ansatz, em coordenadas normalizadas.

    Chamada pela HardConstraintLayer E pela Cell 9c — é o ponto único de
    verdade. Qualquer mudança aqui aparece nos dois lugares por construção.

    Retorna dict com cada D_i, o produto D_vel e decay_freestream.
    """
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
        # max sem suavizar, com as duas folgas na mesma escala.
        # Zero exato em toda a superfície, inclusive na quina.
        d_cyl_3d = tf.maximum(d_radial_sq, (2.0 * R_N) * d_top)
    else:
        # Forma laminar: suave, mas não se anula na quina (D_cyl≈0.98 lá).
        d_cyl_3d = hc_smooth_max(d_radial_sq, d_top, k=HC_K_SMOOTH)
    # O corte em 0 só age DENTRO do cilindro sólido, onde não há ponto
    # algum: seu limite coincide com a parede.
    D_cyl = tf.math.tanh(HC_K_R2 * tf.maximum(d_cyl_3d, 0.0))

    # D_vel NÃO inclui D_outlet: o outlet é pressure-outlet, não parede.
    D_vel = D_inlet * D_ground * D_cyl * D_xmin * D_xmax * D_ztop

    decay_inlet = tf.exp(-HC_RATE * (yn - YN_MIN))
    decay_xmin  = tf.exp(-HC_RATE * (xn - XN_MIN))
    decay_xmax  = tf.exp(-HC_RATE * (XN_MAX - xn))
    decay_ztop  = tf.exp(-HC_RATE * (ZN_MAX - zn))
    decay_freestream = 1.0 - (1.0 - decay_inlet) \
                        * (1.0 - decay_xmin)  \
                        * (1.0 - decay_xmax)  \
                        * (1.0 - decay_ztop)

    return {
        "D_inlet": D_inlet, "D_outlet": D_outlet, "D_ground": D_ground,
        "D_xmin": D_xmin, "D_xmax": D_xmax, "D_ztop": D_ztop,
        "D_cyl": D_cyl, "D_vel": D_vel,
        "decay_inlet": decay_inlet,
        "decay_freestream": decay_freestream,
        # intermediários do cilindro — a Cell 9D inspeciona a cadeia inteira
        # (r² -> folgas -> max -> tanh) sem precisar recalcular nada.
        "r2": r2_normalized,
        "d_radial_sq": d_radial_sq,
        "d_top": d_top,
        "d_top_scaled": (2.0 * R_N) * d_top,
        "d_cyl_3d": d_cyl_3d,
    }


class TanH(layers.Layer):
    def call(self, x):
        return tf.tanh(x)


class SirenLayer(layers.Layer):
    """
    Camada SIREN (Sitzmann et al., 2020): Dense + seno, com a inicialização
    própria que mantém as ativações distribuídas igualmente em todas as
    camadas (com glorot e seno, elas colapsam ou saturam já na 3ª camada).

      is_first=True  -> W ~ U(−1/fan_in, +1/fan_in)
      is_first=False -> W ~ U(−√(6/fan_in)/ω0, +√(6/fan_in)/ω0)

    Saída: sin(ω0·(W·x + b)).

    Adaptada da SIREN laminar (pinn_siren_a_17-07) sem mudar a conta. Os
    pesos se chamam "kernel" e "bias", como num Dense, para o checkpoint ter
    a mesma estrutura de variáveis.
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
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name="bias",
        )

    def call(self, inputs):
        # ω0 multiplica o argumento inteiro, viés incluído — é a forma do
        # artigo e da laminar. Trocar por sin(ω0·W·x + b) mudaria a
        # inicialização efetiva do viés e deixaria de ser a mesma camada.
        return tf.sin(self.omega_0 * (tf.matmul(inputs, self.w) + self.b))

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "is_first": self.is_first,
                       "omega_0": self.omega_0})
        return config


class HardConstraintLayer(layers.Layer):
    """
    Impõe BCs arquiteturalmente via ansatz multiplicativo.

    Modificações vs versão anterior:
      - Substitui sqrt(r²+eps) - R por (r² - R²) na distância ao cilindro
        (sem singularidade na derivada na origem; smooth em todo lugar)
      - Reduz constantes mt/rate/k_smooth para evitar overflow em fp32
      - NOVO: 6 saídas. k e ω recebem BC só no inlet (ver cabeçalho da célula)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def smooth_max(a, b, k=HC_K_SMOOTH):
        """Delega para hc_smooth_max (ponto único de verdade)."""
        return hc_smooth_max(a, b, k=k)

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

        # Ponto único de verdade — a Cell 9c inspeciona ESTAS funções, não
        # uma cópia delas.
        D = hc_distance_functions(xn, yn, zn)
        D_inlet  = D["D_inlet"]
        D_outlet = D["D_outlet"]
        D_ground = D["D_ground"]
        D_cyl    = D["D_cyl"]
        D_vel    = D["D_vel"]
        decay_freestream = D["decay_freestream"]

        # --- ANSATZ HÍBRIDO ---
        # u, w: min-max simétrico (V_REF=5), v=0 nas paredes em coords NORM é v=0 físico
        # Aqui o ansatz simples funciona pq não tem offset
        u_hard = un * D_vel
        w_hard = wn * D_vel

        # v: z-score com offset V_MEAN. Pensa em coords físicas e renormaliza.
        # Queremos: v_phys = 0 nas paredes, = V_INF no inlet, = vn*V_DP+V_MEAN no interior
        v_phys_raw = vn * V_DP + V_MEAN
        v_phys_hard = v_phys_raw * D_vel + V_INF * decay_freestream * D_ground * D_cyl
        v_hard = (v_phys_hard - V_MEAN) / V_DP

        # P: z-score, queremos P_phys = 0 no outlet
        P_phys_raw = Pn * P_DP + P_MEAN
        P_phys_hard = P_phys_raw * D_outlet
        P_hard = (P_phys_hard - P_MEAN) / P_DP

        # --- TURBULÊNCIA ---
        # SÓ inlet. Mistura convexa em espaço log-normalizado:
        #   em y = Y_MIN:  D_inlet = 0  ->  vale exatamente a constante
        #   longe:         D_inlet = 1  ->  vale exatamente a rede
        # A constante entra ADITIVA, então a rede não precisa produzi-la.
        # Sem D_ground, D_cyl, D_xmin...: parede não impõe k nem ω aqui.
        if HARD_BC_K_INLET:
            k_hard = kn * D_inlet + KN_INLET * (1.0 - D_inlet)
        else:
            k_hard = kn

        if HARD_BC_OMEGA_INLET:
            om_hard = omn * D_inlet + LNWN_INLET * (1.0 - D_inlet)
        else:
            om_hard = omn

        # ----------------------------------------------------------------
        # stack+reshape, NAO concat — motivo medido (ver Cell 14, ponto 5)
        # ----------------------------------------------------------------
        # Isto e' identico a tf.concat(..., axis=1) no valor: stack em axis=1
        # de 6 tensores [N,1] da' [N,6,1] e o reshape tira o eixo morto. A
        # troca existe por causa do XLA.
        #
        # O treino precisa de TERCEIRA derivada: duas espaciais (laplaciano
        # dentro do residuo) e uma nos pesos (gradiente da loss). Com concat,
        # a cadeia de diferenciacao e':
        #
        #   forward            Concat
        #   1a derivada        grad(Concat)  -> ops Slice          (array_grad.py, _ConcatGradHelper)
        #   2a derivada        grad(Slice)   -> XlaDynamicUpdateSlice
        #                        (array_grad.py:280 — dentro de um contexto
        #                         XLA o _SliceGrad troca o Pad por esta op)
        #   3a derivada        grad(XlaDynamicUpdateSlice) -> NAO EXISTE
        #                        LookupError: gradient registry has no entry
        #                        for: XlaDynamicUpdateSlice
        #
        # Com stack a cadeia fecha: grad(Pack) = Unpack, grad(Unpack) = Pack,
        # grad(Reshape) = Reshape — todas registradas em qualquer ordem.
        #
        # Medido (bisseccao com este mesmo model, jit_compile=True, 3a ordem):
        #     join=concat take=slice     FALHA
        #     join=concat take=unstack   FALHA
        #     join=stack  take=slice     OK
        #     join=stack  take=unstack   OK
        # e por termo do residuo, antes da troca: lap_u OK, lap_k FALHA,
        # lap_om FALHA, mu_ef*lap_k FALHA, conv_k OK, F1 sozinho OK.
        # (O laplaciano de u passava e o de k nao porque a exponencial de
        #  desnormalizacao poe um fator a mais na cadeia; com q linear na
        #  saida o no' problematico some na poda do grafo. A causa e' a
        #  mesma nos dois casos — o concat.)
        #
        # Se alguem trocar isto de volta por concat, o Cell 14 nao quebra:
        # ele detecta a falha e cai para tf.function sem XLA, so' que mais
        # lento. A troca aqui e' o que permite o XLA ficar ligado.
        return tf.reshape(
            tf.stack([u_hard, v_hard, w_hard, P_hard, k_hard, om_hard],
                     axis=1),
            (-1, 6))


N_OUTPUTS = 6   # (u, v, w, P, k, ω) — era 4 no laminar


def build_pinn(n_layers=6, n_neurons=512, omega_0=None):
    """
    omega_0=None usa o OMEGA_0 global (definido logo abaixo). É o que deixa
    a Cell 16, que reconstrói a rede para conferir o checkpoint, montar a
    MESMA SIREN mesmo chamando build_pinn só com camadas e neurônios.
    """
    if omega_0 is None:
        omega_0 = OMEGA_0

    inp = layers.Input(shape=(4,), name="xyzt")

    # Stack de camadas SIREN, a primeira com a inicialização especial.
    # A entrada vai CRUA, em z-score (sem reescala — ver o cabeçalho).
    h = inp
    for i in range(n_layers):
        h = SirenLayer(
            units=n_neurons,
            is_first=(i == 0),
            omega_0=omega_0,
            name=f"siren_{i+1}",
        )(h)

    # Saída linear (sem ativação) — necessário pra pressão poder ser negativa,
    # e para k e ω, que saem em espaço LOG (a exponencial da desnormalização
    # é que garante positividade; ver Cell 4). glorot_uniform é o default do
    # Keras, igual à sst3 e à SIREN laminar — ver o cabeçalho.
    raw_out = layers.Dense(
        N_OUTPUTS,
        activation='linear',
        bias_initializer="zeros",
        name="raw_fields"
    )(h)

    # Aplica hard constraints (BCs forçadas arquiteturalmente)
    final_out = HardConstraintLayer(name="hard_constraints")([inp, raw_out])

    return Model(inp, final_out, name="PINN_SIREN")


# ============================================================
# Build the model
# ============================================================
MODEL_BLOCKS  = 6
MODEL_NEURONS = 512
OMEGA_0       = 10.0     # decisão do orientador (19/10/2026): manter o valor
                         # que estabilizou a SIREN laminar; é para ser testado.
                         # Entradas SEM reescala (z-score da Cell 4).

# Parâmetros da sst3 (tanh, 6×512). A SIREN tem os mesmos W e b, então tem
# que dar o mesmo número — é o que sustenta "só a ativação mudou".
N_PARAMS_SST3 = 1_318_918

model = build_pinn(MODEL_BLOCKS, MODEL_NEURONS, omega_0=OMEGA_0)

print(f"Architecture: PINN SIREN")
print(f"  SIREN layers:       {MODEL_BLOCKS}")
print(f"  Neurons/layer:      {MODEL_NEURONS}")
print(f"  omega_0:            {OMEGA_0:g}")
print(f"  Total parameters:   {model.count_params():,}"
      f"   (sst3: {N_PARAMS_SST3:,})")
if model.count_params() != N_PARAMS_SST3:
    print(f"  >>> número de parâmetros DIFERENTE da sst3: a comparação deixa de")
    print(f"      ser 'só a ativação mudou'. Conferir MODEL_BLOCKS/MODEL_NEURONS.")
print(f"  Input shape:        (None, 4)")
print(f"  Output shape:       (None, {N_OUTPUTS})   [u, v, w, P, k, ω]")

# Frequência efetiva da 1ª camada por eixo: o argumento do seno é
# ω0·(W·x), e a SIREN foi pensada para |x| ≤ 1. Com o z-score, |x| passa
# disso em y e t. ω0·max|x_i| é o "ω0 equivalente" sobre entrada em [−1, 1].
print(f"\n  Frequência efetiva da 1ª camada, ω0·max|x_i| (artigo: 30 em [−1,1]):")
for _lab, _lo, _hi in [("x", XN_MIN, XN_MAX), ("y", YN_MIN, YN_MAX),
                       ("z", ZN_MIN, ZN_MAX), ("t", TN_MIN, TN_MAX)]:
    _m = max(abs(_lo), abs(_hi))
    print(f"    {_lab}: max|x| = {_m:.3f}  ->  {OMEGA_0 * _m:6.1f}")

print(f"\nBCs duras ativas:")
print(f"  u, w      : zero em TODAS as paredes (D_vel)")
print(f"  v         : zero nas paredes, V_INF={V_INF} m/s no inlet")
print(f"  P         : zero no outlet")
print(f"  k         : {'K_INLET = %.4f m²/s² no inlet' % K_INLET if HARD_BC_K_INLET else 'nenhuma'}"
      f"   (kn = {KN_INLET:+.4f})")
print(f"  ω         : {'OMEGA_INLET = %.0f 1/s no inlet' % OMEGA_INLET if HARD_BC_OMEGA_INLET else 'nenhuma'}"
      f"   (ln ωn = {LNWN_INLET:+.4f})")
print(f"  k, ω em parede: NENHUMA — o CFD usou lei de parede, k não vai a")
print(f"                  zero nem ω a infinito no dado (ver cabeçalho)")

# Escala geométrica de cada função de distância, em METROS. Estava invisível
# no código: as constantes mt/rate/k_r2 vivem em coordenadas normalizadas,
# então não dá para saber, olhando, se a transição acontece em 2 m ou em 20.
# Importa porque tem que ser comparável a BC_TOL/CYL_TOL (3 m e 2 m) — é
# nessas cascas que estão os pontos de colocação que vão sentir a restrição.
_ATANH99 = float(np.arctanh(0.99))          # 2.6467
print(f"\nEspessura da camada onde cada restrição age (até 99%):")
print(f"  tanh(135·Δ)  planos    : {_ATANH99*DP_ISO/135.0:.2f} m   "
      f"(BC_TOL = {BC_TOL} m)")
print(f"  tanh(238·Δ²) cilindro  : {_ATANH99/(238.0*2*R_N)*DP_ISO:.2f} m   "
      f"(CYL_TOL = {CYL_TOL} m)")
print(f"  exp(-72·Δ)   decay v   : {4.6*DP_ISO/72.0:.2f} m   (até 1%)")
print(f"  -> todas da mesma ordem das tolerâncias. Se alguma fosse muito")
print(f"     maior, a restrição vazaria para dentro do domínio.")

if CYL_RIM_FIX:
    _sc = _ATANH99 / (238.0 * 2 * R_N) * DP_ISO
    print(f"\nCYL_RIM_FIX ativo: no-slip exato em toda a superfície do cilindro,")
    print(f"  incluindo a quina (era D_cyl=0.976 lá, ou seja, quase sem restrição).")
    print(f"  Transição de {_sc:.2f} m tanto na lateral quanto no topo "
          f"(CYL_TOL={CYL_TOL} m).")
    print(f"  Na lateral reproduz o laminar valor por valor; muda só perto da quina.")
else:
    print(f"\nCYL_RIM_FIX DESLIGADO: D_cyl ≈ 0.98 na quina do cilindro, ou seja,")
    print(f"  o no-slip praticamente não age naquele canto. Comportamento da")
    print(f"  versão laminar, mantido só para reprodutibilidade.")

# Verificação rápida: output finito (usando range NORMALIZADO VÁLIDO)
test_input = tf.constant(np.column_stack([
    np.random.uniform(XN_MIN, XN_MAX, 10),
    np.random.uniform(YN_MIN, YN_MAX, 10),
    np.random.uniform(ZN_MIN, ZN_MAX, 10),
    np.random.uniform(TN_MIN, TN_MAX, 10),
]).astype(np.float32))

test_output = model(test_input, training=False)
n_finite = tf.reduce_all(tf.math.is_finite(test_output)).numpy()
print(f"\nSanity check:")
print(f"  Output shape: {test_output.shape}")
print(f"  All finite:   {n_finite}")
assert n_finite, "Model produces NaN/Inf on normalized input within valid domain!"

# Diagnóstico em camadas

# 1. Pega input de teste (range normalizado válido)
test_input = tf.constant(np.column_stack([
    np.random.uniform(XN_MIN, XN_MAX, 10),
    np.random.uniform(YN_MIN, YN_MAX, 10),
    np.random.uniform(ZN_MIN, ZN_MAX, 10),
    np.random.uniform(TN_MIN, TN_MAX, 10),
]).astype(np.float32))
print(f"Input: range [{test_input.numpy().min():.3f}, {test_input.numpy().max():.3f}]")
print(f"Input finite: {tf.reduce_all(tf.math.is_finite(test_input)).numpy()}")

# 2. Passa pelo modelo SEM a hard constraint
# Pega o modelo intermediário (até raw_fields)
intermediate_model = Model(
    inputs=model.input,
    outputs=model.get_layer("raw_fields").output
)
raw_pred = intermediate_model(test_input, training=False).numpy()
print(f"\nRaw output (antes da hard constraint):")
print(f"  shape: {raw_pred.shape}")
print(f"  finite: {np.isfinite(raw_pred).all()}")
print(f"  range: [{raw_pred.min():.3f}, {raw_pred.max():.3f}]")
print(f"  mean: {raw_pred.mean():.3f}, std: {raw_pred.std():.3f}")

# 3. Passa pelo modelo COMPLETO (com hard constraint)
final_pred = model(test_input, training=False).numpy()
print(f"\nFinal output (após hard constraint):")
print(f"  shape: {final_pred.shape}")
print(f"  finite: {np.isfinite(final_pred).all()}")
print(f"  range: [{np.nanmin(final_pred):.3f}, {np.nanmax(final_pred):.3f}]")
print(f"  NaN count: {np.isnan(final_pred).sum()} / {final_pred.size}")
print(f"  Inf count: {np.isinf(final_pred).sum()} / {final_pred.size}")
print(f"  por canal:")
for _i, _nm in enumerate(TARGET_NAMES):
    _c = final_pred[:, _i]
    print(f"    {_nm:<6} [{_c.min():+.4f}, {_c.max():+.4f}]")

# 4. k e ω desnormalizados TÊM que ser positivos — é o ponto de sair em log
_k_chk  = denormalize_k(final_pred[:, COL_KN - COL_UN].astype(np.float64))
_om_chk = denormalize_omega(final_pred[:, COL_OMN - COL_UN].astype(np.float64))
print(f"\nPositividade de k e ω na saída (rede não treinada):")
print(f"  k  ∈ [{_k_chk.min():.4g}, {_k_chk.max():.4g}] m²/s²   "
      f"todos > 0: {bool((_k_chk > 0).all())}")
print(f"  ω  ∈ [{_om_chk.min():.4g}, {_om_chk.max():.4g}] 1/s     "
      f"todos > 0: {bool((_om_chk > 0).all())}")
assert (_om_chk > 0).all(), "ω não-positivo — a desnormalização exponencial falhou"


# ============================================================
# Modelo EMA (rodada ema1) — ver o cabeçalho desta célula
# ============================================================
BETA_EMA = 0.999     # decisão do orientador (29/10/2026); a Cell 14 aplica

model_ema = build_pinn(MODEL_BLOCKS, MODEL_NEURONS, omega_0=OMEGA_0)
model_ema.set_weights(model.get_weights())

# --- Conferência: as duas redes TÊM que ser idênticas agora ---
# A Cell 14 junta os pesos com zip(model_ema.trainable_weights,
# model.trainable_weights). O zip não reclama de listas de tamanhos
# diferentes, só corta a maior, e um peso fora de ordem misturaria kernel de
# uma camada com o de outra. Por isso a conferência é de quantidade, forma e
# valor, tensor a tensor, e não só da contagem de parâmetros.
_tw, _tw_ema = model.trainable_weights, model_ema.trainable_weights
assert len(model.non_trainable_weights) == 0 and \
       len(model_ema.non_trainable_weights) == 0, (
    "há pesos não-treináveis: o EMA sobre trainable_weights não cobriria a "
    "rede inteira (e marcar model_ema.trainable=False esvazia o laço da Cell 14)")
assert len(_tw) == len(_tw_ema) == 2 * (MODEL_BLOCKS + 1), (
    f"esperados {2 * (MODEL_BLOCKS + 1)} tensores (kernel e bias de "
    f"{MODEL_BLOCKS} SIREN + raw_fields); model {len(_tw)}, "
    f"model_ema {len(_tw_ema)}")
_dw_max = 0.0
for _w, _we in zip(_tw, _tw_ema):
    assert _w is not _we, "model_ema compartilha variável com model"
    assert tuple(_w.shape) == tuple(_we.shape), (
        f"forma diferente: {_w.name} {tuple(_w.shape)} × "
        f"{_we.name} {tuple(_we.shape)}")
    _dw_max = max(_dw_max, float(np.max(np.abs(_w.numpy() - _we.numpy()))))
assert _dw_max == 0.0, f"pesos copiados diferem: |Δw| máx = {_dw_max:.3e}"

# A mesma conferência na SAÍDA, com a hard constraint junto. Usa um gerador
# próprio, e não np.random, para não deslocar os sorteios globais de quem
# vem depois.
_rng_ema = np.random.default_rng(SEED)
_probe_ema = tf.constant(np.column_stack([
    _rng_ema.uniform(XN_MIN, XN_MAX, 256),
    _rng_ema.uniform(YN_MIN, YN_MAX, 256),
    _rng_ema.uniform(ZN_MIN, ZN_MAX, 256),
    _rng_ema.uniform(TN_MIN, TN_MAX, 256),
]).astype(np.float32))
_dy_max = float(np.max(np.abs(model(_probe_ema, training=False).numpy()
                              - model_ema(_probe_ema, training=False).numpy())))
assert _dy_max == 0.0, f"saídas de model e model_ema diferem: |Δ| = {_dy_max:.3e}"

print(f"\nModelo EMA:")
print(f"  β (decay)           : {BETA_EMA}   (horizonte ~{1/(1-BETA_EMA):,.0f} passos)")
print(f"  atualização         : todo passo aplicado, sem aquecimento (Cell 14)")
print(f"  parâmetros          : {model_ema.count_params():,} "
      f"em {len(_tw_ema)} tensores (cópia de model)")
print(f"  cópia conferida     : |Δw| máx = {_dw_max:.1e}, "
      f"|Δ saída| máx = {_dy_max:.1e} em {_probe_ema.shape[0]} pontos")
del _tw, _tw_ema, _w, _we, _rng_ema, _probe_ema


# ============================================================
# CELL 9b — VALIDAÇÃO NUMÉRICA DAS HARD CONSTRAINTS
# ============================================================
# Confirma que as hard constraints produzem valores corretos nas BCs
# ANTES de iniciar o treino. Modelo ainda não foi treinado — os valores
# nas BCs devem vir direto da hard constraint arquitetural, independente
# dos pesos aleatórios da rede.
#
# ------------------------------------------------------------------------------
# O QUE MUDA
#
# 1. k e ω entram em todas as fronteiras. No inlet têm valor esperado; nas
#    paredes o teste é o OPOSTO — têm que estar LIVRES. Se alguém reintroduzir
#    a BC de parede para k (§2 da Cell 9: ela seria fisicamente errada com
#    este dado), o teste acusa.
#
# 2. ENTRA A QUINA DO CILINDRO como fronteira testada. A versão laminar
#    testava a lateral (`z` uniforme até `H_N`) e nunca a quina — foi
#    exatamente por isso que `D_cyl ≈ 0,98` lá passou despercebido. Um teste
#    que não cobre o canto não encontra o defeito do canto.
#
# 3. VEREDITO AUTOMÁTICO em vez de "olhe os números". São 8 fronteiras × 6
#    campos = 48 valores; ninguém percebe um errado no meio deles. A versão
#    laminar terminava com "If any value differs significantly from expected,
#    hard constraints have bugs" — e o da quina diferia, e ninguém viu.
#
# 4. Os pontos são sorteados LONGE das outras fronteiras. No inlet colado ao
#    chão, `v` não é V_INF: é 0, porque o chão manda. Isso é correto, mas
#    num teste de média vira ruído que esconde defeito. A versão laminar
#    sorteava `z` em todo o range e reportava `mean=16.99` — perto o
#    suficiente de 17 para ninguém questionar, mas é uma média entre 17 e 0.
#
# 5. O tempo é sorteado em [TN_MIN, TN_MAX]. A versão laminar usava
#    `uniform(0, 1)`, que cobre só um pedaço do range normalizado
#    (≈[-1,73, +1,73]) — as BCs não dependem de t, então não escondia nada,
#    mas testar fora do domínio de treino não valida nada tampouco.
# ============================================================

print("="*70)
print("VALIDATING HARD CONSTRAINTS (untrained model)")
print("="*70)

N_TEST = 1000

# Margem para afastar das OUTRAS fronteiras: a restrição mais lenta é o
# exp(-72·Δ) do decay de v, que só cai a 1% em ~5,9 m. 15 m dá folga de 2,5x.
_MARGIN_M = 15.0
_MG = _MARGIN_M / DP_ISO


def _bc_pts(n=N_TEST, x=None, y=None, z=None, seed=None):
    """
    Pontos normalizados numa fronteira. Onde um eixo é dado, fica fixo;
    onde é None, sorteia longe das outras fronteiras.
    """
    r = np.random.default_rng(SEED if seed is None else seed)
    def ax(v, lo, hi):
        if v is not None:
            return np.full(n, v)
        return r.uniform(lo + _MG, hi - _MG, n)
    return tf.constant(np.column_stack([
        ax(x, XN_MIN, XN_MAX),
        ax(y, YN_MIN, YN_MAX),
        ax(z, ZN_MIN, ZN_MAX),
        r.uniform(TN_MIN, TN_MAX, n),     # range real, não uniform(0,1)
    ]).astype(np.float32))


def _cyl_pts(where, n=N_TEST, seed=None):
    """Pontos EXATAMENTE na superfície do cilindro: lateral, topo ou quina."""
    r = np.random.default_rng((SEED if seed is None else seed) + 7)
    th = r.uniform(0, 2*np.pi, n)
    if where == "lateral":
        rad = np.full(n, R_N)
        zz  = r.uniform(ZN_MIN + _MG, H_N - 5.0/DP_ISO, n)
    elif where == "topo":
        rad = R_N * np.sqrt(r.uniform(0, 0.9, n))     # evita a borda
        zz  = np.full(n, H_N)
    else:                                              # "quina"
        rad = np.full(n, R_N)
        zz  = np.full(n, H_N)
    return tf.constant(np.column_stack([
        XC_N + rad*np.cos(th),
        YC_N + rad*np.sin(th),
        zz,
        r.uniform(TN_MIN, TN_MAX, n),
    ]).astype(np.float32))


def _fields(pred):
    """Saída normalizada -> unidades físicas."""
    return {
        "u":     pred[:, 0] * V_REF,
        "v":     pred[:, 1] * V_DP + V_MEAN,
        "w":     pred[:, 2] * V_REF,
        "P":     pred[:, 3] * P_DP + P_MEAN,
        "k":     denormalize_k(pred[:, 4].astype(np.float64)),
        "omega": denormalize_omega(pred[:, 5].astype(np.float64)),
    }


# Cada caso: (rótulo, pontos, {campo: valor esperado}, [campos que têm que
# estar LIVRES]). Campo ausente dos dois = não testado.
# "LIVRE" não é ausência de teste: é o teste de que NENHUMA restrição age ali.
_BC_CASES = [
    ("Inlet (y=Y_MIN)",
     _bc_pts(y=YN_MIN),
     {"u": 0.0, "v": V_INF, "w": 0.0, "k": K_INLET, "omega": OMEGA_INLET},
     ["P"]),

    ("Outlet (y=Y_MAX)",
     _bc_pts(y=YN_MAX),
     {"P": 0.0},
     ["u", "v", "w", "k", "omega"]),

    ("Ground (z=Z_MIN, parede fixa)",
     _bc_pts(z=ZN_MIN),
     {"u": 0.0, "v": 0.0, "w": 0.0},
     ["k", "omega"]),

    ("Side wall (x=X_MIN, parede móvel)",
     _bc_pts(x=XN_MIN),
     {"u": 0.0, "v": V_INF, "w": 0.0},
     ["k", "omega"]),

    ("Top wall (z=Z_MAX, parede móvel)",
     _bc_pts(z=ZN_MAX),
     {"u": 0.0, "v": V_INF, "w": 0.0},
     ["k", "omega"]),

    ("Cilindro — lateral (no-slip)",
     _cyl_pts("lateral"),
     {"u": 0.0, "v": 0.0, "w": 0.0},
     ["k", "omega"]),

    ("Cilindro — topo (no-slip)",
     _cyl_pts("topo"),
     {"u": 0.0, "v": 0.0, "w": 0.0},
     ["k", "omega"]),

    ("Cilindro — QUINA r=R, z=H",
     _cyl_pts("quina"),
     {"u": 0.0, "v": 0.0, "w": 0.0},
     ["k", "omega"]),
]

# Tolerâncias. Absoluta para o que tem que zerar (float32 entrega ~1e-5),
# relativa para os valores não nulos.
_TOL_ABS = {"u": 1e-3, "v": 1e-3, "w": 1e-3, "P": 1e-2}   # m/s e Pa
_TOL_REL = 1e-3                                            # para v=V_INF, k, ω

_n_fail = 0
_fail_list = []

for _lab, _pts, _exp, _free in _BC_CASES:
    _f = _fields(model(_pts, training=False).numpy())
    print(f"\n{_lab}:")

    for _nm, _val in _exp.items():
        _a = _f[_nm]
        _m, _s = float(np.mean(_a)), float(np.std(_a))
        if abs(_val) < 1e-12:
            _ok = abs(_m) < _TOL_ABS.get(_nm, 1e-3) and _s < _TOL_ABS.get(_nm, 1e-3)
        else:
            _ok = abs(_m/_val - 1.0) < _TOL_REL and _s/abs(_val) < _TOL_REL
        if not _ok:
            _n_fail += 1
            _fail_list.append(f"{_lab} :: {_nm}")
        print(f"  {_nm:<6} mean={_m:+12.5f}  std={_s:11.5f}  "
              f"(expected: {_val:g}){'' if _ok else '   <<< FALHOU'}")

    for _nm in _free:
        _a = _f[_nm]
        _m, _s = float(np.mean(_a)), float(np.std(_a))
        # LIVRE = a saída da rede chega aqui. Se std≈0 alguém prendeu o campo.
        _scale = max(abs(_m), 1e-12)
        _ok = _s / _scale > 1e-6
        if not _ok:
            _n_fail += 1
            _fail_list.append(f"{_lab} :: {_nm} deveria estar LIVRE")
        print(f"  {_nm:<6} mean={_m:+12.5f}  std={_s:11.5f}  "
              f"(livre — sem BC){'' if _ok else '   <<< PRESO, não deveria'}")

print("\n" + "="*70)
if _n_fail == 0:
    print(f"TODAS as {sum(len(c[2])+len(c[3]) for c in _BC_CASES)} verificações "
          f"passaram em {len(_BC_CASES)} fronteiras.")
    print("As hard constraints estão corretas — inclusive na quina do cilindro,")
    print("que é onde a versão laminar falhava (D_cyl≈0.98 em vez de 0).")
    print("k e ω livres em TODAS as paredes: é a decisão da §2 da Cell 9, e")
    print("este é o teste que acusa se alguém reintroduzir a BC errada.")
else:
    print(f">>> {_n_fail} VERIFICAÇÃO(ÕES) FALHARAM:")
    for _fl in _fail_list:
        print(f"      {_fl}")
    print("\nA hard constraint tem defeito. NÃO treinar antes de resolver:")
    print("a restrição é multiplicativa e sempre ganha da data loss, então o")
    print("sintoma seria um campo errado que a loss não consegue corrigir.")
print("="*70)


# ================================================================================
# CELL 9c — HARD CONSTRAINT SANITY CHECK
# ================================================================================
# Valida a HardConstraintLayer após recalibração para z-score isotrópico.
#
# Roda 4 testes independentes:
#   (A) Decaimento de cada D_i individual: D=0 na fronteira, D≈1 em ΔTOL
#   (B) Ansatz combinado nas BCs, agora com k e ω
#   (C) Visualização 2D de D_cyl — em XY **e em XZ**
#   (D) Parâmetros recalibrados vs critérios físicos
#
# ------------------------------------------------------------------------------
# O QUE MUDA
#
# 1. ACABOU A RÉPLICA. A versão laminar reimplementava as funções de distância
#    num `compute_distances` próprio, com o aviso "se você mudar os parâmetros
#    internos da layer, atualize aqui também". A réplica já tinha divergido em
#    três pontos ANTES de qualquer mudança:
#
#      | | réplica da Cell 9c | layer de verdade |
#      |---|---|---|
#      | smooth_max | sem subtrair o máximo (exp → 1e162) | com |
#      | D_vel | inclui D_outlet | não inclui |
#      | decay_inlet em Δ=5,6 m | 0,9814 | 0,0186 |
#
#    Ou seja: a Cell 9c aprovava uma função que não era a que a rede usava.
#    Agora `compute_distances` só chama `hc_distance_functions` (Cell 9).
#
# 2. O PLANO XZ ENTRA NA VISUALIZAÇÃO. O teste C laminar plotava só XY em
#    z=H/2 — a meia altura do cilindro, onde D_cyl está correto. O defeito da
#    quina vive no topo, e em XY a meia altura ele é literalmente invisível.
#    Um corte que não passa pelo canto não mostra o problema do canto.
#
# 3. k e ω no teste B.
#
# [18/10/2026] A figura vai para DIAG_DIR (Cell 3), em vez de cair no
# diretório de onde o sbatch foi chamado. Os testes não mudam: a hard
# constraint é a mesma da sst2.
# ================================================================================

# Helper para conversão coords físicas → normalizadas
def to_norm(x_phys, y_phys, z_phys, t_phys=0.0):
    """Converte arrays físicos para tensor (N, 4) de coords normalizadas."""
    x_phys = np.atleast_1d(x_phys).astype(np.float32).ravel()   # garante 1D
    y_phys = np.atleast_1d(y_phys).astype(np.float32).ravel()
    z_phys = np.atleast_1d(z_phys).astype(np.float32).ravel()

    n = len(x_phys)
    assert len(y_phys) == n and len(z_phys) == n, \
        f"Shapes inconsistentes: x={len(x_phys)}, y={len(y_phys)}, z={len(z_phys)}"

    t_arr = np.full(n, t_phys, dtype=np.float32)

    xn = (x_phys - X_MEAN) / DP_ISO
    yn = (y_phys - Y_MEAN) / DP_ISO
    zn = (z_phys - Z_MEAN) / DP_ISO
    tn = (t_arr  - T_MEAN) / T_DP

    return tf.constant(np.column_stack([xn, yn, zn, tn]))   # garante (N, 4)


# Pega referência da HardConstraintLayer já construída dentro do modelo
hc_layer = None
for layer in model.layers:
    if isinstance(layer, HardConstraintLayer):
        hc_layer = layer
        break
if hc_layer is None:
    raise RuntimeError("HardConstraintLayer não encontrada no modelo. "
                       "Construa o modelo antes de rodar este sanity check.")


def compute_distances(coords):
    """
    Funções de distância nos pontos dados.

    NÃO reimplementa nada: chama hc_distance_functions, a MESMA função que a
    HardConstraintLayer usa. É o que garante que este teste valide a rede que
    de fato existe, e não uma cópia que envelheceu.
    """
    return hc_distance_functions(coords[:, 0], coords[:, 1], coords[:, 2])


# ================================================================================
# TESTE A — DECAIMENTO INDIVIDUAL
# ================================================================================
print("="*80)
print("TESTE A — DECAIMENTO DE CADA D_i AO LONGO DE Δ_FÍSICO")
print("="*80)
print(f"Critério: D ≈ 0 na fronteira, D > 0.95 em ΔTOL ({BC_TOL}m para BCs, "
      f"{CYL_TOL}m para cilindro)")
print()

# A.1 — Inlet (y = Y_MIN, varia y para dentro)
deltas = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
D = compute_distances(to_norm(np.full_like(deltas, 140.0),
                              Y_MIN + deltas,
                              np.full_like(deltas, 100.0)))
print(f"A.1) D_inlet ao se afastar de y=Y_MIN (varia +Δy física):")
print(f"     {'Δy [m]':>8} | {'D_inlet':>10}")
for i, dy in enumerate(deltas):
    print(f"     {dy:>8.2f} | {D['D_inlet'][i].numpy():>10.6f}")

# A.2 — Cilindro LATERAL (varia r afastando, a meia altura)
deltas_r = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0])
D = compute_distances(to_norm(XC_CYL + R_CYL + deltas_r,
                              np.full_like(deltas_r, YC_CYL),
                              np.full_like(deltas_r, H_CYL/2)))
print(f"\nA.2) D_cyl ao se afastar da superfície LATERAL (z=H/2, varia +Δr):")
print(f"     {'Δr [m]':>8} | {'D_cyl':>10}")
for i, dr in enumerate(deltas_r):
    print(f"     {dr:>8.2f} | {D['D_cyl'][i].numpy():>10.6f}")

# A.3 — Cilindro TOPO (varia z acima da face superior)  [NOVO]
deltas_z = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0])
D = compute_distances(to_norm(np.full_like(deltas_z, XC_CYL),
                              np.full_like(deltas_z, YC_CYL),
                              H_CYL + deltas_z))
print(f"\nA.3) D_cyl ao subir a partir da face do TOPO (r=0, varia +Δz):")
print(f"     {'Δz [m]':>8} | {'D_cyl':>10}")
for i, dz in enumerate(deltas_z):
    print(f"     {dz:>8.2f} | {D['D_cyl'][i].numpy():>10.6f}")

# A.4 — QUINA: percorre a superfície da lateral até o topo passando pela
#       aresta. É o caminho que a versão laminar nunca testou.  [NOVO]
_zq = np.array([H_CYL-10., H_CYL-5., H_CYL-2., H_CYL-0.5, H_CYL])
_rq = np.array([R_CYL]*5)
_zt = np.array([H_CYL]*4)
_rt = np.array([R_CYL*0.95, R_CYL*0.8, R_CYL*0.5, 0.0])
_th = 0.0
D = compute_distances(to_norm(XC_CYL + np.r_[_rq, _rt]*np.cos(_th),
                              YC_CYL + np.r_[_rq, _rt]*np.sin(_th),
                              np.r_[_zq, _zt]))
print(f"\nA.4) D_cyl SOBRE a superfície, subindo a lateral e cruzando a quina:")
print(f"     (tem que ser 0 em TODOS — é a superfície do sólido)")
print(f"     {'posição':<22} | {'D_cyl':>10}")
_labs = ([f"lateral z={z:.1f}" for z in _zq[:-1]] + ["QUINA r=R, z=H"] +
         [f"topo r={r:.1f}" for r in _rt])
_dq = D['D_cyl'].numpy()
for i, lab in enumerate(_labs):
    _flag = "" if abs(_dq[i]) < 1e-3 else "   <<< não zerou"
    print(f"     {lab:<22} | {_dq[i]:>10.6f}{_flag}")

# A.5 — Decaimento freestream
deltas_far = np.array([0.0, 1.0, 3.0, 5.6, 10.0, 30.0, 100.0])
D = compute_distances(to_norm(np.full_like(deltas_far, 140.0),
                              Y_MIN + deltas_far,
                              np.full_like(deltas_far, 100.0)))
print(f"\nA.5) decay_freestream (1 no inlet, →0 longe de qualquer parede):")
print(f"     {'Δy [m]':>8} | {'decay_fs':>10}")
for i, dy in enumerate(deltas_far):
    print(f"     {dy:>8.2f} | {D['decay_freestream'][i].numpy():>10.6f}")

# ================================================================================
# TESTE B — ANSATZ COMBINADO NAS BCs
# ================================================================================
print("\n" + "="*80)
print("TESTE B — ANSATZ NAS FRONTEIRAS (rede não treinada)")
print("="*80)
print("O ansatz tem que recuperar as BCs corretas independente dos pesos:")
print("u=v=w=0 nas paredes, P=0 no outlet, k e ω livres nas paredes.")
print()

N_BC = 200
_rb = np.random.default_rng(42)

# Inlet — z longe do chão e do topo, senão v não é V_INF (o chão manda)
_m = 15.0
coords_inlet_bc = to_norm(_rb.uniform(X_MIN+_m, X_MAX-_m, N_BC),
                          np.full(N_BC, Y_MIN),
                          _rb.uniform(Z_MIN+_m, Z_MAX-_m, N_BC), t_phys=10.0)
# Ground
xg = _rb.uniform(X_MIN+_m, X_MAX-_m, N_BC); yg = _rb.uniform(Y_MIN+_m, Y_MAX-_m, N_BC)
_ins = ((xg - XC_CYL)**2 + (yg - YC_CYL)**2 < R_CYL**2)
xg, yg = xg[~_ins], yg[~_ins]
coords_ground_bc = to_norm(xg, yg, np.full(len(xg), Z_MIN), t_phys=10.0)
# Cilindro lateral
_thb = _rb.uniform(0, 2*np.pi, N_BC)
coords_cyl_bc = to_norm(XC_CYL + R_CYL*np.cos(_thb), YC_CYL + R_CYL*np.sin(_thb),
                        _rb.uniform(0, H_CYL, N_BC), t_phys=10.0)
# Cilindro QUINA  [NOVO]
coords_rim_bc = to_norm(XC_CYL + R_CYL*np.cos(_thb), YC_CYL + R_CYL*np.sin(_thb),
                        np.full(N_BC, H_CYL), t_phys=10.0)
# Outlet
coords_outlet_bc = to_norm(_rb.uniform(X_MIN+_m, X_MAX-_m, N_BC),
                           np.full(N_BC, Y_MAX),
                           _rb.uniform(Z_MIN+_m, Z_MAX-_m, N_BC), t_phys=10.0)


def denorm(pred):
    """Saída do ansatz -> unidades físicas. 6 campos."""
    u = pred[:, 0] * V_REF
    v = pred[:, 1] * V_DP + V_MEAN
    w = pred[:, 2] * V_REF
    P = pred[:, 3] * P_DP + P_MEAN
    k = denormalize_k(pred[:, 4].astype(np.float64))
    om = denormalize_omega(pred[:, 5].astype(np.float64))
    return u, v, w, P, k, om


_B_CASES = [
    ("B.1) INLET (y=Y_MIN)",      coords_inlet_bc),
    ("B.2) GROUND (z=Z_MIN)",     coords_ground_bc),
    ("B.3) CILINDRO lateral",     coords_cyl_bc),
    ("B.4) CILINDRO QUINA",       coords_rim_bc),
    ("B.5) OUTLET (y=Y_MAX)",     coords_outlet_bc),
]
_B_OUT = {}
for _lab, _c in _B_CASES:
    _B_OUT[_lab] = denorm(model(_c, training=False).numpy())

print(f"{'fronteira':<24}{'|u|max':>10}{'|w|max':>10}{'v méd':>10}"
      f"{'|P|max':>10}{'k méd':>10}{'ω méd':>10}")
print("-"*84)
for _lab, _ in _B_CASES:
    u, v, w, P, k, om = _B_OUT[_lab]
    print(f"{_lab:<24}{np.abs(u).max():>10.5f}{np.abs(w).max():>10.5f}"
          f"{v.mean():>10.4f}{np.abs(P).max():>10.4f}"
          f"{np.median(k):>10.4f}{np.median(om):>10.2f}")
print("-"*84)
print(f"esperado: inlet u,w=0 v={V_INF} k={K_INLET:.4f} ω={OMEGA_INLET:.0f}  |  "
      f"paredes u,v,w=0  |  outlet P=0")

# Avaliação automática
print("\n" + "-"*80)
print("VEREDITO AUTOMÁTICO:")
print("-"*80)

TOL_VEL_BC    = 1e-2   # m/s
TOL_VEL_INLET = 5e-2   # m/s
TOL_P_BC      = 5.0    # Pa
TOL_TURB_REL  = 1e-3   # relativo, para k e ω no inlet

u_in, v_in, w_in, P_in, k_in, om_in = _B_OUT["B.1) INLET (y=Y_MIN)"]
u_gr, v_gr, w_gr, P_gr, k_gr, om_gr = _B_OUT["B.2) GROUND (z=Z_MIN)"]
u_cy, v_cy, w_cy, P_cy, k_cy, om_cy = _B_OUT["B.3) CILINDRO lateral"]
u_ri, v_ri, w_ri, P_ri, k_ri, om_ri = _B_OUT["B.4) CILINDRO QUINA"]
u_ou, v_ou, w_ou, P_ou, k_ou, om_ou = _B_OUT["B.5) OUTLET (y=Y_MAX)"]

checks = {}
checks["inlet_u"]      = np.abs(u_in).max() < TOL_VEL_BC
checks["inlet_w"]      = np.abs(w_in).max() < TOL_VEL_BC
checks["inlet_v"]      = np.abs(v_in - V_INF).max() < TOL_VEL_INLET
checks["inlet_k"]      = (np.abs(k_in/K_INLET - 1).max() < TOL_TURB_REL
                          if HARD_BC_K_INLET else True)
checks["inlet_omega"]  = (np.abs(om_in/OMEGA_INLET - 1).max() < TOL_TURB_REL
                          if HARD_BC_OMEGA_INLET else True)
checks["ground_uvw"]   = (np.abs(u_gr).max() < TOL_VEL_BC and
                          np.abs(v_gr).max() < TOL_VEL_BC and
                          np.abs(w_gr).max() < TOL_VEL_BC)
checks["cylinder_uvw"] = (np.abs(u_cy).max() < TOL_VEL_BC and
                          np.abs(v_cy).max() < TOL_VEL_BC and
                          np.abs(w_cy).max() < TOL_VEL_BC)
checks["rim_uvw"]      = (np.abs(u_ri).max() < TOL_VEL_BC and
                          np.abs(v_ri).max() < TOL_VEL_BC and
                          np.abs(w_ri).max() < TOL_VEL_BC)
checks["outlet_P"]     = np.abs(P_ou).max() < TOL_P_BC
# k e ω LIVRES nas paredes — ver §2 da Cell 9
checks["wall_k_livre"]  = (np.std(k_gr) > 1e-9 and np.std(k_cy) > 1e-9)
checks["wall_om_livre"] = (np.std(om_gr) > 1e-9 and np.std(om_cy) > 1e-9)

for k_, v_ in checks.items():
    print(f"  [{'✓ PASS' if v_ else '✗ FAIL'}]  {k_}")
n_pass = sum(checks.values())
print(f"\n  {n_pass}/{len(checks)} testes passaram")
if n_pass < len(checks):
    print("  >>> a hard constraint tem defeito; ver o detalhe na tabela acima.")

# ================================================================================
# TESTE C — VISUALIZAÇÃO 2D DE D_cyl: XY **E** XZ
# ================================================================================
print("\n" + "="*80)
print("TESTE C — VISUALIZAÇÃO DE D_cyl E D_vel")
print("="*80)

n_grid = 300
x_grid = np.linspace(X_MIN, X_MAX, n_grid)
y_grid = np.linspace(Y_MIN, Y_MAX, n_grid)
z_grid = np.linspace(Z_MIN, Z_MAX, n_grid)

# --- plano XY em z = H/2 ---
XX, YY = np.meshgrid(x_grid, y_grid)
Dg = compute_distances(to_norm(XX.ravel(), YY.ravel(),
                               np.full(XX.size, H_CYL/2), t_phys=10.0))
D_cyl_xy = Dg["D_cyl"].numpy().reshape(XX.shape)
D_vel_xy = Dg["D_vel"].numpy().reshape(XX.shape)

# --- plano XZ em y = YC_CYL (passa pelo eixo E pela quina) ---
XZx, XZz = np.meshgrid(x_grid, z_grid)
Dg2 = compute_distances(to_norm(XZx.ravel(), np.full(XZx.size, YC_CYL),
                                XZz.ravel(), t_phys=10.0))
D_cyl_xz = Dg2["D_cyl"].numpy().reshape(XZx.shape)
D_vel_xz = Dg2["D_vel"].numpy().reshape(XZx.shape)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
theta = np.linspace(0, 2*np.pi, 200)

for _ax, _M, _tit, _lbl in [
        (axes[0,0], D_cyl_xy, f'D_cyl — XY em z={H_CYL/2:.0f} m', 'D_cyl'),
        (axes[0,1], D_vel_xy, f'D_vel — XY em z={H_CYL/2:.0f} m', 'D_vel')]:
    c = _ax.pcolormesh(XX, YY, _M, cmap='Blues', vmin=0, vmax=1, shading='auto')
    plt.colorbar(c, ax=_ax, label=_lbl)
    _ax.plot(XC_CYL + R_CYL*np.cos(theta), YC_CYL + R_CYL*np.sin(theta),
             color='#b03030', lw=2, label=f'parede (r={R_CYL:.0f} m)')
    _ax.plot(XC_CYL + (R_CYL+CYL_TOL)*np.cos(theta),
             YC_CYL + (R_CYL+CYL_TOL)*np.sin(theta),
             color='#b03030', lw=1, ls='--', alpha=.7, label='+CYL_TOL')
    _ax.set_aspect('equal'); _ax.set_xlabel('x [m]'); _ax.set_ylabel('y [m]')
    _ax.set_title(_tit, fontsize=11); _ax.legend(loc='upper right', fontsize=8)

for _ax, _M, _tit, _lbl in [
        (axes[1,0], D_cyl_xz, f'D_cyl — XZ em y={YC_CYL:.0f} m  (corta a QUINA)', 'D_cyl'),
        (axes[1,1], D_vel_xz, f'D_vel — XZ em y={YC_CYL:.0f} m', 'D_vel')]:
    c = _ax.pcolormesh(XZx, XZz, _M, cmap='Blues', vmin=0, vmax=1, shading='auto')
    plt.colorbar(c, ax=_ax, label=_lbl)
    _ax.add_patch(Rectangle((XC_CYL-R_CYL, Z_MIN), 2*R_CYL, H_CYL,
                            fill=False, edgecolor='#b03030', lw=2))
    for _s in (-1, 1):
        _ax.plot([XC_CYL + _s*R_CYL], [H_CYL], marker='o', ms=7,
                 mfc='none', mec='#b03030', mew=2)
    _ax.set_xlabel('x [m]'); _ax.set_ylabel('z [m]')
    _ax.set_xlim(X_MIN, X_MAX); _ax.set_ylim(Z_MIN, Z_MAX)
    _ax.set_title(_tit, fontsize=11)

plt.tight_layout()
_p_hc_png = os.path.join(DIAG_DIR, 'sanity_check_hardconstraint.png')
plt.savefig(_p_hc_png, dpi=120, bbox_inches='tight')
plt.close()
print(f"Figura salva: {_p_hc_png}")
print()
print("Observe na figura:")
print("  XY (linha de cima): D_cyl zero sobre o círculo vermelho, ≈1 fora do")
print("    tracejado. O contorno D=0 tem que ser CIRCULAR, não elipse.")
print("  XZ (linha de baixo): é o corte NOVO, e é onde o defeito da quina")
print("    aparecia. Os dois círculos marcam as arestas superiores. Com")
print("    CYL_RIM_FIX ativo a faixa clara envolve o retângulo INTEIRO,")
print("    incluindo os cantos; sem ele, os cantos ficam escuros — ou seja,")
print("    sem restrição justamente onde o escoamento separa.")

# Diagnóstico numérico: D_cyl sobre toda a superfície, não só a lateral
print()
print("Verificação numérica sobre a superfície do cilindro:")
_tc = np.linspace(0, 2*np.pi, 60)
_surf = [("lateral (z=H/2)", XC_CYL + R_CYL*np.cos(_tc), YC_CYL + R_CYL*np.sin(_tc),
          np.full(60, H_CYL/2)),
         ("QUINA   (z=H)",   XC_CYL + R_CYL*np.cos(_tc), YC_CYL + R_CYL*np.sin(_tc),
          np.full(60, H_CYL)),
         ("topo    (r<R)",   XC_CYL + 0.6*R_CYL*np.cos(_tc),
          YC_CYL + 0.6*R_CYL*np.sin(_tc), np.full(60, H_CYL))]
_all_ok = True
for _lab, _xs, _ys, _zs in _surf:
    _d = compute_distances(to_norm(_xs, _ys, _zs, t_phys=10.0))["D_cyl"].numpy()
    _ok = _d.max() < 1e-3
    _all_ok &= _ok
    print(f"  {_lab:<18} μ={_d.mean():.2e}  max={_d.max():.2e}  "
          f"{'✓' if _ok else '✗ NÃO ZERA'}")
print(f"  {'-> cilindro circular e no-slip em TODA a superfície ✓' if _all_ok else '  >>> a superfície não está totalmente restrita'}")

# ================================================================================
# TESTE D — VERIFICAÇÃO DOS PARÂMETROS RECALIBRADOS
# ================================================================================
print("\n" + "="*80)
print("TESTE D — PARÂMETROS RECALIBRADOS vs CRITÉRIOS FÍSICOS")
print("="*80)
print(f"\nObjetivo: D ≥ 0.95 a ΔTOL da fronteira")
print(f"          (a região de transição da BC tem que caber em ΔTOL)")

_D_TESTS = [
    ("D_inlet  a Δy=%.1fm" % BC_TOL,
     to_norm([140.0], [Y_MIN + BC_TOL], [100.0]), "D_inlet", 0.95),
    ("D_ground a Δz=%.1fm" % BC_TOL,
     to_norm([140.0], [350.0], [Z_MIN + BC_TOL]), "D_ground", 0.95),
    ("D_cyl lateral a Δr=%.1fm" % CYL_TOL,
     to_norm([XC_CYL + R_CYL + CYL_TOL], [YC_CYL], [H_CYL/2]), "D_cyl", 0.95),
    ("D_cyl topo    a Δz=%.1fm" % CYL_TOL,
     to_norm([XC_CYL], [YC_CYL], [H_CYL + CYL_TOL]), "D_cyl", 0.95),
]
print()
_nd_ok = 0
for _lab, _c, _key, _thr in _D_TESTS:
    _v = float(compute_distances(_c)[_key][0].numpy())
    _ok = _v >= _thr
    _nd_ok += _ok
    print(f"  {_lab:<26} {_v:.4f}  {'✓' if _ok else '✗  (< %.2f)' % _thr}")

# decay_freestream tem que estar ~1 no inlet e ~0 no miolo
_v_in  = float(compute_distances(to_norm([140.],[Y_MIN],[100.]))["decay_freestream"][0])
_v_mid = float(compute_distances(to_norm([140.],[350.],[100.]))["decay_freestream"][0])
print(f"  {'decay_fs no inlet':<26} {_v_in:.4f}  {'✓' if _v_in > 0.99 else '✗'}")
print(f"  {'decay_fs no miolo':<26} {_v_mid:.4f}  {'✓' if _v_mid < 0.01 else '✗'}")

print("\n" + "="*80)
print("SANITY CHECK CONCLUÍDO")
print("="*80)


# ============================================================
# CELL 9D — TESTE ISOLADO DE D_cyl NA SUPERFÍCIE
# ============================================================
# Testa se D_cyl realmente zera sobre a parede do cilindro,
# com os parâmetros recalibrados.
#
# ------------------------------------------------------------------------------
# DUAS MUDANÇAS, e a primeira é a razão de o defeito da quina ter sobrevivido
#
# 1. A SUPERFÍCIE INTEIRA, não só a lateral a meia altura.
#
#    A versão laminar gerava 100 pontos em `z = H_CYL/2` — o meio da lateral.
#    É o único lugar da superfície onde D_cyl estava correto. A célula cujo
#    propósito declarado é "testar se D_cyl realmente zera sobre a parede"
#    testava a única região que não tinha problema, e passava.
#
#    Aqui o caminho percorre a superfície toda: sobe a lateral, cruza a
#    QUINA, e entra pelo topo até o eixo. Se D_cyl não zerar em algum ponto
#    do caminho, aparece na tabela com o valor e a posição.
#
# 2. SEM RÉPLICA. A versão laminar recalculava a fórmula aqui, com o
#    comentário "smooth_max manual (não tem acesso ao método interno aqui)".
#    Agora tem: `hc_distance_functions` (Cell 9) devolve também os
#    intermediários (r², folgas, d_cyl_3d), então esta célula é consumidora
#    pura. Se a fórmula mudar, este teste acompanha sozinho.
#
# O valor desta célula, que a 9c não cobre, é mostrar a CADEIA inteira:
#     r²  ->  folga radial e folga vertical  ->  qual das duas manda  ->  tanh
# Quando algo dá errado, é aqui que se vê em qual elo.
# ============================================================

# --- caminho sobre a superfície: base -> lateral -> QUINA -> topo -> eixo ---
_N_LAT, _N_TOP = 7, 6
_z_lat = np.linspace(2.0, H_CYL, _N_LAT)              # sobe a lateral até a quina
_r_lat = np.full(_N_LAT, R_CYL)
_r_top = np.linspace(R_CYL, 0.0, _N_TOP + 1)[1:]      # entra pelo topo
_z_top = np.full(_N_TOP, H_CYL)

_th0 = np.deg2rad(37.0)      # ângulo qualquer: a superfície é axissimétrica
_xs = XC_CYL + np.r_[_r_lat, _r_top] * np.cos(_th0)
_ys = YC_CYL + np.r_[_r_lat, _r_top] * np.sin(_th0)
_zs = np.r_[_z_lat, _z_top]
_labs = ([f"lateral  z={z:5.1f} m" for z in _z_lat[:-1]]
         + ["QUINA    r=R, z=H"]
         + [f"topo     r={r:5.1f} m" for r in _r_top])

_coords = to_norm(_xs, _ys, _zs, t_phys=50.0)
_D = compute_distances(_coords)     # a MESMA função que a rede usa

print("="*84)
print("CELL 9D — D_cyl SOBRE A SUPERFÍCIE DO CILINDRO")
print("="*84)
print(f"  XC_N={XC_N:.6f}  YC_N={YC_N:.6f}  R_N={R_N:.6f}  H_N={H_N:.6f}")
print(f"  R_N² = {R_N**2:.6f}   fator de escala do topo: 2·R_N = {2*R_N:.6f}")
print(f"  caminho: {len(_xs)} pontos, da base da lateral ao eixo do topo,")
print(f"           passando pela quina. θ = {np.rad2deg(_th0):.0f}° (axissimétrico)")
print()
# Qual folga vertical o combinador de fato usa depende do modo: a forma
# corrigida escala por 2·R_N, o smooth_max laminar usa d_top cru. Mostrar a
# errada faria a coluna "manda" descrever uma fórmula que não está rodando.
_TOP_KEY = "d_top_scaled" if CYL_RIM_FIX else "d_top"
_top_lbl = "2R·folga top" if CYL_RIM_FIX else "folga top"

print(f"{'posição':<22}{'r²':>11}{'folga rad':>12}{_top_lbl:>13}"
      f"{'manda':>8}{'d_cyl_3d':>11}{'D_cyl':>10}")
print("-"*85)

_r2   = _D["r2"].numpy()
_drad = _D["d_radial_sq"].numpy()
_dtop = _D[_TOP_KEY].numpy()
_d3d  = _D["d_cyl_3d"].numpy()
_dcyl = _D["D_cyl"].numpy()

_n_bad = 0
for i, lab in enumerate(_labs):
    _manda = "radial" if _drad[i] >= _dtop[i] else "topo"
    _flag = ""
    if abs(_dcyl[i]) > 1e-3:
        _flag = "  <<<"
        _n_bad += 1
    print(f"{lab:<22}{_r2[i]:>11.6f}{_drad[i]:>12.6f}{_dtop[i]:>13.6f}"
          f"{_manda:>8}{_d3d[i]:>11.6f}{_dcyl[i]:>10.6f}{_flag}")
print("-"*85)
print(f"  'folga rad'    = r² − R_N²   (zero na lateral, negativa dentro do raio)")
print(f"  '{_top_lbl}' = {'2·R_N·' if CYL_RIM_FIX else ''}(z − H_N)"
      f"{'   (as duas na mesma escala)' if CYL_RIM_FIX else '   (SEM escala: é o que o smooth_max usa)'}")
print(f"  'manda'        = qual das duas o combinador escolhe — troca NA QUINA")
print(f"  d_cyl_3d       = {'max' if CYL_RIM_FIX else 'smooth_max'} das duas; "
      f"tem que ser ≤ 0 em toda a superfície")
print(f"  D_cyl       = tanh(238 · max(d_cyl_3d, 0)); tem que ser 0")

if _n_bad == 0:
    print(f"\n  ✓ D_cyl zera nos {len(_labs)} pontos da superfície, incluindo a quina.")
else:
    print(f"\n  ✗ D_cyl NÃO zera em {_n_bad} de {len(_labs)} pontos (marcados com <<<).")
    print(f"    O no-slip não age ali. Olhe a coluna 'd_cyl_3d': se for POSITIVO")
    print(f"    sobre a superfície, o problema está na combinação das folgas —")
    print(f"    é o que acontece com CYL_RIM_FIX=False, onde o smooth_max")
    print(f"    devolve log(2)/75 = 0.00924 na quina em vez de 0.")

# --- circularidade: o teste original, mantido ---
# Em coordenadas normalizadas com DP_ISO isotrópico, o cilindro tem que
# continuar CIRCULAR. Se cada eixo tivesse seu próprio desvio padrão,
# viraria elipse e r² deixaria de ser constante ao longo de θ.
_tc = np.linspace(0, 2*np.pi, 100, endpoint=False)
_Dc = compute_distances(to_norm(XC_CYL + R_CYL*np.cos(_tc),
                                YC_CYL + R_CYL*np.sin(_tc),
                                np.full(100, H_CYL/2), t_phys=50.0))
_r2c = _Dc["r2"].numpy()
_spread = float(_r2c.max() - _r2c.min())
print(f"\n  Circularidade — r² ao longo de θ na lateral (100 ângulos):")
print(f"    μ={_r2c.mean():.8f}  (R_N² = {R_N**2:.8f})")
print(f"    dispersão max−min = {_spread:.2e}")
print(f"    {'✓ circular' if _spread < 1e-6 else '✗ ELIPSE — DP_ISO não está isotrópico'}")

# --- onde a troca de regime acontece, em metros ---
# Informação que nenhuma outra célula dá: a partir de que altura, subindo
# pela lateral, a folga do topo passa a ser a restritiva. É a posição da
# superfície de dobra discutida na Cell 9.
_zz = np.linspace(0.0, H_CYL + 20.0, 4001)
_Dz = compute_distances(to_norm(np.full_like(_zz, XC_CYL + R_CYL + 1.0),
                                np.full_like(_zz, YC_CYL), _zz, t_phys=50.0))
_sw = _Dz["d_top_scaled"].numpy() >= _Dz["d_radial_sq"].numpy()
if _sw.any():
    _z_switch = float(_zz[np.argmax(_sw)])
    print(f"\n  Superfície de dobra (onde o max troca de ramo), medida a 1 m")
    print(f"  de gap radial: z = {_z_switch:.2f} m, ou {_z_switch - H_CYL:+.2f} m")
    print(f"  em relação ao topo do cilindro (H = {H_CYL:.0f} m).")
    print(f"  É o cone que sai da quina — a única não-suavidade dentro do fluido.")
print("="*84)


# ============================================================
# CELL 10 — PHYSICS RESIDUALS (URANS + k-ω SST, na forma do Fluent)
# ============================================================
# Equações resolvidas — ρ constante, forma cinemática multiplicada por ρ:
#
#   Continuidade:  ∂u_i/∂x_i = 0
#
#   Momento:       ρ(∂u_i/∂t + u_j ∂u_i/∂x_j)
#                    = −∂p̃/∂x_i + ∂/∂x_j[ μ_eff (∂u_i/∂x_j + ∂u_j/∂x_i) ]
#
#   Transporte k:  ρ(∂k/∂t + u_j ∂k/∂x_j)
#                    = P̃_k − β* ρ k ω + ∂/∂x_j[(μ + σ_k μ_t) ∂k/∂x_j]
#
#   Transporte ω:  ρ(∂ω/∂t + u_j ∂ω/∂x_j)
#                    = α_∞ ρ S² − β_i ρ ω² + ∂/∂x_j[(μ + σ_ω μ_t) ∂ω/∂x_j]
#                      + 2(1−F1) ρ σ_ω2 (1/ω) ∂k/∂x_j ∂ω/∂x_j
#
#   μ_eff = μ + μ_t
#   μ_t   = min( (ρk/ω) / max(1, S·F2/(a1·ω)) , VISC_RATIO_MAX·μ )
#   S²    = 2 S_ij S_ij,   S = √(S²) = strain-rate-mag do Fluent
#   P_k   = μ_t S²,   P̃_k = min(P_k, 10 β* ρ k ω)
#
# ==============================================================================
# REVISÃO DE 28/09/2026 — QUATRO COISAS QUE ESTAVAM ERRADAS
#
# A versão anterior desta célula foi escrita antes dos jobs 6023/6025/6027.
# Cada erro abaixo tem sua correção e o motivo, porque três deles são o tipo
# de coisa que volta sozinha se ninguém escrever por que saiu.
#
# 1. O TERMO (2/3)ρ∇k NÃO EXISTE NA EQUAÇÃO DO SOLVER.
#    Estava ligado. O job 6023 mediu, na caixa, que a coluna `pressure` é a
#    pressão MODIFICADA p̃ = p + (2/3)ρk: o campo ficou em 1e-12 Pa enquanto
#    k decaía, e o resíduo de momento em 1e-13 desde a 1ª iteração. Se a
#    variável já é p̃, escrever −∇(2k/3) conta o termo duas vezes. O flag
#    agora é DERIVADO de PRESSURE_IS_MODIFIED (Cell 2), não digitado aqui:
#    é uma decisão de física do dado, não uma opção do resíduo.
#
# 2. ν_t NÃO É k/ω.
#    Estava assim, com SST_SHEAR_LIMITER=False, apoiado numa mediana de
#    ν_t/(k/ω) igual a 1,000000. A medição estava certa e a conclusão
#    errada: aquela mediana vem de uma amostra dominada pelas células onde o
#    limitador não atua. O job 6027 reconstruiu ν_t célula a célula nas
#    211.579 células e a fórmula COMPLETA — limitador de Bradshaw com F2 e
#    teto de 1e5 — acerta 100% delas. Na malha de teste da Cell 5 os três
#    ramos aparecem: 73,2% livre, 23,9% no teto, 2,9% limitado (a
#    classificação da Cell 5 é exclusiva: o teto esconde o limitador,
#    porque acima dele ν_t vale o teto e a razão ν_t/(k/ω) não diz mais
#    nada sobre o ramo do max).
#
#    O limitador usa S (strain rate), não Ω (vorticidade): nas 24.848
#    células em que as duas hipóteses divergem, S acerta 100% e Ω acerta 0%.
#
# 3. ∂μ_t/∂x_j NÃO TEM MAIS FORMA ANALÍTICA CURTA.
#    Estava ρ(k_x/ω − k ω_x/ω²), que é a derivada de k/ω. Com o limitador e
#    o teto, μ_t passa a depender também de S — ou seja, das PRIMEIRAS
#    derivadas da velocidade — e tem dois pontos de não-diferenciabilidade
#    (o max do limitador e o min do teto). Escrever isso à mão dá uma
#    expressão longa e fácil de errar em silêncio.
#
#    Então μ_t é construído DENTRO da fita externa e ∇μ_t sai por autodiff.
#    Custa uma passagem de segunda ordem a mais e é sempre consistente com a
#    μ_t que o resíduo usa, qualquer que seja o ramo ativo. Nos kinks o
#    autodiff pega um subgradiente, que é o comportamento padrão e correto.
#
#    ARMADILHA DO sqrt: S = √(S²) tem derivada infinita em S²=0, e mesmo
#    quando o max(1,·) zera o caminho, inf·0 = NaN no backward. Por isso
#    S = √(S² + ε) e não √(max(S²,0)).
#
# 4. A FAIXA DE PAREDE NÃO É UM ESCALAR, E A LEI LOG NÃO É O DEFAULT.
#    Estava mascarando por d_wall < Y_WALL_FUNCTION (um escalar) e trocando
#    o resíduo de k pela relação da camada log. Duas correções:
#
#      - o limiar vem POR PONTO (coll_ywall, Cell 7, medido pela Cell 5 em
#        cada parede). Com o escalar, 18% do pool era classificado errado;
#      - o probe (job 6025) mostrou wall-omega-treatment = "correlation".
#        Lendo o §4.18.3 (29/09/2026): ali ω é PRESCRITO (eqs. 4.404-4.407)
#        e k CONTINUA transportado, com ∂k/∂n=0 e produção P_k,wall blendada
#        (eqs. 4.400-4.403) no lugar de μ_t S². Mascarar os dois continua
#        certo, mas por um motivo diferente do que estava escrito: P_k,wall
#        depende de Δy, a altura da CÉLULA, que não existe num ponto de
#        colocação. O balanço que o solver fechou ali é de volume finito
#        com fonte modificada, não uma EDP avaliada ponto a ponto.
#        O default é WALL_RESIDUAL_MODE='mask'; 'loglaw' segue disponível
#        para o caso de a verificação (d) da Cell 5 mostrar razão ≈ 1.
#
# E uma quinta, que é ausência e não erro:
#
# 5. ONDE O PRODUCTION LIMITER CORTA, P_ω É INDEFINIDO.
#    O manual não diz se G_ω usa o P_k limitado ou o bruto. Medido (job
#    6027): 1.132 células, 0,54% do domínio, removendo 0,004% da produção
#    total, todas no ponto de estagnação à frente do cilindro. Resolver a
#    dúvida custa mais que contorná-la, então esses pontos saem do resíduo
#    de ω (MASK_PROD_LIMITED_OMEGA, Cell 2). Aqui eles são DETECTADOS, o que
#    a Cell 5 não conseguia fazer: P_k = μ_t S² precisa de S, que não está
#    no CSV, mas está disponível aqui porque vem da própria rede.
#
# ==============================================================================
# O QUE NÃO MUDOU, e por quê
#
# - O TERMO VISCOSO CONTINUA EXPANDIDO PELA REGRA DO PRODUTO:
#       μ_eff ∇²u_i + (∂μ_eff/∂x_j)(∂u_i/∂x_j + ∂u_j/∂x_i)
#   Escrever μ_eff∇²u e parar por aí é o erro clássico: fica quase certo no
#   freestream, onde μ_t varia pouco, e errado na esteira, que é onde a
#   física acontece. O termo μ_eff ∂(∇·V)/∂x_i fica FORA porque ∇·V = 0 —
#   incluí-lo acoplaria o resíduo de momento ao erro de continuidade da rede
#   sem necessidade.
#
# - OS COEFICIENTES MISTURADOS POR F1 NÃO SÃO DIFERENCIADOS. σ_k, σ_ω, β_i e
#   α_∞ dependem da posição via F1. Derivá-los seria formalmente mais
#   correto, mas nem o Fluent faz isso: na discretização eles são avaliados
#   e congelados localmente.
#
# - P_ω = α_∞ ρ S², SEM dividir por ν_t. Vem de G_ω = (α α*/ν_t)·G_k com
#   G_k = μ_t S² = ρ ν_t S²: o ν_t cancela, inclusive o teto, que aparece em
#   cima e embaixo. NÃO usar a forma α(ω/k)G_k do k-ω padrão (eq. 4.82) —
#   ela só coincide onde ν_t = k/ω, e o limitador existe justamente para que
#   isso não valha em toda parte.
#
# - k E ω SAEM EM ESPAÇO LOG. k_eff = exp(...) = k + K_EPS é positivo por
#   construção; como K_EPS é constante, todas as derivadas são IDÊNTICAS às
#   de k e nenhum clamp entra no caminho do gradiente.
#
# - ESCALA LOCAL DO RESÍDUO DE ω. ω vai de ~1 no interior a 7419 no inlet, e
#   βρω² de 0,098 a 5,4e6 — 5,5e7 de faixa. Com uma escala global única o MSE
#   seria inteiramente dominado pela camada junto à entrada. Dividir pelo
#   PRÓPRIO termo de destruição transforma o resíduo em RELATIVO: a destruição
#   vale exatamente 1 e os outros termos são medidos como fração dela. Não
#   muda o zero, muda o peso entre regiões.
#
#   (Cuidado com a formulação: dividir por ρω daria exatamente a equação de
#   ln ω, porque Dω/Dt = ω·D(ln ω)/Dt. Dividir por ρβω² é aquilo dividido
#   mais uma vez por βω — parecido em espírito, não idêntico. Uma versão
#   anterior deste comentário dizia que era a equação de ln ω; não é.)
#
# ==============================================================================
# ARMADILHA DA FITA — não reordenar as linhas
#
# Ao calcular segunda derivada com fitas aninhadas, TODO fatiamento do
# primeiro gradiente tem que acontecer DENTRO do `with` da fita externa. Se
# sair de lá, ela não registra a operação e `gradient()` devolve None — sem
# erro, sem aviso. Testado: `f = z³` em `z = 3` dá `d²f/dz² = 18` com a
# fatia dentro e `None` com a fatia fora.
#
# Isto ficou MAIS crítico nesta versão: μ_t também é construído dentro da
# fita externa, a partir das fatias. Mover a construção de μ_t para fora faz
# ∇μ_t virar None, e o resíduo perde o termo (∂μ_eff/∂x_j)(...) inteiro —
# silenciosamente, com a loss ainda caindo.
# ==============================================================================


# ---------- USER CONFIG ----------
# DERIVADO da Cell 2, não digitado: se a coluna é p̃, o termo já está dentro
# dela. Ver o ponto 1 do cabeçalho.
MOMENTUM_INCLUDE_K_GRADIENT = not PRESSURE_IS_MODIFIED

# "local" divide o resíduo de ω pelo termo de destruição (ρβω²), tornando-o
# RELATIVO ao maior termo local. Ver o cabeçalho.
OMEGA_RESIDUAL_SCALE = "local"      # "local" | "global"

# ------------------------------------------------------------------------------
# ESCALA DO RESÍDUO DE k — "local" desde 01/10/2026
# ------------------------------------------------------------------------------
# Até aqui r_k era dividido por SCALE_K = ρ·V_inf·K_REF/D = 3,7615 Pa/s, uma
# CONSTANTE, enquanto r_ω já era dividido pelo próprio termo de destruição.
# A assimetria não era intencional — o comentário da adimensionalização, logo
# abaixo, já dizia que a escala local existe para "a faixa do inlet deixar de
# dominar o MSE". O k nunca recebeu o mesmo tratamento.
#
# O que isso fazia, MEDIDO no checkpoint 400 com 48.000 pontos:
#
#     r_k / SCALE_K        MSE 18,50   p50 2,2e-03   p99 2,26   máx 190
#                          faixa dinâmica p99/p50 = 1020×
#                          top 0,1% dos pontos (45 de 44.535) = 83% do MSE
#                          top 1%                             = 99,7%
#                          adj_inlet (4,4% dos pontos)        = 96,2% do MSE
#
#     r_k / (β*ρkω)        MSE 10,05   p50 1,18      p99 12,6  máx 31,7
#                          faixa dinâmica = 10,6×
#                          top 1% = 27,9%
#
# Ou seja: o treino gastava ~14% da loss total (k é 75% da física) em 45
# pontos de uma faixa onde a equação não fecha, e o topo do cilindro — onde
# os termos de k são os maiores do domínio fora do inlet, |r_k| = 2,46 Pa/s
# contra termos de 3 a 7 Pa/s — pesava 0,1%.
#
# POR QUE UM PISO, e não o local puro como o ω. β*ρkω vai a zero no
# escoamento livre (k ≈ 0,005, ω ≈ 0,84 -> 4,5e-04 Pa/s) e 34% das células
# do dataset têm k abaixo de K_EPS. Sem piso, ali o resíduo relativo é ruído
# dividido por ruído, e o treino passaria a perseguir o escoamento livre,
# onde não há física acontecendo. O piso ξ·SCALE_K limita essa amplificação:
#
#     r_k_n = r_k / max( β*ρkω , ξ·SCALE_K )
#
#     ξ = 0  -> local puro (o que o ω faz)
#     ξ = 1  -> praticamente a escala global antiga
#
# ξ NÃO É CHUTE. Varredura MEDIDA no checkpoint 2200 da rodada "sst", com
# 48.000 pontos (diagnostico_residuos.py imprime esta tabela):
#
#      ξ        MSE    |r| p50  p99/p50  top1%   inlet    cil    boi   free
#      0    1.14e+01    1.031     14.0   33.2%    0.6%   5.9%  77.2%  10.9%
#   1e-4    1.14e+01    1.028     14.0   33.2%    0.6%   5.9%  77.2%  10.9%
#   1e-3    7.42e+00    0.595     21.3   44.2%    0.9%   9.1%  80.8%   5.7%
#   1e-2    2.48e+00    0.156     47.2   54.4%    2.7%  18.9%  73.7%   1.7%
#   1e-1    4.29e-01    0.020    140.3   55.7%   15.3%  20.9%  56.5%   0.3%
#      1    6.98e-02    0.002    552.6   56.7%   29.7%  22.8%  30.3%   0.0%
#
# A COLUNA MSE NÃO SERVE PARA ESCOLHER. Dividir por um número maior sempre
# dá um número menor; ξ = 1 tem o menor MSE e é o PIOR ξ da tabela. O que
# importa é a DISTRIBUIÇÃO do gradiente: p99/p50 e top1%.
#
# ξ = 1e-4 dá estatística idêntica ao local puro (ξ = 0) e ainda garante
# denominador não-nulo onde k vai a zero — 34% das células do dataset estão
# abaixo de K_EPS, e durante o treino a rede pode passar por lá. É seguro
# de graça.
#
# O medo de que o escoamento livre tomasse conta NÃO se confirmou: com
# ξ = 0, 'free' fica com 10,9% do MSE tendo 35,9% dos pontos, ou seja
# SUB-representado. Quem concentra é 'boi' (77% do MSE com 36% dos
# pontos) — as caixas de refino em volta do cilindro e na esteira, que é
# exatamente onde a física interessa. Essa concentração é desejada.
#
# Refazer a varredura se a rede mudar muito de regime.
K_RESIDUAL_SCALE    = "local"       # "local" | "global"
K_RESIDUAL_FLOOR_XI = 1.0e-4        # piso do denominador, em unidades de SCALE_K

# ε dentro do sqrt de S. Ver a armadilha do sqrt, ponto 3.
S_SQRT_EPS = 1.0e-16


# ---------- CONSTANTES DE DESNORMALIZAÇÃO COMO TF TENSORS ----------
_DP_ISO  = tf.constant(DP_ISO,   tf.float32)   # σ espacial isotrópico
_T_DP    = tf.constant(T_DP,     tf.float32)   # σ temporal

_V_REF   = tf.constant(V_REF,    tf.float32)
_V_MEAN  = tf.constant(V_MEAN,   tf.float32)
_V_DP    = tf.constant(V_DP,     tf.float32)
_P_MEAN  = tf.constant(P_MEAN,   tf.float32)
_P_DP    = tf.constant(P_DP,     tf.float32)

_LNK_MEAN = tf.constant(LNK_MEAN, tf.float32)
_LNK_DP   = tf.constant(LNK_DP,   tf.float32)
_LNW_MEAN = tf.constant(LNW_MEAN, tf.float32)
_LNW_DP   = tf.constant(LNW_DP,   tf.float32)

_RHO_INF = tf.constant(RHO_INF,  tf.float32)
_MU      = tf.constant(MU,       tf.float32)
_NU      = tf.constant(NU,       tf.float32)

# Constantes do SST (Cell 2: σ na convenção do Menter, derivados dos
# números de Prandtl do Fluent)
_BSTAR   = tf.constant(SST_BETA_STAR, tf.float32)
_KAPPA   = tf.constant(SST_KAPPA,     tf.float32)
_A1      = tf.constant(SST_A1,        tf.float32)
_SK1     = tf.constant(SST_SIGMA_K1,  tf.float32)
_SK2     = tf.constant(SST_SIGMA_K2,  tf.float32)
_SW1     = tf.constant(SST_SIGMA_W1,  tf.float32)
_SW2     = tf.constant(SST_SIGMA_W2,  tf.float32)
_BETA1   = tf.constant(SST_BETA_1,    tf.float32)
_BETA2   = tf.constant(SST_BETA_2,    tf.float32)
_GAMMA1  = tf.constant(SST_GAMMA_1,   tf.float32)
_GAMMA2  = tf.constant(SST_GAMMA_2,   tf.float32)
_PROD_LIM = tf.constant(SST_PROD_LIMIT_COEF, tf.float32)
_CD_FLOOR = tf.constant(SST_CD_KW_FLOOR, tf.float32)
_DW_FLOOR = tf.constant(D_WALL_FLOOR, tf.float32)
_WALL_LAW_C = tf.constant(WALL_LAW_C, tf.float32)
_MUT_MAX  = tf.constant(RHO_INF * NU_T_MAX, tf.float32)   # μ_t máximo [Pa·s]


# ---------- ESCALAS PARA NORMALIZAR RESÍDUOS A O(1) ----------
# Vêm do problema físico, NÃO da normalização da rede.
V_PHYS_SCALE = V_INF                # 17 m/s
L_PHYS_SCALE = 2.0 * R_CYL          # = D_CYL = 54 m

SCALE_CONT = V_PHYS_SCALE / L_PHYS_SCALE                    # ≈ 0.315 [1/s]
SCALE_MOM  = RHO_INF * V_PHYS_SCALE**2 / L_PHYS_SCALE       # ≈ 6.56  [Pa/m]
SCALE_K    = RHO_INF * V_PHYS_SCALE * K_REF     / L_PHYS_SCALE
SCALE_OM   = RHO_INF * V_PHYS_SCALE * OMEGA_REF / L_PHYS_SCALE

_SCALE_CONT = tf.constant(SCALE_CONT, tf.float32)
_SCALE_MOM  = tf.constant(SCALE_MOM,  tf.float32)
_SCALE_K    = tf.constant(SCALE_K,    tf.float32)
_XI_K_FLOOR = tf.constant(K_RESIDUAL_FLOOR_XI * SCALE_K, tf.float32)
_SCALE_OM   = tf.constant(SCALE_OM,   tf.float32)

print(f"\n=== Physics residual scales ===")
print(f"_DP_ISO    = {_DP_ISO.numpy():.4f} m")
print(f"_T_DP      = {_T_DP.numpy():.4f} s")
print(f"SCALE_CONT = {SCALE_CONT:.4f}  [1/s]")
print(f"SCALE_MOM  = {SCALE_MOM:.4f}  [Pa/m]")
print(f"SCALE_K    = {SCALE_K:.4f}  [Pa/s]      (K_REF={K_REF:.3f})")
print(f"SCALE_OM   = {SCALE_OM:.4f}  [kg/(m³s²)] (OMEGA_REF={OMEGA_REF:.3f})")

print(f"\n=== Configuração do resíduo ===")
print(f"  pressão            : {'p̃ (modificada)' if PRESSURE_IS_MODIFIED else 'p (estática)'}"
      f"  ->  termo (2/3)ρ∇k no momento: {MOMENTUM_INCLUDE_K_GRADIENT}")
print(f"  ν_t                : "
      f"{'min((k/ω)/max(1, S·F2/(a1·ω)), teto)' if SST_SHEAR_LIMITER else 'min(k/ω, teto)'}")
print(f"  teto de μ_t        : {_MUT_MAX.numpy():.6f} Pa·s "
      f"(ν_t ≤ {NU_T_MAX:.4f} m²/s)")
print(f"  ∇μ_t               : autodiff (ver ponto 3 do cabeçalho)")
print(f"  limitador de P_k   : {SST_PRODUCTION_LIMITER}")
print(f"  faixa de parede    : WALL_RESIDUAL_MODE = '{WALL_RESIDUAL_MODE}', "
      f"limiar POR PONTO (coll_ywall)")
print(f"  máscara do P_k lim.: MASK_PROD_LIMITED_OMEGA = {MASK_PROD_LIMITED_OMEGA}")
print(f"  escala do resíduo ω: {OMEGA_RESIDUAL_SCALE}")
print(f"  escala do resíduo k: {K_RESIDUAL_SCALE}"
      + (f"  ->  r_k / max(β*ρkω, {K_RESIDUAL_FLOOR_XI:g}·SCALE_K)"
         f"  piso = {K_RESIDUAL_FLOOR_XI*SCALE_K:.4g} Pa/s"
         if K_RESIDUAL_SCALE == "local" else f"  ->  r_k / {SCALE_K:.4f}"))


def sst_F1(k, om, dw, kx, ky, kz, wx, wy, wz):
    """
    Função de mistura F1: 1 junto à parede (ramo k-ω), 0 longe (ramo k-ε
    transformado). É ela que faz o modelo ser "SST" e não um dos dois
    isolados. Eqs. 4.107-4.109.

        CD_kω = max( 2ρσ_ω2 (1/ω) ∇k·∇ω , 1e-10 )
        Φ1    = min( max( √k/(β*ωy) , 500ν/(y²ω) ) , 4ρσ_ω2 k/(CD_kω y²) )
        F1    = tanh(Φ1⁴)

    Os σ_ω2 aqui estão na convenção do Menter (multiplicam), que é o
    recíproco do número de Prandtl do manual — ver Cell 2.
    """
    y  = tf.maximum(dw, _DW_FLOOR)
    y2 = tf.square(y)

    grad_k_dot_grad_w = kx*wx + ky*wy + kz*wz
    CD_kw = tf.maximum(2.0 * _RHO_INF * _SW2 * grad_k_dot_grad_w / om,
                       _CD_FLOOR)

    t1 = tf.sqrt(tf.maximum(k, 0.0)) / (_BSTAR * om * y)
    t2 = 500.0 * _NU / (y2 * om)
    t3 = 4.0 * _RHO_INF * _SW2 * k / (CD_kw * y2)

    arg1 = tf.minimum(tf.maximum(t1, t2), t3)
    return tf.math.tanh(tf.pow(arg1, 4)), CD_kw


def sst_F2(k, om, dw):
    """
    Função de mistura F2, do limitador de tensão cisalhante. Eqs. 4.119-4.120.

        Φ2 = max( 2√k/(β*·ω·y) , 500ν/(y²·ω) )      F2 = tanh(Φ2²)

    Mesma fórmula que sst_F2_np (Cell 5) usa para reconstruir ν_t a partir
    do CSV — se as duas divergirem, a verificação (b) da Cell 5 deixa de
    descrever o que o resíduo faz.

    Note o fator 2 no primeiro termo, que F1 não tem, e o expoente 2 no
    lugar de 4. São funções parecidas e diferentes; trocar uma pela outra
    não quebra nada visivelmente.
    """
    y  = tf.maximum(dw, _DW_FLOOR)
    phi2 = tf.maximum(2.0 * tf.sqrt(tf.maximum(k, 0.0)) / (_BSTAR * om * y),
                      500.0 * _NU / (tf.square(y) * om))
    return tf.math.tanh(tf.square(phi2))


def sst_mu_t(k, om, dw, S2):
    """
    Viscosidade turbulenta DINÂMICA [Pa·s], na forma que o Fluent usa.

        μ_t = min( (ρk/ω) / max(1, S·F2/(a1·ω)) , VISC_RATIO_MAX·μ )

    com S = √(S²) e S² = 2 S_ij S_ij. Eq. 4.118 com α* = 1 (low-Re off),
    mais o teto do solver.

    Devolve (μ_t, F2, limited, capped) — as duas máscaras servem de
    diagnóstico e batem, por construção, com os ramos que a verificação (b)
    da Cell 5 mede no dado.
    """
    mu_base = _RHO_INF * k / om                    # ρk/ω, sem limitador
    F2 = sst_F2(k, om, dw)
    if SST_SHEAR_LIMITER:
        # ARMADILHA: √(S²) tem derivada infinita em S²=0 e, mesmo quando o
        # max(1,·) corta o caminho, inf·0 vira NaN no backward. O ε resolve.
        S = tf.sqrt(S2 + S_SQRT_EPS)
        denom = tf.maximum(1.0, S * F2 / (_A1 * om))
        mu_lim = mu_base / denom
        limited = tf.cast(denom > 1.0 + 1e-6, tf.float32)
    else:
        mu_lim = mu_base
        limited = tf.zeros_like(mu_base)
    mu_t = tf.minimum(mu_lim, _MUT_MAX)
    capped = tf.cast(mu_lim > _MUT_MAX, tf.float32)
    return mu_t, F2, limited, capped


def physics_residuals(coll_pts, coll_dw, coll_yw, model):
    """
    Resíduos de URANS + k-ω SST em pontos de colocação normalizados.

    Args:
      coll_pts : (N, 4) float32 — [xn, yn, zn, tn]
      coll_dw  : (N,)   float32 — distância à parede em METROS (Cell 7)
      coll_yw  : (N,)   float32 — limiar da faixa de tratamento de parede
                 DAQUELE ponto, em metros (Cell 7, medido pela Cell 5 em
                 cada parede). Escalar único classificaria 18% do pool
                 errado — ver Cell 7.
      model    : a PINN

      Os dois arrays de parede são SEPARADOS de coll_pts de propósito: são
      coeficientes do resíduo, não entradas da rede.

    Returns: dict com os resíduos JÁ ADIMENSIONALIZADOS, prontos para MSE,
             mais as máscaras que a Cell 11 precisa.

    Chain rule:
      ∂/∂x_phys = (1/DP_ISO) × ∂/∂xn      ∂/∂t_phys = (1/T_DP) × ∂/∂tn
    """
    dw = tf.reshape(coll_dw, [-1])
    yw = tf.reshape(coll_yw, [-1])

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coll_pts)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coll_pts)

            pred = model(coll_pts, training=True)   # (N, 6)
            un  = pred[:, 0:1]
            vn  = pred[:, 1:2]
            wn  = pred[:, 2:3]
            Pn  = pred[:, 3:4]
            kn  = pred[:, 4:5]
            omn = pred[:, 5:6]

            # Desnormaliza para unidades físicas (esquema híbrido)
            u = un * _V_REF
            v = vn * _V_DP + _V_MEAN
            w = wn * _V_REF
            P = Pn * _P_DP + _P_MEAN
            # k_eff = k + K_EPS: derivadas idênticas, positivo por construção
            k_eff = tf.exp(kn  * _LNK_DP + _LNK_MEAN)
            om    = tf.exp(omn * _LNW_DP + _LNW_MEAN)

        scale_grad = tf.stack([_DP_ISO, _DP_ISO, _DP_ISO, _T_DP])

        du  = tape1.gradient(u,     coll_pts) / scale_grad
        dv  = tape1.gradient(v,     coll_pts) / scale_grad
        dw_ = tape1.gradient(w,     coll_pts) / scale_grad
        dP  = tape1.gradient(P,     coll_pts) / scale_grad
        dk  = tape1.gradient(k_eff, coll_pts) / scale_grad
        dom = tape1.gradient(om,    coll_pts) / scale_grad

        # AS FATIAS TÊM QUE FICAR AQUI DENTRO (ver cabeçalho da célula)
        u_x, u_y, u_z, u_t = du[:, 0],  du[:, 1],  du[:, 2],  du[:, 3]
        v_x, v_y, v_z, v_t = dv[:, 0],  dv[:, 1],  dv[:, 2],  dv[:, 3]
        w_x, w_y, w_z, w_t = dw_[:, 0], dw_[:, 1], dw_[:, 2], dw_[:, 3]
        P_x, P_y, P_z      = dP[:, 0],  dP[:, 1],  dP[:, 2]
        k_x, k_y, k_z, k_t = dk[:, 0],  dk[:, 1],  dk[:, 2],  dk[:, 3]
        o_x, o_y, o_z, o_t = dom[:, 0], dom[:, 1], dom[:, 2], dom[:, 3]

        del tape1

        # --- μ_t TAMBÉM AQUI DENTRO ---
        # Ele depende de S, ou seja, das primeiras derivadas acima. Construir
        # aqui é o que permite ∇μ_t sair da própria tape2 logo abaixo. Mover
        # para fora faz tape2.gradient(mu_t, ·) devolver None e o resíduo
        # perde o termo (∂μ_eff/∂x_j)(...) inteiro, sem avisar.
        k_f_ = tf.squeeze(k_eff, -1)
        o_f_ = tf.squeeze(om,    -1)
        S11_, S22_, S33_ = u_x, v_y, w_z
        S12_ = 0.5*(u_y + v_x)
        S13_ = 0.5*(u_z + w_x)
        S23_ = 0.5*(v_z + w_y)
        S2 = 2.0*(tf.square(S11_) + tf.square(S22_) + tf.square(S33_)
                  + 2.0*(tf.square(S12_) + tf.square(S13_) + tf.square(S23_)))
        mu_t, F2, mask_limited, mask_capped = sst_mu_t(k_f_, o_f_, dw, S2)

    # Laplacianos. u_x já está em unidades físicas, então basta dividir uma
    # vez mais por DP_ISO.
    def lap(fx, fy, fz):
        fxx = tape2.gradient(fx, coll_pts)[:, 0] / _DP_ISO
        fyy = tape2.gradient(fy, coll_pts)[:, 1] / _DP_ISO
        fzz = tape2.gradient(fz, coll_pts)[:, 2] / _DP_ISO
        return fxx + fyy + fzz

    lap_u = lap(u_x, u_y, u_z)
    lap_v = lap(v_x, v_y, v_z)
    lap_w = lap(w_x, w_y, w_z)
    lap_k = lap(k_x, k_y, k_z)
    lap_o = lap(o_x, o_y, o_z)

    # ∇μ_t por autodiff — uma chamada só devolve as quatro parciais.
    dmut  = tape2.gradient(mu_t, coll_pts)
    mut_x = dmut[:, 0] / _DP_ISO
    mut_y = dmut[:, 1] / _DP_ISO
    mut_z = dmut[:, 2] / _DP_ISO

    del tape2

    u_f = tf.squeeze(u, -1)
    v_f = tf.squeeze(v, -1)
    w_f = tf.squeeze(w, -1)
    k_f = tf.squeeze(k_eff, -1)
    o_f = tf.squeeze(om, -1)

    mu_ef = _MU + mu_t

    # ─────────────────────────────────────────────────────────
    # CONTINUIDADE
    # ─────────────────────────────────────────────────────────
    r_cont = u_x + v_y + w_z

    # ─────────────────────────────────────────────────────────
    # MOMENTO URANS
    #   div(τ)_i = μ_eff ∇²u_i + (∂μ_eff/∂x_j)(∂u_i/∂x_j + ∂u_j/∂x_i)
    # ─────────────────────────────────────────────────────────
    div_tau_u = (mu_ef*lap_u + mut_x*(2.0*u_x) + mut_y*(u_y + v_x)
                 + mut_z*(u_z + w_x))
    div_tau_v = (mu_ef*lap_v + mut_x*(v_x + u_y) + mut_y*(2.0*v_y)
                 + mut_z*(v_z + w_y))
    div_tau_w = (mu_ef*lap_w + mut_x*(w_x + u_z) + mut_y*(w_y + v_z)
                 + mut_z*(2.0*w_z))

    conv_u = u_t + u_f*u_x + v_f*u_y + w_f*u_z
    conv_v = v_t + u_f*v_x + v_f*v_y + w_f*v_z
    conv_w = w_t + u_f*w_x + v_f*w_y + w_f*w_z

    r_mu = _RHO_INF*conv_u + P_x - div_tau_u
    r_mv = _RHO_INF*conv_v + P_y - div_tau_v
    r_mw = _RHO_INF*conv_w + P_z - div_tau_w

    if MOMENTUM_INCLUDE_K_GRADIENT:
        # Só entra se a coluna `pressure` for a estática. Ver ponto 1.
        c23 = tf.constant(2.0/3.0, tf.float32)
        r_mu += c23 * _RHO_INF * k_x
        r_mv += c23 * _RHO_INF * k_y
        r_mw += c23 * _RHO_INF * k_z

    # ─────────────────────────────────────────────────────────
    # PRODUÇÃO e MISTURA
    # ─────────────────────────────────────────────────────────
    P_k_raw = mu_t * S2
    if SST_PRODUCTION_LIMITER:
        P_k_cap = _PROD_LIM * _BSTAR * _RHO_INF * k_f * o_f
        P_k = tf.minimum(P_k_raw, P_k_cap)
        mask_prod_lim = tf.cast(P_k_raw > P_k_cap, tf.float32)
    else:
        P_k = P_k_raw
        mask_prod_lim = tf.zeros_like(P_k_raw)

    F1, CD_kw = sst_F1(k_f, o_f, dw, k_x, k_y, k_z, o_x, o_y, o_z)
    # Coeficientes misturados. NÃO são diferenciados (ver cabeçalho).
    sig_k = F1*_SK1    + (1.0 - F1)*_SK2
    sig_w = F1*_SW1    + (1.0 - F1)*_SW2
    beta  = F1*_BETA1  + (1.0 - F1)*_BETA2
    gamma = F1*_GAMMA1 + (1.0 - F1)*_GAMMA2

    # ─────────────────────────────────────────────────────────
    # TRANSPORTE DE k
    #   ∇·[(μ + σ_k μ_t)∇k] com σ_k congelado
    # ─────────────────────────────────────────────────────────
    diff_k = ((_MU + sig_k*mu_t)*lap_k
              + sig_k*(mut_x*k_x + mut_y*k_y + mut_z*k_z))
    conv_k = k_t + u_f*k_x + v_f*k_y + w_f*k_z
    r_k = (_RHO_INF*conv_k - P_k + _BSTAR*_RHO_INF*k_f*o_f - diff_k)

    # ─────────────────────────────────────────────────────────
    # TRANSPORTE DE ω
    #   α_∞ρS² vem de (α_∞ρ/ν_t)·P_k com P_k = μ_t S² — o ν_t cancela
    # ─────────────────────────────────────────────────────────
    diff_o = ((_MU + sig_w*mu_t)*lap_o
              + sig_w*(mut_x*o_x + mut_y*o_y + mut_z*o_z))
    cross  = (2.0*(1.0 - F1)*_RHO_INF*_SW2
              * (k_x*o_x + k_y*o_y + k_z*o_z) / o_f)
    conv_o = o_t + u_f*o_x + v_f*o_y + w_f*o_z
    r_om = (_RHO_INF*conv_o - gamma*_RHO_INF*S2
            + beta*_RHO_INF*tf.square(o_f) - diff_o - cross)

    # ─────────────────────────────────────────────────────────
    # LEI DE PAREDE — só usada no modo 'loglaw'
    #   k = √β*·κ_w²·y²·ω²·(u_τ/u*)²
    #   ⟹  ln k − ln(√β*κ_w²) − 2 ln y − 2 ln ω = 2 ln(u_τ/u*)
    # O lado direito vale 0 em equilíbrio e é negativo onde há separação
    # (u_τ → 0 com u* finito). r_wall mede o lado esquerdo; em espaço log é
    # LINEAR nas saídas da rede e já sai adimensional.
    # κ_w = 0,4187 é o das FUNÇÕES DE PAREDE do Fluent, não o 0,41 do
    # modelo — ver Cell 2. Trocar um pelo outro dá viés de +4,3%.
    # ─────────────────────────────────────────────────────────
    y_saf = tf.maximum(dw, _DW_FLOOR)
    r_wall = (tf.math.log(k_f) - tf.math.log(_WALL_LAW_C)
              - 2.0*tf.math.log(y_saf) - 2.0*tf.math.log(o_f))
    # limiar POR PONTO — a faixa da parede mais próxima daquele ponto
    in_wall = tf.cast(dw <= yw, tf.float32)

    # ─────────────────────────────────────────────────────────
    # ADIMENSIONALIZAÇÃO
    # ─────────────────────────────────────────────────────────
    r_cont_n = r_cont / _SCALE_CONT
    r_mu_n   = r_mu   / _SCALE_MOM
    r_mv_n   = r_mv   / _SCALE_MOM
    r_mw_n   = r_mw   / _SCALE_MOM
    if K_RESIDUAL_SCALE == "local":
        # mesmo princípio do ω: divide pelo termo de destruição, com piso
        # para o escoamento livre não virar o alvo (ver Cell 10, config)
        r_k_n = r_k / tf.maximum(_BSTAR*_RHO_INF*k_f*o_f, _XI_K_FLOOR)
    else:
        r_k_n = r_k / _SCALE_K
    if OMEGA_RESIDUAL_SCALE == "local":
        # divide pelo termo de destruição: o resíduo vira RELATIVO (a
        # destruição vale 1) e a faixa do inlet deixa de dominar o MSE
        r_om_n = r_om / tf.maximum(beta*_RHO_INF*tf.square(o_f), 1e-12)
    else:
        r_om_n = r_om / _SCALE_OM

    # ─────────────────────────────────────────────────────────
    # MÁSCARAS
    # ─────────────────────────────────────────────────────────
    keep = 1.0 - in_wall
    if WALL_RESIDUAL_MODE == "mask":
        # onde o solver não resolveu o transporte, não há equação a impor
        r_k_n  = keep * r_k_n
        r_om_n = keep * r_om_n
    elif WALL_RESIDUAL_MODE == "loglaw":
        r_k_n  = keep * r_k_n + in_wall * r_wall
        r_om_n = keep * r_om_n
    elif WALL_RESIDUAL_MODE != "off":
        raise ValueError(f"WALL_RESIDUAL_MODE inválido: {WALL_RESIDUAL_MODE}")

    if MASK_PROD_LIMITED_OMEGA:
        # P_ω indefinido onde o production limiter corta (ver ponto 5)
        r_om_n = (1.0 - mask_prod_lim) * r_om_n

    # ─────────────────────────────────────────────────────────
    # PONTOS ATIVOS POR RESÍDUO
    # Calculado AQUI, junto com as máscaras, e não recalculado na Cell 11.
    # A lógica de quem foi zerado tem que morar num lugar só: se a Cell 11
    # espelhasse este if/elif e as duas divergissem, o MSE passaria a
    # dividir pelo número errado sem nenhum sintoma.
    # active = 1 onde a equação EXISTE naquele ponto; 0 onde foi zerada.
    # ─────────────────────────────────────────────────────────
    _ones = tf.ones_like(r_cont_n)
    if WALL_RESIDUAL_MODE == "mask":
        act_k, act_om = keep, keep
    elif WALL_RESIDUAL_MODE == "loglaw":
        # r_wall SUBSTITUI o transporte: o ponto continua contribuindo
        act_k, act_om = _ones, keep
    else:                                  # "off"
        act_k, act_om = _ones, _ones
    if MASK_PROD_LIMITED_OMEGA:
        act_om = act_om * (1.0 - mask_prod_lim)

    return {
        # resíduos adimensionais
        "cont": r_cont_n, "mom_u": r_mu_n, "mom_v": r_mv_n, "mom_w": r_mw_n,
        "k": r_k_n, "omega": r_om_n,
        # pontos ATIVOS por resíduo — a Cell 11 divide o MSE por estes.
        # Sem isso o MSE divide por N incluindo os zeros da máscara, e tanto
        # o valor quanto o GRADIENTE de k e ω ficam escalados por
        # n_ativos/N, que muda de batch para batch.
        "active_cont": _ones, "active_mom": _ones,
        "active_k": act_k, "active_omega": act_om,
        # máscaras cruas, para diagnóstico e para a Cell 12
        "in_wall":       in_wall,
        "prod_limited":  mask_prod_lim,
        "visc_limited":  mask_limited,
        "visc_capped":   mask_capped,
        # diagnóstico
        "wall_law": r_wall,
        "F1": F1, "F2": F2, "mu_t": mu_t, "S2": S2,
        "P_k": P_k, "CD_kw": CD_kw,
        "mut_over_base": mu_t / (_RHO_INF * k_f / o_f),
    }


# ============================================================
# CELL 11 — LOSS FUNCTIONS
# ============================================================
# Mesma estrutura do laminar: uma data loss segmentada por zona geométrica,
# uma physics loss por equação, e a montagem ponderada. O que muda vem todo
# de duas coisas — a rede passou de 4 para 6 saídas, e o resíduo agora tem
# pontos MASCARADOS.
#
# ==============================================================================
# 1. O ÍNDICE DE COLUNA QUE TRAVAVA O TREINO EM SILÊNCIO
#
# A versão laminar faz:
#       tgt = batch[:, 4:8]      gid = batch[:, 8]
# Com 6 alvos, a coluna 8 deixou de ser o gid e passou a ser kn (ver o
# layout da Cell 5). Rodar o código antigo aqui não daria erro nenhum: ele
# compararia k contra gid, e o `mean_zone` agruparia por um valor contínuo
# que nunca é igual a 0.0, 1.0, ... — todas as zonas dariam zero, a data
# loss viraria 0 e o treino seguiria "convergindo" sem dado.
#
# Por isso nada aqui usa número solto. Tudo vem de COL_UN, COL_OMN, COL_GID,
# COL_DWALL e SLICE_TARGET, definidos na Cell 5.
#
# ==============================================================================
# 2. O MSE TEM QUE DIVIDIR PELOS PONTOS ATIVOS, NÃO POR N
#
# Este é o ponto que justifica a célula inteira ser reescrita e não só
# estendida.
#
# A Cell 10 ZERA o resíduo de k e de ω onde o solver não resolveu a equação
# (faixa de parede) e onde P_ω é indefinido (production limiter). Um
# `reduce_mean` ingênuo sobre o vetor inteiro divide por N incluindo esses
# zeros. Três consequências, todas ruins:
#
#   (a) O VALOR fica diluído. Com 8,8% dos pontos mascarados, a loss de k
#       sai 8,8% menor sem nada ter melhorado.
#   (b) O GRADIENTE fica escalado por n_ativos/N. Como a fração mascarada
#       muda de batch para batch (o sorteio é aleatório e as regiões têm
#       densidades diferentes), o peso EFETIVO do resíduo de k oscila de
#       passo para passo. É ruído multiplicativo no gradiente, e não
#       aparece em lugar nenhum do log.
#   (c) Comparar configurações vira impossível. Rodar WALL_RESIDUAL_MODE
#       'mask' contra 'off' mudaria a loss de k mesmo que a rede fosse
#       idêntica, só pela contagem.
#
# A correção é `masked_mse`: soma dos quadrados dividida pelo número de
# pontos ativos. Com isso a loss de k tem o mesmo significado nos três
# modos e em qualquer batch.
#
# O vetor de pontos ativos vem PRONTO da Cell 10 ("active_k", "active_omega").
# Espelhar aqui o if/elif de WALL_RESIDUAL_MODE seria criar a quinta réplica
# de lógica deste arquivo, e as anteriores todas divergiram.
#
# ==============================================================================
# 3. QUANTOS COMPONENTES E COM QUE PESO
#
# Os pesos por componente de u, v, w e P ficam EXATAMENTE como no Run 4
# (1,0 / 0,3 / 1,0 / 1,0) — foram testados e são o que dá para comparar
# entre as duas versões. k e ω entram com 1,0, que é chute, e é assim que
# está anotado. Os alvos de k e ω são z-score sobre log (Cell 4), então
# estão na mesma escala de v e P; começar em 1,0 é o neutro.
#
# ATENÇÃO à assimetria herdada: u e w são min-max por V_REF e v, P, kn e omn
# são z-score. Os erros não estão na mesma unidade estatística. Isso já era
# verdade no laminar e não estou mexendo — só registrando, porque é candidato
# a explicar comportamento estranho na Cell 12.
#
# ==============================================================================
# 4. OS DADOS DA FAIXA DE PAREDE CONTINUAM SENDO DADOS
#
# O RESÍDUO de k e ω é mascarado ali; a DATA LOSS não. O valor de k e ω
# naquelas células é a resposta do solver — é o alvo legítimo, mesmo sendo
# saída do tratamento de parede e não física resolvida. A Cell 9 já explica
# por que não há hard constraint de k e ω na parede pelo mesmo motivo.
#
# Fica um flag para excluí-los, porque é uma decisão defensável e agora é
# implementável (a Cell 5 pôs d_wall no data_np). Default False.
# ==============================================================================


# ---------- USER CONFIG ----------
# Pesos por COMPONENTE dentro da data loss.
#   u, v, w, P : herdados do Run 4, testados
#   k, omega   : CHUTE INICIAL. Ajustar depois de ver as magnitudes no log.
DATA_COMP_W = {
    "u":     1.0,
    "v":     0.3,
    "w":     1.0,
    "P":     1.0,
    "k":     1.0,
    "omega": 1.0,
}

# Excluir da DATA loss os alvos de k e ω dentro da faixa de tratamento de
# parede. Ver o ponto 4 do cabeçalho. Default False: o dado é a resposta do
# solver e serve de alvo.
MASK_WALL_DATA_KOMEGA = False

# "active" divide o MSE dos resíduos pelo número de pontos ativos.
# "all" reproduz o comportamento ingênuo (divide por N), só para medir a
# diferença — ver o ponto 2 do cabeçalho.
RESIDUAL_MEAN_MODE = "active"      # "active" | "all"

_COMP_ORDER = ["u", "v", "w", "P", "k", "omega"]
_COMP_W_VEC = tf.constant([DATA_COMP_W[c] for c in _COMP_ORDER], tf.float32)

print(f"\n=== Configuração da loss ===")
print(f"  pesos por componente: "
      + "  ".join(f"{c}={DATA_COMP_W[c]:g}" for c in _COMP_ORDER))
print(f"  MSE dos resíduos    : RESIDUAL_MEAN_MODE = '{RESIDUAL_MEAN_MODE}'"
      f"  ({'divide pelos pontos ativos' if RESIDUAL_MEAN_MODE=='active' else 'divide por N — dilui a máscara'})")
print(f"  k, ω na faixa de parede: data loss "
      f"{'EXCLUI' if MASK_WALL_DATA_KOMEGA else 'usa'} esses pontos")


def mse(x):
    """MSE simples. Usado só onde não há máscara."""
    return tf.reduce_mean(tf.square(x))


def masked_mse(x, active):
    """
    Média dos quadrados sobre os pontos ATIVOS.

        sum(active · x²) / max(sum(active), 1)

    Se `active` for tudo 1, é idêntico a mse(x). O max(·,1) evita 0/0 num
    batch que por azar não tenha ponto ativo nenhum — caso raro mas possível
    num batch pequeno todo dentro da faixa de parede.
    """
    n = tf.reduce_sum(active)
    return tf.reduce_sum(active * tf.square(x)) / tf.maximum(n, 1.0)


def tfp_median(x):
    """Mediana sem depender do tensorflow_probability."""
    n = tf.shape(x)[0]
    srt = tf.sort(tf.reshape(x, [-1]))
    return srt[n // 2]


PHYS_KEYS = ["cont", "mom_u", "mom_v", "mom_w", "k", "omega"]


def wall_band_lookup_tf(batch):
    """
    Limiar da faixa de parede para cada ponto de DADO, em TF.

    Os pontos de colocação já vêm com coll_ywall pronto da Cell 7; os de
    dado não, porque a Cell 5 guarda só d_wall. Aqui o limiar é reconstruído
    a partir das coordenadas, com a mesma tabela por parede.

    Só é chamada se MASK_WALL_DATA_KOMEGA — fora disso não custa nada.
    """
    x = batch[:, COL_XN] * DP_ISO + X_MEAN
    y = batch[:, COL_YN] * DP_ISO + Y_MEAN
    z = batch[:, COL_ZN] * DP_ISO + Z_MEAN
    r = tf.sqrt(tf.square(x - XC_CYL) + tf.square(y - YC_CYL))
    d_cyl = tf.sqrt(tf.square(tf.maximum(r - R_CYL, 0.0))
                    + tf.square(tf.maximum(z - H_CYL, 0.0)))
    d = tf.stack([z - Z_MIN, Z_MAX - z, x - X_MIN, X_MAX - x, d_cyl], axis=1)
    idx = tf.argmin(d, axis=1)
    lut = tf.constant([Y_BAND_BY_WALL.get(nm, Y_WALL_FUNCTION)
                       for nm in WALL_CANDIDATE_NAMES], tf.float32)
    return tf.gather(lut, idx)


def data_loss_grouped(batch, model,
                      w_z_cyl, w_z_bc, w_z_boi_c, w_z_boi_t, w_z_free):
    """
    Data loss SEGMENTADA por zona geométrica, com 6 componentes.

    Para cada ponto:
        sq = Σ_c  DATA_COMP_W[c] · (pred_c − alvo_c)²

    Depois a média por zona e a soma ponderada das zonas.

    batch: (N, N_COLS_DATA) — o layout inteiro da Cell 5, não só as 9
           primeiras colunas. Precisa de COL_GID e, se MASK_WALL_DATA_KOMEGA,
           de COL_DWALL.

    Returns: dict com o total ponderado, as 5 losses por zona e as 6 losses
             por componente (estas últimas só diagnóstico — a Cell 15
             imprime, ninguém deriva delas separadamente).
    """
    xyzt = batch[:, SLICE_INPUT]
    tgt  = batch[:, SLICE_TARGET]          # (N, 6) — u,v,w,P,k,ω normalizados
    gid  = batch[:, COL_GID]

    pred = model(xyzt, training=True)      # (N, 6)
    err2 = tf.square(pred - tgt)           # (N, 6)

    # Máscara por componente. Só k e ω podem ser desligados, e só na faixa.
    if MASK_WALL_DATA_KOMEGA:
        dw_b = batch[:, COL_DWALL]
        yw_b = wall_band_lookup_tf(batch)
        out_band = tf.cast(dw_b > yw_b, tf.float32)          # (N,) 1=fora
        # Forma explícita: os seis nomes na ordem de _COMP_ORDER. O
        # [x]*4 + [y,y] que estava aqui é concatenação de LISTA, não soma,
        # e escondia justamente o acoplamento com a ordem das colunas.
        um = tf.ones_like(out_band)
        comp_mask = tf.stack([um, um, um, um, out_band, out_band], axis=1)
        #                     u   v   w   P     k         ω          (N, 6)
    else:
        comp_mask = tf.ones_like(err2)

    w_err2  = err2 * _COMP_W_VEC * comp_mask                  # (N, 6)
    sq_per_pt = tf.reduce_sum(w_err2, axis=1)                 # (N,)
    # Peso ATIVO por ponto: Σ_c w_c·m_c. Vale Σw quando nada é mascarado.
    w_per_pt  = tf.reduce_sum(_COMP_W_VEC * comp_mask, axis=1)
    _W_FULL   = tf.reduce_sum(_COMP_W_VEC)

    def mean_zone(target_gid):
        """
        Média da zona normalizada pelo PESO ATIVO, não pela contagem de
        pontos. É o mesmo argumento do masked_mse (ponto 2 do cabeçalho):
        se k e ω saem de 8% dos pontos, dividir por N dilui a loss daquela
        zona em ~2,7% sem nada ter melhorado.

            Σ_pts Σ_c w_c m_c e_c²
            ----------------------  ×  Σ_c w_c
              Σ_pts Σ_c w_c m_c

        O fator Σ_c w_c no fim faz isto ser IDÊNTICO à média simples por
        ponto quando comp_mask ≡ 1 — ou seja, com MASK_WALL_DATA_KOMEGA
        desligado a magnitude não muda em nada e segue comparável ao Run 4.
        """
        m = tf.cast(tf.equal(gid, target_gid), tf.float32)
        num = tf.reduce_sum(m * sq_per_pt)
        den = tf.reduce_sum(m * w_per_pt)
        return _W_FULL * num / tf.maximum(den, 1e-12)

    l_cyl   = mean_zone(GID_CYL)
    l_bc    = mean_zone(GID_BC)
    l_boi_c = mean_zone(GID_BOI_C)
    l_boi_t = mean_zone(GID_BOI_T)
    l_free  = mean_zone(GID_FREE)

    total = (w_z_cyl   * l_cyl
             + w_z_bc    * l_bc
             + w_z_boi_c * l_boi_c
             + w_z_boi_t * l_boi_t
             + w_z_free  * l_free)

    # Por componente, SEM o peso — para o log mostrar a magnitude crua de
    # cada alvo. É o número que diz se DATA_COMP_W está desequilibrado.
    per_comp = {c: tf.reduce_sum(err2[:, i] * comp_mask[:, i])
                   / tf.maximum(tf.reduce_sum(comp_mask[:, i]), 1.0)
                for i, c in enumerate(_COMP_ORDER)}

    return {"total": total,
            "cyl": l_cyl, "bc": l_bc, "boi_c": l_boi_c,
            "boi_t": l_boi_t, "free": l_free,
            "comp": per_comp}


def physics_loss(coll_pts, coll_dw, coll_yw, model):
    """
    MSE de cada resíduo da Cell 10, normalizado pelos pontos ATIVOS.

    Returns: dict com as 6 losses, as frações ativas (para o log) e os
             diagnósticos que a Cell 15 acompanha.
    """
    R = physics_residuals(coll_pts, coll_dw, coll_yw, model)

    if RESIDUAL_MEAN_MODE == "active":
        _m = masked_mse
    elif RESIDUAL_MEAN_MODE == "all":
        _m = lambda x, a: mse(x)
    else:
        raise ValueError(f"RESIDUAL_MEAN_MODE inválido: {RESIDUAL_MEAN_MODE}")

    out = {
        "cont":  _m(R["cont"],  R["active_cont"]),
        "mom_u": _m(R["mom_u"], R["active_mom"]),
        "mom_v": _m(R["mom_v"], R["active_mom"]),
        "mom_w": _m(R["mom_w"], R["active_mom"]),
        "k":     _m(R["k"],     R["active_k"]),
        "omega": _m(R["omega"], R["active_omega"]),
    }
    # Frações ativas: o log tem que mostrar isso junto da loss, senão um
    # valor baixo de loss_k é ambíguo entre "aprendeu" e "quase tudo
    # mascarado".
    out["frac_active_k"]  = tf.reduce_mean(R["active_k"])
    out["frac_active_om"] = tf.reduce_mean(R["active_omega"])
    out["frac_in_wall"]   = tf.reduce_mean(R["in_wall"])
    out["frac_prod_lim"]  = tf.reduce_mean(R["prod_limited"])
    out["frac_visc_lim"]  = tf.reduce_mean(R["visc_limited"])
    out["frac_visc_cap"]  = tf.reduce_mean(R["visc_capped"])
    out["mu_t_p50"]       = tfp_median(R["mu_t"])
    return out


def assemble_total(data_parts, phys_parts, W):
    """
    Junta tudo num escalar, e devolve a CONTRIBUIÇÃO de cada termo.

    A contribuição (peso × loss) é o que importa para diagnosticar
    desequilíbrio: uma loss grande com peso pequeno não é problema, e uma
    loss pequena que domina o total é. A Cell 12 escolhe os pesos olhando
    justamente estes números, não as losses cruas.

    No laminar essa soma estava inline na Cell 14. Com 6 termos de física
    em vez de 4, mais as 5 zonas, virou coisa demais para ficar solta no
    train step.
    """
    contrib = {"data": W["data"] * data_parts["total"]}
    for k_ in PHYS_KEYS:
        contrib[k_] = W[k_] * phys_parts[k_]
    total = tf.add_n(list(contrib.values()))
    return total, contrib


# ============================================================
# CELL 12 — LOSS WEIGHTS
# ============================================================
# A versão laminar era um dicionário de 10 números e um print. Aqui são 12
# números, e o que muda de verdade não é a contagem: é que agora a célula
# MEDE o efeito dos pesos que ela declara, em vez de só declará-los.
#
# ==============================================================================
# POR QUE MEDIR, E NÃO SÓ ESCOLHER
#
# O peso não é o que importa — o que importa é a CONTRIBUIÇÃO, peso × loss.
# Uma loss grande com peso pequeno não atrapalha ninguém; uma loss pequena
# que domina o total é que decide o treino. Com 6 termos de física e uma
# data loss de 5 zonas, não dá mais para adivinhar qual é qual olhando o
# dicionário.
#
# Então a célula roda um passo de avaliação (sem treinar) e imprime a tabela
# de contribuições. Foi para isso que assemble_total (Cell 11) devolve o
# dict `contrib`.
#
# ==============================================================================
# O QUE FICA IGUAL AO LAMINAR, DE PROPÓSITO
#
# Os cinco pesos de zona (0,5 / 0,5 / 1,5 / 1,0 / 3,0), o W["data"] = 70 e os
# quatro pesos de física herdados (cont, mom_u, mom_v, mom_w = 1,0) são
# EXATAMENTE os do Run 4/6. Esta etapa existe para isolar o efeito de
# acrescentar turbulência; mudar os pesos ao mesmo tempo misturaria duas
# variáveis, e se o resultado piorasse não daria para saber qual foi.
#
# k e ω entram com 1,0. É chute, e está anotado como chute nos dois lugares
# (aqui e na Cell 11). O que NÃO é chute é a tabela impressa abaixo, que diz
# quanto esse 1,0 realmente pesa.
#
# ==============================================================================
# UMA CORREÇÃO DE LEITURA HERDADA
#
# O comentário da Cell 12 laminar dizia "soma efetiva passou de ~1.0 para
# ~1.5", estimando a soma dos pesos de zona ponderada pelas QUOTAS
# (0,08·w_cyl + 0,12·w_bc + ...). Isso não descreve o que o código faz.
#
# data_loss_grouped calcula a MÉDIA DE CADA ZONA e depois soma ponderado:
#     total = Σ_z  w_z · média_da_zona_z
# A quota não entra. Uma zona com 125 pontos pesa o mesmo que uma com 2680,
# para o mesmo w_z — que é precisamente o motivo de segmentar por zona. Logo
# o ganho efetivo da data loss é W["data"] · Σ_z w_z, e Σ_z w_z = 6,5, não
# 1,5. A célula imprime o número certo.
#
# ==============================================================================
# OS FLAGS DA CELL 1 FINALMENTE FAZEM ALGUMA COISA
#
# ENFORCE_K_EQUATION e ENFORCE_OMEGA_EQUATION existem desde a Cell 1 e até
# agora não eram lidos por ninguém. É aqui que eles agem: desligar um zera o
# peso daquele resíduo. Serve para a etapa intermediária que a própria Cell 1
# descreve — treinar URANS com k e ω vindo só do dado, antes de ligar o
# transporte. TURB_MODEL = "laminar" zera os dois.
# ==============================================================================


# ---------- USER CONFIG ----------
W = {
    # --- data loss ---
    "data":        70.0,   # peso global (Run 6)
    "data_cyl":     0.5,   # hard constraint já cuida do no-slip
    "data_bc":      0.5,   # idem
    "data_boi_c":   1.5,   # detalhamento no entorno do cilindro
    "data_boi_t":   1.0,   # esteira distante
    "data_free":    3.0,   # força v ≈ 17 m/s onde |V| é dominante

    # --- física: herdados do laminar ---
    "cont":         1.0,
    "mom_u":        1.0,
    "mom_v":        1.0,
    "mom_w":        1.0,

    # --- física: turbulência (CHUTE, ver a tabela impressa abaixo) ---
    "k":            1.0,
    "omega":        1.0,
}

# Rodar a medição de balanço ao final desta célula. Custa um forward + um
# passo de segunda ordem; desligar só se estiver com pressa.
MEASURE_LOSS_BALANCE = True
BALANCE_N_DATA  = 4096

# 1.024 pontos de colocação ERAM POUCOS, e isso não era inofensivo.
# O resíduo de k tem cauda pesadíssima: no checkpoint 400, 45 pontos de
# 44.535 (0,1%) respondiam por 83% do MSE. Uma amostra de 1.024 pega, em
# média, UM deles — então a tabela de balanço media k ≈ 1,15 enquanto o
# treino, com 24.000 pontos por passo, media 18 a 25. Fator 16 a 22, e a
# tabela dizia que k era 0,06% da loss quando era 14%.
#
# Com 16.384 a amostra cobre ~17 pontos da cauda e a estimativa para de
# mentir. Custa um passo de segunda ordem a mais na Cell 12, uma vez só.
# (A escala local de r_k, ligada na Cell 10, reduz muito essa cauda — mas
#  a amostra pequena continuaria sendo uma má ideia.)
BALANCE_N_COLL  = 16_384

# Só REPORTA os pesos que equalizariam as contribuições de física. Não
# aplica. Ver a ressalva no texto do relatório.
SUGGEST_BALANCED_WEIGHTS = True


# ---------- os flags da Cell 1 agem aqui ----------
if TURB_MODEL != "sst":
    W["k"] = 0.0
    W["omega"] = 0.0
    print(f"  [pesos] TURB_MODEL='{TURB_MODEL}' -> k e ω fora da physics loss")
else:
    if not ENFORCE_K_EQUATION:
        W["k"] = 0.0
        print(f"  [pesos] ENFORCE_K_EQUATION=False -> W['k'] = 0")
    if not ENFORCE_OMEGA_EQUATION:
        W["omega"] = 0.0
        print(f"  [pesos] ENFORCE_OMEGA_EQUATION=False -> W['omega'] = 0")


# ---------- conferência das chaves ----------
# assemble_total (Cell 11) indexa W por estes nomes. Uma chave a menos seria
# KeyError no primeiro passo de treino; uma a mais, um peso que não faz nada
# e alguém acreditando que faz.
_REQUIRED = {"data"} | set(PHYS_KEYS)
_ZONE_KEYS = {"data_cyl", "data_bc", "data_boi_c", "data_boi_t", "data_free"}
_missing = (_REQUIRED | _ZONE_KEYS) - set(W)
_extra   = set(W) - (_REQUIRED | _ZONE_KEYS)
assert not _missing, f"W sem as chaves {_missing}"
if _extra:
    print(f"  [pesos] AVISO: chaves em W que ninguém lê: {sorted(_extra)}")

# Atalho para passar as 5 zonas de uma vez a data_loss_grouped
ZONE_W = dict(w_z_cyl=W["data_cyl"], w_z_bc=W["data_bc"],
              w_z_boi_c=W["data_boi_c"], w_z_boi_t=W["data_boi_t"],
              w_z_free=W["data_free"])

_SUM_ZONE_W = sum(W[k_] for k_ in _ZONE_KEYS)

print(f"\n=== Pesos da loss ===")
for k_ in ["data"] + sorted(_ZONE_KEYS) + PHYS_KEYS:
    print(f"  W[{k_:12s}] = {W[k_]}")
print(f"\n  Σ pesos de zona = {_SUM_ZONE_W:.1f}  ->  ganho efetivo da data")
print(f"  loss = W['data'] · Σw_z = {W['data'] * _SUM_ZONE_W:.1f}")
print(f"  (a QUOTA não entra: data_loss_grouped usa a média de cada zona,")
print(f"   então uma zona pequena pesa igual a uma grande para o mesmo w_z)")


# ================================================================================
# BALANÇO MEDIDO — quem realmente manda no gradiente
# ================================================================================
def report_loss_balance(n_data=BALANCE_N_DATA, n_coll=BALANCE_N_COLL,
                        seed=SEED, verbose=True):
    """
    Avalia data loss e physics loss numa amostra e imprime a contribuição de
    cada termo.

    RESSALVA QUE NÃO PODE SUMIR: isto é medido na rede COMO ELA ESTÁ. Se for
    chamada logo depois da Cell 9, é uma rede com pesos aleatórios, e as
    magnitudes relativas dizem mais sobre a inicialização do que sobre a
    física. Servem para pegar desequilíbrio GROSSEIRO — um termo 1000× acima
    dos outros — não para calibrar na terceira casa.

    O uso honesto é rodar isto de novo depois de algumas épocas (a Cell 15
    pode chamar) e comparar.
    """
    rng_b = np.random.default_rng(seed)
    ib = rng_b.choice(len(train_np), min(n_data, len(train_np)), replace=False)
    ic = rng_b.choice(len(coll_np),  min(n_coll, len(coll_np)),  replace=False)
    b  = tf.constant(train_np[ib], tf.float32)
    cp = tf.constant(coll_np[ic][:, :4], tf.float32)
    cd = tf.constant(coll_dwall[ic].astype(np.float32))
    cy = tf.constant(coll_ywall[ic].astype(np.float32))

    dp = data_loss_grouped(b, model, **ZONE_W)
    pp = physics_loss(cp, cd, cy, model)
    total, contrib = assemble_total(dp, pp, W)

    tot = float(total)
    if verbose:
        print(f"\n{'='*78}")
        print(f"  BALANÇO DA LOSS — {len(ib):,} pontos de dado, "
              f"{len(ic):,} de colocação")
        print(f"{'='*78}")
        print(f"{'termo':<10}{'loss crua':>14}{'peso':>9}{'contribuição':>16}"
              f"{'% do total':>13}")
        print(f"{'-'*78}")
        _raw = {"data": float(dp["total"])}
        for k_ in PHYS_KEYS:
            _raw[k_] = float(pp[k_])
        for k_ in ["data"] + PHYS_KEYS:
            c = float(contrib[k_])
            print(f"{k_:<10}{_raw[k_]:>14.4e}{W[k_]:>9.1f}{c:>16.4e}"
                  f"{100*c/tot:>12.2f}%")
        print(f"{'-'*78}")
        _phys = sum(float(contrib[k_]) for k_ in PHYS_KEYS)
        print(f"{'TOTAL':<10}{'':>14}{'':>9}{tot:>16.4e}{100.0:>12.2f}%")
        print(f"  data  = {100*float(contrib['data'])/tot:5.2f}%    "
              f"física = {100*_phys/tot:5.2f}%")

        print(f"\n  Dentro da física:")
        for k_ in PHYS_KEYS:
            c = float(contrib[k_])
            bar = "#" * max(1, int(round(40 * c / max(_phys, 1e-30))))
            print(f"    {k_:<7}{100*c/max(_phys,1e-30):>7.2f}%  {bar}")

        print(f"\n  Por zona da data loss (loss crua, antes do peso da zona):")
        for nome in ["cyl", "bc", "boi_c", "boi_t", "free"]:
            _w = W["data_" + nome]
            print(f"    {nome:<7} loss={float(dp[nome]):.4e}  peso={_w:.1f}  "
                  f"contrib={_w*float(dp[nome]):.4e}")

        print(f"\n  Por componente (loss crua, SEM o peso de DATA_COMP_W):")
        for c_ in _COMP_ORDER:
            print(f"    {c_:<7} {float(dp['comp'][c_]):.4e}   "
                  f"(peso {DATA_COMP_W[c_]:g})")

        print(f"\n  Frações mascaradas no lote de colocação:")
        print(f"    na faixa de parede       {100*float(pp['frac_in_wall']):6.2f}%")
        print(f"    production limiter ativo {100*float(pp['frac_prod_lim']):6.2f}%")
        print(f"    resíduo de k   ativo em  {100*float(pp['frac_active_k']):6.2f}%")
        print(f"    resíduo de ω   ativo em  {100*float(pp['frac_active_om']):6.2f}%")

        if SUGGEST_BALANCED_WEIGHTS:
            _l = {k_: max(float(pp[k_]), 1e-30) for k_ in PHYS_KEYS}
            _inv = {k_: 1.0 / _l[k_] for k_ in PHYS_KEYS}
            _norm = len(PHYS_KEYS) / sum(_inv.values())
            print(f"\n  SUGESTÃO (não aplicada): pesos que igualariam a")
            print(f"  contribuição dos 6 termos de física, normalizados para")
            print(f"  média 1:")
            print(f"    " + "  ".join(f"{k_}={_inv[k_]*_norm:.3g}"
                                      for k_ in PHYS_KEYS))
            print(f"  NÃO copie isto sem pensar. Medido numa rede não")
            print(f"  treinada, isto calibra o ruído da inicialização: as")
            print(f"  derivadas são quase aleatórias e ∇·V é grande porque a")
            print(f"  soma de três derivadas sem correlação não se cancela.")
            print(f"  Rodar de novo depois de algumas épocas antes de decidir.")
        print(f"{'='*78}")

    return dict(total=tot, contrib={k_: float(v) for k_, v in contrib.items()},
                data_parts={k_: float(dp[k_]) for k_ in
                            ["total", "cyl", "bc", "boi_c", "boi_t", "free"]},
                phys_parts={k_: float(pp[k_]) for k_ in PHYS_KEYS})


LOSS_BALANCE = report_loss_balance() if MEASURE_LOSS_BALANCE else None


# ============================================================
# CELL 13 — PREPARE TENSORS FOR TRAINING
# ============================================================
# A estratégia de embaralhamento em dois estágios fica IDÊNTICA à laminar:
#
#   1) Pré-embaralhar a ORDEM DOS SNAPSHOTS em train_np (custo de RAM zero
#      além da cópia que já existia). Garante que pontos vizinhos na memória
#      venham de passos de tempo distintos e aleatórios.
#   2) tf.data.Dataset.shuffle por cima, para mistura dentro do batch.
#
# Combinados, cada batch tem pontos de muitos instantes (inicialização,
# regime estabelecido, esteira), e o gradiente fica balanceado entre os
# regimes em vez de ver um instante por vez.
#
# ==============================================================================
# O QUE MUDA: 13 COLUNAS, NÃO 9
#
# 1. O ASSERT DE FORMA ESTAVA COM O NÚMERO CRAVADO.
#    A versão laminar termina com
#        assert _test_batch.shape == (BATCH_SIZE, 9)
#    Com o layout novo isso dispara — e dispara pelo motivo certo, mas com a
#    mensagem errada ("BATCH SHAPE ERRADO" sugere bug no dataset, não uma
#    constante desatualizada). Agora confere contra N_COLS_DATA e diz o que
#    cada coluna deveria ser.
#
# 2. O BATCH TEM QUE CARREGAR O data_np INTEIRO.
#    A Cell 11 lê COL_GID para segmentar por zona e, se
#    MASK_WALL_DATA_KOMEGA, COL_DWALL. Passar só as 9 ou 10 primeiras
#    colunas não daria erro: `gid` viraria outra coisa e a data loss iria a
#    zero em silêncio (é a armadilha documentada no cabeçalho da Cell 11).
#    Por isso o dataset carrega as 13 colunas e o sanity check abaixo
#    verifica que as duas colunas críticas chegaram com valores plausíveis.
#
# 3. O POOL DE COLOCAÇÃO AGORA SÃO TRÊS ARRAYS.
#    coll_np, coll_dwall e coll_ywall são paralelos: a linha i dos três
#    descreve o mesmo ponto. A Cell 14 vai sortear índices e fazer gather
#    NOS TRÊS com os mesmos índices. Se algum dia o pool for reamostrado, os
#    três têm que ser trocados JUNTOS — trocar os pontos e deixar as
#    distâncias velhas daria um resíduo silenciosamente errado justamente na
#    região de parede. Daí resample_collocation_pool(), que faz os três de
#    uma vez; não trocar nada "na mão".
#
# 4. RAM. 13 colunas em vez de 9 é 44% mais memória, e o pico é maior que o
#    estado final porque train_np é montado por concatenação (existe a cópia
#    velha e a nova ao mesmo tempo). Os números reais são impressos abaixo —
#    é a diferença entre caber e não caber no nó do cluster.
# ==============================================================================


# ---------- USER CONFIG ----------
BATCH_SIZE = 48_000     # = N_POINTS_PER_SNAP (um snapshot inteiro por batch)
                        # 16k BC+cilindro + 16k detalhamento + 16k freestream

# COMO O ESTÁGIO 1 EMBARALHA.
#
#   "by_snapshot" — só a ORDEM dos snapshots, como no laminar. Os 48.000
#                   pontos de cada instante continuam CONTÍGUOS na memória,
#                   e a mistura dentro do batch fica por conta do shuffle
#                   buffer do tf.data.
#   "full"        — permutação COMPLETA das linhas de treino. Mistura
#                   perfeita, e custa exatamente a mesma cópia que a
#                   concatenação já custava.
#
# O DEFAULT É "by_snapshot", IGUAL AO LAMINAR, DE PROPÓSITO. A medição no
# bloco "MISTURA TEMPORAL" no fim desta célula mostra que ele mistura muito
# menos do que o cabeçalho laminar afirma — mas trocar isto agora mudaria o
# treino junto com a turbulência, e esta etapa existe para isolar UMA
# variável. Fica registrado e desligado; vira um teste separado depois.
TRAIN_SHUFFLE_MODE = "by_snapshot"     # "by_snapshot" (laminar) | "full"

SHUFFLE_BUFFER = 200_000   # valor do laminar; ver a tabela no fim da célula

# Manter o pool de colocação como tf.Variable permite reamostrar durante o
# treino (ver resample_collocation_pool). Com tf.constant seria um pouco
# mais rápido e imutável.
COLLOCATION_RESAMPLEABLE = True


# --- Estágio 1: pré-embaralhar a ordem dos snapshots ---
# rng LOCAL. A versão laminar rebindava o `rng` global da Cell 1 aqui, o que
# é inofensivo hoje mas quebra a reprodutibilidade de qualquer coisa que
# venha depois e use aquele gerador.
_rng13 = np.random.default_rng(seed=SEED)

train_snap_indices_ordered = [i for i in range(n_snaps) if i not in val_snap_indices]
shuffled_snap_order = _rng13.permutation(train_snap_indices_ordered)

# ÍNDICE das linhas de treino, na ordem dos snapshots embaralhados.
_idx13 = np.concatenate([np.arange(snap_offsets[i][1], snap_offsets[i][2])
                         for i in shuffled_snap_order])
if TRAIN_SHUFFLE_MODE == "full":
    # Permutar o índice ANTES de indexar: é uma cópia só, a mesma que a
    # concatenação já fazia, e sai totalmente misturado.
    # NÃO é o default — ver a nota em TRAIN_SHUFFLE_MODE.
    _rng13.shuffle(_idx13)
elif TRAIN_SHUFFLE_MODE != "by_snapshot":
    raise ValueError(f"TRAIN_SHUFFLE_MODE inválido: {TRAIN_SHUFFLE_MODE}")
train_np = data_np[_idx13]
del _idx13

print(f"Snapshots em treino (10 primeiros na ordem nova): "
      f"{shuffled_snap_order[:10].tolist()}")
print(f"train_np após o pré-embaralhamento: {train_np.shape}")
print(f"  RAM: data_np {data_np.nbytes/1e9:.2f} GB + train_np "
      f"{train_np.nbytes/1e9:.2f} GB + val_np {val_np.nbytes/1e9:.2f} GB")
print(f"       = {(data_np.nbytes+train_np.nbytes+val_np.nbytes)/1e9:.2f} GB "
      f"residentes ({N_COLS_DATA} colunas; eram 9 no laminar, "
      f"{9/N_COLS_DATA*100:.0f}% disso)")
print(f"  pico durante a concatenação ≈ "
      f"{(data_np.nbytes+2*train_np.nbytes+val_np.nbytes)/1e9:.2f} GB")
print(f"  (data_np só é usado daqui para a frente por report_loss_balance e")
print(f"   pelas células de diagnóstico; se faltar RAM no nó, é o primeiro")
print(f"   candidato a `del` depois desta célula)")


# --- Estágio 2: tf.data.Dataset com mistura dentro do batch ---
train_tf = (tf.data.Dataset.from_tensor_slices(train_np)
            .shuffle(min(SHUFFLE_BUFFER, len(train_np)),
                     seed=SEED, reshuffle_each_iteration=True)
            .repeat()                          # dataset infinito
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))


# --- Tensores do pool de colocação: TRÊS arrays paralelos ---
_mk = (lambda a, n: tf.Variable(a, dtype=tf.float32, name=n)) \
      if COLLOCATION_RESAMPLEABLE else \
      (lambda a, n: tf.constant(a, dtype=tf.float32, name=n))

coll_tensor       = _mk(coll_np,    "coll_xyzt")    # (N_COLL, 4)
coll_dwall_tensor = _mk(coll_dwall, "coll_dwall")   # (N_COLL,)  metros
coll_ywall_tensor = _mk(coll_ywall, "coll_ywall")   # (N_COLL,)  metros

val_tensor = tf.constant(val_np, dtype=tf.float32)


def resample_collocation_pool(verbose=True):
    """
    Sorteia um pool de colocação novo e troca os TRÊS tensores de uma vez.

    Existe para que ninguém troque um e esqueça os outros. Um pool novo com
    as distâncias antigas não quebra nada visivelmente: F1, F2 e a máscara
    de parede simplesmente passam a descrever outros pontos, e o resíduo
    fica errado onde mais importa — junto à parede — sem nenhum sintoma.

    Só funciona com COLLOCATION_RESAMPLEABLE = True.
    """
    if not COLLOCATION_RESAMPLEABLE:
        raise RuntimeError("pool criado como tf.constant; "
                           "ligar COLLOCATION_RESAMPLEABLE")
    new_np, new_dw, new_yw = sample_collocation()
    coll_tensor.assign(new_np)
    coll_dwall_tensor.assign(new_dw)
    coll_ywall_tensor.assign(new_yw)
    if verbose:
        print(f"  [colocação] pool reamostrado: {len(new_np):,} pontos, "
              f"d_wall ∈ [{new_dw.min():.4f}, {new_dw.max():.1f}] m")
    return new_np, new_dw, new_yw


steps_per_epoch = int(np.ceil(len(train_np) / BATCH_SIZE))
print(f"\nBatch size: {BATCH_SIZE:,}   ->  steps/época (teórico): {steps_per_epoch}")
print(f"Pool de colocação: {N_COLL:,} pontos, "
      f"{N_COLL_PER_STEP:,} sorteados por passo")
print(f"  coll_tensor       {tuple(coll_tensor.shape)}   "
      f"{'tf.Variable (reamostrável)' if COLLOCATION_RESAMPLEABLE else 'tf.constant'}")
print(f"  coll_dwall_tensor {tuple(coll_dwall_tensor.shape)}")
print(f"  coll_ywall_tensor {tuple(coll_ywall_tensor.shape)}")


# ================================================================================
# SANITY CHECK DO BATCH — o que a Cell 11 vai receber
# ================================================================================
_b = next(iter(train_tf))
print(f"\n{'='*78}")
print(f"  PRIMEIRO BATCH")
print(f"{'='*78}")
print(f"  shape {tuple(_b.shape)}   dtype {_b.dtype}")

assert _b.shape == (BATCH_SIZE, N_COLS_DATA), (
    f"batch com {_b.shape[1]} colunas, esperado {N_COLS_DATA}. "
    f"O layout do data_np (Cell 5) e este número têm que andar juntos: "
    f"0-3 xn,yn,zn,tn | 4-9 alvos | {COL_GID} gid | {COL_VLIM} vlim | "
    f"{COL_DWALL} d_wall.")

# --- as colunas que a Cell 11 lê, e que não podem vir erradas em silêncio ---
_gid = _b[:, COL_GID].numpy()
_dw  = _b[:, COL_DWALL].numpy()
_uni = sorted(np.unique(_gid).tolist())
_esperado = sorted([GID_CYL, GID_BC, GID_BOI_C, GID_BOI_T, GID_FREE])
print(f"\n  coluna {COL_GID} (gid): valores distintos = {_uni}")
print(f"           esperado {_esperado}")
assert set(_uni) <= set(_esperado), f"gid com valores fora do esperado: {_uni}"
if set(_uni) != set(_esperado):
    print(f"    AVISO: faltam zonas neste batch — possível com batch pequeno,")
    print(f"    suspeito com {BATCH_SIZE:,} pontos.")
for _g, _n in zip(_esperado, ["cyl", "bc", "boi_c", "boi_t", "free"]):
    _f = float((_gid == _g).mean())
    print(f"    {_n:<7} {100*_f:5.2f}%  ({int((_gid==_g).sum()):,} pontos)")

print(f"\n  coluna {COL_DWALL} (d_wall): [{_dw.min():.4f}, {_dw.max():.2f}] m  "
      f"p50={np.percentile(_dw,50):.2f}")
assert _dw.min() >= 0.0, "d_wall negativo no batch"
assert _dw.max() <= D_WALL_MAX_EST * 1.01, (
    f"d_wall de {_dw.max():.1f} m acima do teto geométrico "
    f"{D_WALL_MAX_EST:.1f} m — a coluna não é d_wall")

print(f"\n  coluna {COL_VLIM} (ν_t no teto): "
      f"{100*float(_b[:, COL_VLIM].numpy().mean()):.2f}% dos pontos")

# --- mistura temporal ---
# ESTE NÚMERO ERA O PROBLEMA. A estratégia laminar embaralhava só a ORDEM
# dos snapshots e deixava os 48.000 pontos de cada instante CONTÍGUOS,
# contando com o shuffle buffer de 200.000 do tf.data para misturar. Não
# mistura: o buffer cobre ~4 snapshots, então um batch de 48.000 sai de
# meia dúzia de instantes. Medido, com 900 snapshots de treino:
#
#     buffer          RAM      instantes distintos no 1º batch
#       200.000     10 MB                6 / 900
#     1.000.000     52 MB               22 / 900
#     5.000.000    260 MB              106 / 900
#    20.000.000   1040 MB              418 / 900
#
# Ou seja: com o valor herdado, "cada batch tem pontos de muitos instantes"
# — que é a justificativa escrita no cabeçalho — era falso. Cada batch via
# essencialmente um instante.
#
# A correção não é subir o buffer. É embaralhar de verdade no estágio 1:
# permutar o ÍNDICE antes de indexar custa a mesma cópia que a concatenação
# já custava, dá mistura perfeita e deixa o buffer livre para o que ele
# serve (variar os batches entre épocas). Medido nesta malha:
#
#     TRAIN_SHUFFLE_MODE = "by_snapshot"   16 de 108 instantes   σ_t = 0,809
#     TRAIN_SHUFFLE_MODE = "full"         108 de 108 instantes   σ_t = 0,998
#
# ESTÁ DESLIGADO POR DECISÃO, NÃO POR DÚVIDA. Ligar mudaria o treino ao
# mesmo tempo que a turbulência, e aí um resultado pior não teria culpado
# identificável. A comparação com a PINN laminar exige o mesmo
# embaralhamento dos dois lados. Depois que o SST estiver estável, trocar
# para "full" é um experimento de uma linha — e, pelos números acima,
# provavelmente vale.
_t = _b[:, COL_TN].numpy()
_n_uni = len(np.unique(_t))
_n_tr  = len(train_snap_indices_ordered)
print(f"\n  Mistura temporal (TRAIN_SHUFFLE_MODE = '{TRAIN_SHUFFLE_MODE}'):")
print(f"    instantes distintos no batch: {_n_uni} de {_n_tr} de treino "
      f"({100*_n_uni/max(_n_tr,1):.1f}%)")
print(f"    t normalizado ∈ [{_t.min():+.4f}, {_t.max():+.4f}]   "
      f"σ = {_t.std():.4f}  (alto = bom mix)")
if _n_uni < 0.5 * min(_n_tr, BATCH_SIZE):
    print(f"    >>> MISTURA POBRE — e ESPERADA com 'by_snapshot'. O gradiente")
    print(f"        de cada passo vem de poucos instantes, e o treino vê um")
    print(f"        regime por vez. Mantido assim para ficar comparável à")
    print(f"        PINN laminar; TRAIN_SHUFFLE_MODE='full' resolve, e é um")
    print(f"        experimento para depois do SST estabilizar.")
elif TRAIN_SHUFFLE_MODE == "full":
    print(f"    (mistura completa — NÃO é a configuração do laminar)")

# --- os alvos chegaram normalizados? ---
print(f"\n  Alvos (colunas {COL_UN}-{COL_OMN}), μ e σ no batch:")
for _i, _n in zip(range(COL_UN, COL_OMN + 1), _COMP_ORDER):
    _c = _b[:, _i].numpy()
    print(f"    {_n:<7} μ={_c.mean():+.4f}  σ={_c.std():.4f}  "
          f"[{_c.min():+.2f}, {_c.max():+.2f}]")
print(f"{'='*78}")
del _b, _gid, _dw, _t


# ================================================================================
# CELL 14 — TRAIN STEP COM SORTEIO ESTRATIFICADO EM 11 REGIÕES
# ================================================================================
# O sorteio do pool e a estrutura do passo ficam IDÊNTICOS ao laminar: um
# `tf.random.uniform` por região, concatena, gather, uma fita, clip do
# gradiente global em 5.0, Adam. Nada disso muda.
#
# ==============================================================================
# 1. O GATHER TEM QUE SER NOS TRÊS ARRAYS
#
# O pool virou três arrays paralelos (Cell 13): coordenadas, d_wall e o
# limiar da faixa de parede. Os MESMOS índices têm que ser aplicados aos
# três. Gatherar só as coordenadas e passar o d_wall inteiro daria erro de
# forma na hora — sorte. Gatherar dois e esquecer o terceiro, não: o resíduo
# usaria o limiar de um ponto qualquer e a máscara de parede ficaria
# aleatória, com a loss caindo normalmente.
#
# ==============================================================================
# 2. OS PESOS VIRARAM tf.Variable
#
# A versão laminar passava os 10 pesos como ARGUMENTOS do train_step, para
# poder mudá-los sem forçar retracing. Com 12 pesos isso vira uma assinatura
# de 15 parâmetros posicionais, que é exatamente o tipo de coisa em que se
# troca dois de lugar.
#
# A mesma propriedade sai de graça com tf.Variable: o grafo captura a
# referência, não o valor, então `W_TF["k"].assign(0.1)` muda o peso no
# passo seguinte sem retracer nada. É isso que permite rampa de peso na
# Cell 15 — subir o peso da física de 0 até o valor final ao longo das
# primeiras épocas — sem reescrever o train_step.
#
# ==============================================================================
# 3. GUARDA DE NaN
#
# Numa rodada de 50.000 épocas no cluster, um único passo com NaN envenena
# TODOS os pesos, e a partir dali a loss vira nan e o resto da rodada é
# tempo de fila jogado fora. O resíduo de ω é o candidato natural: a escala
# local divide por ρβω², e ω vem de uma exponencial.
#
# Então: se a norma do gradiente não for finita, o passo é DESCARTADO em vez
# de aplicado. Isso não conserta a causa e não pode ser silencioso — o
# contador `skipped` volta no dict e a Cell 15 tem que imprimir. Um treino
# que pula 3 passos em 5 milhões é robustez; um que pula 30% é um bug com
# band-aid em cima.
#
# ==============================================================================
# 4. O PASSO DEVOLVE UM DICT
#
# O laminar devolvia uma tupla de 12. Com 6 termos de física, 5 zonas, 6
# componentes e as frações mascaradas, seria uma tupla de ~25 — e a Cell 15
# desempacotando por posição. Dict, e a Cell 15 indexa por nome.
#
# ==============================================================================
# [29/10/2026] RODADA ema1 — A ATUALIZAÇÃO DO EMA ENTRA NO PASSO
#
# Depois de cada apply_gradients, cada peso do model_ema (Cell 9) vai para
#
#     w_ema ← β·w_ema + (1 − β)·w        β = BETA_EMA = 0,999
#
# com w o peso correspondente da rede de treino, JÁ ATUALIZADO pelo Adam.
# É a fórmula do orientador. O laço é sobre os pares
# zip(model_ema.trainable_weights, model.trainable_weights), na mesma ordem
# porque as duas redes saem do mesmo build_pinn (a Cell 9 confere). Dentro
# do tf.function o laço Python é desenrolado no traçado: são 14 assign, um
# por tensor, e cada um soma elemento a elemento.
#
# ONDE: dentro de _apply, o ramo do tf.cond que aplica o passo. Um passo
# descartado pela guarda de NaN (ponto 3) não mexe na rede de treino, e
# portanto também não entra na média. Se entrasse, o EMA andaria (1 − β) em
# direção a um θ que não mudou: inofensivo, mas contaria um passo que não
# existiu. No ramo sem guarda (SKIP_NONFINITE_STEP = False) a atualização vem
# logo depois do apply_gradients, do mesmo jeito.
#
# ORDEM: o assign do EMA lê w DEPOIS do apply_gradients escrever nele. Isso
# não depende da ordem das linhas no Python. O tf.function põe dependência
# de controle entre operações que tocam a mesma variável, na ordem em que
# aparecem, e o XLA respeita essa ordem.
#
# XLA: só assign, multiplicação e soma, nenhuma derivada. Não toca na
# terceira derivada que já quebrou o traçado (ponto 5). As variáveis do EMA
# nascem na Cell 9, antes do traçado, como os slots do Adam aqui embaixo.
# Mesmo assim, conferir o "[train_step] XLA ... traçado OK" na calibração.
#
# CUSTO: 1.318.918 multiplicações e somas por passo, desprezível perto do
# passo de 3ª ordem. O custo real do EMA está na validação dobrada (Cell 15).
# ==============================================================================


# ---------- USER CONFIG ----------
LEARNING_RATE_INIT = 1e-4      # a Cell 15 sobrescreve
GRAD_CLIP_NORM     = 5.0
SKIP_NONFINITE_STEP = True     # ver o ponto 3
USE_XLA            = True      # jit_compile no train_step


# --- Limites do pool como constantes TF (evitam retracing) ---
_REGION_STARTS_TF = [tf.constant(r["start"], tf.int32) for r in COLLOCATION_REGIONS]
_REGION_ENDS_TF   = [tf.constant(r["end"],   tf.int32) for r in COLLOCATION_REGIONS]
_REGION_NSTEPS    = [r["per_step"] for r in COLLOCATION_REGIONS]

# --- Optimizer ---
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_INIT)

# Os slots do Adam (momento e velocidade, um par por variavel) sao criados na
# PRIMEIRA chamada de apply_gradients. Se essa primeira chamada acontecer
# dentro do tf.function, o traçado morre com
#
#     ValueError: tf.function only supports singleton tf.Variables created
#     on the first call.
#
# Isto nao aparecia antes so' porque o traçado de teste do XLA falhava ANTES
# de chegar no apply_gradients (no gradiente, ver ponto 5). Com o XLA
# funcionando, o apply_gradients passa a ser alcançado e o erro aparece.
# Construir os slots aqui, fora do grafo, resolve de uma vez para os dois
# caminhos (com e sem XLA).
optimizer.build(model.trainable_variables)
print(f"  [optimizer] Adam com slots construidos: "
      f"{len(optimizer.variables)} variaveis de estado")

# --- Pesos como Variables (ver ponto 2) ---
W_TF = {k_: tf.Variable(float(v), trainable=False, dtype=tf.float32,
                        name=f"w_{k_}")
        for k_, v in W.items()}
ZONE_W_TF = dict(w_z_cyl=W_TF["data_cyl"], w_z_bc=W_TF["data_bc"],
                 w_z_boi_c=W_TF["data_boi_c"], w_z_boi_t=W_TF["data_boi_t"],
                 w_z_free=W_TF["data_free"])

# Contador de passos descartados. Variable e não int Python porque é
# incrementado de dentro do grafo.
n_skipped_steps = tf.Variable(0, trainable=False, dtype=tf.int64,
                              name="n_skipped_steps")


def _sample_collocation_indices():
    """
    Sorteio estratificado: n_step índices dentro do bloco de cada região.
    Idêntico ao laminar — com reposição dentro do bloco, o que é irrelevante
    com pool 3× maior que o passo.
    """
    pieces = [tf.random.uniform((n,), s, e, dtype=tf.int32)
              for s, e, n in zip(_REGION_STARTS_TF, _REGION_ENDS_TF,
                                 _REGION_NSTEPS)]
    return tf.concat(pieces, axis=0)


def _update_ema():
    """
    θ_ema ← β·θ_ema + (1 − β)·θ, peso a peso (ver o bloco ema1 do cabeçalho).
    Uma função só, chamada nos dois ramos do passo, para a fórmula não
    existir em duas cópias que podem divergir.
    """
    for w_ema, w in zip(model_ema.trainable_weights, model.trainable_weights):
        w_ema.assign(BETA_EMA * w_ema + (1.0 - BETA_EMA) * w)


def _train_step_impl(batch, coll_xyzt, coll_dw, coll_yw):
    """
    Um passo de treino.

    Args:
      batch      : (B, N_COLS_DATA) — o layout INTEIRO da Cell 5
      coll_xyzt  : (N_COLL, 4)   coll_tensor
      coll_dw    : (N_COLL,)     coll_dwall_tensor  [m]
      coll_yw    : (N_COLL,)     coll_ywall_tensor  [m]

    Os três últimos são o pool COMPLETO; o sorteio acontece aqui dentro.
    """
    idx = _sample_collocation_indices()
    # OS TRÊS, COM OS MESMOS ÍNDICES (ver ponto 1)
    cp = tf.gather(coll_xyzt, idx)
    cd = tf.gather(coll_dw,   idx)
    cy = tf.gather(coll_yw,   idx)

    with tf.GradientTape() as tape:
        dp = data_loss_grouped(batch, model, **ZONE_W_TF)
        pp = physics_loss(cp, cd, cy, model)
        total, contrib = assemble_total(dp, pp, W_TF)

    grads = tape.gradient(total, model.trainable_variables)
    grads, gnorm = tf.clip_by_global_norm(grads, GRAD_CLIP_NORM)

    if SKIP_NONFINITE_STEP:
        ok = tf.math.is_finite(gnorm)
        def _apply():
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            _update_ema()      # ema1: só passo aplicado entra na média
            return tf.constant(0, tf.int64)
        def _skip():
            return tf.constant(1, tf.int64)
        n_skipped_steps.assign_add(tf.cond(ok, _apply, _skip))
    else:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        _update_ema()          # ema1

    out = {"total": total, "gnorm": gnorm,
           "data": dp["total"],
           "data_cyl": dp["cyl"], "data_bc": dp["bc"],
           "data_boi_c": dp["boi_c"], "data_boi_t": dp["boi_t"],
           "data_free": dp["free"]}
    for c_ in _COMP_ORDER:
        out["comp_" + c_] = dp["comp"][c_]
    for k_ in PHYS_KEYS:
        out[k_] = pp[k_]
        out["contrib_" + k_] = contrib[k_]
    out["contrib_data"] = contrib["data"]
    out["phys_sum"] = tf.add_n([contrib[k_] for k_ in PHYS_KEYS])
    for k_ in ["frac_active_k", "frac_active_om", "frac_in_wall",
               "frac_prod_lim", "frac_visc_lim", "frac_visc_cap", "mu_t_p50"]:
        out[k_] = pp[k_]
    return out


# Chaves que a Cell 15 vai logar, em ordem de leitura.
LOSS_KEYS_LOG = (["total", "data"] + PHYS_KEYS + ["phys_sum", "gnorm"]
                 + ["data_cyl", "data_bc", "data_boi_c", "data_boi_t", "data_free"]
                 + ["comp_" + c for c in _COMP_ORDER]
                 + ["contrib_data"] + ["contrib_" + k for k in PHYS_KEYS]
                 + ["frac_active_k", "frac_active_om", "frac_in_wall",
                    "frac_prod_lim", "frac_visc_lim", "frac_visc_cap",
                    "mu_t_p50"])


# ==============================================================================
# 5. O XLA COMPILA ESTE PASSO — DEPOIS DE UMA TROCA DE UMA LINHA NA CELL 9
#
# Historico, porque a causa e' contra-intuitiva e vale ficar registrada.
#
# Sintoma: @tf.function(jit_compile=True) falhava no TRACADO, com
#
#     LookupError: gradient registry has no entry for: XlaDynamicUpdateSlice
#
# Bissecao 1 — por termo do residuo, compilando o gradiente de cada um
# isoladamente:
#
#     cont      OK        F1        OK        P_k       OK
#     mom_u     OK        F2        OK        S2        OK
#     omega   FALHA       CD_kw     OK        wall_law  OK
#     k       FALHA
#
# Nao era a mascara, nem o tf.cond da guarda de NaN, nem o tf.sort da
# mediana, nem o clip, nem o apply_gradients, nem a data loss.
#
# Bissecao 2 — dentro do residuo de k, termo a termo:
#
#     lap_u   OK      mu_ef*lap_u  OK    conv_k           OK
#     lap_k FALHA     mu_ef*lap_k  FALHA grad_mut.grad_k  OK
#     lap_om FALHA    F1*lap_u     OK    F1*lap_k         FALHA
#
# Bissecao 3 — o que difere u de k e' a exponencial da desnormalizacao.
# Trocando so' isso, com a mesma rede:
#
#     laplaciano de (saida * cte)      canal 0 OK, canal 4 OK, canal 5 OK
#     laplaciano de exp(saida)         canal 0, 3, 4 e 5: TODOS FALHAM
#     laplaciano de square(saida)      FALHA
#
# Ou seja: nao e' o canal, nem a exponencial em particular — e' QUALQUER
# nao-linearidade depois da saida da rede. Isso derrubou a hipotese antiga
# ("segunda derivada atraves da exponencial") e apontou para a rede.
#
# CAUSA, confirmada:  tf.concat na HardConstraintLayer (Cell 9).
#
# O treino precisa de TERCEIRA derivada — duas espaciais (o laplaciano
# dentro do residuo) e uma nos pesos (o gradiente da loss). A cadeia era:
#
#     forward        Concat
#     1a derivada    grad(Concat)  -> ops Slice
#     2a derivada    grad(Slice)   -> XlaDynamicUpdateSlice
#     3a derivada    grad(XlaDynamicUpdateSlice) -> nao existe -> LookupError
#
# O passo do meio esta' no proprio TensorFlow, em
# tensorflow/python/ops/array_grad.py, _SliceGrad:
#
#     if control_flow_util.GraphOrParentsInXlaContext(...):
#         return gen_xla_ops.xla_dynamic_update_slice(
#             array_ops.zeros_like(input_vec), grad, begin_vec), None, None
#
# Dentro de um contexto XLA ele troca o Pad (diferenciavel) por
# XlaDynamicUpdateSlice (sem gradiente registrado). Fora do XLA a op nunca
# aparece — por isso o codigo laminar, que tambem tem concat mas so' precisa
# de terceira derivada nos mesmos moldes, nunca batia nisso com jit
# desligado, e por isso o repro minimo sem concat compilava.
#
# Com q linear na saida o no' problematico e' podado do grafo e o XLA passa;
# com qualquer nao-linearidade ele fica no caminho e o traçado morre. Por
# isso lap_u passava e lap_k nao: u e' linear na saida da rede, k e' exp().
#
# CORRECAO (Cell 9, HardConstraintLayer.call): trocar
#
#     tf.concat([u,v,w,P,k,om], axis=1)
# por
#     tf.reshape(tf.stack([u,v,w,P,k,om], axis=1), (-1, 6))
#
# Mesmo valor, mesma forma, mesma ordem de colunas — zero efeito numerico.
# A diferenca e' so' a cadeia de gradientes: grad(Pack)=Unpack,
# grad(Unpack)=Pack, grad(Reshape)=Reshape, todas registradas em qualquer
# ordem. Verificacao isolada (mesma rede, jit_compile=True, 3a ordem):
#
#     join=concat take=slice     FALHA        join=stack take=slice    OK
#     join=concat take=unstack   FALHA        join=stack take=unstack  OK
#
# EFEITO COLATERAL que apareceu junto: com o XLA passando, o traçado chega
# no apply_gradients, que ate' entao nunca era alcancado. Os slots do Adam
# eram criados ali dentro e o TF recusa
# ("tf.function only supports singleton tf.Variables created on the first
# call"). Resolvido com optimizer.build(...) antes do traçado — ver o bloco
# do optimizer no inicio desta celula.
#
# A IDENTIDADE DO ESPACO LOG nao e' mais necessaria. Ficava pendente como
# possivel conserto:
#
#     ∇²k = k·( a²|∇kn|² + a·∇²kn )
#
# Medido: bate com o autodiff em 4,6e-07 (mediana do erro relativo), que e'
# ruido de fp32 — e' literalmente a mesma soma. Nao e' mais preciso, nao e'
# melhor condicionado (os dois termos se cancelam nos mesmos pontos), e
# amarraria a Cell 10 a K_NORM_MODE = "logz", ficando errada em silencio
# nos modos "scale" e "z". Sem ganho: nao usar.
#
# O fallback abaixo continua ligado. Se alguem mexer na Cell 9 e o concat
# voltar, o treino nao quebra — cai para tf.function sem XLA e avisa qual
# dos dois esta' valendo. Em GPU o XLA costuma dar 2-3x, entao vale olhar
# essa linha no log antes de deixar um job de dias rodando.
# ==============================================================================
# ==============================================================================
# 6. O QUE MATOU O JOB 6072 — E POR QUE A PROVA DE XLA MUDOU DE FORMA
#
# Com o concat corrigido (ponto 5), o XLA passou a COMPILAR. O job seguinte
# morreu mesmo assim, e a causa é outra:
#
#   stdout:  [train_step] XLA falhou no traçado: Out of memory while trying
#            to allocate 39.06GiB. [Op:__inference__train_step_impl_49079]
#            [train_step] -> tf.function sem XLA
#   stderr:  F bfc_allocator.cc:808] Check failed: c->in_use() && ...
#            @ tensorflow::AssignVariableOp<>::Compute()
#            *** Check failure stack trace: ***   (core dumped)
#
# Leitura na ordem dos fatos:
#   1. a prova EXECUTAVA um passo de treino de verdade, com o lote cheio;
#   2. o XLA fundiu o passo inteiro num programa só e pediu 39,06 GiB;
#   3. a A40 tem 48 GB, mas não 39 GiB livres depois do resto — recusou;
#   4. o except daqui pegou a ResourceExhausted e trocou para o fallback;
#   5. a linha SEGUINTE restaurava os pesos (`_v.assign(_a)`), e essa op
#      abortou o processo: um OOM dentro da compilação XLA deixa o
#      BFCAllocator inconsistente, e o Check do alocador derruba tudo.
#
# Ou seja: o treino nunca começou, e quem derrubou foi a rede de
# segurança, não o erro. Duas consequências, as duas aplicadas:
#
#   (a) a prova virou get_concrete_function — TRAÇA e para. Pega o
#       LookupError do ponto 5, que é erro de traçado, e não aloca nada.
#       Não há OOM possível ali, logo não há alocador para corromper. E
#       de brinde some o salva-restaura de pesos, que era o que abortava.
#
#   (b) memória passou a ser dimensionada ANTES, não descoberta por core
#       dump: COLL_STEP_SCALE (Cell 7) controla quantos pontos de
#       colocação entram por passo, a estimativa é impressa no traçado e
#       o pico real vai para o log nas 3 primeiras épocas
#       (_gpu_mem_report).
#
# Por que 39 GiB: o passo precisa de terceira derivada sobre uma rede de
# 1,32 M de parâmetros em 48.000 pontos, e o SST multiplicou o número de
# varreduras da fita (6 saídas em vez de 4; laplacianos de k e ω; ∇μ_t por
# autodiff; F1, F2, CD_kω, limitadores). O laminar cabia com os mesmos
# 48.000 pontos porque fazia perto de um terço do trabalho. A memória é
# ~linear no nº de pontos — por isso o botão é esse, e não a rede.
#
# NÃO tente sobreviver a um OOM de XLA dentro do processo. Não dá: o
# alocador já está inconsistente quando a exceção chega no Python.
# ==============================================================================
def _build_train_step(use_xla):
    return tf.function(_train_step_impl, jit_compile=use_xla)


train_step = _build_train_step(USE_XLA)
if USE_XLA:
    # ------------------------------------------------------------------
    # PROVA DE TRAÇADO — get_concrete_function, NÃO uma execução
    # ------------------------------------------------------------------
    # A versão anterior EXECUTAVA um passo de verdade para descobrir se o
    # XLA compilava. Isso matou o job 6072 (ver ponto 6 do cabeçalho): o
    # passo executou, o XLA pediu 39 GiB, o alocador recusou, o except
    # daqui pegou a exceção — e o processo morreu no `_v.assign(_a)` da
    # linha seguinte, com
    #
    #     F bfc_allocator.cc:808] Check failed: c->in_use() && ...
    #     *** Check failure stack trace: ***   (core dumped)
    #
    # Um OOM DENTRO da compilação XLA deixa o BFCAllocator inconsistente.
    # A próxima op de GPU aborta o processo inteiro, e o fallback nunca
    # chega a rodar. Ou seja: a rede de segurança derrubava o avião.
    #
    # get_concrete_function CONSTRÓI o grafo e para aí. É exatamente onde
    # o LookupError do XlaDynamicUpdateSlice aparecia (é um erro de
    # traçado, em Python, no registro de gradientes), então a prova
    # continua valendo para aquilo que ela existe para pegar. E como nada
    # executa:
    #   - não aloca memória de GPU     -> não há OOM para corromper nada;
    #   - não mexe nos pesos           -> não precisa salvar/restaurar;
    #   - não mexe no estado do Adam   -> não precisa zerar nada.
    #
    # O que ela NÃO pega mais é falta de memória. Isso é de propósito:
    # não existe jeito seguro de sobreviver a um OOM de XLA no mesmo
    # processo. Memória se dimensiona ANTES, com COLL_STEP_SCALE (Cell 7)
    # — a estimativa está impressa logo abaixo.
    #
    # Traça com as formas REAIS (lote cheio), então o primeiro passo de
    # treino reaproveita este traçado em vez de refazer outro.
    try:
        _ = train_step.get_concrete_function(
                tf.zeros((BATCH_SIZE, N_COLS_DATA), tf.float32),
                coll_tensor, coll_dwall_tensor, coll_ywall_tensor)
        print(f"  [train_step] XLA (jit_compile=True): grafo traçado OK")
    except Exception as _exc:
        _m = [l.strip() for l in str(_exc).splitlines() if "Error" in l]
        print(f"  [train_step] XLA falhou no traçado: "
              f"{_m[-1][:100] if _m else str(_exc)[:100]}")
        print(f"  [train_step] -> tf.function sem XLA (ver ponto 5 do cabeçalho)")
        train_step = _build_train_step(False)
        USE_XLA = False
    n_skipped_steps.assign(0)


def _gpu_mem_report(tag=""):
    """
    Pico de memória de GPU até agora, em GiB. Devolve string vazia se não
    houver GPU ou se a versão do TF não expuser o contador.

    Serve para transformar "coube ou não coube" em número: é o que faltava
    no job 6072, onde a primeira notícia de memória foi o core dump.
    """
    try:
        _g = tf.config.list_physical_devices("GPU")
        if not _g:
            return ""
        _mi = tf.config.experimental.get_memory_info("GPU:0")
        _cur = _mi.get("current", 0) / 1024**3
        _pk  = _mi.get("peak", 0) / 1024**3
        return (f"  [memoria GPU{(' ' + tag) if tag else ''}] "
                f"em uso {_cur:.2f} GiB | pico {_pk:.2f} GiB")
    except Exception:
        return ""


# Estimativa, para não descobrir por core dump. A medida é do job 6072:
# 48.000 pontos de colocação sob XLA pediram 39,06 GiB e não couberam numa
# A40 de 48 GB. A memória do grafo de 3a ordem escala ~linear no numero de
# pontos, entao a regra de tres serve de ordem de grandeza.
_MEM_MEDIDO_GIB, _MEM_MEDIDO_PTS = 39.06, 48_000
_mem_est = _MEM_MEDIDO_GIB * N_COLL_PER_STEP / _MEM_MEDIDO_PTS
print(f"  [memoria] colocação {N_COLL_PER_STEP:,}/passo  ->  "
      f"~{_mem_est:.1f} GiB estimados sob XLA "
      f"(medido: {_MEM_MEDIDO_GIB:.2f} GiB com {_MEM_MEDIDO_PTS:,}, job 6072)")
if _mem_est > 30.0:
    print(f"  [memoria] >>> ACIMA DE 30 GiB. Numa A40 (48 GB) isto tende a "
          f"estourar. Baixar COLL_STEP_SCALE na Cell 7.")


print(f"\n=== Train step ===")
print(f"  batch de dado     : {BATCH_SIZE:,} × {N_COLS_DATA} colunas")
print(f"  colocação/passo   : {N_COLL_PER_STEP:,} de {N_COLL:,} "
      f"({len(COLLOCATION_REGIONS)} regiões)")
print(f"  gather nos três   : coll_tensor, coll_dwall_tensor, coll_ywall_tensor")
print(f"  clip global norm  : {GRAD_CLIP_NORM}")
print(f"  passo não-finito  : {'DESCARTADO (contado em n_skipped_steps)' if SKIP_NONFINITE_STEP else 'aplicado (comportamento laminar)'}")
print(f"  EMA               : β = {BETA_EMA}, depois de cada passo aplicado "
      f"({len(model_ema.trainable_weights)} tensores)"
      + (", passo descartado fica fora" if SKIP_NONFINITE_STEP else ""))
print(f"  XLA               : {USE_XLA}")
print(f"  pesos             : tf.Variable — mudar com W_TF['k'].assign(...)")
print(f"                      sem retracing (rampa na Cell 15)")
print(f"  learning rate     : {LEARNING_RATE_INIT:.1e} (a Cell 15 sobrescreve)")
print(f"  retorno           : dict com {len(LOSS_KEYS_LOG)} escalares")


# ============================================================
# CELL 15 — TRAINING LOOP
# ============================================================
# A estrutura fica a do laminar: DEBUG_MODE escolhe o tamanho da rodada, LR
# constante, acumuladores como tensores TF dentro da época (sem sync
# GPU→CPU por passo), CSV por época, console espaçado, checkpoint por
# SAVE_EVERY e best-checkpoint por val.
#
# ==============================================================================
# 1. A RÉPLICA DA FÓRMULA DA LOSS NA VALIDAÇÃO
#
# O laminar tem `evaluate_val_alinhado`, que promete usar "a MESMA fórmula
# do treino" e a REESCREVE à mão:
#
#     sq_per_pt = 1.0*err[:,0]² + 0.3*err[:,1]² + 1.0*err[:,2]² + 1.0*err[:,3]²
#
# Quatro componentes e quatro pesos cravados. Com 6 saídas essa réplica
# passa a medir outra coisa que não a loss de treino, e — pior — continua
# rodando e imprimindo um número plausível. É a quinta réplica de lógica
# que este arquivo produz, e todas as anteriores divergiram.
#
# Aqui a validação acumula SOMAS por zona (numerador e denominador do peso
# ativo) usando o mesmo _COMP_W_VEC da Cell 11, e o resultado é CONFERIDO
# contra data_loss_grouped num chunk logo na primeira chamada. Se algum dia
# as duas divergirem, quebra na hora.
#
# ==============================================================================
# 2. DOIS NÚMEROS DE VALIDAÇÃO, E ISSO É DE PROPÓSITO
#
# `val_mse6` é o MSE sobre as 6 saídas. `val_mse4` é sobre u, v, w e P
# apenas — é o número DIRETAMENTE comparável com o log da PINN laminar,
# que só tinha essas quatro. Sem ele, a comparação entre as duas versões
# some, e é ela que justifica esta etapa existir.
#
# ==============================================================================
# 3. RAMPA DE PESO DA FÍSICA — disponível, DESLIGADA
#
# A Cell 12 mediu que, com peso 1 em tudo, o resíduo de k sozinho é ~87% do
# gradiente físico numa rede não treinada. A prática usual em PINN é subir
# o peso da física de 0 até o valor final ao longo das primeiras épocas,
# deixando a rede ajustar o dado primeiro. Com W_TF (Cell 14) isso é um
# assign por época, sem retracing.
#
# Fica DESLIGADO por padrão pelo mesmo motivo de sempre: o laminar não
# tinha rampa, e ligar mudaria o treino junto com a turbulência.
#
# ==============================================================================
# 4. O QUE PASSOU A SER OBRIGATÓRIO NO LOG
#
#   n_skipped_steps   passos descartados pela guarda de NaN (Cell 14). Se
#                     isso subir, o treino está mascarando um problema.
#   frac_active_k/ω   fração de pontos onde o resíduo existe. Uma loss_k
#                     baixa com frac_active baixa não é aprendizado.
#   frac_in_wall      quanto do lote cai na faixa de parede.
#   frac_visc_cap     quanto bateu no teto de ν_t.
#   contrib_*         peso × loss, que é o que manda no gradiente.
#
# São as grandezas que, faltando no log, transformam um treino ruim em
# mistério. O CSV leva todas.
#
# ==============================================================================
# [29/10/2026] RODADA ema1 — DUAS REDES NO LAÇO
#
# O laço de treino não muda: o EMA é atualizado dentro do train_step
# (Cell 14). O que muda é o que se faz com as duas redes fora dele.
#
# VALIDAÇÃO DAS DUAS. evaluate_val_mse, _val_zone_sums e evaluate_val_aligned
# usavam o `model` global. Agora recebem a rede como argumento OBRIGATÓRIO:
# um default para `model` faria a validação "do EMA" medir, em silêncio, a
# rede de treino. As colunas da rede de treino ficam com o nome de sempre
# (val_mse6, val_aligned, val_<zona>), comparáveis com o CSV da siren1, e as
# do EMA ganham "_ema" no fim. As novas vão para o FIM do CSV: as antigas
# continuam na mesma posição, e quem lê por nome (plot_loss_history da
# inferência) não quebra, só não desenha as novas até ser mudado.
#
# val_time: segundos gastos na validação naquela época (0 nas épocas sem
# validação). Validar as duas redes dobra esse custo, e é com essa coluna
# que se decide o VAL_EVERY do "full" depois da calibração. Ela não entra
# em `elapsed`, que é medido ANTES da validação, como sempre foi.
#
# DOIS BEST, cada um pela val_aligned da própria rede:
#   pinn_<tag>_ema_best.weights.h5    pesos do EMA, menor val_aligned_ema.
#                                     É o best OFICIAL, porque a inferência
#                                     roda no EMA. best_val_loss/best_epoch
#                                     passam a ser dele (a Cell 16 os grava).
#   pinn_<tag>_siren_best.weights.h5  pesos da rede de treino, menor
#                                     val_aligned. Serve para comparar best
#                                     com best contra a siren1.
#
# CHECKPOINT A CADA SAVE_EVERY: as duas redes, pinn_<tag>_siren_NNNNNN e
# pinn_<tag>_ema_NNNNNN (nomes na Cell 3). Gravar o segundo arquivo custa
# menos de um segundo a cada ~1 h de treino; o custo é de disco (~5,3 MB por
# arquivo, ~5,3 GB no full).
#
# NA CALIBRAÇÃO, val_aligned_ema TEM QUE SAIR PIOR que val_aligned: na época
# 20 o EMA ainda carrega ~13% da inicialização aleatória (β^2000, Cell 9).
# Não é o EMA sendo pior. O mesmo vale para o best do EMA nas primeiras
# dezenas de épocas do full.
#
# ==============================================================================
# [09/11/2026] RETOMADA NA ÉPOCA 18.100 (ver o bloco RETOMADA da Cell 3)
#
# Ligada por WARM_START_FROM_EMA na Cell 3. Quatro coisas mudam, todas só
# numa retomada (warm start antigo e treino do zero seguem como antes):
#   1. N_EPOCHS: o laço roda só o que falta até o total do modo (50.000 −
#      18.100 = 31.900 no full), e aborta se a época de retomada não for
#      múltipla de SAVE_EVERY e VAL_EVERY (a grade dos checkpoints).
#   2. EMA: carregado do arquivo dele, não recopiado da rede de treino, com
#      a conferência de que mudou e de que é diferente da rede de treino.
#   3. CSV: as linhas depois da época de retomada saem (cópia de segurança
#      do original ao lado), e o monitoramento ganha uma linha de separação.
#      O treino emenda no mesmo arquivo: a inferência vê 1 → 50.000.
#   4. BEST: retomado do CSV original, para os _best só serem trocados por
#      algo melhor que o de antes da queda.
# O estado do Adam continua sendo zerado (não está no .h5).
# ==============================================================================

import csv
import shutil      # [09/11/2026] cópia de segurança do CSV na retomada

# [19/10/2026] siren1: a PRIMEIRA submissão é "calibration" (20 épocas).
# A SIREN muda as derivadas que a física usa (até a 3ª ordem passa por
# sin(ω0·…)), e nada do que foi medido na sst3 vale automaticamente aqui.
# Antes do "full", ler no .out da calibração:
#   - [train_step] XLA traçado OK (Cell 14) e o pico de memória das ép. 1-3;
#   - o tempo por época — é ele que dimensiona o --time do .slurm
#     (na sst3: 42,8 s/época, 50.000 épocas ≈ 24,8 dias);
#   - o balanço da Cell 12 (data × física, e o peso de cada resíduo);
#   - |g|, 'teto ν_t', μ_t p50 e passos descartados nas linhas do log.
# Depois, voltar para "full" e resubmeter.
#
# [29/10/2026] ema1: de novo "calibration" primeiro. Ler, além da lista acima:
#   - o "[train_step] XLA ... traçado OK" com a atualização do EMA dentro;
#   - pico de memória (siren1: 20,25 GiB) e s/época (siren1: ~35,9 s);
#   - val_time nas épocas 1, 5, 10, 15 e 20 — o custo da validação dobrada;
#   - a conferência |Δ| = 0 das DUAS redes na Cell 16;
#   - val_al_ema PIOR que val_al (esperado com 20 épocas, ver o cabeçalho).
#
# [09/11/2026] ema1, RETOMADA do full na época 18.100 (Cell 3): "full". O
# N_EPOCHS do modo passa a ser o TOTAL da rodada, e o laço roda só o que
# falta (ver o bloco logo abaixo da tabela de modos).
DEBUG_MODE = "full"   # "fast" | "calibration" | "intermediate" | "medium" | "full"

LR_CONSTANT = 3e-5    # LR constante (decisão do orientador, mantida)

if DEBUG_MODE == "fast":
    N_EPOCHS, LOG_EVERY, SAVE_EVERY, VAL_EVERY = 5, 1, 999_999, 1
    STEPS_PER_EPOCH, CONV_ENABLED = 100, False
elif DEBUG_MODE == "calibration":
    N_EPOCHS, LOG_EVERY, SAVE_EVERY, VAL_EVERY = 20, 1, 999_999, 5
    STEPS_PER_EPOCH, CONV_ENABLED = 100, False
elif DEBUG_MODE == "intermediate":
    N_EPOCHS, LOG_EVERY, SAVE_EVERY, VAL_EVERY = 500, 25, 50, 10
    STEPS_PER_EPOCH, CONV_ENABLED = 100, True
    CONV_PATIENCE, CONV_MIN_PCT, CONV_WARMUP = 100, 1.0, 200
elif DEBUG_MODE == "medium":
    N_EPOCHS, LOG_EVERY, SAVE_EVERY, VAL_EVERY = 1000, 25, 50, 10
    STEPS_PER_EPOCH, CONV_ENABLED = 100, True
    CONV_PATIENCE, CONV_MIN_PCT, CONV_WARMUP = 100, 1.0, 200
elif DEBUG_MODE == "full":
    N_EPOCHS, LOG_EVERY, SAVE_EVERY, VAL_EVERY = 50_000, 50, 100, 50
    STEPS_PER_EPOCH, CONV_ENABLED = 100, True
    CONV_PATIENCE, CONV_MIN_PCT, CONV_WARMUP = 200, 0.5, 500

# [09/11/2026] RETOMADA (WARM_START_FROM_EMA na Cell 3). O N_EPOCHS de cada
# modo é o TOTAL da rodada (50.000 no full). Numa retomada o laço roda só o
# que falta, para a rodada terminar na mesma época de sempre, e não 50.000
# épocas DEPOIS da retomada (68.100). Num warm start antigo (sem
# WARM_START_FROM_EMA) nada muda: N_EPOCHS épocas novas, como antes.
#
# O laço testa VAL_EVERY e SAVE_EVERY na época LOCAL (1, 2, ...). Com a
# época de retomada múltipla dos dois, local e global caem nos mesmos
# múltiplos (18.150, 18.200, ...). Se não for múltipla, os checkpoints
# sairiam fora da grade de 100 em 100: aborta aqui, em 2 segundos.
N_EPOCHS_TOTAL = N_EPOCHS
if WARM_START_FROM_EMA:
    N_EPOCHS = N_EPOCHS_TOTAL - int(WARM_START_EPOCH)
    if N_EPOCHS <= 0:
        raise ValueError(f"retomada na época {WARM_START_EPOCH}, mas o modo "
                         f"{DEBUG_MODE!r} tem {N_EPOCHS_TOTAL} épocas no total: "
                         f"não falta nada para treinar")
    for _nm, _v in (("SAVE_EVERY", SAVE_EVERY), ("VAL_EVERY", VAL_EVERY)):
        if _v < N_EPOCHS_TOTAL and int(WARM_START_EPOCH) % _v:
            raise ValueError(f"WARM_START_EPOCH = {WARM_START_EPOCH} não é "
                             f"múltiplo de {_nm} = {_v}: as épocas de "
                             f"{_nm[:-6].lower()} sairiam da grade da rodada")
    print(f"[retomada] {DEBUG_MODE}: {N_EPOCHS_TOTAL:,} épocas no total, "
          f"{int(WARM_START_EPOCH):,} feitas, {N_EPOCHS:,} a rodar")

RESAMPLE_COLL_EVERY = 999_999    # praticamente desativado, como no laminar
BALANCE_EVERY       = 500        # report_loss_balance() a cada N épocas (0 = nunca)

# --- rampa de peso da física (ver ponto 3) ---
PHYS_RAMP_ENABLED = False
PHYS_RAMP_EPOCHS  = 200          # sobe linearmente de PHYS_RAMP_START até 1
PHYS_RAMP_START   = 0.0
_W_PHYS_FINAL     = {k_: float(W[k_]) for k_ in PHYS_KEYS}

VAL_CHUNK = 200_000

# ------------------------------------------------------------------------------
# RUN_TAG E WARM START
# ------------------------------------------------------------------------------
# A rodada "sst" (job 6083) treinou 400+ épocas com a escala GLOBAL de r_k e
# com as BCs duras de k e ω no inlet. As duas coisas mudaram. Continuar
# escrevendo no mesmo training_log_sst.csv juntaria, numa coluna só, duas
# grandezas que não são a mesma: r_k/SCALE_K e r_k/max(β*ρkω, ξSCALE_K). O
# gráfico mostraria um degrau que pareceria resultado e seria mudança de
# régua. Por isso: RUN_TAG novo, CSV novo, pasta de checkpoint nova.
#
# O WARM START aproveita os pesos da rodada anterior, e isso é seguro aqui
# por um motivo específico, não por costume:
#
#   - nenhuma das mudanças cria ou remove parâmetro (1.318.918 nos dois);
#   - desligar a BC dura do inlet altera a SAÍDA só onde D_inlet < 1. A
#     3 m do inlet D_inlet = tanh(135·3/101,32) = 0,99933, ou seja, o termo
#     da constante vale 6,7e-04 e muda ω em 0,5%. Fora da faixa de 3 m a
#     rede continua calculando exatamente o que calculava;
#   - mudar a escala de r_k não toca na rede, só na loss.
#
# Logo: o que precisa reaprender é a faixa do inlet. O resto do domínio —
# onde u, v, P e ω já estão em R² de 0,84 a 0,96 — parte de onde parou.
#
# O estado do Adam NÃO é restaurado (não é salvo no .h5). Recomeça zerado,
# o que é o normal num warm start e é conservador: os primeiros passos são
# menores, não maiores.
# RUN_TAG, WARM_START_FROM e WARM_START_EPOCH estão na Cell 3, junto do
# SNAPSHOT_DIR, e NÃO aqui. O motivo é prático: um caminho errado tem que
# derrubar o job em 2 segundos, não depois dos ~30 minutos de leitura dos
# 999 CSVs. A Cell 3 confere a existência do arquivo na hora.
#
# [18/10/2026] RODADA "sst3" — DADO NOVO, COLD START
#
# O que está escrito acima sobre o warm start valia para sst -> sst2, em
# que só a loss mudava e a rede era lida com a mesma convenção. Para a sst3
# NÃO vale: o dado é outro (job 6125, teto de ν_t em 1e7), a Cell 4 remede
# LNK/LNW nele, e a saída 'kn' de um peso da sst2 deixaria de significar o
# mesmo k. WARM_START_FROM = None na Cell 3; o bloco de warm start abaixo
# fica no código, desligado.
#
# A comparação sst2 × sst3 é ESTATÍSTICA (as rodadas de CFD divergem depois
# de t ≈ 40 s), e o CSV da sst3 é novo de qualquer jeito.
#
# [19/10/2026] RODADA "siren1" — MESMO DADO DA sst3, COLD START
#
# Nada do laço muda. O warm start continua desligado, agora porque a
# arquitetura é outra (Cell 9): os pesos da sst3 não têm onde entrar. O CSV
# e o monitoramento levam a tag nova e vão para a pasta da siren1 (Cell 3).
# Como o dado é o mesmo da sst3, as colunas dos dois CSVs são comparáveis
# época a época, com média móvel de 9 épocas.
#
# DESTINO DOS ARQUIVOS. Até a sst2 os três caminhos eram definidos AQUI e
# eram RELATIVOS ("./pinn_<tag>_checkpoints", "training_log_<tag>.csv"),
# ou seja, caíam onde o sbatch tivesse sido chamado. Agora:
#   CHECKPOINT_DIR  vem da Cell 3 (saida/pinn_<tag>_checkpoints/), absoluto,
#                   já criado e com permissão de escrita conferida. NÃO é
#                   redefinido aqui — redefinir era exatamente o que fazia
#                   os checkpoints voltarem para o cwd;
#   CSV_LOG_PATH e MONITOR_PATH vão para HISTORICO_DIR (saida/historico/).
# Os nomes dos arquivos não mudam.
CSV_LOG_PATH   = os.path.join(HISTORICO_DIR, f"training_log_{RUN_TAG}.csv")
MONITOR_PATH   = os.path.join(HISTORICO_DIR, f"monitoramento_pinn_{RUN_TAG}.txt")

optimizer.learning_rate.assign(LR_CONSTANT)
# Redundante com a Cell 3 de propósito: se alguém rodar esta célula
# isolada num notebook, a pasta continua existindo.
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
assert os.path.isabs(CHECKPOINT_DIR), \
    "CHECKPOINT_DIR relativo: ele vem da Cell 3 e tem que ser absoluto"

if WARM_START_FROM:
    # a existência do arquivo já foi conferida na Cell 3
    # Conferir que os pesos realmente entraram: load_weights silencioso com
    # arquitetura compatível mas camadas renomeadas devolveria a rede
    # inalterada, e o treino recomeçaria do zero sem avisar.
    _w_antes = model.get_weights()[0].copy()
    model.load_weights(WARM_START_FROM)
    _w_depois = model.get_weights()[0]
    _mudou = float(np.abs(_w_depois - _w_antes).max())
    print(f"\n[warm start] {WARM_START_FROM}")
    print(f"[warm start] pesos carregados: |Δ| máx na 1ª camada = "
          f"{_mudou:.4e}  ({'OK' if _mudou > 0 else 'NADA MUDOU — SUSPEITO'})")
    assert _mudou > 0, ("load_weights não alterou nenhum peso. O arquivo "
                        "existe mas não bateu com a arquitetura.")
    # Zera o estado do Adam para não carregar momento de uma loss que não
    # existe mais (o r_k mudou de escala).
    for _ov in optimizer.variables:
        _ov.assign(tf.zeros_like(_ov))
    optimizer.learning_rate.assign(LR_CONSTANT)
    if WARM_START_FROM_EMA:
        # [09/11/2026] RETOMADA: o EMA vem do PRÓPRIO arquivo, da mesma época
        # (a Cell 3 conferiu nomes e épocas). Recopiar da rede de treino
        # jogaria fora a média acumulada até aqui — que é o que a rodada mede.
        # O estado do Adam acima é zerado mesmo assim: ele não está no .h5.
        _we_antes = model_ema.get_weights()[0].copy()
        model_ema.load_weights(WARM_START_FROM_EMA)
        _we_mudou = float(np.abs(model_ema.get_weights()[0] - _we_antes).max())
        _d_redes = max(float(np.abs(a_ - b_).max()) for a_, b_ in
                       zip(model.get_weights(), model_ema.get_weights()))
        print(f"[retomada] EMA: {WARM_START_FROM_EMA}")
        print(f"[retomada] EMA carregado: |Δ| máx na 1ª camada = "
              f"{_we_mudou:.4e} | EMA × rede de treino: |Δw| máx = "
              f"{_d_redes:.4e} (tem que ser > 0)")
        assert _we_mudou > 0, ("load_weights não alterou o EMA: o arquivo não "
                               "bateu com a arquitetura.")
        assert _d_redes > 0, ("o EMA carregado é IGUAL à rede de treino: os "
                              "dois arquivos têm os mesmos pesos.")
        del _we_antes
    else:
        # ema1: o model_ema foi copiado da rede ALEATÓRIA na Cell 9. Sem recopiar
        # aqui, ele começaria longe da rede de treino e levaria centenas de
        # épocas para esquecer aquela inicialização. Recopiar mantém o
        # "inicializados iguais" também num warm start.
        model_ema.set_weights(model.get_weights())
        print(f"[warm start] model_ema recopiado da rede de treino")
    del _w_antes, _w_depois


# ================================================================================
# Validação
# ================================================================================
def evaluate_val_mse(val_t, net, chunk_size=VAL_CHUNK):
    """
    MSE neutro (sem pesos de componente nem de zona) sobre o conjunto de
    validação, em chunks para não estourar a memória da GPU.

    net: a rede avaliada (model ou model_ema). Obrigatório, sem default —
    ver o bloco ema1 do cabeçalho.

    Devolve (mse6, mse4):
      mse6 — sobre as 6 saídas
      mse4 — sobre u, v, w, P apenas; é o número comparável com o log da
             PINN laminar (ver ponto 2 do cabeçalho)
    """
    n_total = int(val_t.shape[0])
    s6 = 0.0; s4 = 0.0; n = 0
    for start in range(0, n_total, chunk_size):
        ch = val_t[start:start + chunk_size]
        pred = net(ch[:, SLICE_INPUT], training=False)
        err2 = tf.square(pred - ch[:, SLICE_TARGET])
        s6 += float(tf.reduce_sum(err2))
        s4 += float(tf.reduce_sum(err2[:, :4]))
        n  += int(ch.shape[0])
    return s6 / max(n * N_TARGETS, 1), s4 / max(n * 4, 1)


_ZONE_NAMES = ["cyl", "bc", "boi_c", "boi_t", "free"]
_ZONE_GIDS  = [GID_CYL, GID_BC, GID_BOI_C, GID_BOI_T, GID_FREE]
_W_FULL_PY  = float(tf.reduce_sum(_COMP_W_VEC))

_VAL_CHECKED = False


def _val_zone_sums(ch, net):
    """
    Somas por zona de UM chunk: numerador Σ w_c·m_c·e_c² e denominador
    Σ w_c·m_c. São exatamente as duas somas que o mean_zone da Cell 11
    usa — por isso a validação não reescreve fórmula nenhuma.

    net: a rede avaliada (model ou model_ema).
    """
    pred = net(ch[:, SLICE_INPUT], training=False)
    err2 = tf.square(pred - ch[:, SLICE_TARGET])
    # Mesmos nomes do data_loss_grouped (Cell 11) de propósito: é a mesma
    # máscara, e abreviar aqui faria parecer outra coisa.
    if MASK_WALL_DATA_KOMEGA:
        out_band = tf.cast(ch[:, COL_DWALL] > wall_band_lookup_tf(ch), tf.float32)
        um = tf.ones_like(out_band)
        comp_mask = tf.stack([um, um, um, um, out_band, out_band], axis=1)
        #                     u   v   w   P     k         ω
    else:
        comp_mask = tf.ones_like(err2)
    sq  = tf.reduce_sum(err2 * _COMP_W_VEC * comp_mask, axis=1)
    wpp = tf.reduce_sum(_COMP_W_VEC * comp_mask, axis=1)
    g   = ch[:, COL_GID]
    num, den = {}, {}
    for nm, gv in zip(_ZONE_NAMES, _ZONE_GIDS):
        m = tf.cast(tf.equal(g, gv), tf.float32)
        num[nm] = float(tf.reduce_sum(m * sq))
        den[nm] = float(tf.reduce_sum(m * wpp))
    return num, den




def _combine_zone_sums(num, den):
    per_zone = {nm: _W_FULL_PY * num[nm] / max(den[nm], 1e-12)
                for nm in _ZONE_NAMES}
    total = sum(W["data_" + nm] * per_zone[nm] for nm in _ZONE_NAMES)
    return total, per_zone


def evaluate_val_aligned(val_t, net, chunk_size=VAL_CHUNK):
    """
    Data loss no conjunto de validação, com A MESMA conta do treino
    (ver ponto 1 do cabeçalho). Chunkado para não estourar a memória.

    net: a rede avaliada (model ou model_ema). Obrigatório, sem default.

    Na primeira chamada confere o resultado de um chunk contra
    data_loss_grouped. Se divergirem, quebra aqui e não no gráfico final.
    Uma vez basta: o que se confere é a FÓRMULA, que é a mesma para as duas
    redes. A conferência usa a mesma `net` dos dois lados.
    """
    global _VAL_CHECKED
    n_total = int(val_t.shape[0])
    num = {nm: 0.0 for nm in _ZONE_NAMES}
    den = {nm: 0.0 for nm in _ZONE_NAMES}
    for start in range(0, n_total, chunk_size):
        n_c, d_c = _val_zone_sums(val_t[start:start + chunk_size], net)
        for nm in _ZONE_NAMES:
            num[nm] += n_c[nm]; den[nm] += d_c[nm]
    total, per_zone = _combine_zone_sums(num, den)

    if not _VAL_CHECKED:
        _VAL_CHECKED = True
        _n = min(chunk_size, n_total)
        _ref = data_loss_grouped(val_t[:_n], net, **ZONE_W)
        _t1, _pz = _combine_zone_sums(*_val_zone_sums(val_t[:_n], net))
        _errs = {nm: abs(_pz[nm] - float(_ref[nm])) / max(abs(float(_ref[nm])), 1e-30)
                 for nm in _ZONE_NAMES}
        _worst = max(_errs.values())
        print(f"  [val] conferência contra data_loss_grouped em {_n:,} pontos: "
              f"pior erro relativo por zona = {_worst:.2e}")
        assert _worst < 1e-4, (
            f"evaluate_val_aligned e data_loss_grouped discordam: {_errs}. "
            f"As duas TÊM que calcular a mesma coisa.")

    return total, per_zone


# ================================================================================
# Histórico e CSV
# ================================================================================
_VAL_ZONES = ["cyl", "bc", "boi_c", "boi_t", "free"]
HIST_KEYS = (["epoch", "lr", "elapsed", "skipped"]
             + LOSS_KEYS_LOG
             + ["val_mse6", "val_mse4", "val_aligned"]
             + ["val_" + z for z in _VAL_ZONES]
             # ema1: no FIM, para as colunas antigas não mudarem de posição
             + ["val_mse6_ema", "val_mse4_ema", "val_aligned_ema"]
             + ["val_" + z + "_ema" for z in _VAL_ZONES]
             + ["val_time"])

if "history" not in globals() or len(history.get("epoch", [])) == 0:
    history = {k_: [] for k_ in HIST_KEYS}
    # O offset só numera as épocas no log e nos nomes de checkpoint, para
    # a continuidade com a rodada anterior ficar legível. O histórico em si
    # começa vazio de propósito: as colunas de loss da rodada velha foram
    # medidas com outra régua.
    epoch_offset = int(WARM_START_EPOCH) if WARM_START_FROM else 0
    print(f"Histórico novo"
          + (f", numerando a partir da época {epoch_offset} "
             f"(continuação de uma rodada anterior)" if epoch_offset
             else " — treino do zero"))
else:
    # Continuação: chaves novas são preenchidas com nan para o passado, para
    # o CSV e os gráficos não desalinharem.
    _n_old = len(history["epoch"])
    for k_ in HIST_KEYS:
        history.setdefault(k_, [float("nan")] * _n_old)
    epoch_offset = max(history["epoch"])
    print(f"Continuando treino. Última época registrada: {epoch_offset}")

# [09/11/2026] RETOMADA: o CSV continua o MESMO arquivo, mas as épocas
# entre o checkpoint de retomada e a queda do job já estão nele e vão ser
# refeitas. Elas saem antes do append, e o original vai para uma cópia de
# segurança (a inferência só lê training_log_<tag>.csv). As linhas lidas
# aqui também dão o best de antes da queda (mais abaixo).
_csv_linhas_orig = None
if WARM_START_FROM_EMA:
    if not os.path.exists(CSV_LOG_PATH):
        raise FileNotFoundError(f"retomada sem o CSV da rodada: {CSV_LOG_PATH}")
    with open(CSV_LOG_PATH, newline="") as f:
        _rd = list(csv.reader(f))
    if not _rd or _rd[0] != HIST_KEYS:
        raise ValueError("o cabeçalho do CSV existente não é o desta versão "
                         "do código (HIST_KEYS): não dá para emendar")
    _csv_linhas_orig = [dict(zip(HIST_KEYS, map(float, r_)))
                        for r_ in _rd[1:] if r_]
    _ficam = [r_ for r_ in _rd[1:] if r_ and float(r_[0]) <= epoch_offset]
    _saem  = len(_rd) - 1 - len(_ficam)
    _ult   = int(float(_ficam[-1][0])) if _ficam else 0
    _bkp = CSV_LOG_PATH.replace(".csv", f".antes_retomada_{epoch_offset:06d}.csv")
    _i = 2
    while os.path.exists(_bkp):          # nunca sobrescreve uma cópia antiga
        _bkp = CSV_LOG_PATH.replace(
            ".csv", f".antes_retomada_{epoch_offset:06d}_{_i}.csv")
        _i += 1
    shutil.copy2(CSV_LOG_PATH, _bkp)
    with open(CSV_LOG_PATH, "w", newline="") as f:
        _w = csv.writer(f)
        _w.writerow(HIST_KEYS)
        _w.writerows(_ficam)
    print(f"[retomada] CSV: cópia do original em {_bkp}")
    print(f"[retomada] CSV: {len(_ficam):,} linhas ficam (até a época {_ult}), "
          f"{_saem:,} saem (épocas depois de {epoch_offset}, refeitas agora)")
    if _ult != epoch_offset:
        print(f"[retomada] >>> a última época no CSV é {_ult}, não "
              f"{epoch_offset}: o histórico vai ter um buraco entre elas.")
    with open(MONITOR_PATH, "a") as fl:
        fl.write(f"\n{'='*78}\n  RETOMADA na época {epoch_offset} "
                 f"({time.strftime('%Y-%m-%d %H:%M')}): as linhas acima depois "
                 f"da época {epoch_offset} foram refeitas abaixo\n{'='*78}\n")

if not os.path.exists(CSV_LOG_PATH) or epoch_offset == 0:
    with open(CSV_LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(HIST_KEYS)
    print(f"CSV de log criado: {CSV_LOG_PATH}  ({len(HIST_KEYS)} colunas)")
    # O monitoramento segue o CSV. Ele é escrito em modo "a" (append) no
    # laço, então, sem isto, rodar de novo com o mesmo RUN_TAG emendaria a
    # rodada nova no fim da velha, enquanto o CSV recomeçava do zero — dois
    # registros da mesma rodada contando histórias diferentes. A Cell 3
    # avisa quando a pasta de checkpoints já tem pesos, que é esse caso.
    open(MONITOR_PATH, "w").close()
    print(f"Monitoramento zerado: {MONITOR_PATH}")
else:
    print(f"CSV existente, fará append: {CSV_LOG_PATH}")
    print(f"Monitoramento, idem:        {MONITOR_PATH}")

# ema1: best_val_loss/best_epoch/best_ckpt_path são do EMA — é o best
# oficial (a inferência roda no EMA) e é o que a Cell 16 grava na metadata.
# O best da rede de treino tem as próprias variáveis, com sufixo _siren.
best_val_loss = float("inf")
best_epoch = 0
best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"pinn_{RUN_TAG}_ema_best.weights.h5")
best_val_loss_siren = float("inf")
best_epoch_siren = 0
best_ckpt_path_siren = os.path.join(CHECKPOINT_DIR,
                                    f"pinn_{RUN_TAG}_siren_best.weights.h5")

# [09/11/2026] RETOMADA: o best não recomeça em infinito, senão a primeira
# validação depois da retomada sobrescreveria os _best de antes da queda
# mesmo sendo pior. Ele vem do CSV ORIGINAL inteiro (antes do corte), porque
# é ele que descreve o arquivo _best que está no disco: se o best caiu numa
# época entre o checkpoint e a queda, o arquivo é daquela época. Isso só é
# avisado — o arquivo fica, e só é trocado por algo melhor.
if _csv_linhas_orig is not None:
    def _best_do_csv(col):
        val = [(r_[col], int(r_["epoch"])) for r_ in _csv_linhas_orig
               if r_.get("val_time", 1.0) > 0 and np.isfinite(r_[col])]
        return min(val) if val else (float("inf"), 0)
    for _rede, _col, _path in (("ema", "val_aligned_ema", best_ckpt_path),
                               ("siren", "val_aligned", best_ckpt_path_siren)):
        _v, _ep = _best_do_csv(_col)
        if not os.path.exists(_path):
            print(f"[retomada] >>> {os.path.basename(_path)} não existe: o best "
                  f"da rede {_rede} recomeça do zero")
            continue
        if _rede == "ema":
            best_val_loss, best_epoch = _v, _ep
        else:
            best_val_loss_siren, best_epoch_siren = _v, _ep
        print(f"[retomada] best {_rede:<5}: {_col} = {_v:.4e} na época {_ep} "
              f"({os.path.basename(_path)})")
        if _ep > epoch_offset:
            print(f"[retomada] >>> esse best é de uma época DEPOIS de "
                  f"{epoch_offset}, que saiu do CSV (rodou antes da queda e é "
                  f"refeita agora). O arquivo continua sendo o best até "
                  f"aparecer um melhor.")

print(f"\n{'='*78}")
print(f"  Treino: {N_EPOCHS:,} épocas novas (a partir da {epoch_offset})")
print(f"  LR {LR_CONSTANT:.2e} constante | {STEPS_PER_EPOCH} passos/época")
print(f"  Dado: {len(train_np):,} treino, {len(val_np):,} val")
print(f"  Colocação: {N_COLL:,} no pool, {N_COLL_PER_STEP:,} por passo")
print(f"  XLA: {USE_XLA} | guarda de NaN: {SKIP_NONFINITE_STEP}")
print(f"  EMA: β = {BETA_EMA} | best oficial pela val_aligned_ema | "
      f"checkpoints _siren e _ema a cada {SAVE_EVERY}")
print(f"  Rampa da física: {PHYS_RAMP_ENABLED}"
      + (f" ({PHYS_RAMP_START:g} -> 1 em {PHYS_RAMP_EPOCHS} épocas)"
         if PHYS_RAMP_ENABLED else " (pesos fixos, como no laminar)"))
print(f"  Checkpoints em {CHECKPOINT_DIR}")
print(f"  Histórico em   {HISTORICO_DIR}")
print(f"{'='*78}\n")


# ================================================================================
# Loop
# ================================================================================
train_start = time.time()
train_iter = iter(train_tf)
_skip_before = int(n_skipped_steps.numpy())

for epoch in range(1, N_EPOCHS + 1):

    # --- rampa de peso (no-op se desligada) ---
    if PHYS_RAMP_ENABLED:
        _f = min(1.0, PHYS_RAMP_START
                 + (1.0 - PHYS_RAMP_START) * (epoch / max(PHYS_RAMP_EPOCHS, 1)))
        for k_ in PHYS_KEYS:
            W_TF[k_].assign(_f * _W_PHYS_FINAL[k_])

    # --- acumuladores como tensores TF (sem sync por passo) ---
    acc = {k_: tf.constant(0.0) for k_ in LOSS_KEYS_LOG}
    n_steps = 0
    while n_steps < STEPS_PER_EPOCH:
        out = train_step(next(train_iter), coll_tensor,
                         coll_dwall_tensor, coll_ywall_tensor)
        for k_ in LOSS_KEYS_LOG:
            acc[k_] += out[k_]
        n_steps += 1

    # --- um sync por época ---
    avg = {k_: float(acc[k_] / n_steps) for k_ in LOSS_KEYS_LOG}
    lr_now = float(optimizer.learning_rate.numpy())
    elapsed = time.time() - train_start
    _skip_now = int(n_skipped_steps.numpy())

    # --- validação (espaçada), das DUAS redes ---
    if epoch % VAL_EVERY == 0 or epoch == 1:
        _t_val0 = time.time()
        val6, val4 = evaluate_val_mse(val_tensor, model)
        val_al, val_zone = evaluate_val_aligned(val_tensor, model)
        val6_ema, val4_ema = evaluate_val_mse(val_tensor, model_ema)
        val_al_ema, val_zone_ema = evaluate_val_aligned(val_tensor, model_ema)
        val_time = time.time() - _t_val0
    else:
        # Repete o último valor medido, para o CSV não ter buracos. Vale
        # para as duas redes igual.
        def _last(k_):
            return history[k_][-1] if history[k_] else float("nan")
        val6, val4, val_al = _last("val_mse6"), _last("val_mse4"), _last("val_aligned")
        val_zone = {z: _last("val_" + z) for z in _VAL_ZONES}
        val6_ema, val4_ema = _last("val_mse6_ema"), _last("val_mse4_ema")
        val_al_ema = _last("val_aligned_ema")
        val_zone_ema = {z: _last("val_" + z + "_ema") for z in _VAL_ZONES}
        val_time = 0.0

    # --- histórico ---
    row = dict(epoch=epoch + epoch_offset, lr=lr_now, elapsed=elapsed,
               skipped=_skip_now, val_mse6=val6, val_mse4=val4,
               val_aligned=val_al,
               **{"val_" + z: val_zone[z] for z in val_zone},
               val_mse6_ema=val6_ema, val_mse4_ema=val4_ema,
               val_aligned_ema=val_al_ema,
               **{"val_" + z + "_ema": val_zone_ema[z] for z in val_zone_ema},
               val_time=val_time)
    row.update(avg)
    for k_ in HIST_KEYS:
        history[k_].append(row[k_])

    # --- melhor checkpoint (pelo val alinhado, que é a loss de verdade) ---
    # ema1: dois best, cada rede julgada pela própria validação. O do EMA é
    # o oficial (ver o cabeçalho).
    if val_al_ema < best_val_loss:
        best_val_loss = val_al_ema
        best_epoch = epoch + epoch_offset
        model_ema.save_weights(best_ckpt_path)
        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"  ✓ novo melhor val_aligned_ema: {val_al_ema:.4e}  (EMA)",
                  flush=True)
    if val_al < best_val_loss_siren:
        best_val_loss_siren = val_al
        best_epoch_siren = epoch + epoch_offset
        model.save_weights(best_ckpt_path_siren)
        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"  ✓ novo melhor val_aligned: {val_al:.4e}  (rede de treino)",
                  flush=True)

    # --- CSV toda época ---
    with open(CSV_LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([row[k_] for k_ in HIST_KEYS])

    # --- console espaçado ---
    if epoch % LOG_EVERY == 0 or epoch == 1:
        l1 = (f"Ep {epoch + epoch_offset:6d} | total {avg['total']:.3e} "
              f"| data {avg['data']:.3e} | fis {avg['phys_sum']:.3e} "
              f"| val6 {val6:.3e} | val4 {val4:.3e} | val_al {val_al:.3e} "
              f"| |g| {avg['gnorm']:.2f} | {elapsed:.0f}s")
        l2 = ("       fis: " + "  ".join(f"{k_} {avg[k_]:.2e}" for k_ in PHYS_KEYS))
        l3 = ("       zonas: " + "  ".join(
                f"{z} {avg['data_' + z]:.2e}"
                for z in ["cyl", "bc", "boi_c", "boi_t", "free"]))
        l4 = (f"       ativo k {100*avg['frac_active_k']:.1f}% "
              f"ω {100*avg['frac_active_om']:.1f}% | parede "
              f"{100*avg['frac_in_wall']:.1f}% | teto ν_t "
              f"{100*avg['frac_visc_cap']:.2f}% | μ_t p50 {avg['mu_t_p50']:.3e}")
        # ema1: a validação do EMA numa linha própria, alinhada com a l1, e
        # o tempo da validação das duas redes (0 s = época sem validação,
        # valores repetidos da última medida).
        l5 = (f"       ema: val6 {val6_ema:.3e} | val4 {val4_ema:.3e} "
              f"| val_al {val_al_ema:.3e} | validação {val_time:.1f}s")
        for _l in (l1, l2, l3, l4, l5):
            print(_l, flush=True)
        # Memória de GPU nas primeiras épocas: é o número que faltava no
        # job 6072. Depois de 3 épocas o pico já estabilizou e a linha só
        # atrapalharia a leitura do log.
        if epoch <= 3:
            _mr = _gpu_mem_report(f"ep {epoch}")
            if _mr:
                print(_mr, flush=True)
        if _skip_now > _skip_before:
            print(f"       >>> {_skip_now - _skip_before} passo(s) DESCARTADO(S) "
                  f"por gradiente não-finito desde o último log. "
                  f"Total {_skip_now}.", flush=True)
        _skip_before = _skip_now
        with open(MONITOR_PATH, "a") as fl:
            fl.write("\n".join((l1, l2, l3, l4, l5)) + "\n")

    # --- balanço de contribuições, espaçado ---
    if BALANCE_EVERY and epoch % BALANCE_EVERY == 0:
        print(f"\n  [balanço na época {epoch + epoch_offset}]", flush=True)
        report_loss_balance()

    # --- reamostrar o pool de colocação ---
    if epoch % RESAMPLE_COLL_EVERY == 0:
        resample_collocation_pool()

    # --- checkpoint ---
    # ema1: as duas redes, cada uma com o sufixo dela (Cell 3). O sufixo vem
    # ANTES do número da época, que é onde a regex da inferência o espera.
    if epoch % SAVE_EVERY == 0:
        _ep = f"{epoch + epoch_offset:06d}"
        _p = os.path.join(CHECKPOINT_DIR,
                          f"pinn_{RUN_TAG}_siren_{_ep}.weights.h5")
        _pe = os.path.join(CHECKPOINT_DIR,
                           f"pinn_{RUN_TAG}_ema_{_ep}.weights.h5")
        model.save_weights(_p)
        model_ema.save_weights(_pe)
        print(f"  checkpoint: {_p}", flush=True)
        print(f"  checkpoint: {_pe}", flush=True)

print(f"\nTreino concluído em {time.time() - train_start:.1f}s")
print(f"Épocas totais: {epoch_offset + N_EPOCHS}")
print(f"Melhor val_aligned_ema: {best_val_loss:.4e} na época {best_epoch}"
      f"   (EMA, best oficial)")
print(f"Melhor val_aligned:     {best_val_loss_siren:.4e} na época "
      f"{best_epoch_siren}   (rede de treino)")
_vt = [v for v in history["val_time"] if v > 0]
if _vt:
    print(f"Validação (duas redes): {np.mean(_vt):.1f} s em média, "
          f"{len(_vt)} vezes, {sum(_vt):.0f} s no total")
print(f"Passos descartados por gradiente não-finito: {int(n_skipped_steps.numpy())}"
      f" de {N_EPOCHS * STEPS_PER_EPOCH:,}")
print(f"CSV: {CSV_LOG_PATH}   ({len(HIST_KEYS)} colunas)")


# ============================================================
# CELL 16 — SAVE FINAL MODEL & METADATA
# ============================================================
# Salva pesos, metadata e histórico, e fecha com uma tabela de erro em
# unidades FÍSICAS sobre o conjunto de validação.
#
# ==============================================================================
# 1. A METADATA DO LAMINAR NÃO ROLA, E NÃO É SÓ POR CAUSA DA TURBULÊNCIA
#
# O bloco laminar começa com
#
#     L_ref = float(L_REF),  P_ref = float(P_REF),  time_ref = float(TIME_REF),
#
# e NENHUM desses três nomes existe no arquivo. São de uma normalização
# anterior (comprimento/pressão/tempo de referência), abandonada quando as
# Cells 4 e 5 passaram para z-score isotrópico. Rodar a célula laminar
# levanta NameError na primeira linha.
#
# Ou seja: a metadata daquela versão já não descrevia o próprio modelo dela.
# Quem carregasse os pesos teria os números da rede e nenhuma forma de
# transformá-los em m/s e Pa.
#
# ==============================================================================
# 2. O QUE A METADATA PRECISA TER PARA O MODELO SER UTILIZÁVEL
#
# A rede devolve seis números adimensionais. Sem as constantes de
# desnormalização eles não significam nada, e elas estão espalhadas por
# quatro células. Então a metadata guarda TODAS, e mais: guarda as FÓRMULAS
# como texto, para quem abrir o JSON daqui a dois anos não ter que
# reconstruir o esquema híbrido (u e w por V_REF, v e P por z-score, k e ω
# por z-score no log) lendo o código.
#
# ==============================================================================
# 3. SALVAR SEM CONFERIR NÃO É SALVAR
#
# No fim a célula constrói uma rede NOVA com build_pinn, carrega o arquivo
# de pesos e compara as predições com as do modelo em memória. Se o
# checkpoint estiver incompleto ou a arquitetura tiver mudado, o erro
# aparece aqui — e não daqui a um mês, quando alguém for usar o modelo.
#
# Confere também o round-trip das constantes: desnormaliza uma predição com
# os valores TIRADOS DA METADATA e compara com denormalize_k/denormalize_omega.
#
# ==============================================================================
# 4. AS TABELAS FINAIS DO LAMINAR FORAM SUBSTITUÍDAS
#
# A Cell 16 laminar termina com dois blocos de "resumo" que usam
# errors_by_slice, SUMMARY_DIR, datasets, last_t e n — variáveis que vêm de
# um notebook de pós-processamento separado, não deste arquivo. São código
# morto aqui.
#
# No lugar entra uma tabela AUTOSSUFICIENTE: erro por variável em unidades
# físicas sobre o conjunto de validação, que é dado que o modelo nunca viu.
# MAE, RMSE, viés e correlação, mais |V|. Para k e ω o erro também sai em
# escala log, porque MAE em m²/s² num campo que varre três ordens de
# grandeza não diz nada.
#
# ==============================================================================
# 5. [18/10/2026] PROCEDÊNCIA — de qual CFD e para onde foi cada arquivo
#
# A sst3 é a primeira rodada treinada em outro dado de CFD (job 6125, teto
# de ν_t em 1e7) com a mesma malha, o mesmo nº de passos e o mesmo nome de
# arquivo da sst2. Sem procedência na metadata, um checkpoint da sst3 e um
# da sst2 são indistinguíveis por dentro — e são lidos com dicionários de
# normalização diferentes. Entram:
#
#   procedencia  job do CFD, CFD_DIR, arquivo de faces do inlet, as pastas
#                de saída (Cell 3) e a nota de que a comparação com a sst2
#                é estatística;
#   turbulencia  o campo de ν_t MEDIDO no job 6125 ao lado do teto, e as
#                escalas de referência (NU_T_REF virou valor absoluto na
#                Cell 2 — sem registrar, não dá para saber depois que ela
#                não segue mais o teto);
#   verificacoes o resultado da verificação (c) da Cell 5 (INLET_CHECK):
#                a BC do inlet comparada com o valor de FACE do solver.
#
# A metadata, o histórico e as métricas continuam na pasta de checkpoints,
# junto dos pesos — é o que o inferencia_sst.py espera encontrar ali.
#
# ==============================================================================
# 6. [19/10/2026] RODADA siren1 — A METADATA TEM QUE DESCREVER A SIREN
#
# Os pesos de uma SIREN só reproduzem o modelo se forem carregados numa rede
# com a MESMA ativação e o MESMO ω0. [Corrigido em 29/10/2026, com o teste de
# 27/10 (TF 2.21, Keras 3).] Numa MLP tanh 6×512, o load_weights de um
# .weights.h5 da siren1 DÁ ERRO, apesar de as formas dos tensores serem
# idênticas (1.318.918 parâmetros): o .h5 indexa as variáveis pelo caminho
# da CLASSE da camada (layers/dense × layers/siren_layer) e pela ordem de
# criação, e não pelo nome da camada. O que passa EM SILÊNCIO é uma SIREN
# com outro ω0: carrega sem erro e devolve um campo errado (|Δ| ≈ 5,8 na
# saída crua com ω0 = 30). O mesmo vale para EMA × rede de treino (ema1):
# mesma classe, mesmas formas, então trocar uma pela outra também passa
# calado. A versão anterior deste texto dizia o contrário sobre a MLP tanh.
# Por isso model_config deixa de
# dizer "MLP_simple" e passa a guardar tudo o que o build_pinn da Cell 9
# precisa: tipo, ω0, regra de inicialização e o fato de as entradas NÃO
# serem reescaladas (z-score da Cell 4 direto na 1ª camada). É também o que
# o inferencia_sst.py vai ler para escolher a arquitetura.
#
# A conferência do checkpoint (ponto 3) passa o ω0 explicitamente ao
# build_pinn. O default None também cairia no OMEGA_0 global, mas aqui o
# teste é justamente "a rede descrita pela metadata reproduz o modelo", e
# por isso ela é montada com os valores que ESTÃO na metadata.
#
# procedencia["comparacao"] passa a descrever siren1 × sst3: mesmo CFD
# (job 6125), mesma normalização, mesmo split — comparação instante a
# instante, ao contrário de sst2 × sst3.
#
# ==============================================================================
# 7. [29/10/2026] RODADA ema1 — DUAS REDES PARA SALVAR, CONFERIR E MEDIR
#
# PESOS FINAIS DAS DUAS, cada um com o sufixo da Cell 3:
#   pinn_<tag>_ema.weights.h5     o EMA — é nele que a inferência roda
#   pinn_<tag>_siren.weights.h5   a rede de treino, para comparação
#
# METADATA E MÉTRICAS POR REDE, com o mesmo sufixo dos pesos
# (pinn_<tag>_ema.metadata.json, pinn_<tag>_siren.metadata.json, idem
# .val_metrics.csv). A inferência acha o perfil pela tag no nome do
# checkpoint ("ema1_ema" ou "ema1_siren"). Com a metadata batizada do mesmo
# jeito, cada arquivo de pesos tem a sua ao lado, e nenhum arquivo da rodada
# fica com nome "sem rede". As duas metadatas são iguais, a não ser pelo
# campo ema["rede_destes_pesos"]. O histórico é da rodada, não de uma
# rede: fica um só, pinn_<tag>.history.json.
#
# BLOCO "ema" NA METADATA: β, quando atualiza, a ausência de aquecimento, a
# inicialização como cópia, o critério do best e os nomes dos arquivos.
# Pesos do EMA e da rede de treino têm formas idênticas: sem esse bloco, a
# metadata não diria de qual das duas os pesos são.
#
# CONFERÊNCIA DO CHECKPOINT NAS DUAS: cada uma é recarregada numa rede
# montada pela metadata e tem que dar |Δ| = 0. Além disso, os dois arquivos
# têm que DIFERIR entre si. Se derem iguais, a mesma rede foi salva duas
# vezes, ou o EMA nunca andou (o zip vazio da Cell 9).
#
# TABELA PINN × CFD NAS DUAS: a do EMA é o resultado, e a da rede de treino
# vem logo abaixo para comparar. final_metrics recebe a rede como argumento
# obrigatório, pelo mesmo motivo da validação da Cell 15.
#
# ==============================================================================
# 8. [09/11/2026] RETOMADA NA ÉPOCA 18.100 (Cells 3 e 15)
#
# Numa retomada, esta célula é executada só pelo job que retomou, e a
# memória dele só tem as épocas DEPOIS da retomada. O que descreve o treino
# INTEIRO (1 → 50.000) passa a vir do CSV, que a Cell 15 emendou:
#   - pinn_<tag>.history.json: montado a partir do training_log_<tag>.csv,
#     e não do `history` em memória (que começaria na 18.101);
#   - treino.skipped_steps: total das duas execuções. O contador de passos
#     descartados recomeça em 0 a cada execução, então o da primeira vem
#     da coluna `skipped` do CSV na época de retomada;
#   - ema.n_atualizacoes: total de passos aplicados desde a época 1. O da
#     execução atual fica em ema.n_atualizacoes_nesta_execucao;
#   - treino.retomada: época, os dois arquivos de partida e a cópia de
#     segurança do CSV, para a metadata dizer que houve uma.
# treino.n_epochs já saía certo (é a maior época registrada, 50.000).
# Sem retomada, tudo segue como antes.
# ==============================================================================

final_weights_path      = os.path.join(CHECKPOINT_DIR, f"pinn_{RUN_TAG}_ema.weights.h5")
final_weights_path_siren = os.path.join(CHECKPOINT_DIR,
                                        f"pinn_{RUN_TAG}_siren.weights.h5")
final_metadata_path     = os.path.join(CHECKPOINT_DIR,
                                       f"pinn_{RUN_TAG}_ema.metadata.json")
final_metadata_path_siren = os.path.join(CHECKPOINT_DIR,
                                         f"pinn_{RUN_TAG}_siren.metadata.json")
final_history_path      = os.path.join(CHECKPOINT_DIR, f"pinn_{RUN_TAG}.history.json")
final_metrics_path      = os.path.join(CHECKPOINT_DIR,
                                       f"pinn_{RUN_TAG}_ema.val_metrics.csv")
final_metrics_path_siren = os.path.join(CHECKPOINT_DIR,
                                        f"pinn_{RUN_TAG}_siren.val_metrics.csv")

model_ema.save_weights(final_weights_path)
model.save_weights(final_weights_path_siren)


def _get(name, default=None):
    """Lê um global opcional (as verificações da Cell 5 usam globals())."""
    v = globals().get(name, default)
    return v


def _jsonable(obj):
    """
    Converte recursivamente para tipos que o json aceita.

    As verificações da Cell 5 devolvem dicts com escalares do NumPy
    (np.bool_ e np.int64 não são serializáveis; np.float64 é, por herdar de
    float, mas é melhor não depender disso) e dicts aninhados. Um TypeError
    no json.dump aconteceria DEPOIS de dias de treino, com os pesos salvos
    e a metadata pela metade — por isso a conversão é feita por inteiro.
    """
    if isinstance(obj, dict):
        return {str(k_): _jsonable(v_) for k_, v_ in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v_) for v_ in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return None        # arrays grandes (p.ex. S implícito) não vão para a metadata
    return obj


# [09/11/2026] retomada: histórico e contadores do treino INTEIRO (ponto 8)
_retomada = bool(WARM_START_FROM_EMA)
if _retomada and os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, newline="") as f:
        _rd16 = list(csv.reader(f))
    historico_completo = {k_: [] for k_ in _rd16[0]}
    for _r in _rd16[1:]:
        if not _r:
            continue
        for k_, v_ in zip(_rd16[0], _r):
            historico_completo[k_].append(float(v_))
    historico_completo["epoch"] = [int(e_) for e_ in historico_completo["epoch"]]
    _i_ret = [i_ for i_, e_ in enumerate(historico_completo["epoch"])
              if e_ == int(epoch_offset)]
    _skip_antes = (int(historico_completo["skipped"][_i_ret[0]])
                   if _i_ret else 0)
    print(f"[retomada] histórico do CSV: épocas "
          f"{historico_completo['epoch'][0]} → "
          f"{historico_completo['epoch'][-1]} "
          f"({len(historico_completo['epoch']):,} linhas)")
    del _rd16
else:
    historico_completo = history
    _skip_antes = 0
_skip_total = _skip_antes + int(n_skipped_steps.numpy())
_passos_total = (int(epoch_offset) + N_EPOCHS) * STEPS_PER_EPOCH \
                if _retomada else N_EPOCHS * STEPS_PER_EPOCH

metadata = dict(
    run_tag = RUN_TAG,
    criado_em = time.strftime("%Y-%m-%d %H:%M:%S"),
    tensorflow = tf.__version__,

    # ----------------------------------------------------------------
    # PROCEDÊNCIA — qual CFD gerou o dado e onde ficou cada saída
    # ----------------------------------------------------------------
    procedencia = dict(
        cfd_job_id=int(CFD_JOB_ID),
        cfd_dir=CFD_DIR,
        snapshot_dir=SNAPSHOT_DIR,
        inlet_faces_csv=INLET_FACES_CSV,
        has_inlet_faces=bool(HAS_INLET_FACES),
        visc_ratio_max_cfd=float(VISC_RATIO_MAX),
        comparacao=("ema1: EMA x rede de treino da própria rodada (mesma "
                    "inicialização, mesmos passos; comparação exata, na mesma "
                    "época). Rede de treino x siren1: mesmo CFD (job 6125), "
                    "mesma normalização e mesmo split, mas inicialização "
                    "diferente (o Keras 3 sorteia a semente dos "
                    "inicializadores com o random do Python, que a Cell 1 não "
                    "fixa); comparar a tendência, com média móvel de 9 épocas"),
        train_dir=TRAIN_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        historico_dir=HISTORICO_DIR,
        diagnosticos_dir=DIAG_DIR,
        csv_log=CSV_LOG_PATH,
        monitor=MONITOR_PATH,
    ),

    # ----------------------------------------------------------------
    # DESNORMALIZAÇÃO — sem isto a saída da rede não significa nada
    # ----------------------------------------------------------------
    normalizacao = dict(
        formulas = dict(
            entrada_x = "xn = (x - X_MEAN) / DP_ISO",
            entrada_y = "yn = (y - Y_MEAN) / DP_ISO",
            entrada_z = "zn = (z - Z_MEAN) / DP_ISO",
            entrada_t = "tn = (t - T_MEAN) / T_DP",
            saida_u   = "u = un * V_REF",
            saida_v   = "v = vn * V_DP + V_MEAN",
            saida_w   = "w = wn * V_REF",
            saida_P   = "P = Pn * P_DP + P_MEAN",
            saida_k   = "k = exp(kn * LNK_DP + LNK_MEAN) - K_EPS",
            saida_om  = "omega = exp(omn * LNW_DP + LNW_MEAN)",
            ordem_saida = list(_COMP_ORDER),
        ),
        X_MEAN=float(X_MEAN), Y_MEAN=float(Y_MEAN), Z_MEAN=float(Z_MEAN),
        DP_ISO=float(DP_ISO), T_MEAN=float(T_MEAN), T_DP=float(T_DP),
        V_REF=float(V_REF), V_MEAN=float(V_MEAN), V_DP=float(V_DP),
        P_MEAN=float(P_MEAN), P_DP=float(P_DP),
        LNK_MEAN=float(LNK_MEAN), LNK_DP=float(LNK_DP),
        LNW_MEAN=float(LNW_MEAN), LNW_DP=float(LNW_DP),
        K_EPS=float(K_EPS), K_NORM_MODE=K_NORM_MODE, LOG_OMEGA=bool(LOG_OMEGA),
    ),

    # ----------------------------------------------------------------
    # GEOMETRIA E ESCOAMENTO
    # ----------------------------------------------------------------
    geometria = dict(
        X_min=X_MIN, X_max=X_MAX, Y_min=Y_MIN, Y_max=Y_MAX,
        Z_min=Z_MIN, Z_max=Z_MAX,
        D_cyl=D_CYL, H_cyl=H_CYL, xc_cyl=XC_CYL, yc_cyl=YC_CYL, zc_base=ZC_BASE,
        BOI_C_X=list(BOI_C_X), BOI_C_Y=list(BOI_C_Y), BOI_C_Z=list(BOI_C_Z),
        BOI_T_V1=list(BOI_T_V1), BOI_T_V2=list(BOI_T_V2),
        BOI_T_V3=list(BOI_T_V3), BOI_T_Z=list(BOI_T_Z),
        BC_TOL=BC_TOL, CYL_TOL=CYL_TOL,
        wall_zones=list(WALL_ZONES),
    ),
    escoamento = dict(
        V_inf=V_INF, rho_inf=float(RHO_INF), mu=MU, nu=float(NU),
        P_op=P_OP, R_sp=R_SP, g=G_ACC,
        Re_D=float(RE_D), Re_H=float(RE_H),
        T_flowthrough=float(T_FLOWTHROUGH),
    ),

    # ----------------------------------------------------------------
    # MODELO DE TURBULÊNCIA — o que o resíduo impôs
    # ----------------------------------------------------------------
    turbulencia = dict(
        modelo="k-omega SST (Fluent 2025 R2, Theory Guide §4.4.3)",
        sigma_prandtl_fluent=dict(k1=SST_PR_K1, w1=SST_PR_W1,
                                  k2=SST_PR_K2, w2=SST_PR_W2),
        sigma_menter=dict(k1=float(SST_SIGMA_K1), w1=float(SST_SIGMA_W1),
                          k2=float(SST_SIGMA_K2), w2=float(SST_SIGMA_W2)),
        beta_i1=SST_BETA_1, beta_i2=SST_BETA_2, beta_star=SST_BETA_STAR,
        kappa_modelo=SST_KAPPA, kappa_parede=KAPPA_WALL, a1=SST_A1,
        alpha_inf1=float(SST_GAMMA_1), alpha_inf2=float(SST_GAMMA_2),
        production_limiter=bool(SST_PRODUCTION_LIMITER),
        prod_limit_coef=SST_PROD_LIMIT_COEF,
        shear_limiter=bool(SST_SHEAR_LIMITER),
        visc_ratio_max=VISC_RATIO_MAX, nu_t_max=float(NU_T_MAX),
        # campo MEDIDO no job 6125, para mostrar que o teto não morde
        visc_ratio_measured_max=float(VISC_RATIO_MEASURED_MAX),
        nu_t_measured=dict(p50=float(NU_T_MEASURED_P50),
                           p90=float(NU_T_MEASURED_P90),
                           max=float(NU_T_MEASURED_MAX)),
        visc_limited_pct_expected=float(VISC_LIMITED_PCT_EXPECTED),
        k_inlet=float(K_INLET), omega_inlet=float(OMEGA_INLET),
        TI_inlet=TI_INLET, visc_ratio_inlet=VISC_RATIO_INLET,
        # escalas de referência (Cell 2). NU_T_REF é valor ABSOLUTO desde a
        # sst3 — não segue mais 0,25·NU_T_MAX
        escalas_referencia=dict(TI_ref=float(TI_REF), K_ref=float(K_REF),
                                NU_T_ref=float(NU_T_REF),
                                OMEGA_ref=float(OMEGA_REF)),
    ),
    residuo = dict(
        pressao_modificada=bool(PRESSURE_IS_MODIFIED),
        momentum_include_k_gradient=bool(MOMENTUM_INCLUDE_K_GRADIENT),
        omega_residual_scale=OMEGA_RESIDUAL_SCALE,
        k_residual_scale=K_RESIDUAL_SCALE,
        k_residual_floor_xi=float(K_RESIDUAL_FLOOR_XI),
        k_residual_formula=("r_k / max(beta_star*rho*k*omega, "
                            "xi*SCALE_K)" if K_RESIDUAL_SCALE == "local"
                           else "r_k / SCALE_K"),
        scale_k=float(SCALE_K), scale_om=float(SCALE_OM),
        warm_start_from=(WARM_START_FROM or ""),
        warm_start_from_ema=(WARM_START_FROM_EMA or ""),
        warm_start_epoch=int(WARM_START_EPOCH),
        wall_residual_mode=WALL_RESIDUAL_MODE,
        mask_prod_limited_omega=bool(MASK_PROD_LIMITED_OMEGA),
        mask_visc_limited=bool(MASK_VISC_LIMITED),
        mask_wall_data_komega=bool(MASK_WALL_DATA_KOMEGA),
        residual_mean_mode=RESIDUAL_MEAN_MODE,
        wall_law_C=float(WALL_LAW_C),
        Y_WALL_FUNCTION=float(Y_WALL_FUNCTION),
        Y_BAND_BY_WALL={k_: float(v) for k_, v in Y_BAND_BY_WALL.items()},
        Y_FIRST_BY_WALL={k_: float(v) for k_, v in Y_FIRST_BY_WALL.items()},
        d_wall_floor=D_WALL_FLOOR,
        escalas=dict(cont=float(SCALE_CONT), mom=float(SCALE_MOM),
                     k=float(SCALE_K), omega=float(SCALE_OM)),
    ),

    # ----------------------------------------------------------------
    # O QUE FOI MEDIDO NO DADO (Cell 5) — fica junto do modelo
    # ----------------------------------------------------------------
    verificacoes = dict(
        nu_t = ({kk: (float(vv) if isinstance(vv, (int, float)) else None)
                 for kk, vv in _get("NU_T_CHECK", {}).items()
                 if kk != "S_implied"} if _get("NU_T_CHECK") else None),
        lei_de_parede = ({kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                          for kk, vv in _get("WALL_LAW_CHECK", {}).items()
                          if kk in ("y_first_global", "ratio_med", "ratio_p10",
                                    "ratio_p99", "y_wall_function")}
                         if _get("WALL_LAW_CHECK") else None),
        pressao = ({kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                    for kk, vv in _get("PRESSURE_CHECK", {}).items()}
                   if _get("PRESSURE_CHECK") else None),
        d_wall_formula_vs_fluent = ({kk: float(vv) for kk, vv
                                     in _get("WALL_DIST_CHECK", {}).items()}
                                    if _get("WALL_DIST_CHECK") else None),
        omega_wall = ({kk: (float(vv) if isinstance(vv, (int, float, bool))
                            else vv)
                       for kk, vv in _get("OMEGA_WALL_CHECK", {}).items()}
                      if _get("OMEGA_WALL_CHECK") else None),
        # (c) BC do inlet contra o valor de FACE do solver. mode "faces"
        # com ok=True é o resultado esperado; mode "cells" significa que o
        # arquivo de faces faltou e a BC NÃO foi verificada.
        inlet = (_jsonable(_get("INLET_CHECK"))
                 if _get("INLET_CHECK") else None),
    ),

    # ----------------------------------------------------------------
    # DADO E ARQUITETURA
    # ----------------------------------------------------------------
    dado = dict(
        snapshot_dir=SNAPSHOT_DIR,
        n_snaps=int(n_snaps),
        t_min=float(times_all[0]), t_max=float(times_all[-1]),
        n_cells_cfd=int(N_CELLS_CFD), dt_cfd=DT_CFD,
        csv_columns=list(CSV_COLUMNS_CFD),
        data_np_layout={"xn": COL_XN, "yn": COL_YN, "zn": COL_ZN, "tn": COL_TN,
                        "un": COL_UN, "vn": COL_VN, "wn": COL_WN, "Pn": COL_PN,
                        "kn": COL_KN, "omn": COL_OMN, "gid": COL_GID,
                        "vlim": COL_VLIM, "d_wall": COL_DWALL},
        n_train=int(len(train_np)), n_val=int(len(val_np)),
        sampling_quotas=dict(QUOTAS),
        train_shuffle_mode=TRAIN_SHUFFLE_MODE,
    ),
    model_config = dict(
        type="SIREN", n_layers=MODEL_BLOCKS, n_neurons=MODEL_NEURONS,
        n_outputs=N_OUTPUTS, n_params=int(model.count_params()),
        # --- SIREN (Cell 9). Sem isto os pesos não são reutilizáveis: numa
        #     SIREN com outro ω0 eles carregam sem erro e dão campo errado
        #     (numa MLP tanh o load_weights dá erro — ver o ponto 6) ---
        activation="sin(omega_0*(W·x + b)) nas camadas ocultas",
        omega_0=float(OMEGA_0),
        init=dict(first="U(-1/fan_in, +1/fan_in)",
                  hidden="U(-sqrt(6/fan_in)/omega_0, +sqrt(6/fan_in)/omega_0)",
                  bias="zeros",
                  output="glorot_uniform (Dense linear, default do Keras)"),
        input_scaling="none (z-score da Cell 4 direto na 1a camada)",
        layer_names=[f"siren_{i+1}" for i in range(MODEL_BLOCKS)] + ["raw_fields"],
        n_params_sst3=int(N_PARAMS_SST3),
        hard_bc_k_inlet=bool(HARD_BC_K_INLET),
        hard_bc_omega_inlet=bool(HARD_BC_OMEGA_INLET),
        cyl_rim_fix=bool(CYL_RIM_FIX),
        hc=dict(mt=HC_MT, rate=HC_RATE, k_smooth=HC_K_SMOOTH, k_r2=HC_K_R2),
    ),
    colocacao = dict(
        pool_total=int(N_COLL), per_step=int(N_COLL_PER_STEP),
        pool_mult=POOL_MULT,
        por_regiao={r["name"]: r["per_step"] for r in COLLOCATION_REGIONS},
        reject_below_first_cell=bool(COLL_REJECT_BELOW_FIRST_CELL),
    ),
    pesos = dict(W=dict(W), comp=dict(DATA_COMP_W)),
    treino = dict(
        batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
        lr=LR_CONSTANT, grad_clip=GRAD_CLIP_NORM, xla=bool(USE_XLA),
        n_epochs=int(max(history["epoch"])) if history["epoch"] else 0,
        # [09/11/2026] total das duas execuções numa retomada (ponto 8)
        skipped_steps=int(_skip_total),
        retomada=(dict(epoca=int(epoch_offset),
                       rede_de_treino=os.path.basename(WARM_START_FROM),
                       ema=os.path.basename(WARM_START_FROM_EMA),
                       csv_original=os.path.basename(globals().get("_bkp", "")),
                       adam="zerado na retomada (não está no .h5)")
                  if _retomada else None),
        phys_ramp=bool(PHYS_RAMP_ENABLED),
        # ema1: best oficial = o do EMA (Cell 15). O da rede de treino vai
        # no bloco "ema", com sufixo _siren.
        best_epoch=int(best_epoch), best_val_aligned=float(best_val_loss),
    ),

    # ----------------------------------------------------------------
    # EMA (rodada ema1) — sem isto não dá para saber de qual rede são os
    # pesos: as duas têm formas idênticas (ver o ponto 7)
    # ----------------------------------------------------------------
    ema = dict(
        rede_destes_pesos="ema",          # trocado para "siren" na cópia abaixo
        beta=float(BETA_EMA),
        formula="w_ema <- beta*w_ema + (1 - beta)*w, peso a peso, "
                "w já atualizado pelo Adam",
        atualizacao="todo passo aplicado; passo descartado pela guarda de "
                    "NaN fica fora da média",
        aquecimento="nenhum",
        inicializacao="cópia exata da rede de treino (set_weights na Cell 9, "
                      "|Δ| = 0 conferido) antes do primeiro passo",
        # passos aplicados. Num warm start antigo o EMA é recopiado (Cell
        # 15), então a conta começa ali; numa RETOMADA ele continua, e a
        # conta vai desde a época 1 (ponto 8).
        n_atualizacoes=int(_passos_total - _skip_total),
        n_atualizacoes_nesta_execucao=int(N_EPOCHS * STEPS_PER_EPOCH
                                          - int(n_skipped_steps.numpy())),
        best_criterio="menor val_aligned de cada rede",
        best_epoch_ema=int(best_epoch),
        best_val_aligned_ema=float(best_val_loss),
        best_epoch_siren=int(best_epoch_siren),
        best_val_aligned_siren=float(best_val_loss_siren),
        arquivos=dict(
            ema=dict(final=os.path.basename(final_weights_path),
                     best=os.path.basename(best_ckpt_path),
                     periodico=f"pinn_{RUN_TAG}_ema_NNNNNN.weights.h5",
                     metadata=os.path.basename(final_metadata_path),
                     val_metrics=os.path.basename(final_metrics_path)),
            siren=dict(final=os.path.basename(final_weights_path_siren),
                       best=os.path.basename(best_ckpt_path_siren),
                       periodico=f"pinn_{RUN_TAG}_siren_NNNNNN.weights.h5",
                       metadata=os.path.basename(final_metadata_path_siren),
                       val_metrics=os.path.basename(final_metrics_path_siren)),
            historico=os.path.basename(final_history_path),
        ),
    ),
)

# Passa tudo por _jsonable: as verificações trazem escalares do NumPy, e
# um TypeError aqui viria depois do treino inteiro.
metadata = _jsonable(metadata)
with open(final_metadata_path, "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
# A da rede de treino é a MESMA metadata, com o campo que diz de qual rede
# são os pesos trocado. Copia profunda via json, para não mexer na do EMA.
metadata_siren = json.loads(json.dumps(metadata))
metadata_siren["ema"]["rede_destes_pesos"] = "siren"
with open(final_metadata_path_siren, "w") as f:
    json.dump(metadata_siren, f, indent=2, ensure_ascii=False)
with open(final_history_path, "w") as f:
    # [09/11/2026] o treino inteiro, também numa retomada (ponto 8)
    json.dump(historico_completo, f, indent=2)


# ================================================================================
# CONFERÊNCIA DO QUE FOI SALVO (ver ponto 3)
# ================================================================================
print(f"\n{'='*78}")
print(f"  CONFERÊNCIA DO CHECKPOINT")
print(f"{'='*78}")

_probe = tf.constant(val_np[:2048, SLICE_INPUT], tf.float32)
# ema1: as duas redes, cada uma contra o próprio arquivo. _ref_pred fica
# com a do EMA, que é a que o round-trip das constantes abaixo usa.
_ref_pred_siren = model(_probe, training=False).numpy()
_ref_pred = model_ema(_probe, training=False).numpy()

for _rotulo, _net_ref, _path, _meta in (
        ("EMA",            _ref_pred,       final_weights_path,
         metadata),
        ("rede de treino", _ref_pred_siren, final_weights_path_siren,
         metadata_siren)):
    # Montada com o ω0 que FOI PARA A METADATA, não com o global: o que se
    # confere aqui é que a rede descrita pela metadata reproduz o modelo.
    _model2 = build_pinn(_meta["model_config"]["n_layers"],
                         _meta["model_config"]["n_neurons"],
                         omega_0=_meta["model_config"]["omega_0"])
    _ = _model2(_probe, training=False)               # constrói os pesos
    _model2.load_weights(_path)
    _new_pred = _model2(_probe, training=False).numpy()
    _dmax = float(np.max(np.abs(_net_ref - _new_pred)))
    print(f"  {_rotulo:<15}: rede nova + load_weights  ->  |Δ| máx = {_dmax:.3e}"
          f"   ({os.path.basename(_path)})")
    assert _dmax == 0.0, (f"os pesos recarregados ({_rotulo}) não reproduzem o "
                          f"modelo. Checkpoint incompleto ou arquitetura diferente.")
    del _model2, _new_pred

# Os dois arquivos TÊM que ser de redes diferentes (ver o ponto 7).
_d_redes = float(np.max(np.abs(_ref_pred - _ref_pred_siren)))
print(f"  EMA × rede de treino      ->  |Δ| máx = {_d_redes:.3e}   "
      f"(tem que ser > 0)")
assert _d_redes > 0.0, ("EMA e rede de treino dão a MESMA saída: a mesma rede "
                        "foi salva duas vezes, ou o EMA nunca foi atualizado.")

# round-trip das constantes DA METADATA
_nm = metadata["normalizacao"]
# float64 nos DOIS lados: comparar float32 com float64 daria 1e-7 de
# diferença só pela precisão, e o que se quer testar aqui é se as
# constantes são as mesmas, não a aritmética.
_kn64 = _ref_pred[:, 4].astype(np.float64)
_on64 = _ref_pred[:, 5].astype(np.float64)
_k_meta = np.exp(_kn64 * _nm["LNK_DP"] + _nm["LNK_MEAN"]) - _nm["K_EPS"]
_o_meta = np.exp(_on64 * _nm["LNW_DP"] + _nm["LNW_MEAN"])
_k_code = denormalize_k(_kn64)
_o_code = denormalize_omega(_on64)
_ek = float(np.max(np.abs(_k_meta - _k_code) / np.maximum(np.abs(_k_code), 1e-30)))
_eo = float(np.max(np.abs(_o_meta - _o_code) / np.maximum(np.abs(_o_code), 1e-30)))
print(f"  desnormalização pela metadata × pelo código: k {_ek:.2e}  ω {_eo:.2e}")
assert max(_ek, _eo) < 1e-10, "as constantes salvas não reproduzem a desnormalização"
del _probe, _ref_pred, _ref_pred_siren

print(f"\n  Salvos:")
print(f"    pesos EMA       {final_weights_path}")
print(f"    pesos treino    {final_weights_path_siren}")
print(f"    best EMA        {best_ckpt_path}   (época {best_epoch})")
print(f"    best treino     {best_ckpt_path_siren}   (época {best_epoch_siren})")
print(f"    metadata EMA    {final_metadata_path}  "
      f"({os.path.getsize(final_metadata_path)/1024:.1f} kB)")
print(f"    metadata treino {final_metadata_path_siren}")
print(f"    histórico {final_history_path}  "
      f"({len(history['epoch'])} épocas)")
print(f"    CSV       {CSV_LOG_PATH}")
print(f"    monitor   {MONITOR_PATH}")
print(f"    diagnóst. {DIAG_DIR}")


# ================================================================================
# TABELA FINAL — erro em unidades FÍSICAS no conjunto de validação
# ================================================================================
def final_metrics(val_t, net, chunk_size=VAL_CHUNK):
    """
    PINN × CFD no conjunto de validação (snapshots que o treino não viu),
    em unidades físicas.

    net: a rede avaliada (model_ema ou model). Obrigatório, sem default —
    ver o ponto 7 do cabeçalho.

    Para k e ω o erro sai TAMBÉM em escala log: um MAE de 2 m²/s² não diz
    nada num campo que vai de 0,1 a 100, e o erro relativo é o que a rede
    de fato otimizou.

    [18/10/2026] PARA k, O LOG É ln(k + K_EPS), NÃO ln k. É a variável em
    que a rede é treinada (Cell 4). A versão anterior usava ln k cru e
    dava, na calibração da sst3 (job 6389), MAE(ln) = 5,5 e "fator de erro
    ×243" — número que não descreve a rede:
      - 34% das células têm k < K_EPS = 0,0108, com k até 1e-10. O piso
        K_EPS existe justamente para a rede NÃO gastar esforço separando
        1e-6 de 1e-10; medir em ln k cobra dela exatamente essa separação
        (k real 1e-10 previsto como 0,005 conta 18 em ln);
      - exp(kn·σ + μ) − K_EPS pode dar k levemente NEGATIVO perto do piso,
        e o np.maximum(·, 1e-30) transformava isso em ln = −69. Era daí que
        saía o RMSE(ln) = 16.
    Em ln(k + K_EPS) os dois somem: o argumento é exp(...) > 0 por
    construção e a escala é a do treino. Para ω não há piso (ω = exp(...)),
    então ln ω fica como estava.
    """
    nomes = ["u", "v", "w", "P", "k", "omega", "|V|"]
    unid  = ["m/s", "m/s", "m/s", "Pa", "m²/s²", "1/s", "m/s"]
    acc = {n: dict(n=0, se=0.0, ae=0.0, bias=0.0,
                   sx=0.0, sy=0.0, sxx=0.0, syy=0.0, sxy=0.0) for n in nomes}
    acc_log = {n: dict(n=0, ae=0.0, se=0.0) for n in ["k", "omega"]}

    def _fis(arr):
        u = arr[:, 0] * V_REF
        v = arr[:, 1] * V_DP + V_MEAN
        w = arr[:, 2] * V_REF
        P = arr[:, 3] * P_DP + P_MEAN
        k = denormalize_k(arr[:, 4])
        o = denormalize_omega(arr[:, 5])
        return dict(u=u, v=v, w=w, P=P, k=k, omega=o,
                    **{"|V|": np.sqrt(u*u + v*v + w*w)})

    n_total = int(val_t.shape[0])
    for start in range(0, n_total, chunk_size):
        ch = val_t[start:start + chunk_size]
        pr = net(ch[:, SLICE_INPUT], training=False).numpy().astype(np.float64)
        tg = ch[:, SLICE_TARGET].numpy().astype(np.float64)
        P_, T_ = _fis(pr), _fis(tg)
        for n in nomes:
            x, y = T_[n], P_[n]            # x = CFD, y = PINN
            d = y - x
            a = acc[n]
            a["n"] += len(d); a["se"] += float((d*d).sum())
            a["ae"] += float(np.abs(d).sum()); a["bias"] += float(d.sum())
            a["sx"] += float(x.sum()); a["sy"] += float(y.sum())
            a["sxx"] += float((x*x).sum()); a["syy"] += float((y*y).sum())
            a["sxy"] += float((x*y).sum())
        for n in ["k", "omega"]:
            # k: ln(k + K_EPS), a variável do treino (ver docstring);
            # ω: ln ω. O maximum só protege os modos "scale"/"z" de
            # K_NORM_MODE — no "logz" o argumento já é exp(...) > 0.
            _eps = K_EPS if n == "k" else 0.0
            lx = np.log(np.maximum(T_[n] + _eps, 1e-30))
            ly = np.log(np.maximum(P_[n] + _eps, 1e-30))
            d = ly - lx
            acc_log[n]["n"] += len(d)
            acc_log[n]["ae"] += float(np.abs(d).sum())
            acc_log[n]["se"] += float((d*d).sum())

    rows = []
    for n, u_ in zip(nomes, unid):
        a = acc[n]; N = max(a["n"], 1)
        mae = a["ae"]/N; rmse = np.sqrt(a["se"]/N); bias = a["bias"]/N
        cov = a["sxy"]/N - (a["sx"]/N)*(a["sy"]/N)
        vx = max(a["sxx"]/N - (a["sx"]/N)**2, 0.0)
        vy = max(a["syy"]/N - (a["sy"]/N)**2, 0.0)
        corr = cov/np.sqrt(vx*vy) if vx > 0 and vy > 0 else float("nan")
        r = dict(var=n, unidade=u_, MAE=mae, RMSE=rmse, bias=bias,
                 corr=corr, sigma_CFD=np.sqrt(vx),
                 RMSE_rel_pct=100*rmse/np.sqrt(vx) if vx > 0 else float("nan"))
        if n in acc_log:
            M = max(acc_log[n]["n"], 1)
            r["MAE_ln"] = acc_log[n]["ae"]/M
            r["RMSE_ln"] = np.sqrt(acc_log[n]["se"]/M)
        rows.append(r)
    return pd.DataFrame(rows)


# ema1: a tabela é a mesma para as duas redes; o EMA primeiro, porque é o
# resultado, e a rede de treino depois, para comparar.
def _print_val_table(df, rotulo, path):
    print(f"\n{'='*78}")
    print(f"  PINN × CFD NA VALIDAÇÃO — {rotulo}  ({len(val_np):,} pontos, "
          f"{len(_get('val_snap_indices', []))} snapshots não vistos)")
    print(f"{'='*78}")
    print(f"{'var':<7}{'unid':<8}{'MAE':>12}{'RMSE':>12}{'viés':>12}"
          f"{'corr':>8}{'σ_CFD':>12}{'RMSE/σ':>9}")
    print(f"{'-'*78}")
    for _, r in df.iterrows():
        print(f"{r['var']:<7}{r['unidade']:<8}{r['MAE']:>12.4g}{r['RMSE']:>12.4g}"
              f"{r['bias']:>+12.4g}{r['corr']:>8.3f}{r['sigma_CFD']:>12.4g}"
              f"{r['RMSE_rel_pct']:>8.1f}%")
    print(f"{'-'*78}")
    print(f"  'RMSE/σ' é o erro dividido pelo desvio padrão do campo no CFD:")
    print(f"  100% significa 'errou tanto quanto quem chutasse a média'.")
    for n in ["k", "omega"]:
        _r = df[df["var"] == n].iloc[0]
        _lab = f"ln(k + {K_EPS:.4g})" if n == "k" else "ln ω"
        print(f"  {n}: em {_lab:<16} MAE = {_r['MAE_ln']:.4f}  "
              f"RMSE = {_r['RMSE_ln']:.4f}   "
              f"(fator típico de erro: ×{np.exp(_r['MAE_ln']):.3f})")
    print(f"  (k em ln(k + K_EPS), a variável do treino: abaixo de K_EPS a rede")
    print(f"   não distingue valores de propósito — ver final_metrics)")
    print(f"\n  tabela salva em {path}")
    print(f"{'='*78}")


df_val = final_metrics(val_tensor, model_ema)
df_val.to_csv(final_metrics_path, index=False)
df_val_siren = final_metrics(val_tensor, model)
df_val_siren.to_csv(final_metrics_path_siren, index=False)

_print_val_table(df_val, f"EMA (β = {BETA_EMA})", final_metrics_path)
_print_val_table(df_val_siren, "REDE DE TREINO", final_metrics_path_siren)

# Resumo lado a lado: RMSE/σ é a coluna que se lê primeiro nas duas
# tabelas. Negativo em Δ = EMA melhor.
print(f"\n  RMSE/σ   {'EMA':>8}  {'treino':>8}  {'Δ (EMA − treino)':>18}")
for (_, re_), (_, rs_) in zip(df_val.iterrows(), df_val_siren.iterrows()):
    print(f"  {re_['var']:<7}{re_['RMSE_rel_pct']:>8.1f}%"
          f"{rs_['RMSE_rel_pct']:>9.1f}%"
          f"{re_['RMSE_rel_pct'] - rs_['RMSE_rel_pct']:>+17.1f} pp")
print(f"  (na calibração o EMA ainda carrega parte da inicialização: Δ > 0 é")
print(f"   esperado ali — ver o cabeçalho da Cell 15)")
print(f"{'='*78}")
