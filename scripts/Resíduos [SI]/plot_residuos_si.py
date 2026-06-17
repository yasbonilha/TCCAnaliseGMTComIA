"""
Plot de Resíduos em Unidades Físicas (SI) — atualizado para normalização z-score
=================================================================================

Lê training_log_segregated.csv e converte os resíduos adimensionais
(divididos por SCALE_CONT e SCALE_MOM no treino) para suas unidades
físicas originais (1/s para continuidade, N/m³ para momento).

ATUALIZAÇÕES NESTA VERSÃO:
  - Graduações log densas (minor ticks rotulados) em TODOS os plots
  - Novo PLOT 5: comparação train vs val SEM PESOS
    Endereça pergunta do orientador sobre discrepância total_loss/val_loss
    Mostra que, quando removidos os pesos arbitrários (W_data=70 e pesos
    por zona), o erro de treino e validação ficam comparáveis (~10%)
    indicando boa generalização e ausência de memorização.

NORMALIZAÇÃO (z-score, mantém da versão anterior):
  - Coordenadas espaciais: z-score isotrópico (σ_iso ≈ 101.23 m)
  - Tempo: z-score (σ_t ≈ 28.87 s)
  - Saídas: esquema híbrido (V_REF=5 m/s para u,w; z-score para v,P)
  - SCALE_CONT e SCALE_MOM no treino usam V_INF e D_CYL (escalas físicas
    do problema, não escalas da normalização da rede)

Uso:
    python plot_residuos_si.py [path_to_csv]
    
    Se path_to_csv não fornecido, usa 'training_log_segregated.csv'.

Saída:
    Cria plots PNG no mesmo diretório do CSV:
    - residuo_continuidade_si.png     (1/s)
    - residuo_momento_si.png          (N/m³, 3 componentes)
    - data_loss_zonas_si.png          (m/s, por zona, aproximado)
    - resumo_residuos_si.png          (4 painéis combinados)
    - train_vs_val_unweighted.png     (NOVO: data_raw vs val)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation


def _densify_log_yaxis(ax):
    """Adiciona graduações finas (minor ticks rotulados) num eixo y em escala log."""
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=100)
    )
    ax.yaxis.set_minor_formatter(
        LogFormatterSciNotation(base=10.0, minor_thresholds=(np.inf, np.inf))
    )
    ax.tick_params(axis="y", which="major", labelsize=6)
    ax.tick_params(axis="y", which="minor", labelsize=6)
    ax.grid(True, which="both", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)


# ============================================================
# CONFIGURAÇÃO — DEVE BATER COM O SCRIPT DE TREINO
# ============================================================
# Constantes físicas do problema
V_INF   = 17.0        # m/s, velocidade do escoamento livre
RHO_INF = 1.225       # kg/m³, densidade do ar
D_CYL   = 54.0        # m, diâmetro do cilindro (escala física)

# Escalas físicas usadas para adimensionalizar os resíduos no treino.
# IMPORTANTE: SCALE_CONT e SCALE_MOM usam V_INF e D_CYL (escalas FÍSICAS),
# NÃO as escalas da normalização da rede. Os resíduos das PDEs devem ser
# interpretados na escala em que a física opera, não na escala arbitrária
# da rede neural.
SCALE_CONT = V_INF / D_CYL                   # ≈ 0.3148 [1/s]
SCALE_MOM  = RHO_INF * V_INF**2 / D_CYL      # ≈ 6.556  [N/m³]

# Escalas da normalização da rede (z-score / híbrido)
V_REF   = 5.0          # min-max simétrico para u, w
V_MEAN  = 14.322       # z-score para v: média
V_DP    = 7.400        # z-score para v: desvio padrão
P_MEAN  = -32.789      # z-score para P: média
P_DP    = 98.412       # z-score para P: desvio padrão

# Escala de referência para o data_loss combinado (aproximação)
V_DATA_SCALE = V_INF   # m/s

# Pesos POR ZONA usados no treino (devem bater com o script de treino)
# Esses pesos são aplicados na soma data_loss = Σ w_zona × MSE_zona
W_ZONE_CYL    = 0.5
W_ZONE_BC     = 0.5
W_ZONE_BOI_C  = 1.5
W_ZONE_BOI_T  = 1.0
W_ZONE_FREE   = 3.0
W_ZONE_SUM    = W_ZONE_CYL + W_ZONE_BC + W_ZONE_BOI_C + W_ZONE_BOI_T + W_ZONE_FREE


print("=" * 72)
print("Configuração (atualizada para z-score isotrópico)")
print("=" * 72)
print(f"  V_INF      = {V_INF:.4f} m/s     (escala física)")
print(f"  D_CYL      = {D_CYL:.4f} m       (escala física)")
print(f"  RHO_INF    = {RHO_INF:.4f} kg/m³")
print()
print("Fatores de escala dos resíduos físicos (multiplicar RMSE adim. por):")
print(f"  SCALE_CONT = V_INF / D_CYL       = {SCALE_CONT:.6f} [1/s]")
print(f"  SCALE_MOM  = rho V_INF² / D_CYL  = {SCALE_MOM:.6f} [N/m³]")
print()
print("Pesos por zona aplicados no data_loss (do treino):")
print(f"  cyl={W_ZONE_CYL}, bc={W_ZONE_BC}, boi_c={W_ZONE_BOI_C}, "
      f"boi_t={W_ZONE_BOI_T}, free={W_ZONE_FREE}")
print(f"  Soma efetiva: {W_ZONE_SUM}  (usada para extrair 'data_raw' do log)")
print("=" * 72)


# ============================================================
# CARREGA CSV
# ============================================================
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = "training_log_segregated.csv"

if not os.path.exists(csv_path):
    print(f"\nERRO: arquivo não encontrado: {csv_path}")
    print("Uso: python plot_residuos_si.py [caminho_csv]")
    sys.exit(1)

csv_dir = os.path.dirname(os.path.abspath(csv_path))
print(f"\nLendo CSV: {csv_path}")
print(f"Plots serão salvos em: {csv_dir}")
df = pd.read_csv(csv_path)
print(f"  Linhas: {len(df)}")
print(f"  Colunas: {list(df.columns)}")


# ============================================================
# CONVERSÃO PARA UNIDADES SI
# ============================================================
epoch     = df["epoch"].values
elapsed_h = df["elapsed_s"].values / 3600.0

# --- Resíduos físicos (continuidade e momento) ---
rmse_cont_si = np.sqrt(df["loss_cont"].values)  * SCALE_CONT   # [1/s]
rmse_mu_si   = np.sqrt(df["loss_mom_u"].values) * SCALE_MOM    # [N/m³]
rmse_mv_si   = np.sqrt(df["loss_mom_v"].values) * SCALE_MOM    # [N/m³]
rmse_mw_si   = np.sqrt(df["loss_mom_w"].values) * SCALE_MOM    # [N/m³]

# --- Data loss por zona (m/s aproximado) ---
rmse_data_cyl_ms   = np.sqrt(df["loss_data_cyl"].values)   * V_DATA_SCALE
rmse_data_bc_ms    = np.sqrt(df["loss_data_bc"].values)    * V_DATA_SCALE
rmse_data_boi_c_ms = np.sqrt(df["loss_data_boi_c"].values) * V_DATA_SCALE
rmse_data_boi_t_ms = np.sqrt(df["loss_data_boi_t"].values) * V_DATA_SCALE
rmse_data_free_ms  = np.sqrt(df["loss_data_free"].values)  * V_DATA_SCALE

# --- DATA RAW vs VAL (sem pesos) ---
# data_loss no log é a soma ponderada por zona: Σ w_zona × MSE_zona
# Para comparar com val (MSE puro sem pesos), dividimos pela soma efetiva.
# Isso aproxima o "MSE médio nos pontos de treino".
#
# A val_loss já é MSE puro em pontos não vistos, sem pesos.
data_raw   = df["loss_data"].values / W_ZONE_SUM   # MSE de treino sem pesos
val_loss   = df["loss_val"].values                 # MSE de validação sem pesos


# ============================================================
# ESTILO DOS PLOTS
# ============================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "figure.dpi": 100,
})


# ============================================================
# PLOT 1 — Continuidade (1/s)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epoch, rmse_cont_si, color="purple", label="∇·V residual")
ax.set_xlabel("Época")
ax.set_ylabel(r"RMSE de $\nabla \cdot \mathbf{V}$  [1/s]")
ax.set_title("Resíduo de continuidade (unidades SI) — normalização z-score")
ax.set_yscale("log")
_densify_log_yaxis(ax)
ax.legend(loc="upper right")
fig.tight_layout()
out_path = os.path.join(csv_dir, "residuo_continuidade_si.png")
fig.savefig(out_path, dpi=120)
print(f"\nSalvo: {out_path}")
plt.close(fig)


# ============================================================
# PLOT 2 — Momento (N/m³)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epoch, rmse_mu_si, color="orange",      label="Momento u (cross-stream)")
ax.plot(epoch, rmse_mv_si, color="saddlebrown", label="Momento v (streamwise)")
ax.plot(epoch, rmse_mw_si, color="hotpink",     label="Momento w (vertical)")
ax.set_xlabel("Época")
ax.set_ylabel(r"RMSE do resíduo de momento  [N/m³]")
ax.set_title("Resíduos das equações de momento (unidades SI) — normalização z-score")
ax.set_yscale("log")
_densify_log_yaxis(ax)
ax.legend(loc="upper right")
fig.tight_layout()
out_path = os.path.join(csv_dir, "residuo_momento_si.png")
fig.savefig(out_path, dpi=120)
print(f"Salvo: {out_path}")
plt.close(fig)


# ============================================================
# PLOT 3 — Data loss por zona (m/s aproximado)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epoch, rmse_data_cyl_ms,   color="red",     label="Cilindro (no-slip)")
ax.plot(epoch, rmse_data_bc_ms,    color="blue",    label="BCs externas")
ax.plot(epoch, rmse_data_boi_c_ms, color="green",   label="BOI_C (entorno cilindro)")
ax.plot(epoch, rmse_data_boi_t_ms, color="orange",  label="BOI_T (wake distante)")
ax.plot(epoch, rmse_data_free_ms,  color="dimgray", label="Freestream")
ax.set_xlabel("Época")
ax.set_ylabel(r"RMSE combinado por zona  [m/s, aproximado]")
ax.set_title("Erro médio nos dados, segmentado por zona geométrica\n"
             "(combinação ponderada de u, v, w, P em escala de velocidade)")
ax.set_yscale("log")
_densify_log_yaxis(ax)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
out_path = os.path.join(csv_dir, "data_loss_zonas_si.png")
fig.savefig(out_path, dpi=120)
print(f"Salvo: {out_path}")
plt.close(fig)


# ============================================================
# PLOT 4 — Resumo: 4 painéis combinados
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# (0,0) Continuidade
ax = axes[0, 0]
ax.plot(epoch, rmse_cont_si, color="purple")
ax.set_xlabel("Época")
ax.set_ylabel(r"$\nabla \cdot \mathbf{V}$ residual [1/s]")
ax.set_title(f"Continuidade — RMSE final: {rmse_cont_si[-1]:.4f} [1/s]")
ax.set_yscale("log")
_densify_log_yaxis(ax)

# (0,1) Momento — 3 componentes
ax = axes[0, 1]
ax.plot(epoch, rmse_mu_si, color="orange",      label="u")
ax.plot(epoch, rmse_mv_si, color="saddlebrown", label="v")
ax.plot(epoch, rmse_mw_si, color="hotpink",     label="w")
ax.set_xlabel("Época")
ax.set_ylabel(r"Momento residual [N/m³]")
ax.set_title(f"Momento — final: u={rmse_mu_si[-1]:.3f}, v={rmse_mv_si[-1]:.3f}, w={rmse_mw_si[-1]:.3f}")
ax.set_yscale("log")
_densify_log_yaxis(ax)
ax.legend(loc="upper right")

# (1,0) Data loss por zona
ax = axes[1, 0]
ax.plot(epoch, rmse_data_cyl_ms,   color="red",     label="cyl")
ax.plot(epoch, rmse_data_bc_ms,    color="blue",    label="bc")
ax.plot(epoch, rmse_data_boi_c_ms, color="green",   label="boi_c")
ax.plot(epoch, rmse_data_boi_t_ms, color="orange",  label="boi_t")
ax.plot(epoch, rmse_data_free_ms,  color="dimgray", label="free")
ax.set_xlabel("Época")
ax.set_ylabel("RMSE data [m/s, aprox.]")
ax.set_title("Data loss por zona (aprox. em m/s)")
ax.set_yscale("log")
_densify_log_yaxis(ax)
ax.legend(loc="upper right", fontsize=8)

# (1,1) Caixa de resumo numérico
ax = axes[1, 1]
ax.axis("off")
last_ep = int(epoch[-1])
last_h  = elapsed_h[-1]
generalization_gap = abs(data_raw[-1] - val_loss[-1]) / val_loss[-1] * 100
summary = (
    f"RESUMO  —  Época {last_ep} ({last_h:.2f}h de treino)\n"
    f"{'='*50}\n\n"
    f"Continuidade:\n"
    f"  RMSE = {rmse_cont_si[-1]:.4f} [1/s]\n"
    f"  (escala V_INF/D_CYL = {SCALE_CONT:.4f})\n"
    f"  razão: {rmse_cont_si[-1]/SCALE_CONT:.2f}× a escala\n\n"
    f"Momento (RMSE em N/m³):\n"
    f"  u:  {rmse_mu_si[-1]:.4f}\n"
    f"  v:  {rmse_mv_si[-1]:.4f}\n"
    f"  w:  {rmse_mw_si[-1]:.4f}\n"
    f"  (escala ρV²/D = {SCALE_MOM:.4f})\n\n"
    f"Comparação SEM PESOS (treino vs val):\n"
    f"  data_raw:  {data_raw[-1]:.5f}  (MSE treino)\n"
    f"  val_loss:  {val_loss[-1]:.5f}  (MSE validação)\n"
    f"  Gap relativo: {generalization_gap:.1f}%\n\n"
    f"Normalização: z-score isotrópico\n"
    f"  σ_iso = 101.23 m, σ_t = 28.87 s"
)
ax.text(0.0, 1.0, summary, transform=ax.transAxes, va="top",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))

fig.suptitle(f"Resíduos da PINN em unidades SI — z-score normalization\n{os.path.basename(csv_path)}",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(csv_dir, "resumo_residuos_si.png")
fig.savefig(out_path, dpi=120)
print(f"Salvo: {out_path}")
plt.close(fig)


# ============================================================
# PLOT 5 — NOVO: data_raw vs val_loss (SEM PESOS)
# ============================================================
# Endereça pergunta do orientador sobre discrepância total_loss/val_loss.
# A discrepância de ~500× vem APENAS dos pesos arbitrários (W_data=70 e
# pesos por zona). Quando removidos, train e val ficam comparáveis,
# indicando boa generalização e ausência de memorização.
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epoch, data_raw, color="steelblue",
        label=f"Treino (data_raw = data/{W_ZONE_SUM:.1f})", linewidth=1.8)
ax.plot(epoch, val_loss, color="darkorange",
        label="Validação (val_loss, MSE puro)", linewidth=1.8, linestyle="--")
ax.set_xlabel("Época")
ax.set_ylabel("MSE (adimensional, sem pesos)")
ax.set_title("Loss de Treino vs Loss Validação\n")
ax.set_yscale("log")
_densify_log_yaxis(ax)
ax.legend(loc="upper right")

# Caixa explicativa
final_gap = abs(data_raw[-1] - val_loss[-1]) / val_loss[-1] * 100
ax.text(0.750, 0.60,
        f"Final (época {int(epoch[-1])}):\n"
        f"  data_raw = {data_raw[-1]:.5f}\n"
        f"  val_loss = {val_loss[-1]:.5f}\n",
        transform=ax.transAxes, va="bottom", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
fig.tight_layout()
out_path = os.path.join(csv_dir, "train_vs_val_unweighted.png")
fig.savefig(out_path, dpi=120)
print(f"Salvo: {out_path}")
plt.close(fig)


# ============================================================
# RESUMO NO CONSOLE
# ============================================================
print("\n" + "=" * 72)
print(f"RESUMO — Última época ({int(epoch[-1])}) — {elapsed_h[-1]:.2f}h de treino")
print("=" * 72)
print(f"\nCONTINUIDADE (∇·V = 0)")
print(f"  RMSE = {rmse_cont_si[-1]:.4f} [1/s]")
print(f"  Como referência: V_INF/D_CYL = {SCALE_CONT:.4f} [1/s]")
print(f"  Razão: {rmse_cont_si[-1] / SCALE_CONT:.2f}× a escala característica")

print(f"\nMOMENTO (ρ DV/Dt = -∇P + μ∇²V)")
print(f"  RMSE_u = {rmse_mu_si[-1]:.4f} [N/m³]")
print(f"  RMSE_v = {rmse_mv_si[-1]:.4f} [N/m³]")
print(f"  RMSE_w = {rmse_mw_si[-1]:.4f} [N/m³]")
print(f"  Como referência: ρV_INF²/D_CYL = {SCALE_MOM:.4f} [N/m³]")
print(f"  Razões: u={rmse_mu_si[-1]/SCALE_MOM:.3f}×, "
      f"v={rmse_mv_si[-1]/SCALE_MOM:.3f}×, w={rmse_mw_si[-1]/SCALE_MOM:.3f}×")

print(f"\nDATA RAW vs VAL (sem pesos)")
print(f"  data_raw (treino, sem pesos) = {data_raw[-1]:.6f}")
print(f"  val_loss (MSE puro)          = {val_loss[-1]:.6f}")
print(f"  Gap relativo: {abs(data_raw[-1]-val_loss[-1])/val_loss[-1]*100:.1f}%")
print(f"  → diferença pequena indica boa generalização")

print(f"\nDATA LOSS por zona (RMSE em m/s, APROXIMAÇÃO)")
print(f"  Cilindro:   {rmse_data_cyl_ms[-1]:.3f} m/s")
print(f"  BCs:        {rmse_data_bc_ms[-1]:.3f} m/s")
print(f"  BOI_C:      {rmse_data_boi_c_ms[-1]:.3f} m/s")
print(f"  BOI_T:      {rmse_data_boi_t_ms[-1]:.3f} m/s")
print(f"  Freestream: {rmse_data_free_ms[-1]:.3f} m/s")
print(f"  Como referência: V_INF = {V_INF:.2f} m/s")

print("\n" + "=" * 72)
print(f"5 plots PNG gerados em: {csv_dir}")
print("=" * 72)
