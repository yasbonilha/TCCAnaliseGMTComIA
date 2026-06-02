"""
Plot de Resíduos em Unidades Físicas (SI) — atualizado para normalização z-score
=================================================================================

Lê training_log_segregated.csv e converte os resíduos adimensionais
(divididos por SCALE_CONT e SCALE_MOM no treino) para suas unidades
físicas originais (1/s para continuidade, N/m³ para momento).

ATUALIZAÇÃO (normalização nova):
  - Coordenadas espaciais: z-score isotrópico (σ_iso ≈ 101.23 m)
  - Tempo: z-score (σ_t ≈ 28.87 s)
  - Saídas: esquema híbrido (V_REF=5 m/s para u,w; z-score para v,P)
  - SCALE_CONT e SCALE_MOM no treino usam V_INF e D_CYL (escalas físicas
    do problema, não escalas da normalização da rede)
  - SCALE_CONT mudou de 0.0411 → 0.3148 (~7.6× maior)
  - SCALE_MOM  mudou de 1.452  → 6.556  (~4.5× maior)

Uso:
    python plot_residuos_si.py [path_to_csv]
    
    Se path_to_csv não fornecido, usa 'training_log_segregated.csv'.

Saída:
    Cria plots PNG no mesmo diretório do CSV:
    - residuo_continuidade_si.png   (1/s)
    - residuo_momento_si.png        (N/m³, 3 componentes)
    - data_loss_zonas_si.png        (m/s, por zona, aproximado)
    - resumo_residuos_si.png        (4 painéis combinados)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO — DEVE BATER COM O SCRIPT DE TREINO
# ============================================================
# Constantes físicas do problema
V_INF   = 17.0        # m/s, velocidade do escoamento livre
RHO_INF = 1.225       # kg/m³, densidade do ar
D_CYL   = 54.0        # m, diâmetro do cilindro (escala física)

# Escalas físicas usadas para adimensionalizar os resíduos no treino.
# IMPORTANTE: no script atual, SCALE_CONT e SCALE_MOM usam V_INF e D_CYL
# (escalas FÍSICAS do problema), NÃO as escalas da normalização da rede.
# Isso é metodologicamente correto: os resíduos das PDEs devem ser
# interpretados na escala em que a física opera, não na escala arbitrária
# da rede neural.
SCALE_CONT = V_INF / D_CYL                   # ≈ 0.3148 [1/s]
SCALE_MOM  = RHO_INF * V_INF**2 / D_CYL      # ≈ 6.556  [N/m³]

# Escalas da normalização da rede (z-score / híbrido)
# Estas são usadas APENAS para o data_loss por zona, como aproximação grosseira.
# Para análise rigorosa por componente, usar o script de inferência.
V_REF   = 5.0          # min-max simétrico para u, w
V_MEAN  = 14.322       # z-score para v: média
V_DP    = 7.400        # z-score para v: desvio padrão
P_MEAN  = -32.789      # z-score para P: média
P_DP    = 98.412       # z-score para P: desvio padrão

# Escala de referência para o data_loss combinado (aproximação)
# Usamos V_INF como escala "natural" do problema, NÃO V_REF=5 que é só
# uma escolha de normalização da rede.
V_DATA_SCALE = V_INF   # m/s, usado como aproximação para data_loss combinado


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
print("Mudança vs normalização antiga:")
print(f"  SCALE_CONT: 0.0411 → {SCALE_CONT:.4f}  ({SCALE_CONT/0.0411:.1f}× maior)")
print(f"  SCALE_MOM:  1.452  → {SCALE_MOM:.4f}  ({SCALE_MOM/1.452:.1f}× maior)")
print()
print("Escala de referência para data_loss aproximado em m/s:")
print(f"  V_DATA_SCALE = V_INF = {V_DATA_SCALE:.2f} m/s")
print("  (NOTA: aproximação — para rigor por componente, use script de inferência)")
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
# O CSV armazena MSE adimensional (resíduo normalizado por SCALE_CONT ou SCALE_MOM).
# Para reportar em unidades físicas:
#   RMSE_SI = sqrt(MSE_normalizado) × ESCALA

epoch     = df["epoch"].values
elapsed_h = df["elapsed_s"].values / 3600.0

# --- Resíduos físicos (continuidade e momento) ---
rmse_cont_si = np.sqrt(df["loss_cont"].values)  * SCALE_CONT   # [1/s]
rmse_mu_si   = np.sqrt(df["loss_mom_u"].values) * SCALE_MOM    # [N/m³]
rmse_mv_si   = np.sqrt(df["loss_mom_v"].values) * SCALE_MOM    # [N/m³]
rmse_mw_si   = np.sqrt(df["loss_mom_w"].values) * SCALE_MOM    # [N/m³]

# --- Data loss por zona (m/s aproximado) ---
# ATENÇÃO METODOLÓGICA: o data_loss por zona no CSV é a média do MSE entre
# as 4 saídas (u, v, w, P), cada uma com sua própria escala de normalização:
#   - u, w: min-max V_REF=5 m/s  → desvio típico em m/s ≈ √(MSE)×5
#   - v:    z-score σ=7.4 m/s    → desvio típico em m/s ≈ √(MSE)×7.4
#   - P:    z-score σ=98.4 Pa    → desvio típico em Pa  ≈ √(MSE)×98.4
# Como o data_loss MISTURA isso, multiplicar por uma única escala é
# APROXIMAÇÃO ilustrativa. Usamos V_INF=17 m/s como referência "natural"
# do problema. Para valores rigorosos, ver o script de inferência.
rmse_data_cyl_ms   = np.sqrt(df["loss_data_cyl"].values)   * V_DATA_SCALE
rmse_data_bc_ms    = np.sqrt(df["loss_data_bc"].values)    * V_DATA_SCALE
rmse_data_boi_c_ms = np.sqrt(df["loss_data_boi_c"].values) * V_DATA_SCALE
rmse_data_boi_t_ms = np.sqrt(df["loss_data_boi_t"].values) * V_DATA_SCALE
rmse_data_free_ms  = np.sqrt(df["loss_data_free"].values)  * V_DATA_SCALE


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
ax.legend(loc="upper right")
ax.text(0.02, 0.95,
        f"Escala característica: V_INF/D_CYL = {SCALE_CONT:.4f} [1/s]\n"
        f"Final ({len(df)} ép): {rmse_cont_si[-1]:.4f} [1/s]\n"
        f"Razão: {rmse_cont_si[-1] / SCALE_CONT:.2f}× a escala",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
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
ax.legend(loc="upper right")
ax.text(0.02, 0.95,
        f"Escala característica: ρV_INF²/D_CYL = {SCALE_MOM:.4f} [N/m³]\n"
        f"Final ({len(df)} ép):\n"
        f"  u: {rmse_mu_si[-1]:.4f} | v: {rmse_mv_si[-1]:.4f} | w: {rmse_mw_si[-1]:.4f} [N/m³]",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
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
ax.legend(loc="upper right", fontsize=9)
ax.text(0.02, 0.05,
        f"Escala de referência: V_INF = {V_DATA_SCALE:.2f} m/s\n"
        f"NOTA: data_loss combina u, v, w, P normalizados com escalas diferentes.\n"
        f"Esta conversão é APROXIMAÇÃO. Para RMSE por componente,\n"
        f"use o script de inferência (separa u, v, w, P explicitamente).",
        transform=ax.transAxes, va="bottom", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
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

# (0,1) Momento — 3 componentes
ax = axes[0, 1]
ax.plot(epoch, rmse_mu_si, color="orange",      label="u")
ax.plot(epoch, rmse_mv_si, color="saddlebrown", label="v")
ax.plot(epoch, rmse_mw_si, color="hotpink",     label="w")
ax.set_xlabel("Época")
ax.set_ylabel(r"Momento residual [N/m³]")
ax.set_title(f"Momento — final: u={rmse_mu_si[-1]:.3f}, v={rmse_mv_si[-1]:.3f}, w={rmse_mw_si[-1]:.3f}")
ax.set_yscale("log")
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
ax.legend(loc="upper right", fontsize=8)

# (1,1) Caixa de resumo numérico
ax = axes[1, 1]
ax.axis("off")
last_ep = int(epoch[-1])
last_h  = elapsed_h[-1]
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
    f"Data loss por zona (m/s, aprox.):\n"
    f"  cilindro:   {rmse_data_cyl_ms[-1]:.3f}\n"
    f"  BCs:        {rmse_data_bc_ms[-1]:.3f}\n"
    f"  boi_c:      {rmse_data_boi_c_ms[-1]:.3f}\n"
    f"  boi_t:      {rmse_data_boi_t_ms[-1]:.3f}\n"
    f"  freestream: {rmse_data_free_ms[-1]:.3f}\n"
    f"  (escala V_INF = {V_DATA_SCALE:.1f} m/s)\n\n"
    f"Normalização: z-score isotrópico\n"
    f"  σ_iso = 101.23 m (espacial)\n"
    f"  σ_t   = 28.87 s\n"
    f"  Esquema híbrido nas saídas"
)
ax.text(0.0, 1.0, summary, transform=ax.transAxes, va="top",
        family="monospace", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))

fig.suptitle(f"Resíduos da PINN em unidades SI — z-score normalization\n{os.path.basename(csv_path)}",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(csv_dir, "resumo_residuos_si.png")
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
print(f"  (i.e., a divergência média é {rmse_cont_si[-1]/SCALE_CONT*100:.1f}% da escala física)")

print(f"\nMOMENTO (ρ DV/Dt = -∇P + μ∇²V)")
print(f"  RMSE_u = {rmse_mu_si[-1]:.4f} [N/m³]")
print(f"  RMSE_v = {rmse_mv_si[-1]:.4f} [N/m³]")
print(f"  RMSE_w = {rmse_mw_si[-1]:.4f} [N/m³]")
print(f"  Como referência: ρV_INF²/D_CYL = {SCALE_MOM:.4f} [N/m³]")
print(f"  Razões: u={rmse_mu_si[-1]/SCALE_MOM:.3f}×, "
      f"v={rmse_mv_si[-1]/SCALE_MOM:.3f}×, w={rmse_mw_si[-1]/SCALE_MOM:.3f}×")

print(f"\nDATA LOSS por zona (RMSE em m/s, APROXIMAÇÃO)")
print(f"  Cilindro:   {rmse_data_cyl_ms[-1]:.3f} m/s")
print(f"  BCs:        {rmse_data_bc_ms[-1]:.3f} m/s")
print(f"  BOI_C:      {rmse_data_boi_c_ms[-1]:.3f} m/s")
print(f"  BOI_T:      {rmse_data_boi_t_ms[-1]:.3f} m/s")
print(f"  Freestream: {rmse_data_free_ms[-1]:.3f} m/s")
print(f"  Como referência: V_INF = {V_INF:.2f} m/s")
print(f"  (NOTA: data_loss mistura u, v, w, P com escalas diferentes;")
print(f"   para RMSE rigoroso por componente, ver final_metrics_lastsnap.csv)")

print("\n" + "=" * 72)
print(f"4 plots PNG gerados em: {csv_dir}")
print("=" * 72)