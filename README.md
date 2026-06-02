# PINN 3D Transiente para Escoamento ao Redor de Cilindro Aquecido

> Trabalho de Conclusão de Curso - Instituto Mauá de Tecnologia (IMT)
> Validação metodológica de Physics-Informed Neural Networks (PINN) contra ground-truth de CFD para escoamento externo 3D transiente.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Resultados Principais](#resultados-principais)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Configuração CFD](#configuração-cfd)
- [Arquitetura da PINN](#arquitetura-da-pinn)
- [Como Reproduzir](#como-reproduzir)
- [Acesso aos Dados](#acesso-aos-dados)
- [Limitações Conhecidas](#limitações-conhecidas)
- [Referências](#referências)

---

## Visão Geral

Este repositório contém a implementação completa e os resultados da validação metodológica de uma Physics-Informed Neural Network (PINN) treinada para reproduzir o escoamento 3D transiente ao redor de um cilindro vertical em um domínio do tipo túnel de vento. O trabalho usa simulação CFD de alta resolução como ground-truth e a rede aprende as variáveis primitivas `(u, v, w, P)` em função das coordenadas espaço-temporais `(x, y, z, t)`.

**Caso de aplicação.** A geometria é inspirada em estruturas alvo de engenharia eólica e astronômica (cilindro vertical em escoamento atmosférico). A escolha proposital de regime laminar — não realista para o número de Reynolds do problema — permite isolar e validar a pipeline metodológica sem interferência de modelos de turbulência, priorizando reprodutibilidade arquitetural sobre fidelidade fenomenológica.

**Configuração física**

| Parâmetro | Valor |
|---|---|
| Domínio computacional | 280 × 700 × 190 m |
| Diâmetro do cilindro D | 54 m |
| Altura do cilindro H | 65 m |
| Posição do cilindro (base) | (140, 200, 0) m |
| Velocidade do escoamento V∞ | 17 m/s (+Y) |
| Densidade ρ | 1.225 kg/m³ |
| Viscosidade μ | 1.789×10⁻⁵ Pa·s |
| Reynolds Re = ρV∞D/μ | ≈ 6.3×10⁸ |
| Regime numérico adotado | Laminar (escolha metodológica) |
| Tempo simulado | 100 s, Δt = 0.1 s, 1000 snapshots |

---

## Resultados Principais

A versão final da PINN (pasta `pinn_d_banca_sem1`) atinge correlação `r > 0.90` em todas as componentes de velocidade e `r = 0.993` em pressão contra o último snapshot do CFD (t = 100s), avaliada sobre 213.146 pontos de um domínio 3D.

### Métricas finais — modelo `pinn_d_banca_sem1` (1000 épocas, z-score isotrópico)

Avaliação em t = 100s, n = 213.146 pontos:

| Var | MAE | RMSE | Bias | r | R² | Acc(20%) |
|---|---|---|---|---|---|---|
| u | 0.62 m/s | 1.29 m/s | +0.03 | 0.946 | 0.895 | 96.3% |
| v | 1.06 m/s | 1.77 m/s | −0.02 | 0.973 | 0.945 | 91.4% |
| w | 0.47 m/s | 0.97 m/s | −0.00 | 0.902 | 0.812 | 97.9% |
||V|| | 1.10 m/s | 1.83 m/s | −0.19 | 0.951 | 0.892 | 91.1% |
| P | 5.05 Pa | 9.18 Pa | +0.75 | 0.993 | 0.986 | 98.7% |

A coluna **Acc(20%)** representa o percentual de pontos com erro absoluto inferior a 20% da escala física do problema (V∞ = 17 m/s para velocidades; ½ρV∞² ≈ 177 Pa para pressão).

### Evolução metodológica

O repositório preserva duas versões do trabalho para fins de comparação metodológica:

| Versão | Normalização | Resultado |
|---|---|---|
| `pinn_c_pre_banca` | Min-max anisotrópico (L_REF=700m, V_REF=V_max=28.8) | r(u)≈0, r(v)=0.33, r(P)=0.90, **bias em v = −9.5 m/s** |
| `pinn_d_banca_sem1` | **z-score isotrópico + esquema híbrido nas saídas** | r(u)=0.946, r(v)=0.973, r(P)=0.993, bias praticamente nulo |

A migração metodológica resolveu três patologias críticas presentes na versão `pinn_c_pre_banca`:

1. **Modo trivial em u e w** — a versão antiga colapsava para `u ≈ 0` e `w ≈ 0` por todo o domínio (r ≈ 0). A nova versão captura essas componentes com r > 0.90.
2. **Bias sistemático em v** — a versão antiga subestimava v sistematicamente em ~9 m/s (mais da metade da escala física). O bias na nova versão é < 0.03 m/s.
3. **Cilindro distorcido no espaço normalizado** — a normalização anisotrópica deformava o cilindro circular em uma elipse de razão 3.16:1 no espaço de entrada da rede, comprometendo a `HardConstraintLayer`. O z-score isotrópico preserva a geometria circular.

### Verificação de aprendizado real (não memorização)

A documentação completa da verificação está no PDF `Material Suplementar Artigo/Materiais Suplementares.pdf`. Evidências centrais:

- **Generalização temporal uniforme:** 99 snapshots foram reservados para validação (nunca vistos no treino). O erro nesses snapshots é estatisticamente idêntico ao do conjunto de treino, indicando que a rede não decorou pares específicos.
- **Loss de validação monotonicamente decrescente:** sem U-shape, sem divergência entre train e val loss ao longo de 1000 épocas — incompatível com overfitting/memorização.
- **Resíduos das PDEs em pontos de collocation:** a continuidade e as três equações de momento são satisfeitas em pontos aleatórios sem rótulos. Como esses pontos não têm ground-truth para decorar, satisfazê-los exige aprender estrutura física.
- **Hierarquia espacial dos erros consistente com a física:** os maiores erros estão exatamente onde a física é mais complexa (zona do cilindro, ponto de estagnação, esteira próxima), reduzindo monotonicamente para regiões de escoamento livre.

---

## Estrutura do Repositório

```
TCC/
├── README.md                                  ← este arquivo
│
├── CFD/                                       ← simulação CFD (ground-truth)
│   ├── SIMULACAO_CFD.txt                      ← documentação completa do setup Fluent
│   ├── Dados/
│   │   └── link zip dados.txt                 ← link para dataset (Google Drive, ~8.4 GB)
│   ├── Prints Ansys/                          ← screenshots da malha e zoneamento
│   │   ├── 0_Zonas de Detalhamento.jpg
│   │   ├── 1_Malha Superficial.jpg
│   │   ├── 2_Malha Hexagonal Isométrica.jpg
│   │   └── 3_Malha Hexagonal.jpg
│   ├── escoamento vel e pressao 4k.mp4        ← vídeo do escoamento (CFD)
│   └── t100 escoamento final 4k.png           ← snapshot final do CFD
│
├── pinn_codes/                                ← código-fonte da PINN
│   ├── pinn_c_pre_banca/
│   │   └── PINN - Artigo.ipynb                ← versão antiga (min-max anisotrópico)
│   └── pinn_d_banca_sem1/
│       └── pinn_normalizada.py                ← versão final (z-score isotrópico)
│
├── pinn_results/                              ← saídas de inferência
│   ├── pinn_c_pre_banca/                      ← resultados versão antiga
│   │   ├── loss_history.png
│   │   ├── pinn_vs_cfd_scatter.png
│   │   ├── pinn_vs_cfd_errors_hist.png
│   │   ├── pinn_vs_cfd_xz_y*_t*s.png          ← slices CFD vs PINN
│   │   ├── residuo_continuidade_si.png
│   │   ├── residuo_momento_si.png
│   │   ├── resumo_residuos_si.png
│   │   ├── data_loss_zonas_si.png
│   │   ├── errors_by_slice.csv
│   │   ├── final_metrics_lastsnap.csv
│   │   └── training_log_segregated.csv
│   └── pinn_d_banca_sem1/                     ← resultados versão final
│       ├── histograms/
│       ├── loss_history/
│       ├── scatter/
│       ├── slices/
│       ├── summary/
│       │   ├── errors_by_slice.csv
│       │   └── final_metrics_lastsnap.csv
│       └── training_log_segregated.csv
│
├── scripts/                                   ← ferramentas auxiliares
│   ├── Cálculo Média e Desvio Padrão Features/
│   │   ├── calculo_media_dp_features.py       ← gera normalization_stats.json
│   │   └── normalization_stats.json           ← μ e σ dos 213M pontos
│   ├── Classificação de Pontos no Domínio/
│   │   ├── domain_points_classification.mlx   ← MATLAB Live Script
│   │   ├── domain_points_classification.pdf   ← exportação para revisão
│   │   ├── fig1_camadas_adjacentes.png
│   │   ├── fig2_BOIs_volumetricos.png
│   │   ├── fig3_freestream.png
│   │   ├── fig4_visao_global.png
│   │   └── fig5_contagem_por_BC.png
│   ├── Inferência e Plots Estatísticos/
│   │   └── inferencia_new_norm.py             ← script standalone de pós-processamento
│   └── Resíduos [SI]/
│       ├── plot_residuos_si.py                ← converte resíduos adimensionais → SI
│       ├── residuo_continuidade_si.png
│       ├── residuo_momento_si.png
│       ├── resumo_residuos_si.png
│       └── data_loss_zonas_si.png
│
└── Material Suplementar Artigo/
    └── Materiais Suplementares.pdf            ← apêndices, derivações, verificações
```

---

## Configuração CFD

A simulação CFD é o ground-truth contra o qual a PINN é treinada e validada. A configuração completa está documentada em `CFD/SIMULACAO_CFD.txt`. Resumo dos pontos críticos:

**Software:** ANSYS Fluent 2026 R1 (Student License — 213.146 cells = 20% do limite de 1.048.576).

**Malha:** Watertight Workflow do Fluent Meshing, células poliédricas. Local sizing por face (1m no cilindro, 4–8m nas fronteiras externas), refinamento volumétrico via Bodies of Influence (4.5m no entorno do cilindro, 4.8m na esteira). Min Orthogonal Quality = 0.37, reprodutibilidade < 0.3% entre runs.

**Solver:** Pressure-Based, Transient, Laminar, sem energia. Time step 0.1s, 1000 passos, 20 iterações máximas por passo.

**Boundary Conditions:**

| BC | Tipo | Configuração |
|---|---|---|
| inlet (y = 0) | velocity-inlet | 17 m/s na direção +Y |
| outlet (y = 700) | pressure-outlet | Gauge 0 Pa |
| cylinder_wall, cylinder_top | wall | Stationary, no-slip |
| ground (z = 0) | wall | Stationary, no-slip |
| lateral_min, lateral_max, top | wall | Moving Translational, 17 m/s em +Y |

Os três últimos slip walls (Moving Translational) são essenciais para representar o freestream sem desenvolver camada limite artificial nas fronteiras laterais e superior.

**Exportação dos snapshots:** comando TUI executado a cada time step:
```
/file/export/ascii "CSVs/timestep_%t.csv" no domain () no yes pressure x-velocity y-velocity z-velocity quit
```
Geração de 1000 arquivos `.csv`, totalizando aproximadamente 8.4 GB.

**Classificação dos pontos de malha** (script MATLAB em `scripts/Classificação de Pontos no Domínio/`):

| Categoria | Cells | Densidade vs freestream |
|---|---|---|
| Adjacente ao cilindro | 40.483 (19.0%) | — |
| Interior do BOI Cilindro | 44.638 (20.9%) | 15.8× |
| Interior do BOI Triangle (esteira) | 20.379 (9.6%) | 2.5× |
| Adjacente às BCs externas | 23.961 (11.2%) | — |
| Freestream | 83.685 (39.3%) | 1× (base) |

---

## Arquitetura da PINN

**Modelo:** MLP residual com 6 blocos de 256 neurônios, ativação tanh, 331.268 parâmetros totais. Entrada `(x, y, z, t)` ∈ ℝ⁴, saída `(u, v, w, P)` ∈ ℝ⁴.

**Normalização (versão `pinn_d_banca_sem1`).** Esquema híbrido cuidadosamente calibrado:

| Quantidade | Tipo | Parâmetros |
|---|---|---|
| Coords x, y, z | z-score isotrópico | μ por eixo, σ_iso = 101.234 m comum a todos |
| Tempo t | z-score | μ_t = 50.05 s, σ_t = 28.867 s |
| Saídas u, w | min-max simétrico | V_REF = 5 m/s |
| Saída v | z-score | μ_v = 14.32 m/s, σ_v = 7.40 m/s |
| Saída P | z-score | μ_P = −32.79 Pa, σ_P = 98.41 Pa |

A escolha de σ_iso espacialmente isotrópico é fundamental: ela preserva a geometria circular do cilindro no espaço normalizado, o que é requisito para que a `HardConstraintLayer` opere corretamente sobre as distâncias geométricas. O esquema híbrido nas saídas decorre da assimetria estatística de `v` (com média não-nula devido ao escoamento ser predominantemente em +Y) e de `P` (gauge com offset).

**HardConstraintLayer.** Impõe as condições de contorno arquiteturalmente via ansatz multiplicativo, eliminando o termo de boundary loss e garantindo BCs exatas. Para `u, w` (sem offset) a forma é `u_pred · D_velocity(x, y, z)`; para `v, P` (com offset) o ansatz opera em coordenadas físicas antes de renormalizar, evitando que `v_norm = 0` seja interpretado como `v_phys = μ_v ≠ 0`.

**Loss function.** Combinação ponderada de:
- Data loss (MSE em pontos do CFD, peso 70, com sub-pesos por zona)
- Resíduo da continuidade `∇ · V = 0` (peso 1)
- Resíduos das três equações de momento de Navier-Stokes (peso 1 cada)

**Treino.** 1000 épocas, learning rate constante 3×10⁻⁵, batch size 48.000, gradient clipping com threshold 1.0. Adam optimizer. 43.2 milhões de pontos de treino (900 snapshots) + 4.7 milhões de validação (99 snapshots retidos). Tempo total de treino: aproximadamente 5h50min em 1 GPU NVIDIA A40.

---

## Como Reproduzir

### Pré-requisitos

```
Python 3.11
TensorFlow 2.21
NumPy, Pandas, Matplotlib, SciPy
Acesso a GPU (recomendado A40 ou equivalente para o treino completo)
```

### Pipeline completo

**1. Obtenção do dataset.** Veja a seção [Acesso aos Dados](#acesso-aos-dados) abaixo.

**2. Geração das estatísticas de normalização.** Executar uma vez sobre o dataset completo:
```bash
cd scripts/Cálculo\ Média\ e\ Desvio\ Padrão\ Features/
python calculo_media_dp_features.py
```
Saída: `normalization_stats.json` com μ e σ de todas as features sobre os 213 milhões de pontos. Este arquivo já está disponível no repositório.

**3. Treino da PINN.**
```bash
cd pinn_codes/pinn_d_banca_sem1/
python pinn_normalizada.py
```
Configurações relevantes no topo do script:
- `TEST_MODE = True`: usa 40 snapshots subamostrados (treino rápido, ~30min em A40, para sanity-check)
- `TEST_MODE = False`: usa os 999 snapshots completos (~6h em A40, resultado de produção)

Saídas geradas durante o treino:
- `training_log_segregated.csv` — log de loss por época
- `pinn_checkpoints_segregated/pinn_segregated_ep000XXX.weights.h5` — checkpoints a cada 50 épocas
- `pinn_checkpoints_segregated/pinn_best_segregated.weights.h5` — melhor val_loss

**4. Inferência e plots de pós-processamento.**
```bash
cd scripts/Inferência\ e\ Plots\ Estatísticos/
python inferencia_new_norm.py
```
Gera scatter plots, histogramas de erro, slices 2D do domínio (CFD vs PINN) em múltiplos tempos físicos, e tabela quantitativa de métricas. Por padrão usa o checkpoint mais recente; pode-se especificar via `--checkpoint`.

**5. Plots de resíduos físicos em unidades SI.**
```bash
cd scripts/Resíduos\ \[SI\]/
python plot_residuos_si.py /caminho/para/training_log_segregated.csv
```
Converte os resíduos adimensionais armazenados no log para unidades físicas (`1/s` para continuidade, `N/m³` para momento), permitindo interpretação dimensional dos termos da PDE.

---

## Acesso aos Dados

O dataset CFD completo (1000 snapshots × 213.146 cells, ~8.4 GB compactado) está disponível via Google Drive devido ao tamanho. O link público está em `CFD/Dados/link zip dados.txt`. Cada arquivo tem o nome `timestep-NNNN.csv` com as colunas:

```
cellnumber, x-coordinate, y-coordinate, z-coordinate, pressure, x-velocity, y-velocity, z-velocity
```

Após download, descompactar para `/caminho/local/CSVs/` e ajustar a variável `SNAPSHOT_DIR` no script de treino e nos scripts de inferência.

---

## Limitações Conhecidas

A documentação honesta das limitações é parte central da metodologia deste trabalho. As principais ressalvas:

**1. Regime físico simplificado.** A simulação CFD foi rodada em regime laminar mesmo com Re ≈ 6.3×10⁸ — fisicamente turbulento. Esta escolha foi metodológica: priorizar reprodutibilidade da pipeline PINN sobre fidelidade fenomenológica. Como consequência, o resultado tende a Euler invíscido e a esteira de von Kármán característica não é capturada fielmente. Para aplicações reais, seria necessário usar LES ou DES.

**2. Domínio downstream curto.** A esteira tem apenas ~9D de comprimento disponível (cilindro em y = 200m, outlet em y = 700m). Para von Kármán plenamente desenvolvida recomenda-se 15–20D.

**3. Sem inflation layers.** Para problemas turbulentos reais, faltaria malha estruturada nas primeiras camadas das paredes para resolver y⁺.

**4. Estudo de independência de malha não formalizado.** A escolha da malha v5 (213k cells) foi baseada em iteração informal entre 6 versões. Para publicação científica formal seria necessário gerar três malhas em fator √2 de refinamento e comparar grandezas integrais (C_d, C_l, St) — verificação clássica de convergência GCI.

**5. Treino com V∞ único.** A PINN foi treinada com um único valor de V∞ = 17 m/s. Para usar como surrogate paramétrico (variando velocidade), seria necessário treinar em múltiplos valores ou tornar V∞ um input adicional da rede.

**6. Artefato numérico na aresta superior do cilindro.** A função `smooth_max` da `HardConstraintLayer` não vai estritamente a zero na curva de medida zero correspondente à aresta circular do topo do cilindro (interseção entre superfície lateral e topo). O resultado é uma pequena região com BC imperfeita, visualmente quase imperceptível e sem impacto mensurável no aprendizado. Documentado em detalhe no material suplementar.

**7. Validação fenomenológica limitada.** As métricas reportadas comparam a PINN contra o CFD, não contra dados experimentais. Por construção, a PINN herda todos os erros e simplificações do CFD usado como ground-truth.

---

## Referências

**Software**
- ANSYS Fluent 2026 R1
- TensorFlow 2.21
- ANSYS DesignModeler e Fluent Meshing (Watertight Workflow)

**Hardware**
- Cluster Mauá HPC, nó gn01 (3× NVIDIA A40 48GB)
- Workstation local: Intel i5-11400H, 6 cores físicos, MPI Intel

**Documentação técnica complementar**
- `CFD/SIMULACAO_CFD.txt` — setup completo do Fluent (geometria, malha, BCs, solver, exportação)
- `Material Suplementar Artigo/Materiais Suplementares.pdf` — derivações matemáticas, verificação de aprendizado, sanity checks
- `scripts/Classificação de Pontos no Domínio/domain_points_classification.pdf` — taxonomia geométrica dos pontos

---

## Citação

Se este trabalho for útil para sua pesquisa, considere citá-lo. A referência bibliográfica completa será adicionada após a defesa.

## Licença

Este repositório é parte de um Trabalho de Conclusão de Curso. O uso do código e dos dados está sujeito às políticas da instituição e aos termos da licença ANSYS Student usada para a geração do CFD.

## Contato

Para questões técnicas ou colaborações, abrir uma issue neste repositório.
