# HANDOFF — nova rodada de CFD (teto de ν_t em 1e7)

**Para:** conversa nova sobre o CFD.
**De:** conversa da PINN k-ω SST (TCC, escoamento sobre a montanha do GMT, IMT Mauá).
**Data:** 03/10/2026.

---

## 1. O que é esta rodada

Refazer a simulação **mudando uma única coisa no solver**:

```
turb-viscosity/max-limit:  1e5  ->  1e7
```

Tudo o mais permanece: malha, passo de tempo, esquemas, modelo de turbulência,
domínio, condições de contorno. A malha **não muda** — isso é importante e está
explicado na §5.

**Escopo do trabalho:** replicar com a PINN os campos de **pressão e
velocidade** (mais k, ω e μ_t, que são o estado do modelo SST). Cálculo de
forças, momentos, Cd/Cl **está fora do escopo** — não exportar, não monitorar
para esse fim.

---

## 2. Por que mudar o teto

O teto de 1e5 **estava sendo atingido** e distorcendo o dado:

| medida | valor | fonte |
|---|---|---|
| células com ν_t no teto, t=300 s | 4,3% | job 6027 |
| idem, conjunto amostrado (999 snapshots) | 5,36% | job 6083 |
| idem, **na caixa de refino da esteira (boi_t)** | **24,1%** | job 6083 |
| fração do recorte que cai em boi_c + boi_t | 80,7% | job 6083 |

Ou seja: o teto mordia justamente nas zonas de refino, que existem porque é ali
que a física interessa.

E ele é baixo demais para a escala do problema. Com D = 54 m e U = 17 m/s, a
estimativa usual para a esteira dá

```
ν_t ~ 0,005 a 0,01 · U · D  =  4,6 a 9,2 m²/s   ->   μ_t/μ = 3e5 a 6e5
teto 1e5  =    1,46 m²/s    <- abaixo da escala física
teto 1e7  =  146,1  m²/s    <- folgado, não deve morder
```

### Previsão falseável (conferir quando o dado sair)

1. **~0% das células no teto.** Se ainda houver fração apreciável, 1e7 também
   é baixo, ou há outro problema.
2. **ν_t na esteira entre ~5 e ~10 m²/s.** Se subir para 100+ m²/s em todo
   lugar, o teto não era o problema.

### Dois riscos a vigiar durante a rodada

- **Convergência do SIMPLE.** Hoje são 13,8 iterações por passo contra teto de
  50. μ_t cem vezes maior deixa a difusão mais rígida. Se o contador encostar
  no teto, a qualidade do dado cai e o resultado não serve.
- **A esteira pode morrer.** O teto prendia 24% das células da esteira; soltar
  isso pode difundir o campo a ponto de suprimir o desprendimento de vórtices.
  **Verificação barata e dentro do escopo:** um monitor de ponto (vertex
  monitor) de pressão estática ou de velocidade-x num ponto da esteira, por
  exemplo em torno de (x, y, z) = (140, 300, 32) m. Se o sinal virar uma reta,
  o teto estava fazendo trabalho físico real e 1e7 é frouxo demais. Não é
  entregável, é só termômetro.

---

## 3. O que ACRESCENTAR na exportação

Nenhum destes itens muda a solução. São só exportação — custam disco e nada
mais. Estão em ordem de valor.

### 3.1 As 9 derivadas de velocidade + `strain-rate-mag`  — PRIORIDADE MÁXIMA

As nove ∂u_i/∂x_j (ficam na categoria **Derivatives** da lista de campos de
exportação do Fluent) e a magnitude da taxa de deformação.

**Por quê.** Hoje o pior campo da PINN é o ν_t: 0,48 dex de erro e R² = 0,670,
contra k em 0,13 dex e ω em 0,03 dex. Como ν_t = (k/ω)/denominador e k e ω
estão quase perfeitos, o erro está no **ramo** do modelo — o limitador de
cisalhamento dispara em 78% dos pontos da PINN junto à parede contra 34% no
CFD. O limitador depende de **S = √(2 S_ij S_ij)**, e **S não existe no CSV**.
Não dá para consertar o que não dá para medir.

**Segundo motivo, igualmente importante:** com as derivadas dá para avaliar o
resíduo da EDP **no próprio dado do CFD**, e aí descobrir o **piso da physics
loss** — quanto do resíduo da PINN é erro da rede e quanto é erro de
discretização do solver (SIMPLE, BDF2, MUSCL, upwind de 2ª ordem). Hoje esse
piso é desconhecido, o que enfraquece qualquer afirmação de "a física foi
imposta" na escrita do TCC.

**Se dobrar o dataset for um problema:** exportar as derivadas só em **1 de
cada 10 snapshots**. Cem instantes bastam para a estatística de S e para medir
o piso da EDP.

**Antes de rodar tudo:** confirmar os nomes exatos dos campos numa malha
pequena (o mesmo procedimento dos probes anteriores). Os nomes variam entre
versões e o exportador aceita nomes que não existem sem reclamar.

### 3.2 Os valores de contorno do inlet, tirados do próprio solver

Um dump das condições de contorno (TI, razão de viscosidade, k e ω impostos no
inlet) salvo em texto junto com os CSVs.

**Por quê.** A verificação (c) do pipeline da PINN continua inutilizável: ela
compara a BC com valores **já decaídos**. Com ω_inlet = 7419 1/s o comprimento
de decaimento é **0,26 mm**, então a 0,2 m do inlet o dado já caiu 8×, e o
teste conclui "TI ≈ 1,63%, μ_t/μ ≈ 24,8" que é artefato. Um `print` do solver
encerra a dúvida de vez e transforma inferência em medida.

### 3.3 `cell-volume` no arquivo de malha estática

Uma coluna a mais no `static_mesh.csv` (que já tem `cell-wall-distance`).

**Por quê.** Todas as estatísticas atuais são "% de células", e célula na BOI é
minúscula enquanto no campo livre é enorme. "5,36% das células no teto"
ponderado por volume pode ser 0,5% ou 30% — hoje não sabemos qual.

### 3.4 y+ e tensão cisalhante nas zonas de parede

Arquivo separado, por zona de parede (são variáveis de face, não de célula).

**Por quê.** Hoje o y+ (~1e4) é inferido invertendo k. Exportado, vira
verificação direta do tratamento de parede, que é a parte menos defensável do
dado hoje.

### 3.5 Journal, resumo do `.cas` e histórico de resíduos, guardados junto

**Por quê.** O metadata da PINN registra o que ela **supôs**; nada hoje
registra o que o solver **fez**. A banca vai perguntar.

---

## 4. O que NÃO mudar

Malha, passo de tempo (0,1 s), esquemas (SIMPLE, transiente 2ª ordem implícito,
momento Third-Order MUSCL, k e ω em 2nd Order Upwind), modelo (k-ω SST),
tratamento de parede (`wall-omega-treatment = correlation`), domínio,
velocidade de entrada, velocidade das paredes móveis, e os valores de k e ω do
inlet.

Já se muda uma variável (o teto). As exportações da §3 não mudam nenhuma.

---

## 5. O caso atual, para referência

### Geometria e escoamento

```
Domínio   X[0, 280]  Y[0, 700]  Z[0, 190] m
Cilindro  D = 54 m, H = 65 m, centro (140, 200, 0), base no chão
BOI_C     X(100,180) Y(160,240) Z(0,100)
BOI_T     V1=(100,240) V2=(180,240) V3=(140,600), Z(0,100)   [prisma triangular]
Escoamento  V_inf = 17 m/s em +Y
Ar          rho = 1,225 kg/m³   mu = 1,7894e-5   nu = 1,4607e-5 m²/s
Re_D = 6,285e7     Re_H = 7,565e7
```

**Contornos.** Inlet (y=0) velocity-inlet; outlet (y=700) pressure-outlet;
**chão e cilindro são paredes FIXAS**; **topo, x_min e x_max são paredes MÓVEIS
a 17 m/s em +Y**. Inlet e outlet não contam como parede para a distância à
parede.

**Turbulência no inlet:** TI = 5%, μ_t/μ = 10 → k = 1,08375 m²/s²,
ω = 7419,21 1/s. (Ver §3.2 e §7.)

### Dado atual (job 5518, pós-processado nos jobs 6023/6025/6027/6083)

```
células      211.579
passos       1000 × 0,1 s = 100 s físicos
travessia    41,2 s  ->  2,4 travessias
desprendimento  St ≈ 0,2 -> f = 0,063 Hz, T = 15,9 s -> 159 passos por ciclo
SIMPLE       13,8 iterações/passo (teto 50)
t_00001      é a hyb-initialization, NÃO é escoamento — a PINN pula esse arquivo
```

### Formato dos CSV (manter idêntico)

Vírgula como separador, 11 colunas, **nesta ordem** (que é invertida em relação
ao journal, com `x-velocity` por último):

```
cellnumber, x-coordinate, y-coordinate, z-coordinate,
viscosity-turb, specific-diss-rate, turb-kinetic-energy,
pressure, z-velocity, y-velocity, x-velocity
```

Arquivo de malha estática, exportado DENTRO da sessão do solve (portanto
alinhado célula a célula com os `t_*.csv`):

```
static_mesh.csv:  cellnumber, x-coordinate, y-coordinate, z-coordinate,
                  cell-wall-distance    [+ cell-volume, ver §3.3]
```

Nomes de arquivo `t_00001.csv` … `t_01000.csv`, índice × 0,1 s = tempo físico.

### Convenções já MEDIDAS do dado atual (não re-litigar)

| o quê | veredito | como foi medido |
|---|---|---|
| coluna `pressure` | é a pressão **MODIFICADA** p̃ = p + (2/3)ρk | job 6023 e verificação (e): margem de 170× contra a hipótese estática |
| coluna `viscosity-turb` | é **μ_t DINÂMICA** [Pa·s], não ν_t | verificação (a): mediana de ν_t/(k/ω) = 1,000000 |
| limitador de cisalhamento | usa **S**, não Ω | 24.848 células decidem: S 100%, Ω 0% |
| ω na 1ª camada | está **CONGELADO** em ω_wall nas 5 paredes | verificação (f), razão 1,0000 em todas |
| distância à parede analítica | confere com o Fluent | correlação 1,000000, erro p99 = 0,0098 m |

### Faixas físicas medidas (teto 1e5)

```
x-velocity       min -18,79   p50 0,00053   max  18
y-velocity       min -16,34   p50 17,01     max  35,69
z-velocity       min -17,47   p50 0,0023    max  35,18
pressure         min -1093    p50 -0,207    max  191,3   [Pa, gauge]
turb-kin-energy  min 1e-10    p50 0,0466    max  37,81   [m²/s²]
specific-diss    min 0,128    p50 2,03      max  1320    [1/s]
viscosity-turb   min 9,1e-15  p50 0,0389    max  1,789   [Pa·s]
```

---

## 6. O que muda do lado da PINN quando o dado novo chegar

Não é tarefa desta conversa, mas registre para não esquecer:

- **`VISC_RATIO_MAX = 1e5` → `1e7` na Cell 2.** O resíduo da Cell 10 aplica
  `min(..., teto)`. Se o dado for até 1e7 e a PINN continuar cortando em 1e5,
  ela impõe um teto que o dado não tem — erro silencioso.
- Grandezas derivadas que mudam junto (só informativo, nenhuma é crítica):
  `NU_T_MAX` 1,461 → 146,1; `NU_T_REF` 0,365 → 36,5; `OMEGA_REF` 26,7 →
  **0,267**; `SCALE_OM` 10,30 → **0,103**. `K_REF` e `SCALE_K` **não mudam**.
- As constantes de normalização são remedidas sozinhas (`AUTO_NORM_STATS`).
- As verificações (a)–(f) da Cell 5 remedem tudo e vão acusar divergência.

---

## 7. Perguntas em aberto que esta conversa pode ajudar a resolver

**(a) O nível de turbulência do inlet está certo?** Com TI = 5% e μ_t/μ = 10,
ω_inlet = 7419 1/s e a turbulência da entrada morre em **0,26 mm**. Na prática
o domínio é alimentado com escoamento praticamente laminar, e toda a
turbulência é gerada pelo cilindro e pelo chão. Para um sítio astronômico com
camada limite atmosférica isso pode não ser o que se quer — mas mudar implica
repensar o perfil de velocidade da entrada também, o que é decisão de
modelagem, não ajuste. **Vale discutir com o orientador antes.**

**(b) Rodar mais tempo?** 100 s são ~6,3 ciclos de desprendimento. Com ~200 s
daria para descartar a primeira travessia **e** reter blocos contíguos de
snapshots para validação. Hoje a validação da PINN é intercalada, a 2,3° de
fase do treino — ela mede suavidade no tempo, não generalização, e isso é um
ponto fraco conhecido para a banca. Custa o dobro de tempo de solver.

---

## 8. Armadilhas de medição já identificadas

Para a conversa nova não repetir erros que já custaram tempo aqui:

1. **A verificação (c) do inlet é inválida como está.** Compara a BC com
   valores já decaídos. Qualquer veredito dela sobre TI e μ_t/μ deve ser
   ignorado até o teste ser refeito ajustando a lei de decaimento.
2. **"% de células" não é "% do domínio".** Sem `cell-volume`, as frações são
   dominadas pelas zonas de refino.
3. **Comparar frações exige o mesmo limiar nos dois lados.** O script de
   inferência classificava o limitador com tolerância 1e-6 enquanto o teste do
   CSV usava 1e-4 — 100× mais frouxo. Comparação sem sentido até alinhar.
4. **Nomes de campo no exportador do Fluent.** Já houve exportação com nome
   errado passando em silêncio. Conferir numa malha pequena antes.

---

## 9. Resumo em uma lista

- [ ] `turb-viscosity/max-limit` de 1e5 para 1e7 — **a única mudança no solver**
- [ ] exportar as 9 derivadas de velocidade + `strain-rate-mag` (tudo, ou 1 em 10)
- [ ] salvar em texto as condições de contorno do inlet, lidas do solver
- [ ] acrescentar `cell-volume` ao `static_mesh.csv`
- [ ] exportar y+ e tensão de parede por zona de parede
- [ ] guardar journal, resumo do `.cas` e histórico de resíduos junto dos CSVs
- [ ] monitor de ponto na esteira, só para confirmar que o desprendimento sobreviveu
- [ ] conferir nomes dos campos numa malha pequena antes de rodar os 1000 passos
- [ ] vigiar o contador de iterações do SIMPLE (hoje 13,8, teto 50)
- [ ] manter malha, dt, esquemas, modelo e contornos idênticos
