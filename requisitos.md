# Especificação de Requisitos do Sistema (ML API Backend)
## Projeto: Autenticação por Dinâmica de Digitação (*Keystroke Dynamics*)
**Contexto Acadêmico:** UFABC — Sistemas Inteligentes (ESZI014-17)

---

## 1. Visão Geral

Este documento especifica os requisitos funcionais, não-funcionais, arquiteturais, modelos de dados e contratos HTTP da API de Aprendizado de Máquina (`projeto_si_ml_api`), desenvolvida em **Python 3.12 + Django 5.2 LTS + Django REST Framework + scikit-learn**.

A API provê o backend oficial para cadastro de credenciais (`/api/register`), coleta e calibração biométrica inicial com 5 amostras (`/api/enrollment`), autenticação híbrida por senha e dinâmica de digitação (`/api/authenticate`), monitoramento de saúde (`/api/health`) e retreinamento adaptativo periódico em janela deslizante com limiar dinâmico dos modelos individuais de cada usuário.

---

## 2. Fundamentação Experimental e Algoritmo de ML Escolhido (`M2005_Pisani2024_Sliding_Dynamic`)

A escolha do classificador biométrico baseia-se nos experimentos comparativos conduzidos no submódulo `projeto_si_experimentos` (`results/exp_metric_averages.csv`) sobre os *benchmarks* públicos **CMU Keystroke Dynamics (`DSL-StrongPasswordData.csv`)** e **Keyrecs** no cenário de **janela deslizante (*Sliding Window*)**, priorizando o equilíbrio entre FRR e FAR com tendência de baixa FRR ao longo do tempo e FAR controlada:

| Algoritmo (Janela Deslizante) | BACC (CMU) | FRR (CMU) | FAR (CMU) | BACC (Keyrecs) | FRR (Keyrecs) | FAR (Keyrecs) | Média BACC |
|---|---|---|---|---|---|---|---|
| **`M2005_Pisani2024_Sliding_Dynamic`** | **0.8619** | **0.0782** | **0.1980** | **0.7187** | **0.0549** | **0.5077** | **0.7903** |
| `SelfDet_Sliding` (`radius=0.02`) | 0.8625 | 0.2347 | 0.0403 | 0.8038 | 0.3368 | 0.0556 | 0.8332 |
| `M2005_Pisani2024_Sliding_Fixed` | 0.8613 | 0.0759 | 0.2016 | 0.6786 | 0.0343 | 0.6085 | 0.7699 |
| `ScaledManhattan_Sliding` | 0.7944 | 0.1290 | 0.2822 | 0.7110 | 0.1670 | 0.4110 | 0.7527 |
| `Mhenni_Sliding_Fixed` | 0.6764 | 0.6085 | 0.0387 | 0.7141 | 0.3563 | 0.2156 | 0.6952 |
| `KNN_Sliding` | 0.5382 | 0.9226 | 0.0010 | 0.5821 | 0.8213 | 0.0144 | 0.5602 |

Enquanto o `SelfDet_Sliding` apresenta FRR elevada (`23.47%` no CMU e `33.68%` no Keyrecs — rejeitando entre 1 em cada 4 e 1 em cada 3 acessos legítimos), o **`M2005_Pisani2024_Sliding_Dynamic`** (`M2005Alg(window_strategy="sliding", threshold_method="min_non_outlier", dynamic_threshold=True)`) atinge praticamente a mesma BACC no CMU (`0.8619` vs `0.8625`), mas reduz a **FRR em 3× no CMU (`7.82%`) e em 6× no Keyrecs (`5.49%`)**, mantendo a FAR baixa no CMU (`19.80%` na média, caindo de `27.47%` na 1ª metade para **`12.57%`** na 2ª metade do fluxo adaptativo).

### Formulação do `M2005_Pisani2024_Sliding_Dynamic` (Magalhães et al., 2005; Pisani, SBSeg 2024)

1. **Estatísticas por Dígrafo na Janela Deslizante (`DigraphsMetrics`):**
   - Para cada atributo temporal $k$ (`H.*`, `DD.*`, `UD.*`) sobre a galeria deslizante $\mathcal{G}$ (até `window_size = 50` amostras genuínas mais recentes), calculam-se média ($\mu_k$), desvio-padrão amostral ($\sigma_k$) e mediana ($m_k$).
2. **Cálculo de Escore Ponderado por Acertos Consecutivos (`calculate_score`):**
   - Cada atributo $k$ obtém *hit* se $x_k \in [\min(b_1, b_2), \max(b_1, b_2)]$, onde:
     $$b_1 = \min(\mu_k, m_k) \cdot \left(0.95 - \frac{\sigma_k}{\mu_k}\right), \quad b_2 = \max(\mu_k, m_k) \cdot \left(1.05 + \frac{\sigma_k}{\mu_k}\right)$$
   - O primeiro acerto (ou acerto após erro) soma `1.0`, e acertos consecutivos somam `1.5`, normalizados por $(L - 1) \cdot 1.5 + 1.0 \in [0, 1]$.
3. **Calibração Dinâmica de Limiar sem Outliers (`calibrate_threshold`, Pisani SBSeg 2024):**
   - Avaliam-se os escores $S_j = \{\text{score}(g) \mid g \in \mathcal{G}\}$ das amostras da galeria deslizante e descartam-se outliers inferiores a $\bar{S}_j - 2.5 \cdot \sigma(S_j)$:
     $$S_j^n = \{s \in S_j \mid s > \text{mean}(S_j) - 2.5 \cdot \text{std}(S_j)\}$$
   - O limiar individual do usuário é recalibrado dinamicamente a cada atualização da janela como $\min(\min(S_j^n), \tau_{\text{base}})$, adaptando-se à variabilidade natural do usuário ao longo do tempo e reduzindo falsos rejeites (FRR).

---

## 3. Requisitos Funcionais (RF)

- **RF01 - Cadastro de Usuário (`POST /api/register`):** Registra um novo usuário utilizando o modelo `User` do Django (`PBKDF2` password hashing). Aceita tanto `{username, password}` quanto `{username, targetPassword, confirmPassword}`. Retorna `201` em caso de sucesso e `409` caso o `username` já exista.
- **RF02 - Coleta Inicial e Calibração (`POST /api/enrollment`):** Recebe exatamente `ENROLLMENT_SAMPLES = 5` amostras válidas do usuário recém-cadastrado (no formato §5.3 com `samples: [{repetition, features, rawTimings}]` ou lista direta de `features`), fixa a ordem canônica lexicográfica das features em `BiometricProfile`, grava as amostras como `ENROLLMENT` / `GENUINE` e treina de forma síncrona a versão `v1` do modelo `M2005_Pisani2024_Sliding_Dynamic`.
- **RF03 - Validação de Features Temporais:** Valida que todas as chaves seguem o padrão `H.*`, `DD.*` ou `UD.*` (incluindo sufixos sequenciais como `H.t.0`), possuem valores numéricos finitos e respeitam os limites físicos: `H, DD ∈ (0, 5]` segundos e `UD ∈ [-5, 5]` segundos.
- **RF04 - Autenticação Biométrica (`POST /api/authenticate`):** Verifica primeiramente a senha textual via hash do Django (`401 INVALID_CREDENTIALS` se usuário inexistente ou senha incorreta, sem gravar amostra e sem permitir enumeração de usuários). Se a senha estiver correta, vetoriza as features segundo o schema do usuário, executa a inferência cronometrada com `time.perf_counter()` sobre o modelo ativo (carregado via cache LRU com verificação de hash SHA-256) e retorna `200` com `decision` (`GENUINE` ou `IMPOSTOR`), `confidence`, `threshold`, `inferenceTimeMs`, `meanHoldTimeMs` e `modelVersion`.
- **RF05 - Auto-Rotulagem com Banda de Confiança:** Tentativas de login com senha textual correta são gravadas em `KeystrokeSample` com `source=LOGIN` e rótulo determinado pela margem de confiança `LABEL_CONFIDENCE_MARGIN = 0.15` em torno do limiar do modelo:
  - `GENUINE` se $\text{score} \ge \text{threshold} + \text{margem}$;
  - `IMPOSTOR` se $\text{score} \le \text{threshold} - \text{margem}$;
  - `UNCERTAIN` caso contrário (excluída de retreinamentos futuros para mitigar *feedback loop* / envenenamento).
- **RF06 - Geração de Negativos Sintéticos:** Disponibiliza gerador de amostras negativas sintéticas (`biometrics/ml/synthetic.py`) combinando perturbação de ritmo global + ruído multiplicativo log-normal (com filtro de distância mínima Manhattan escalada até a média genuína) e amostragem por distribuições populacionais (`population_priors.json`, quando disponível), garantindo avaliação de separabilidade (`oob_score` / BACC) em todo treinamento.
- **RF07 - Retreinamento Periódico em Janela Deslizante e sob Demanda:** Retreina usuários que acumularam ao menos `RETRAIN_MIN_NEW_SAMPLES` amostras rotuladas novas desde o modelo ativo, utilizando a janela deslizante das `RETRAIN_GENUINE_WINDOW` (`50`) amostras `GENUINE` mais recentes e até `RETRAIN_IMPOSTOR_WINDOW` amostras `IMPOSTOR` mais recentes (ignorando `UNCERTAIN`), recalibrando dinamicamente o limiar `min_non_outlier`, com lock `select_for_update(skip_locked=True)` para segurança concorrente.
- **RF08 - Versionamento, Integridade e Rollback de Modelos:** Todos os modelos treinados são persistidos em `models_data/<user_id>/v<version>.joblib` com hash `sha256` registrado em `TrainedModel`. O carregamento valida obrigatoriamente o `sha256` antes de `joblib.load`. O comando `activate_model <username> <version>` permite rollback imediato.

---

## 4. Requisitos Não-Funcionais (RNF)

- **RNF01 - Segurança no Armazenamento e Carregamento:** Senhas armazenadas exclusivamente com PBKDF2 (`django.contrib.auth`). Arquivos `.joblib` têm hash SHA-256 verificado antes da desserialização.
- **RNF02 - Proteção contra Força Bruta:** Endpoint `/api/authenticate` protegido por `ScopedRateThrottle` (`authenticate`).
- **RNF03 - Isolamento de Camadas:** O pacote `biometrics/ml/` não possui qualquer dependência de HTTP/DRF, podendo ser exercitado diretamente por testes unitários, comandos CLI e agendadores.
- **RNF04 - Containerização e Orquestração (RNF06/RNF09):** `Dockerfile` e `docker-compose.yml` com serviços `db` (PostgreSQL 17), `api` (Gunicorn na porta 8000) e `scheduler` (APScheduler em processo dedicado), compartilhando o volume `models_data` e a rede `keystroke-network`.
- **RNF05 - Automação via Makefile (RNF07):** `Makefile` com os alvos `install`, `dev`, `test`, `lint`, `migrate`, `retrain`, `docker-build`, `compose-up` e `compose-down`.

---

## 5. Governança e Histórico de Revisões

| Versão | Data | Autor / Agente | Descrição das Modificações |
|---|---|---|---|
| `1.0.0` | 2026-11-14 | Thiago Dias / Antigravity | Especificação inicial da ML API com Django 5.2, DRF, registro de modelos com SHA-256, retreinamento agendado via APScheduler e suíte completa de testes. |
| `1.1.0` | 2026-11-14 | Thiago Dias / Antigravity | Avaliação do cenário de janela deslizante com `SelfDet_Sliding`. |
| `1.2.0` | 2026-11-14 | Thiago Dias / Antigravity | Adoção do `M2005_Pisani2024_Sliding_Dynamic` (`M2005Alg` com `window_strategy="sliding"`, `threshold_method="min_non_outlier"`, `dynamic_threshold=True`), proporcionando melhor equilíbrio entre FRR e FAR com redução acentuada de FRR (`7.82%` no CMU e `5.49%` no Keyrecs) e queda progressiva da FAR ao longo do tempo. |
