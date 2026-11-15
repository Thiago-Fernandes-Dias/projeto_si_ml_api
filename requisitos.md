# Especificação de Requisitos do Sistema (ML API Backend)
## Projeto: Autenticação por Dinâmica de Digitação (*Keystroke Dynamics*)
**Contexto Acadêmico:** UFABC — Sistemas Inteligentes (ESZI014-17)

---

## 1. Visão Geral

Este documento especifica os requisitos funcionais, não-funcionais, arquiteturais, modelos de dados e contratos HTTP da API de Aprendizado de Máquina (`projeto_si_ml_api`), desenvolvida em **Python 3.12 + Django 5.2 LTS + Django REST Framework + scikit-learn**.

A API provê o motor biométrico para coleta e calibração biométrica inicial com 5 amostras (`/api/enrollment`), autenticação biométrica por dinâmica de digitação (`/api/authenticate`), monitoramento de saúde (`/api/health`) e retreinamento adaptativo incremental periódico dos modelos individuais de cada usuário (`M2005_IDB` com submodelo crescente via estatísticas online de Welford + janela deslizante de 50 amostras). O cadastro de usuários e senhas é responsabilidade exclusiva da API `projeto-si-backend` no banco PostgreSQL compartilhado.

---

## 2. Fundamentação Experimental e Algoritmo de ML Escolhido (`M2005_IDB`)

A escolha do classificador biométrico baseia-se nos 52 experimentos comparativos completos executados no submódulo `projeto_si_experimentos` (`results/exp_metric_averages.csv`) sobre os *benchmarks* públicos **CMU Keystroke Dynamics (`DSL-StrongPasswordData.csv`)** e **Keyrecs**, priorizando BACC global e equilíbrio entre FRR e FAR ao longo da evolução temporal da referência biométrica:

| Algoritmo | BACC (CMU) | FRR (CMU) | FAR (CMU) | BACC (Keyrecs) | FRR (Keyrecs) | FAR (Keyrecs) | Média BACC |
|---|---|---|---|---|---|---|---|
| **`M2005_IDB`** | **0.8909** | **0.1524** | **0.0658** | **0.7980** | **0.0903** | **0.3137** | **0.8445** |
| `M2005_DB` | 0.8907 | 0.1464 | 0.0722 | 0.7822 | 0.0868 | 0.3487 | 0.8365 |
| `SelfDet_Sliding` (`radius=0.02`) | 0.8655 | 0.2341 | 0.0349 | 0.8068 | 0.3325 | 0.0538 | 0.8362 |
| `M2005_Pisani2024_Sliding_Dynamic` | 0.8659 | 0.0784 | 0.1898 | 0.7219 | 0.0528 | 0.5033 | 0.7939 |
| `M2005_Pisani2024_Sliding_Fixed` | 0.8661 | 0.0773 | 0.1905 | 0.6792 | 0.0349 | 0.6067 | 0.7727 |
| `ScaledManhattan_Sliding` | 0.7997 | 0.1289 | 0.2717 | 0.7138 | 0.1710 | 0.4014 | 0.7568 |
| `Mhenni_Sliding_Fixed` | 0.6764 | 0.6057 | 0.0416 | 0.7127 | 0.3594 | 0.2153 | 0.6945 |
| `KNN_Sliding` | 0.5381 | 0.9230 | 0.0009 | 0.5819 | 0.8204 | 0.0159 | 0.5600 |

O **`M2005_IDB`** (`AdaptiveM2005Alg(threshold=0.7, variant="idb", window_size=50)`, com limiar calibrado por usuário via `min`/`min_non_outlier`) lidera em BACC médio (`0.8445`), G-mean (`0.8400`) e menor diferença média `|FRR - FAR|` (`0.1550`): reduz a FAR ~3× no CMU em relação ao `M2005_Pisani2024_Sliding_Dynamic` (`6.58%` vs `18.98%`) e em `-18.96 p.p.` no Keyrecs (`31.37%` vs `50.33%`), ao mesmo tempo em que evita o excesso de falsos rejeites do `SelfDet_Sliding` (`15.24%`/`9.03%` FRR vs `23.41%`/`33.25%`).

### Formulação do `M2005_IDB` (*Improved Double Parallel*, Pisani et al., IJCNN 2015)

1. **Arquitetura Paralela Dupla (`AdaptiveM2005Alg`):**
   - **Submodelo Deslizante (*Sliding*):** Fila FIFO limitada a `window_size = 50` amostras genuínas mais recentes, mantendo média ($\mu_k$), desvio-padrão amostral ($\sigma_k$) e mediana ($m_k$) por dígrafo (`DigraphsMetrics`) para capturar adaptações recentes do usuário.
   - **Submodelo Crescente Incremental (*Growing* com Algoritmo de Welford):** Atualiza média ($\mu_k$) e desvio-padrão amostral ($\sigma_k$, com mediana igual à média na variante `idb`) em memória constante $O(1)$ sobre todo o histórico genuíno desde o cadastro, evitando deriva ou contaminação transitória.
2. **Cálculo de Escore Ponderado e Média dos Submodelos (`calculate_score`):**
   - Em cada submodelo, o atributo $k$ obtém *hit* se $x_k \in [\min(b_1, b_2), \max(b_1, b_2)]$, onde:
     $$b_1 = \min(\mu_k, m_k) \cdot \left(0.95 - \frac{\sigma_k}{\mu_k}\right), \quad b_2 = \max(\mu_k, m_k) \cdot \left(1.05 + \frac{\sigma_k}{\mu_k}\right)$$
   - Acertos somam `1.0` (após erro ou início) ou `1.5` (consecutivos), normalizados em $[0, 1]$, e o escore final do `M2005_IDB` é $\frac{\text{Score}_{\text{sliding}} + \text{Score}_{\text{growing}}}{2}$, comparado ao limiar calibrado do usuário (`min`/`min_non_outlier`, limitado ao teto base `0.7`).
3. **Compatibilidade Retroativa (`M2005Alg`):**
   - A classe `M2005Alg` permanece disponível via configuração `BIOMETRICS["M2005"]["algorithm"] = "M2005_Pisani2024_Sliding_Dynamic"` (ou `"M2005_DB"` para `AdaptiveM2005Alg(variant="db")`) e para carregar modelos `.joblib` legados.

---

## 3. Requisitos Funcionais (RF)

- **RF01 - Cadastro de Usuário (Responsabilidade Externa):** O cadastro de novos usuários e o armazenamento inicial de credenciais pertencem exclusivamente à API `projeto-si-backend`. A ML API não expõe endpoint de registro (`/api/register`) e consome usuários previamente existentes na tabela compartilhada `usuarios`.
- **RF02 - Coleta Inicial e Calibração (`POST /api/enrollment`):** Recebe exclusivamente o campo `vectors` do backend, com exatamente `ENROLLMENT_SAMPLES = 5` vetores posicionais `3n-2`, monta o `DataFrame` com schema posicional (`H.0, DD.0.1, UD.0.1, ...`) e treina de forma síncrona a versão `v1` do modelo configurado (`M2005_IDB` por padrão, `AdaptiveM2005Alg` com `window_size = 50`, `threshold` base `0.7` e calibração `min`/`min_non_outlier`) em `modelos` + `.joblib`, sem gravar linhas em `amostras` nem alterar `usuarios.biometria_cadastrada_em`.
- **RF03 - Validação de Vetores Posicionais:** Valida exclusivamente vetores posicionais no formato CMU `3n - 2` (`comprimento >= 4`, `(len + 2) % 3 == 0`), finitos, respeitando os limites físicos por slot: `H, DD ∈ (0, 5]` segundos e `UD ∈ [-5, 5]` segundos. Não há suporte a mapas de `features`, `samples` ou `rawTimings`.
- **RF04 - Autenticação Biométrica (`POST /api/authenticate`):** Recebe exclusivamente o campo `vector` do backend. Não lê nem verifica senha textual do usuário (`senha_hash`); valida a existência do usuário (`404`) e de modelo ativo (`409`), valida o vetor posicional `3n-2` e a dimensão esperada pelo modelo ativo (`400`), executa a inferência cronometrada com `time.perf_counter()` (cache LRU + SHA-256) sem gravar linhas em `amostras` e retorna `200` com `decision` (`GENUINE` ou `IMPOSTOR`), `confidence`, `threshold` calibrado do usuário, `inferenceTimeMs`, `meanHoldTimeMs`, `userAverageHoldTimeMs`, `modelVersion` e `shouldRecordForRetraining` (booleano indicando se a amostra deve ser persistida em `amostras` pelo backend para futuro retreinamento).
- **RF05 - Auto-Rotulagem e Elegibilidade para Retreinamento:** Na resposta do `/api/authenticate`, `shouldRecordForRetraining` é `true` somente quando a amostra for `GENUINE` com confiança acima da banda de incerteza ($\text{score} \ge \text{threshold} + \text{LABEL\_CONFIDENCE\_MARGIN}$, padrão `0.15`), e `false` para `UNCERTAIN` e `IMPOSTOR`. A ML API apenas consulta `amostras` (nunca cria novas linhas); o backend `.NET` utiliza essa flag para persistir somente amostras elegíveis e o retreinamento consome apenas amostras `Cadastro` e `Login` `Genuine` acima da banda de confiança:
  - `GENUINE` se $\text{score} \ge \text{threshold} + \text{margem}$;
  - `IMPOSTOR` se $\text{score} \le \text{threshold} - \text{margem}$;
  - `UNCERTAIN` caso contrário (excluída de retreinamentos futuros para mitigar *feedback loop* / envenenamento).
- **RF06 - Treinamento Exclusivo com Amostras Genuínas (One-Class):** Tanto o enrollment quanto o retreinamento utilizam **exclusivamente** amostras genuínas do usuário (sem geração de impostores sintéticos e sem inclusão de amostras classificadas como `IMPOSTOR` ou `UNCERTAIN`), alinhando a pipeline ao modelo estatístico univariado configurado (`M2005_IDB` por padrão).
- **RF07 - Retreinamento Periódico Incremental/Deslizante e sob Demanda:** Retreina usuários que acumularam ao menos `RETRAIN_MIN_NEW_SAMPLES` novas amostras `GENUINE` de alta confiança ($\text{confianca} \ge \text{threshold} + \text{margem}$) desde o modelo ativo, adaptando incrementalmente o submodelo crescente (estatísticas online de Welford sobre todo o histórico genuíno) e avançando a janela deslizante de tamanho `RETRAIN_GENUINE_WINDOW` (`50`) amostras genuínas mais recentes (ignorando `IMPOSTOR` e `UNCERTAIN`), com lock `select_for_update(skip_locked=True)` para segurança concorrente.
- **RF08 - Versionamento, Integridade e Rollback de Modelos:** Todos os modelos treinados são persistidos em `models_data/<user_id>/v<version>.joblib` com hash `sha256` registrado em `TrainedModel`. O carregamento valida obrigatoriamente o `sha256` antes de `joblib.load`. O comando `activate_model <username> <version>` permite rollback imediato.
- **RF09 - Calibração de Limiar por Usuário (`min`/`min_non_outlier`):** No enrollment e em cada retreinamento, o limiar do `M2005_IDB` é calibrado a partir dos scores da galeria genuína: `min` usa o menor score; `min_non_outlier` descarta scores abaixo de `mean(S) - 2.5 * std(S)` antes de tomar o mínimo. O resultado é limitado ao teto base `BIOMETRICS_DECISION_THRESHOLD` (`0.7`) e usado na decisão, em `label_for`/`shouldRecordForRetraining` e retornado em `threshold` por `/api/authenticate`. O método é configurável por `BIOMETRICS_THRESHOLD_METHOD` (`min` ou `min_non_outlier`) e a recalibração após adaptação por `BIOMETRICS_THRESHOLD_DYNAMIC`.
- **RF10 - Isolamento de Modelos após Troca de Senha:** `get_active_model` considera exclusivamente versões com `criado_em >= data_mudanca_senha`; não há fallback para modelos anteriores. O `active_version.json` também só pode selecionar versões pós-troca. `activate_model_version` rejeita versões anteriores à troca, e `retrain_user` não reutiliza a galeria de um modelo antigo. Para compensação de falha na troca de senha, `POST /api/models/rollback` remove modelos criados após um timestamp e restaura a versão ativa anterior.

---

## 4. Requisitos Não-Funcionais (RNF)

- **RNF01 - Separação de Credenciais e Segurança de Modelos:** A ML API não armazena, lê ou verifica hashes de senha (`senha_hash`); toda gestão de senha e bloqueio pertence ao `projeto-si-backend`. Arquivos `.joblib` têm hash SHA-256 verificado antes da desserialização.
- **RNF02 - Proteção contra Força Bruta:** Endpoint `/api/authenticate` protegido por `ScopedRateThrottle` (`authenticate`).
- **RNF03 - Isolamento de Camadas:** O pacote `biometrics/ml/` não possui qualquer dependência de HTTP/DRF, podendo ser exercitado diretamente por testes unitários, comandos CLI e agendadores.
- **RNF04 - Containerização e Orquestração (RNF06/RNF09):** `Dockerfile` e `docker-compose.yml` com serviços `db` (PostgreSQL 17), `api` (Gunicorn na porta 8000) e `scheduler` (APScheduler em processo dedicado), compartilhando o volume `models_data` e a rede `keystroke-network`.
- **RNF05 - Automação via Makefile (RNF07):** `Makefile` com os alvos `install`, `dev`, `test`, `lint`, `migrate`, `retrain`, `docker-build`, `compose-up` e `compose-down`.
- **RNF06 - Logs Estruturados e Diagnóstico de Fluxos:** A ML API e o `ml-scheduler` emitem logs estruturados em JSON para `stdout` com correlação via cabeçalho `X-Correlation-ID` cobrindo `enrollment`, `authenticate` e ciclos de retreinamento, sem registrar senhas, chaves nominais de features ou segredos. Os logs de nível `DEBUG` ficam desativados por padrão quando `DJANGO_DEBUG=False` (produção), podendo ser habilitados em execução via Docker Compose/desenvolvimento por `BIOMETRICS_DEBUG_LOGS=True` ou `DEBUG_LOGS=true`.

---

## 5. Melhorias Futuras

- **Recuperação após queda durante troca de senha:** a compensação atual cobre exceções, mas não uma queda do processo/container entre o commit da nova senha e o treinamento. Uma evolução deve adicionar um estado pendente e reconciliação automática ou um endpoint de reenrollment para concluir o modelo.
- **Autenticação do endpoint interno `/api/models/rollback`:** hoje ele assume a rede Docker privada. Uma evolução deve exigir segredo/header interno ou usar um identificador de requisição por modelo para limitar o rollback.

---

## 6. Governança e Histórico de Revisões

| Versão | Data | Autor / Agente | Descrição das Modificações |
|---|---|---|---|
| `1.0.0` | 2026-11-14 | Thiago Dias / Antigravity | Especificação inicial da ML API com Django 5.2, DRF, registro de modelos com SHA-256, retreinamento agendado via APScheduler e suíte completa de testes. |
| `1.1.0` | 2026-11-14 | Thiago Dias / Antigravity | Avaliação do cenário de janela deslizante com `SelfDet_Sliding`. |
| `1.2.0` | 2026-11-14 | Thiago Dias / Antigravity | Adoção do `M2005_Pisani2024_Sliding_Dynamic` (`M2005Alg` com `window_strategy="sliding"`, `threshold_method="min_non_outlier"`, `dynamic_threshold=True`), proporcionando melhor equilíbrio entre FRR e FAR com redução acentuada de FRR (`7.82%` no CMU e `5.49%` no Keyrecs) e queda progressiva da FAR ao longo do tempo. |
| `1.3.0` | 2026-11-14 | Thiago Dias / Antigravity | Remoção completa da geração de dados sintéticos de impostores (`synthetic.py`, `fit_population_priors.py`) e do uso de amostras de impostores no treinamento e retreinamento; treinamento e retreinamento passam a utilizar exclusivamente amostras genuínas do usuário. |
| `1.4.0` | 2026-11-14 | Thiago Dias / Antigravity | Remoção do endpoint `/api/register` e do serviço `register_user`; o cadastro de usuários passa a ser responsabilidade exclusiva da API `projeto-si-backend`, mantendo na ML API apenas `/api/enrollment`, `/api/authenticate` e `/api/health`, com suporte a `recordSamples=False` no enrollment. |
| `1.5.0` | 2026-11-14 | Thiago Dias / Antigravity | Remoção completa de leitura e verificação de senha (`senha_hash`, `check_password`, `passwordAttempt`); `/api/enrollment` cria exclusivamente o modelo do usuário (sem gravar em `amostras` ou alterar `usuarios`); `/api/authenticate` valida estritamente estrutura `3n-2`, limites por slot, dimensão do modelo e ordem exata das features sem gravar em `amostras`. |
| `1.6.0` | 2026-11-14 | Thiago Dias / Antigravity | Inclusão de `shouldRecordForRetraining` na resposta de `/api/authenticate` (`true` apenas para `GENUINE` de alta confiança acima de `threshold + LABEL_CONFIDENCE_MARGIN`) e bloqueio de criação de linhas em `amostras` também no Django Admin (`AmostraAdmin`), reforçando acesso estritamente de leitura em `amostras`. |
| `1.7.0` | 2026-11-15 | Thiago Dias / Antigravity | Migração do classificador padrão para `M2005_IDB` (`AdaptiveM2005Alg` com `variant="idb"`, `window_size=50`, `threshold=0.7`) com retreinamento incremental para o ramo crescente Welford e janela deslizante limitada, mantendo suporte configurável a `M2005_DB` e `M2005_Pisani2024_Sliding_Dynamic` (`M2005Alg`). |
| `1.8.0` | 2026-11-15 | Thiago Dias / Antigravity | Remoção de `extract_vector_from_raw_timings` e do suporte a `rawTimings`; a ML API consome exclusivamente vetores posicionais ou mapas `features` já extraídos pelo backend. |
| `1.9.0` | 2026-11-15 | Thiago Dias / Antigravity | Adição de logs de depuração estruturados em JSON (`stdout`) com propagação de `X-Correlation-ID`, mascaramento de dados sensíveis e controle por ambiente (`BIOMETRICS_DEBUG_LOGS`/`DEBUG_LOGS`, desativados em produção e habilitados em Docker Compose dev). |
| `1.10.0` | 2026-11-15 | Thiago Dias / Antigravity | Adoção da calibração de limiar `min`/`min_non_outlier` no `M2005_IDB`, com teto em `0.7`, recalibração dinâmica após retreinamento e variáveis `BIOMETRICS_THRESHOLD_METHOD`/`BIOMETRICS_THRESHOLD_DYNAMIC`. |
| `1.11.0` | 2026-11-15 | Thiago Dias / Antigravity | Remoção do suporte a `samples`/`features`/raw fallback: `/api/enrollment` aceita somente `vectors` e `/api/authenticate` somente `vector`, com schema posicional `3n-2`; gravação de modelo com compensação de arquivos em falha de banco. |
| `1.12.0` | 2026-11-15 | Thiago Dias / Antigravity | Isolamento estrito de modelos por `data_mudanca_senha`, bloqueio de rollback para versões antigas e endpoint interno `/api/models/rollback` para compensar falhas na troca de senha. |
| `1.13.0` | 2026-11-15 | Thiago Dias / Antigravity | Registro de melhorias futuras para recuperação após queda durante troca de senha e proteção do endpoint interno de rollback. |
