# API de Autenticação por Dinâmica de Digitação (`projeto_si_ml_api`)

API RESTful desenvolvida em **Python 3.12**, **Django 5.2 LTS**, **Django REST Framework** e **scikit-learn** para autenticação biométrica comportamental baseada em dinâmica de digitação (*Keystroke Dynamics*).

## Algoritmo Biométrico: `M2005_Pisani2024_Sliding_Dynamic`

A API utiliza o classificador estatístico adaptativo **`M2005_Pisani2024_Sliding_Dynamic`** (`M2005Alg(window_strategy="sliding", window_size=50, threshold_method="min_non_outlier", dynamic_threshold=True)` — Magalhães et al., 2005; Pisani, SBSeg 2024), selecionado pelo seu equilíbrio superior entre FRR e FAR no cenário de janela deslizante (`projeto_si_experimentos/results/exp_metric_averages.csv`):

- **Janela Deslizante (`window_strategy="sliding"`):** Atualiza média, mediana e desvio-padrão por dígrafo sobre as `50` amostras genuínas mais recentes do usuário.
- **Limiar Dinâmico sem Outliers (`threshold_method="min_non_outlier"`, `dynamic_threshold=True`):** Recalibra o limiar individual filtrando outliers abaixo de $\mu_S - 2.5\sigma_S$, reduzindo falsas rejeições (FRR) conforme o usuário ganha fluência.
- **Desempenho Experimental em Janela Deslizante (`projeto_si_experimentos`):**
  - **CMU (`DSL-StrongPasswordData.csv`):** `BACC = 0.8619`, com **FRR de apenas `7.82%`** (3× menor que `SelfDet_Sliding`) e **FAR de `19.80%`** (caindo de `27.47%` na 1ª metade para **`12.57%`** na 2ª metade do fluxo).
  - **Keyrecs:** `BACC = 0.7187`, com **FRR de apenas `5.49%`** (6× menor que `SelfDet_Sliding`, caindo de `5.61%` para `5.40%` ao longo do tempo).

---

## Arquitetura do Projeto

```text
projeto_si_ml_api/
├── biometrics/
│   ├── management/commands/
│   │   ├── activate_model.py        # Rollback/ativação de versão específica de modelo
│   │   ├── retrain_models.py        # Retreinamento manual ou via CLI (--username, --force)
│   │   └── run_scheduler.py         # BlockingScheduler (APScheduler) para retreino periódico
│   ├── migrations/
│   │   └── 0001_initial.py
│   ├── ml/
│   │   ├── features.py              # Validação de H/DD/UD, schema canônico e vetorização
│   │   ├── inference.py             # Inferência cronometrada com time.perf_counter()
│   │   ├── m2005.py                 # Implementação de DigraphsMetrics e M2005Alg (Pisani 2024)
│   │   ├── registry.py              # Persistência .joblib, validação SHA-256 e cache LRU
│   │   └── training.py              # Pipeline de treino one-class usando apenas amostras genuínas
│   ├── admin.py
│   ├── models.py                    # Usuario, Amostra, ModeloBiometrico
│   ├── serializers.py
│   ├── services.py                  # Orquestração de negócio e controle transacional
│   ├── urls.py
│   └── views.py
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_registry.py
│   ├── test_retrain.py
│   └── test_training.py
├── AGENTS.md
├── Dockerfile
├── Makefile
├── docker-compose.yml
├── manage.py
├── pyproject.toml
└── requisitos.md
```

---

## Endpoints da API

| Método | Rota | Descrição | Códigos HTTP |
|---|---|---|---|
| `GET` | `/api/health` | Status da aplicação e conectividade com o banco | `200`, `503` |
| `POST` | `/api/enrollment` | Recebe 5 amostras/vetores (ou lê do banco compartilhado) e cria apenas o modelo `v1` | `201`, `400`, `404`, `409` |
| `POST` | `/api/authenticate` | Valida features, dimensão e ordem posicional `3n-2` e classifica no modelo ativo | `200`, `400`, `404`, `409`, `429` |

---

## Comandos Rápidos (`Makefile`)

```bash
make install       # Cria .venv (ou usa uv) e instala dependências + dev
make migrate       # Aplica migrações do banco de dados
make dev           # Inicia o servidor local em http://localhost:8000
make test          # Executa a suíte de testes automatizados (pytest)
make lint          # Executa o linter (ruff check .)
make retrain       # Executa o retreinamento dos modelos elegíveis
make docker-build  # Constrói a imagem Docker da API
make compose-up    # Inicia db (Postgres 17), api e scheduler via Docker Compose
make compose-down  # Encerra os containers do Docker Compose
```
