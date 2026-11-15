# API de Autenticação por Dinâmica de Digitação (`projeto_si_ml_api`)

API RESTful desenvolvida em **Python 3.12**, **Django 5.2 LTS**, **Django REST Framework** e **scikit-learn** para autenticação biométrica comportamental baseada em dinâmica de digitação (*Keystroke Dynamics*).

## Algoritmo Biométrico: `M2005_IDB`

A API utiliza o classificador estatístico adaptativo **`M2005_IDB`** (`AdaptiveM2005Alg(threshold=0.7, variant="idb", window_size=50)` — *Improved Double Parallel*, Pisani et al., IJCNN 2015), mantendo compatibilidade retroativa com `M2005Alg` (`M2005_Pisani2024_Sliding_Dynamic`). O `M2005_IDB` foi selecionado por liderar os 52 experimentos do `projeto_si_experimentos` (`results/exp_metric_averages.csv`) em BACC médio (`0.8445`), G-mean (`0.8400`) e equilíbrio entre FRR e FAR:

- **Submodelo Deslizante (`window_size=50`):** Mantém média, mediana e desvio-padrão por dígrafo sobre as `50` amostras genuínas mais recentes, capturando a evolução recente do usuário.
- **Submodelo Crescente Incremental (Welford $O(1)$):** Acumula média e desvio-padrão online sobre **todo** o histórico genuíno do usuário desde o cadastro (preservado incrementalmente no retreinamento), impedindo deriva ou esquecimento catastrófico.
- **Decisão e Desempenho Experimental (`projeto_si_experimentos`):** Score final dado pela média aritmética dos submodelos deslizante e crescente (`threshold = 0.7`):
  - **CMU (`DSL-StrongPasswordData.csv`):** `BACC = 0.8909`, **FRR = `15.24%`** e **FAR = `6.58%`** (3× menor que os `18.98%` do `M2005_Pisani2024_Sliding_Dynamic`).
  - **Keyrecs:** `BACC = 0.7980`, **FRR = `9.03%`** e **FAR = `31.37%`** (`+7.61 p.p.` de BACC e `-18.96 p.p.` de FAR frente ao `M2005_Pisani2024_Sliding_Dynamic`).

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
