# API de Autenticação por Dinâmica de Digitação (`projeto_si_ml_api`)

API RESTful desenvolvida em **Python 3.12**, **Django 5.2 LTS**, **Django REST Framework** e **scikit-learn** para autenticação biométrica comportamental baseada em dinâmica de digitação (*Keystroke Dynamics*).

A API é o motor biométrico do sistema: recebe vetores posicionais do `projeto-si-backend`, treina modelos individuais por usuário e responde à autenticação. **Ela não cadastra usuários, não lê nem verifica senhas** e não escreve novas linhas em `amostras`.

---

## Algoritmo Biométrico: `M2005_IDB`

A API utiliza o classificador estatístico adaptativo **`M2005_IDB`** (`AdaptiveM2005Alg(threshold=0.7, variant="idb", window_size=50)` — *Improved Double Parallel*, Pisani et al., IJCNN 2015), mantendo compatibilidade retroativa com `M2005Alg` (`M2005_Pisani2024_Sliding_Dynamic`). O `M2005_IDB` foi selecionado por liderar os 52 experimentos do `projeto_si_experimentos` (consolidados localmente em `results/exp_metric_averages.csv` por `generate_exp_metric_averages.py`) em BACC médio (`0.8445`), G-mean (`0.8400`) e equilíbrio entre FRR e FAR:

- **Submodelo Deslizante (`window_size=50`):** Mantém média, mediana e desvio-padrão por dígrafo sobre as `50` amostras genuínas mais recentes, capturando a evolução recente do usuário.
- **Submodelo Crescente Incremental (Welford $O(1)$):** Acumula média e desvio-padrão online sobre **todo** o histórico genuíno do usuário desde o cadastro (preservado incrementalmente no retreinamento), impedindo deriva ou esquecimento catastrófico.
- **Calibração de Limiar (`min` / `min_non_outlier`):** Após o treinamento (e a cada retreinamento, com `dynamic_threshold=True`), o limiar do usuário é calibrado a partir dos scores da galeria genuína: `min` usa o menor score; `min_non_outlier` descarta scores abaixo de `mean - 2.5 * std` antes de tomar o mínimo. O valor é limitado ao teto base `0.7`. O limiar calibrado é usado na decisão e na resposta de `/api/authenticate`.
  - O `M2005_IDB` original de Pisani et al. (2015) usa threshold fixo; a calibração e a recalibração dinâmica vêm de Pisani (2024).
  - O teto `0.7` é uma decisão deste projeto. O `M2005Alg` do `projeto_si_experimentos` segue Pisani (2024) sem esse cap (o threshold pode chegar ao mínimo dos scores da galeria).
- **Decisão e Desempenho Experimental (`projeto_si_experimentos`):** Score final dado pela média aritmética dos submodelos deslizante e crescente (resultados medidos com `threshold = 0.7`):
  - **CMU (`DSL-StrongPasswordData.csv`):** `BACC = 0.8909`, **FRR = `15.24%`** e **FAR = `6.58%`** (3× menor que os `18.98%` do `M2005_Pisani2024_Sliding_Dynamic`).
  - **Keyrecs:** `BACC = 0.7980`, **FRR = `9.03%`** e **FAR = `31.37%`** (`+7.61 p.p.` de BACC e `-18.96 p.p.` de FAR frente ao `M2005_Pisani2024_Sliding_Dynamic`).
- **Treinamento One-Class:** o enrollment usa apenas os vetores genuínos fornecidos; no retreinamento, entram as amostras de `Cadastro` e as de `Login` que sejam `Genuine` e superem a banda de confiança. `IMPOSTOR` e `UNCERTAIN` são descartadas.

### Configuração do algoritmo

| Variável | Padrão | Descrição |
|---|---|---|
| `BIOMETRICS_ALGORITHM` | `M2005_IDB` | `M2005_IDB`, `M2005_DB` ou `M2005_Pisani2024_Sliding_Dynamic`. |
| `BIOMETRICS_M2005_VARIANT` | `idb` | Variante do `AdaptiveM2005Alg` (`idb` ou `db`) quando aplicável. |
| `BIOMETRICS_DECISION_THRESHOLD` | `0.7` | Teto do limiar calibrado por usuário. |
| `BIOMETRICS_THRESHOLD_METHOD` | `min_non_outlier` | Método de calibração (`min` ou `min_non_outlier`). |
| `BIOMETRICS_THRESHOLD_DYNAMIC` | `True` | Recalibra o limiar após cada adaptação/retreinamento. |
| `BIOMETRICS_RETRAIN_GENUINE_WINDOW` | `50` | Tamanho da janela deslizante de amostras genuínas. |

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
│   │   ├── features.py              # Validação de vetores posicionais 3n-2 e schema CMU
│   │   ├── inference.py             # Inferência cronometrada com time.perf_counter()
│   │   ├── m2005.py                 # M2005Alg (Pisani 2024) e AdaptiveM2005Alg (DB/IDB)
│   │   ├── registry.py              # Persistência .joblib, SHA-256, cache LRU e compensação atômica
│   │   └── training.py              # Pipeline de treino one-class usando apenas amostras genuínas
│   ├── admin.py
│   ├── logging_utils.py             # Logs JSON estruturados, correlação e redação de dados sensíveis
│   ├── middleware.py                # Propaga X-Correlation-ID e registra requisições HTTP
│   ├── models.py                    # User, Amostra e ModeloBiometrico
│   ├── serializers.py
│   ├── services.py                  # Orquestração de negócio, retreinamento e rollback
│   ├── urls.py
│   └── views.py
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_logging.py
│   ├── test_registry.py
│   ├── test_retrain.py
│   └── test_training.py
├── .env.example
├── AGENTS.md
├── Dockerfile
├── Makefile
├── docker-compose.yml
├── manage.py
├── pyproject.toml
└── requisitos.md
```

---

## Pré-requisitos

| Ferramenta | Versão |
|---|---|
| Python | 3.12+ |
| PostgreSQL | 17 (compartilhado com o `projeto-si-backend`) |
| Docker + Docker Compose | Opcional, para execução em container |

A ML API usa o mesmo banco `projeto_si` do backend. As tabelas `usuarios`, `modelos` e `amostras` são criadas pelo `schema.sql`/migrações do backend; por isso o comando de migração do Django usa `--fake-initial`.

---

## Executando Localmente

```bash
# Copie as variáveis de ambiente
cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env

# Crie e ative o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -e ".[dev]"
```

Antes de iniciar, garanta que o PostgreSQL está em execução e que as tabelas do backend já existem (suba o backend uma vez ou aplique `projeto-si-backend/schema.sql`).

```bash
# Marca a migração inicial do Django sobre as tabelas existentes
python manage.py migrate --fake-initial

# Inicia a API em http://localhost:8000
python manage.py runserver 0.0.0.0:8000
```

Para habilitar o retreinamento periódico em um processo separado:

```bash
python manage.py run_scheduler
```

Ou dispare manualmente:

```bash
# Todos os usuários elegíveis
python manage.py retrain_models

# Usuário específico
python manage.py retrain_models --username thiago_dias

# Força o retreinamento mesmo sem novas amostras suficientes
python manage.py retrain_models --username thiago_dias --force

# Ativa uma versão anterior (rollback manual)
python manage.py activate_model thiago_dias 1
```

> Os testes usam SQLite automaticamente quando `DJANGO_TEST=1` ou quando `DATABASE_URL` não está definida.

---

## Executando com Docker

Na raiz deste submódulo:

```bash
docker compose up --build
```

Sobem os serviços `db` (PostgreSQL 17), `api` (Gunicorn em `http://localhost:8000`) e `scheduler` (APScheduler), compartilhando o volume `models_data` e a rede `keystroke-network`.

```bash
# Em segundo plano
docker compose up --build -d

# Executar comandos dentro da API
docker compose exec api python manage.py retrain_models
docker compose exec api python manage.py activate_model thiago_dias 1

# Encerrar
docker compose down
```

O Compose da raiz do projeto reaproveita esta imagem e usa o serviço `postgres` compartilhado; nesse caso a ML API também fica em `http://localhost:8000`.

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `DJANGO_SECRET_KEY` | chave insegura de desenvolvimento | Chave secreta do Django; obrigatória em produção. |
| `DJANGO_DEBUG` | `True` local / `False` no Compose | Modo debug. |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1,testserver,api` | Hosts permitidos. |
| `DATABASE_URL` | `postgres://projeto_si:projeto_si_dev@localhost:5432/projeto_si` | Banco compartilhado. Sem ela, usa SQLite. |
| `BIOMETRICS_ENROLLMENT_SAMPLES` | `5` | Quantidade exata de vetores no enrollment. |
| `BIOMETRICS_DECISION_THRESHOLD` | `0.7` | Teto do limiar calibrado. |
| `BIOMETRICS_THRESHOLD_METHOD` | `min_non_outlier` | Método de calibração do limiar. |
| `BIOMETRICS_THRESHOLD_DYNAMIC` | `True` | Recalibração dinâmica após adaptação. |
| `BIOMETRICS_LABEL_CONFIDENCE_MARGIN` | `0.15` | Banda de incerteza para rotulagem/retreinamento. |
| `BIOMETRICS_RETRAIN_INTERVAL_MINUTES` | `60` | Intervalo do scheduler. |
| `BIOMETRICS_RETRAIN_MIN_NEW_SAMPLES` | `5` | Novas amostras genuínas necessárias para retreinar. |
| `BIOMETRICS_RETRAIN_GENUINE_WINDOW` | `50` | Janela deslizante de amostras genuínas. |
| `BIOMETRICS_MODELS_DIR` | `models_data` | Diretório dos artefatos `.joblib`. |
| `BIOMETRICS_AUTH_THROTTLE_RATE` | `120/minute` | Rate limit de `/api/authenticate`. |
| `BIOMETRICS_DEBUG_LOGS` | derivado de `DJANGO_DEBUG` | Habilita logs estruturados de depuração. |
| `BIOMETRICS_ALGORITHM` | `M2005_IDB` | Algoritmo de produção. |
| `BIOMETRICS_M2005_VARIANT` | `idb` | Variante DB/IDB. |

---

## Endpoints da API

| Método | Rota | Descrição | Códigos HTTP |
|---|---|---|---|
| `GET` | `/api/health` | Status da aplicação e conectividade com o banco | `200`, `503` |
| `POST` | `/api/enrollment` | Recebe exatamente 5 `vectors` posicionais do backend e cria apenas o modelo `v1` | `201`, `400`, `404`, `409` |
| `POST` | `/api/authenticate` | Recebe um `vector` posicional `3n-2`, valida dimensão e classifica no modelo ativo | `200`, `400`, `404`, `409`, `429`, `500` |
| `POST` | `/api/models/rollback` | Endpoint interno: remove modelos criados após uma troca de senha e restaura a versão ativa anterior | `200`, `400`, `404` |

### Exemplo de enrollment

```json
{
  "username": "thiago_dias",
  "userId": "0199f3a0-6f2b-7c3e-8a41-123456789abc",
  "vectors": [
    [0.1491, 0.3979, 0.2488, 0.1069, 0.1674, 0.0605, 0.1150],
    [0.1510, 0.3902, 0.2392, 0.1098, 0.1701, 0.0642, 0.1162],
    [0.1487, 0.4010, 0.2523, 0.1051, 0.1650, 0.0599, 0.1138],
    [0.1502, 0.3944, 0.2442, 0.1077, 0.1688, 0.0620, 0.1155],
    [0.1499, 0.3961, 0.2471, 0.1064, 0.1669, 0.0611, 0.1147]
  ]
}
```

Resposta `201` (resumida):

```json
{
  "userId": "0199f3a0-6f2b-7c3e-8a41-123456789abc",
  "username": "thiago_dias",
  "modelId": 1,
  "modelVersion": 1,
  "featureCount": 7,
  "samplesRegistered": 5,
  "calculatedAverageHoldMs": 112.3,
  "success": true,
  "message": "Enrollment concluído e modelo biométrico calibrado com sucesso."
}
```

### Exemplo de autenticação

```json
{
  "username": "thiago_dias",
  "vector": [0.1510, 0.3902, 0.2392, 0.1098, 0.1701, 0.0642, 0.1162]
}
```

Resposta `200` (resumida):

```json
{
  "decision": "GENUINE",
  "confidence": 0.931,
  "confidenceScore": 0.931,
  "threshold": 0.68,
  "thresholdScore": 0.68,
  "inferenceTimeMs": 4.82,
  "meanHoldTimeMs": 109.4,
  "userAverageHoldTimeMs": 112.0,
  "modelId": 1,
  "modelVersion": 1,
  "shouldRecordForRetraining": true,
  "message": "Padrão biométrico compatível com o perfil legítimo cadastrado."
}
```

`shouldRecordForRetraining` é `true` somente quando a decisão é `GENUINE` e o score atinge `threshold + LABEL_CONFIDENCE_MARGIN` (`0.15` por padrão), ou seja, quando a amostra é confiável o bastante para alimentar o retreinamento. A ML API **não** grava a amostra; o backend .NET decide persisti-la.

---

## Ciclo de Vida dos Modelos

- Cada modelo é persistido em `models_data/<user_id>/v<version>_<sha256>.joblib`; o hash SHA-256 fica no nome do arquivo e no `.joblib.meta.json`, junto com a versão do scikit-learn, o gatilho e as estatísticas de treino. O carregamento valida o hash antes de `joblib.load`.
- A tabela `modelos` guarda apenas `versao`, `algoritmo`, `caminho_arquivo` e `criado_em`; o hash não é uma coluna do banco.
- A versão ativa é a maior `versao` com `criado_em >= usuarios.data_mudanca_senha`, respeitando `active_version.json` quando presente.
- Após uma troca de senha, modelos anteriores são ignorados e `activate_model_version` rejeita versões pré-troca.
- Em falha de treinamento durante a troca de senha, o backend chama `POST /api/models/rollback` para remover modelos criados após o timestamp e restaurar a versão ativa anterior.
- O retreinamento pode ser **incremental** (adapta o `AdaptiveM2005Alg` com as novas amostras genuínas, preservando o ramo crescente Welford e a janela deslizante) ou **completo** (reconstrói o modelo com as amostras genuínas posteriores à troca; no caminho legado `M2005Alg`, a galeria é limitada a `RETRAIN_GENUINE_WINDOW`).

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

---

## Testes e Qualidade

```bash
make test
make lint
```

Ou diretamente:

```bash
pytest
ruff check .
```

A suíte cobre validação de vetores posicionais, treinamento/adaptação incremental do `AdaptiveM2005Alg`, calibração de limiar, integridade SHA-256/cache, endpoints de enrollment/autenticação/rollback e regras de retreinamento.

---

## Observabilidade e Logs

A ML API e o `ml-scheduler` emitem logs estruturados em JSON no `stdout`, com correlação por `X-Correlation-ID` nos fluxos de enrollment, autenticação, retreinamento e troca de senha. Senhas, chaves nominais de features e segredos não são registrados. O nível de depuração é controlado por `BIOMETRICS_DEBUG_LOGS` (ou `DEBUG_LOGS` no Compose) e fica desativado por padrão quando `DJANGO_DEBUG=False`.
