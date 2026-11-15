# Diretrizes e Governança do Projeto para Agentes de IA (AGENTS.md)

Este documento estabelece as diretrizes operacionais, arquiteturais e de qualidade que o **Antigravity CLI** e quaisquer agentes de IA devem seguir rigorosamente ao atuar neste repositório (`projeto_si_ml_api`).

---

## 1. Governança e Fonte Única da Verdade

- **Fonte Única da Verdade:** O arquivo [requisitos.md](file:///Users/thiago/Repositories/projeto_si/projeto_si_ml_api/requisitos.md) na raiz do repositório é a especificação oficial de requisitos da API de Aprendizado de Máquina.
- **Sincronização Obrigatória:** Qualquer alteração de contrato HTTP, regra de negócio, hiperparâmetro biométrico ou fluxo de retreinamento deve ser refletida em `requisitos.md` (incluindo registro na tabela de histórico de revisões) antes ou em conjunto com a modificação do código.

---

## 2. Arquitetura e Engenharia de Software (Stack Python / Django / ML)

### 2.1. Separação Estrita de Camadas
- **`biometrics/ml/` (Núcleo de Machine Learning):**
  - Módulos puros de ciência de dados e aprendizado de máquina (`features.py`, `m2005.py`, `training.py`, `registry.py`, `inference.py`).
  - **Proibido** importar classes de `rest_framework` ou objetos HTTP (`Request`, `Response`) dentro de `biometrics/ml/`.
- **`biometrics/services.py` (Camada de Aplicação / Orquestração):**
  - Concentra toda a lógica transacional, regras de domínio (`enroll_user`, `authenticate_user`, `label_for`, `retrain_user`, `retrain_due_users`, `activate_model_version`) e controle de concorrência (`select_for_update(skip_locked=True)`). O cadastro de usuários é responsabilidade exclusiva do `projeto-si-backend`.
- **`biometrics/views.py` e `biometrics/serializers.py` (Camada HTTP):**
  - Responsáveis exclusivamente por validar formatos de entrada, delegar a execução para `biometrics/services.py` e traduzir resultados ou exceções de domínio nos códigos HTTP do contrato (`200`, `201`, `400`, `401`, `404`, `409`).
- **`biometrics/models.py` (Persistência):**
  - Define `BiometricProfile`, `KeystrokeSample` e `TrainedModel` com constraints de integridade no banco de dados (`unique_model_version` e `single_active_model`).

### 2.2. Qualidade de Código e Segurança
- **Tipagem e Estilo:** Utilizar *type hints* do Python 3.12+ (`list[str]`, `dict[str, float]`, `X | None`) e manter conformidade estrita com `ruff check .` (zero avisos ou erros).
- **Segurança de Modelos:** Nunca carregar arquivos `.joblib` sem antes validar o hash `sha256` registrado em `TrainedModel` via `biometrics/ml/registry.py`.
- **Precisão de Cronometragem:** Toda medição de tempo de inferência deve utilizar `time.perf_counter()`.

---

## 3. Testes Automatizados Obrigatórios

- Toda funcionalidade do núcleo de ML, serviços, endpoints HTTP e comandos de gerenciamento deve possuir cobertura automatizada na pasta `tests/`:
  - `tests/test_features.py`: validação de faixas (`H`, `DD`, `UD`), estrutura `3n-2`, preservação/validação estrita de ordem e vetorização.
  - `tests/test_training.py`: treinamento do `M2005Alg` (`M2005_Pisani2024_Sliding_Dynamic`) utilizando exclusivamente amostras genuínas, probabilidades altas para amostras genuínas e baixas para ritmo 2× mais lento.
  - `tests/test_registry.py`: versionamento incremental, constraint de modelo único ativo, bloqueio de arquivo adulterado (SHA-256) e cache LRU.
  - `tests/test_api.py`: fluxo de enrollment (criando apenas o modelo) e autenticação sem senha (com validação estrita de dimensão/ordem de features, retorno de `shouldRecordForRetraining` e zero escrita em `amostras`), remoção de `/api/register` (`404`) e códigos HTTP (`200`, `201`, `400`, `404`, `409`).
  - `tests/test_retrain.py`: janelas deslizantes de amostras, exclusão de `UNCERTAIN`, `--force` e rollback via `activate_model`.
- Antes de encerrar qualquer tarefa, execute obrigatoriamente:
  ```bash
  make test
  make lint
  ```
