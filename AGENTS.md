# Diretrizes e Governança do Projeto para Agentes de IA (AGENTS.md)

Este documento estabelece as diretrizes operacionais, arquiteturais e de qualidade que o **Antigravity CLI** e quaisquer agentes de IA devem seguir rigorosamente ao atuar neste repositório (`projeto_si_ml_api`).

---

## 1. Governança e Fonte Única da Verdade

- **Fonte Única da Verdade:** O arquivo [requisitos.md](requisitos.md) na raiz do repositório é a especificação oficial de requisitos da API de Aprendizado de Máquina.
- **Sincronização Obrigatória:** Qualquer alteração de contrato HTTP, regra de negócio, hiperparâmetro biométrico ou fluxo de retreinamento deve ser refletida em `requisitos.md` (incluindo registro na tabela de histórico de revisões) antes ou em conjunto com a modificação do código.

---

## 2. Arquitetura e Engenharia de Software (Stack Python / Django / ML)

### 2.1. Separação Estrita de Camadas
- **`biometrics/ml/` (Núcleo de Machine Learning):**
  - Módulos puros de ciência de dados e aprendizado de máquina (`features.py`, `m2005.py`, `training.py`, `registry.py`, `inference.py`).
  - **Proibido** importar classes de `rest_framework` ou objetos HTTP (`Request`, `Response`) dentro de `biometrics/ml/`.
- **`biometrics/services.py` (Camada de Aplicação / Orquestração):**
  - Concentra toda a lógica transacional, regras de domínio (`enroll_user`, `authenticate_user`, `label_for`, `retrain_user`, `retrain_due_users`, `activate_model_version`, `rollback_models_after`) e controle de concorrência (`select_for_update(skip_locked=True)`). O cadastro de usuários é responsabilidade exclusiva do `projeto-si-backend`.
- **`biometrics/views.py` e `biometrics/serializers.py` (Camada HTTP):**
  - Responsáveis exclusivamente por validar formatos de entrada, delegar a execução para `biometrics/services.py` e traduzir resultados ou exceções de domínio nos códigos HTTP do contrato (`200`, `201`, `400`, `404`, `409`, `429`, `500`, `503`).
- **`biometrics/models.py` (Persistência):**
  - Define `User`, `Amostra` e `ModeloBiometrico` com `unique(usuario, versao)` (`ix_modelos_usuario_id_versao`) e `versao > 0` (`ck_modelos_versao_positiva`); a versão ativa é controlada por `active_version.json` e por `get_active_model`, não por uma constraint de banco.

### 2.2. Qualidade de Código e Segurança
- **Tipagem e Estilo:** Utilizar *type hints* do Python 3.12+ (`list[str]`, `dict[str, float]`, `X | None`) e manter conformidade estrita com `ruff check .` (zero avisos ou erros).
- **Segurança de Modelos:** Nunca carregar arquivos `.joblib` sem antes validar o hash `sha256` registrado no `.joblib.meta.json`/nome do artefato via `biometrics/ml/registry.py`.
- **Precisão de Cronometragem:** Toda medição de tempo de inferência deve utilizar `time.perf_counter()`.

---

## 3. Testes Automatizados Obrigatórios

- Toda funcionalidade do núcleo de ML, serviços, endpoints HTTP e comandos de gerenciamento deve possuir cobertura automatizada na pasta `tests/`:
  - `tests/test_features.py`: validação de faixas (`H`, `DD`, `UD`), estrutura `3n-2`, schema posicional e rejeição de vetores inválidos.
  - `tests/test_training.py`: treinamento do `AdaptiveM2005Alg` (`M2005_IDB`) utilizando exclusivamente amostras genuínas, preservação do crescimento incremental Welford vs janela deslizante limitada e compatibilidade com `M2005Alg`.
  - `tests/test_registry.py`: versionamento incremental, troca da versão ativa via `active_version.json`, bloqueio de arquivo adulterado (SHA-256) e cache LRU.
  - `tests/test_api.py`: fluxo de enrollment (criando apenas o modelo), autenticação sem senha (com validação estrita de dimensão/ordem de features, retorno de `shouldRecordForRetraining` e zero escrita em `amostras`), remoção de `/api/register` (`404`) e códigos HTTP (`200`, `201`, `400`, `404`, `409`, `429`, `500`), incluindo o endpoint interno `/api/models/rollback`.
  - `tests/test_retrain.py`: janelas deslizantes de amostras, exclusão de amostras abaixo do threshold, `--force` e rollback via `activate_model`.
  - `tests/test_logging.py`: redação de dados sensíveis, propagação de `X-Correlation-ID` e ativação de logs de depuração.
- Antes de encerrar qualquer tarefa, execute obrigatoriamente:
  ```bash
  make test
  make lint
  ```
