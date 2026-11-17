import json
import re
import uuid
from pathlib import Path
from typing import Any

from django.db import models
from django.utils import timezone

FILENAME_SHA256_RE = re.compile(r"_([0-9a-f]{64})\.joblib$")


class FloatArrayField(models.Field):
    description = "Vetor de pontos flutuantes (double precision[] no PostgreSQL, JSON no SQLite)"

    def db_type(self, connection: Any) -> str:
        if connection.vendor == "postgresql":
            return "double precision[]"
        return "text"

    def from_db_value(self, value: Any, expression: Any, connection: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, str):
            parsed = json.loads(value)
            return [float(x) for x in parsed]
        return [float(x) for x in value]

    def to_python(self, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, str):
            parsed = json.loads(value)
            return [float(x) for x in parsed]
        return [float(x) for x in value]

    def get_prep_value(self, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, str):
            parsed = json.loads(value)
            return [float(x) for x in parsed]
        return [float(x) for x in value]

    def get_db_prep_value(self, value: Any, connection: Any, prepared: bool = False) -> Any:
        arr = self.get_prep_value(value)
        if connection.vendor == "postgresql":
            return arr
        return json.dumps(arr)


class OrigemAmostra(models.TextChoices):
    CADASTRO = "Cadastro", "Cadastro"
    LOGIN = "Login", "Login"


class SampleSource:
    ENROLLMENT = OrigemAmostra.CADASTRO
    CADASTRO = OrigemAmostra.CADASTRO
    LOGIN = OrigemAmostra.LOGIN
    choices = OrigemAmostra.choices


class DecisaoBiometrica(models.TextChoices):
    GENUINE = "Genuine", "Genuine"
    IMPOSTOR = "Impostor", "Impostor"
    INVALID_CREDENTIALS = "InvalidCredentials", "InvalidCredentials"


class SampleLabel(models.TextChoices):
    GENUINE = "GENUINE", "Genuine"
    IMPOSTOR = "IMPOSTOR", "Impostor"


class TrainingTrigger(models.TextChoices):
    ENROLLMENT = "ENROLLMENT", "Enrollment"
    SCHEDULED = "SCHEDULED", "Scheduled"
    MANUAL = "MANUAL", "Manual"


class User(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    nome = models.CharField(max_length=50, unique=True)
    data_mudanca_senha = models.DateTimeField(default=timezone.now)
    biometria_cadastrada_em = models.DateTimeField(null=True, blank=True)
    tentativas_falhas = models.IntegerField(default=0)
    bloqueado_ate = models.DateTimeField(null=True, blank=True)
    criado_em = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "usuarios"

    def __str__(self) -> str:
        return f"Usuario({self.nome})"

    @property
    def username(self) -> str:
        return self.nome

    @property
    def possui_biometria(self) -> bool:
        return self.biometria_cadastrada_em is not None


class ModeloBiometrico(models.Model):
    id = models.BigAutoField(primary_key=True)
    usuario = models.ForeignKey(
        User,
        db_column="usuario_id",
        on_delete=models.CASCADE,
        related_name="modelos",
    )
    versao = models.IntegerField()
    algoritmo = models.CharField(max_length=50)
    caminho_arquivo = models.CharField(max_length=500)
    criado_em = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "modelos"
        constraints = [
            models.UniqueConstraint(
                fields=["usuario", "versao"],
                name="ix_modelos_usuario_id_versao",
            ),
            models.CheckConstraint(
                condition=models.Q(versao__gt=0),
                name="ck_modelos_versao_positiva",
            ),
        ]

    def __str__(self) -> str:
        return f"ModeloBiometrico({self.usuario.nome}, v{self.versao}, {self.algoritmo})"

    @property
    def user(self) -> User:
        return self.usuario

    @property
    def version(self) -> int:
        return self.versao

    @property
    def file_path(self) -> str:
        return self.caminho_arquivo

    @property
    def created_at(self) -> Any:
        return self.criado_em

    @property
    def is_active(self) -> bool:
        if self.caminho_arquivo:
            active_file = Path(self.caminho_arquivo).parent / "active_version.json"
            if active_file.is_file():
                try:
                    data = json.loads(active_file.read_text(encoding="utf-8"))
                    active_v = data.get("active_version")
                    if isinstance(active_v, int):
                        return self.versao == active_v
                except Exception:
                    pass
        latest_id = (
            ModeloBiometrico.objects.filter(usuario_id=self.usuario_id)
            .order_by("-versao", "-id")
            .values_list("id", flat=True)
            .first()
        )
        return bool(latest_id is not None and latest_id == self.id)

    def _meta_path(self) -> Path:
        return Path(f"{self.caminho_arquivo}.meta.json")

    def _read_meta(self) -> dict[str, Any]:
        meta_file = self._meta_path()
        if meta_file.is_file():
            try:
                return json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _write_meta(self, updates: dict[str, Any]) -> None:
        meta_file = self._meta_path()
        current = self._read_meta()
        current.update(updates)
        if meta_file.parent.exists():
            meta_file.write_text(json.dumps(current, indent=2), encoding="utf-8")

    @property
    def sha256(self) -> str:
        override = self.__dict__.get("_sha256_override")
        if override:
            return str(override)
        match = FILENAME_SHA256_RE.search(self.caminho_arquivo or "")
        if match:
            return match.group(1)
        meta = self._read_meta()
        return str(meta.get("sha256", ""))

    @sha256.setter
    def sha256(self, value: str) -> None:
        self.__dict__["_sha256_override"] = value
        self._write_meta({"sha256": value})

    @property
    def sklearn_version(self) -> str:
        override = self.__dict__.get("_sklearn_version_override")
        if override:
            return str(override)
        meta = self._read_meta()
        return str(meta.get("sklearn_version", ""))

    @sklearn_version.setter
    def sklearn_version(self, value: str) -> None:
        self.__dict__["_sklearn_version_override"] = value
        self._write_meta({"sklearn_version": value})

    @property
    def training_stats(self) -> dict[str, Any]:
        override = self.__dict__.get("_training_stats_override")
        if isinstance(override, dict):
            return override
        meta = self._read_meta()
        stats = meta.get("training_stats")
        return stats if isinstance(stats, dict) else {}

    @training_stats.setter
    def training_stats(self, value: dict[str, Any]) -> None:
        self.__dict__["_training_stats_override"] = value
        self._write_meta({"training_stats": value})

    @property
    def trigger(self) -> str:
        override = self.__dict__.get("_trigger_override")
        if override:
            return str(override)
        meta = self._read_meta()
        return str(meta.get("trigger", TrainingTrigger.ENROLLMENT))

    @trigger.setter
    def trigger(self, value: str) -> None:
        self.__dict__["_trigger_override"] = value
        self._write_meta({"trigger": value})


class Amostra(models.Model):
    id = models.BigAutoField(primary_key=True)
    usuario = models.ForeignKey(
        User,
        db_column="usuario_id",
        on_delete=models.CASCADE,
        related_name="amostras",
    )
    origem = models.CharField(max_length=20, choices=OrigemAmostra.choices)
    vetor = FloatArrayField()
    repeticao = models.IntegerField(null=True, blank=True)
    duracao_total_ms = models.FloatField(default=0.0)
    decisao = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        choices=DecisaoBiometrica.choices,
    )
    confianca = models.FloatField(null=True, blank=True)
    modelo = models.ForeignKey(
        ModeloBiometrico,
        db_column="modelo_id",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="amostras",
    )
    coletada_em = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "amostras"
        indexes = [
            models.Index(fields=["modelo"], name="ix_amostras_modelo_id"),
            models.Index(
                fields=["usuario", "origem", "coletada_em"],
                name="ix_amostras_usr_orig_coletada",
            ),
        ]
        constraints = [
            models.CheckConstraint(
                condition=~models.Q(origem="Login") | models.Q(decisao__isnull=False),
                name="ck_amostras_login_com_decisao",
            ),
        ]

    def __str__(self) -> str:
        return f"Amostra({self.usuario.nome}, {self.origem}, {self.decisao})"


TrainedModel = ModeloBiometrico
KeystrokeSample = Amostra
