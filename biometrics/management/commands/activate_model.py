from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from biometrics.ml.registry import ModelIntegrityError
from biometrics.services import (
    ModelVersionNotFoundError,
    UserNotFoundError,
    activate_model_version,
)


class Command(BaseCommand):
    help = "Ativa uma versão específica de modelo biométrico (rollback/switch) para um usuário."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("username", type=str, help="Username do usuário.")
        parser.add_argument("version", type=int, help="Versão do modelo a ativar.")

    def handle(self, *args: Any, **options: Any) -> None:
        username = str(options["username"])
        version = int(options["version"])

        try:
            activated = activate_model_version(username=username, version=version)
        except (UserNotFoundError, ModelVersionNotFoundError, ModelIntegrityError) as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"Modelo v{activated.version} ativado com sucesso para '{username}'."
            )
        )
