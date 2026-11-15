from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from biometrics.models import TrainingTrigger, Usuario
from biometrics.services import normalize_username, retrain_due_users, retrain_user


class Command(BaseCommand):
    help = "Retreina modelos biométricos dos usuários elegíveis ou de um usuário específico."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--username",
            type=str,
            default=None,
            help="Username específico para retreinamento.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Força o retreinamento mesmo sem atingir RETRAIN_MIN_NEW_SAMPLES.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        username: str | None = options.get("username")
        force: bool = bool(options.get("force"))

        if username:
            clean = normalize_username(username)
            user = Usuario.objects.filter(nome__iexact=clean).first()
            if user is None:
                raise CommandError(f"Usuário '{username}' não encontrado.")
            trained = retrain_user(
                user=user,
                trigger=TrainingTrigger.MANUAL,
                force=force,
            )
            if trained is None:
                msg = (
                    f"Usuário '{username}' ignorado "
                    "(sem amostras novas suficientes ou sem enrollment)."
                )
                self.stdout.write(self.style.WARNING(msg))
            else:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Modelo retreinado para '{user.nome}': v{trained.versao}."
                    )
                )
            return

        retrained_models = retrain_due_users(force=force)
        self.stdout.write(
            self.style.SUCCESS(
                f"Retreinamento concluído. Modelos gerados: {len(retrained_models)}."
            )
        )
