from typing import Any

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError, CommandParser

from biometrics.models import TrainingTrigger
from biometrics.services import retrain_due_users, retrain_user


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
            user = User.objects.filter(username=username).first()
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
                    self.style.SUCCESS(f"Modelo retreinado para '{username}': v{trained.version}.")
                )
            return

        retrained_models = retrain_due_users(force=force)
        self.stdout.write(
            self.style.SUCCESS(
                f"Retreinamento concluído. Modelos gerados: {len(retrained_models)}."
            )
        )
