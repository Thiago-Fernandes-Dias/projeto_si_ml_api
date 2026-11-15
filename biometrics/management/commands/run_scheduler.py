import logging
import uuid
from typing import Any

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import close_old_connections
from django.utils import timezone

from biometrics.logging_utils import correlation_context, log_structured
from biometrics.services import retrain_due_users

logger = logging.getLogger(__name__)


def scheduled_retrain_job() -> int:
    close_old_connections()
    correlation_id = f"sched-{uuid.uuid4().hex[:12]}"
    try:
        with correlation_context(correlation_id, flow="retrain"):
            log_structured(
                logger,
                logging.DEBUG,
                "scheduler.retrain_cycle_started",
                "Ciclo agendado de retreinamento iniciado.",
                flow="retrain",
            )
            retrained = retrain_due_users(now=timezone.now(), force=False)
            log_structured(
                logger,
                logging.INFO,
                "scheduler.retrain_cycle_completed",
                "Execução agendada concluída.",
                flow="retrain",
                retrainedCount=len(retrained),
            )
            return len(retrained)
    finally:
        close_old_connections()


class Command(BaseCommand):
    help = "Inicia o BlockingScheduler do APScheduler para retreinamento periódico dos modelos."

    def handle(self, *args: Any, **options: Any) -> None:
        interval_minutes = int(settings.BIOMETRICS["RETRAIN_INTERVAL_MINUTES"])
        scheduler = BlockingScheduler(timezone=str(settings.TIME_ZONE))
        scheduler.add_job(
            scheduled_retrain_job,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id="retrain_due_users",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        msg = (
            f"Scheduler iniciado (intervalo={interval_minutes} min, "
            "max_instances=1, coalesce=True)."
        )
        self.stdout.write(self.style.SUCCESS(msg))
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.stdout.write(self.style.WARNING("Encerrando scheduler..."))
            scheduler.shutdown(wait=False)
