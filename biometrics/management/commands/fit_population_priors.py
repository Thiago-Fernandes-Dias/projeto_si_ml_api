import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, CommandParser


class Command(BaseCommand):
    help = (
        "Ajusta distribuições populacionais (log-normal para H/DD, normal para UD) "
        "a partir do dataset CMU e grava population_priors.json."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "csv_path",
            type=str,
            help="Caminho para o arquivo CSV do dataset CMU (DSL-StrongPasswordData.csv).",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Caminho de saída para population_priors.json.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        csv_path = Path(options["csv_path"])
        if not csv_path.is_file():
            raise CommandError(f"Arquivo CSV não encontrado: '{csv_path}'.")

        output_path = (
            Path(options["output"])
            if options.get("output")
            else Path(settings.BIOMETRICS["POPULATION_PRIORS_PATH"])
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)
        h_cols = [c for c in df.columns if str(c).startswith("H.")]
        dd_cols = [c for c in df.columns if str(c).startswith("DD.")]
        ud_cols = [c for c in df.columns if str(c).startswith("UD.")]

        if not h_cols and not dd_cols and not ud_cols:
            raise CommandError("Nenhuma coluna H.*, DD.* ou UD.* encontrada no CSV informado.")

        priors: dict[str, dict[str, float]] = {}

        for prefix, cols in (("H", h_cols), ("DD", dd_cols)):
            if cols:
                vals = df[cols].to_numpy(dtype=np.float64).ravel()
                positive_vals = vals[np.isfinite(vals) & (vals > 1e-5)]
                log_vals = np.log(positive_vals)
                priors[prefix] = {
                    "log_mean": float(np.mean(log_vals)),
                    "log_std": float(np.std(log_vals, ddof=1)),
                }

        if ud_cols:
            ud_vals = df[ud_cols].to_numpy(dtype=np.float64).ravel()
            finite_ud = ud_vals[np.isfinite(ud_vals)]
            priors["UD"] = {
                "mean": float(np.mean(finite_ud)),
                "std": float(np.std(finite_ud, ddof=1)),
            }

        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(priors, fp, indent=2, sort_keys=True)

        self.stdout.write(self.style.SUCCESS(f"Priors populacionais gravados em '{output_path}'."))
