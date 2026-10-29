"""训练日志：写 CSV（可被 tools/plot_results.py 直接读取）。"""

from __future__ import annotations

import csv
import os


class CSVLogger:
    def __init__(self, path: str, fields, append: bool = False):
        self.path = path
        self.fields = list(fields)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        is_new = not (append and os.path.exists(path) and os.path.getsize(path) > 0)
        self._fh = open(path, "w" if is_new else "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fields, extrasaction="ignore")
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, row: dict) -> None:
        self._writer.writerow({key: row.get(key, "") for key in self.fields})
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
