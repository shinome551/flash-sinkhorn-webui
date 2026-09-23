"""マッチ結果に添える警告 (``MatchResponse.warnings``)。

フロントはコードと ``params`` から自前の文言 (日本語) を組み立て、未知のコードは ``message``
(英語) をそのまま出す。コードを増やしたら ``frontend/src/lib/warnings.ts`` にも文言を足す。
"""

from dataclasses import dataclass, field
from enum import StrEnum


class WarningCode(StrEnum):
    PCA_DIM_REDUCED = "PCA_DIM_REDUCED"  # params: requested, actual
    ROW_MASS_ERROR = "ROW_MASS_ERROR"  # params: error, tolerance, converged
    NOT_CONVERGED = "NOT_CONVERGED"  # params: iterations


@dataclass(frozen=True)
class Notice:
    code: WarningCode
    message: str
    params: dict[str, int | float | bool] = field(default_factory=dict)
