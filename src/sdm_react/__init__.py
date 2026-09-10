"""SDM-REACT 初期実装 (設計書 v1.0, §22, §26 Stage 0-2, §30).

研究開発用の設計仕様の実装であり、実装済み・実験検証済みではない
(設計書 §0 文書状態を参照)。物性値・実験条件の既定値は含まない。
"""

from sdm_react.core import (
    ALL_COMPONENTS,
    ELEMENTS,
    OPTIONAL_COMPONENTS,
    BaseModel,
    Capability,
    DomainStatus,
    ExecStatus,
    IdentStatus,
    MaterialState,
    ModelOutput,
    OutputNature,
    Provenance,
    ReachabilityVerdict,
    ResultStatus,
    Verdict3,
    combine_verdicts,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_COMPONENTS",
    "ELEMENTS",
    "OPTIONAL_COMPONENTS",
    "BaseModel",
    "Capability",
    "DomainStatus",
    "ExecStatus",
    "IdentStatus",
    "MaterialState",
    "ModelOutput",
    "OutputNature",
    "Provenance",
    "ReachabilityVerdict",
    "ResultStatus",
    "Verdict3",
    "combine_verdicts",
]
