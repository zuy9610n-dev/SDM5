"""GoalCompiler: 研究目標を必要条件・観測条件へ分解する (設計書 §5).

目標情報 Q1-Q8 の仕様・品質判定 (三値) を扱う。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from sdm_react.core import Verdict3, combine_verdicts


class InformationType(str, Enum):
    """目標情報の種類 (§5.1)。"""

    Q1 = "Q1"  # 五元素が定量された反応領域
    Q2 = "Q2"  # 混合影響を評価できる局所平均組成
    Q3 = "Q3"  # 同定相の相内部組成
    Q4 = "Q4"  # 相共存・相間分配
    Q5 = "Q5"  # 第五元素操作への応答
    Q6 = "Q6"  # 反応機構または失敗機構の識別
    Q7 = "Q7"  # 未説明材料状態の候補
    Q8 = "Q8"  # 候補の独立再現・形成経路の確認


INFO_DESCRIPTIONS: Dict[InformationType, str] = {
    InformationType.Q1: "五元素が定量された反応領域",
    InformationType.Q2: "混合影響を評価できる局所平均組成",
    InformationType.Q3: "同定相の相内部組成",
    InformationType.Q4: "相共存・相間分配",
    InformationType.Q5: "第五元素操作への応答",
    InformationType.Q6: "反応機構または失敗機構の識別",
    InformationType.Q7: "未説明材料状態の候補",
    InformationType.Q8: "候補の独立再現・形成経路の確認",
}


@dataclass
class Criterion:
    """成否判定基準。測定値が None (未測定) の場合は unknown。"""

    id: str
    description: str
    metric: str
    threshold: Optional[float] = None
    comparator: str = ">="  # >=, <=, == のいずれか
    required: bool = True

    def judge(self, measured: Optional[float]) -> Verdict3:
        if measured is None or self.threshold is None:
            return "unknown"
        if self.comparator == ">=":
            return "pass" if measured >= self.threshold else "fail"
        if self.comparator == "<=":
            return "pass" if measured <= self.threshold else "fail"
        if self.comparator == "==":
            return "pass" if measured == self.threshold else "fail"
        raise ValueError(f"unknown comparator: {self.comparator}")


@dataclass
class TargetSpec:
    """目標情報の仕様 (§5.2)。"""

    id: str
    information_type: InformationType
    scientific_question: str = ""
    composition_region: Dict[str, Any] = field(default_factory=dict)
    temperature_reference: Dict[str, Any] = field(default_factory=dict)
    material_state_required: List[str] = field(default_factory=list)
    minimum_observation_requirements: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    competing_interpretations: List[str] = field(default_factory=list)
    success_criteria: List[Criterion] = field(default_factory=list)
    failure_criteria: List[Criterion] = field(default_factory=list)
    inconclusive_conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def c2d(c: Criterion) -> Dict[str, Any]:
            return {
                "id": c.id,
                "description": c.description,
                "metric": c.metric,
                "threshold": c.threshold,
                "comparator": c.comparator,
                "required": c.required,
            }

        return {
            "target": {
                "id": self.id,
                "information_type": self.information_type.value,
                "scientific_question": self.scientific_question,
                "composition_region": self.composition_region,
                "temperature_reference": self.temperature_reference,
                "material_state_required": self.material_state_required,
                "minimum_observation_requirements": (
                    self.minimum_observation_requirements
                ),
                "required_evidence": self.required_evidence,
                "competing_interpretations": self.competing_interpretations,
                "success_criteria": [c2d(c) for c in self.success_criteria],
                "failure_criteria": [c2d(c) for c in self.failure_criteria],
                "inconclusive_conditions": self.inconclusive_conditions,
            }
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TargetSpec":
        t = d.get("target", d)

        def d2c(x: Dict[str, Any]) -> Criterion:
            return Criterion(
                id=str(x.get("id", "")),
                description=str(x.get("description", "")),
                metric=str(x.get("metric", "")),
                threshold=x.get("threshold"),
                comparator=str(x.get("comparator", ">=")),
                required=bool(x.get("required", True)),
            )

        return cls(
            id=str(t.get("id", "")),
            information_type=InformationType(str(t.get("information_type", "Q1"))),
            scientific_question=str(t.get("scientific_question", "")),
            composition_region=dict(t.get("composition_region", {})),
            temperature_reference=dict(t.get("temperature_reference", {})),
            material_state_required=list(t.get("material_state_required", [])),
            minimum_observation_requirements=list(
                t.get("minimum_observation_requirements", [])
            ),
            required_evidence=list(t.get("required_evidence", [])),
            competing_interpretations=list(t.get("competing_interpretations", [])),
            success_criteria=[d2c(x) for x in t.get("success_criteria", [])],
            failure_criteria=[d2c(x) for x in t.get("failure_criteria", [])],
            inconclusive_conditions=list(t.get("inconclusive_conditions", [])),
        )


@dataclass
class CompiledGoal:
    """分解された目標: 必要条件の一覧。"""

    target: TargetSpec
    material_requirements: List[str]
    observation_requirements: List[str]
    evidence_requirements: List[str]
    notes: List[str] = field(default_factory=list)


@dataclass
class Judgement:
    target_id: str
    overall: Verdict3
    per_criterion: Dict[str, Verdict3]
    failed: List[str]
    unknown: List[str]
    notes: List[str] = field(default_factory=list)


class GoalCompiler:
    """研究目標の分解と三値品質判定 (§5.2-5.3)。"""

    def compile(self, target: TargetSpec) -> CompiledGoal:
        """目標を材料・観測・証拠の必要条件へ分解する。"""
        notes = [
            f"情報種別: {target.information_type.value} "
            f"({INFO_DESCRIPTIONS[target.information_type]})"
        ]
        if INFO_DESCRIPTIONS.get(target.information_type) is None:
            notes.append("未知の情報種別。成果指標を分けて記録すること。")
        return CompiledGoal(
            target=target,
            material_requirements=list(target.material_state_required),
            observation_requirements=list(target.minimum_observation_requirements),
            evidence_requirements=list(target.required_evidence),
            notes=notes,
        )

    def judge(
        self,
        target: TargetSpec,
        measurements: Dict[str, Optional[float]],
    ) -> Judgement:
        """成功基準の三値判定。失敗基準のいずれかが pass なら全体 fail。

        判定不能は合格・不合格へ自動変換しない (§5.3)。
        """
        per: Dict[str, Verdict3] = {}
        # 失敗基準: 基準を満たす = 失敗が確定
        for c in target.failure_criteria:
            v = c.judge(measurements.get(c.metric))
            per[f"failure:{c.id}"] = v
            if v == "pass":
                return Judgement(
                    target_id=target.id,
                    overall="fail",
                    per_criterion=per,
                    failed=[f"failure:{c.id}"],
                    unknown=[],
                    notes=[f"失敗基準 {c.id} が成立したため fail"],
                )
        required_vs: List[Verdict3] = []
        failed, unknown = [], []
        for c in target.success_criteria:
            v = c.judge(measurements.get(c.metric))
            per[c.id] = v
            if c.required:
                required_vs.append(v)
                if v == "fail":
                    failed.append(c.id)
                elif v == "unknown":
                    unknown.append(c.id)
            elif v == "unknown":
                unknown.append(c.id)
        overall = combine_verdicts(required_vs) if required_vs else "unknown"
        return Judgement(
            target_id=target.id,
            overall=overall,
            per_criterion=per,
            failed=failed,
            unknown=unknown,
        )
