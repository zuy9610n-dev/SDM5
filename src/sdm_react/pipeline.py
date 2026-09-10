"""全体連鎖: 目標定義から独立検証までの 12 段階 (設計書 §6, §29)。

① 目標情報・設備・予算の定義
② 支配機構とデータ適用範囲の監査
③ 作製可能な候補設計の生成
④ 焼結後初期状態の候補集合を構築
⑤ 到達可能性の外側評価
⑥ 有限供給・反応経路・熱履歴の予測
⑦ 冷却・仮想測定
⑧ 対立説明と識別可能性の評価
⑨ 情報増分・費用・頑健性の比較
⑩ 作製/追加測定/識別実験を選択
⑪ 実験前予測を固定
⑫ 独立検証・モデル更新

各処理は自身に必要な入力だけを要求する (§6)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sdm_react.core import ExecStatus, ResultStatus
from sdm_react.goals import CompiledGoal, GoalCompiler, TargetSpec
from sdm_react.planner import ActionTicket, ExperimentPlanner
from sdm_react.process import FabDesign, InitialStateEnsemble, ProcessState
from sdm_react.scope import ScopeAudit


@dataclass
class StageResult:
    id: str
    title: str
    status: ResultStatus = field(default_factory=ResultStatus)
    output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CampaignResult:
    """最終成果物の入れ物 (§29)。"""

    applicability_map: Dict[str, Any] = field(default_factory=dict)
    path_map: Dict[str, Any] = field(default_factory=dict)
    time_windows: Dict[str, Any] = field(default_factory=dict)
    protocol: Dict[str, Any] = field(default_factory=dict)
    ledger: Dict[str, Any] = field(default_factory=dict)
    effectiveness: Dict[str, Any] = field(default_factory=dict)
    stages: List[StageResult] = field(default_factory=list)


# 各段階で使う計算の型: ctx (共有文脈) -> output dict
StageFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class Pipeline:
    """12 段階の縦断的連鎖。初期実装は一つの作製介入を対象とする (§30)。"""

    STAGES = [
        ("s01", "目標情報・設備・予算の定義"),
        ("s02", "支配機構とデータ適用範囲の監査"),
        ("s03", "作製可能な候補設計の生成"),
        ("s04", "焼結後初期状態の候補集合を構築"),
        ("s05", "到達可能性の外側評価"),
        ("s06", "有限供給・反応経路・熱履歴の予測"),
        ("s07", "冷却・仮想測定"),
        ("s08", "対立説明と識別可能性の評価"),
        ("s09", "情報増分・費用・頑健性の比較"),
        ("s10", "作製/追加測定/識別実験を選択"),
        ("s11", "実験前予測を固定"),
        ("s12", "独立検証・モデル更新"),
    ]

    def __init__(self) -> None:
        self.goals = GoalCompiler()
        self.process = ProcessState()
        self.scope = ScopeAudit()
        self.planner = ExperimentPlanner()
        self._fns: Dict[str, StageFn] = {}

    def on(self, stage_id: str, fn: StageFn) -> "Pipeline":
        """段階の計算を登録する。未登録の段階は skipped となる。"""
        self._fns[stage_id] = fn
        return self

    def run(self, ctx: Optional[Dict[str, Any]] = None) -> CampaignResult:
        ctx = dict(ctx or {})
        result = CampaignResult()
        for sid, title in self.STAGES:
            fn = self._fns.get(sid)
            if fn is None:
                result.stages.append(StageResult(
                    sid, title,
                    ResultStatus(execution=ExecStatus.INSUFFICIENT_INPUT,
                                 notes=["未登録のため skip"]),
                ))
                continue
            try:
                out = fn(ctx)
                ctx[sid] = out
                result.stages.append(StageResult(
                    sid, title,
                    ResultStatus(notes=["completed"]),
                    out,
                ))
            except Exception as exc:  # noqa: BLE001 — 段階失敗を記録して停止
                result.stages.append(StageResult(
                    sid, title,
                    ResultStatus(execution=ExecStatus.NUMERICAL_FAILURE,
                                 notes=[f"{type(exc).__name__}: {exc}"]),
                ))
                break
        ctx["campaign_result"] = result
        return result

    # -- 段階 ①④ の標準実装 (他は研究課題ごとに登録) ---------------------
    def compile_goals(self, targets: List[TargetSpec]) -> Dict[str, Any]:
        compiled: List[CompiledGoal] = [self.goals.compile(t) for t in targets]
        return {"compiled": compiled}

    def build_initial_ensembles(
        self, designs: List[FabDesign], candidates_fn: StageFn, ctx: Dict[str, Any]
    ) -> Dict[str, InitialStateEnsemble]:
        out: Dict[str, InitialStateEnsemble] = {}
        for d in designs:
            cands = candidates_fn({**ctx, "design": d}).get("candidates", [])
            out[d.id] = self.process.predict_initial(d, cands)
        return {"ensembles": out}

    @staticmethod
    def summarize(result: CampaignResult) -> Dict[str, Any]:
        return {
            s.id: {"title": s.title, "execution": s.status.execution.value,
                   "notes": s.status.notes}
            for s in result.stages
        }


def default_ticket_for(
    action_id: str, target_ids: List[str], role: str, kind: str
) -> ActionTicket:
    """行動票の雛形 (§23.3)。"""
    from sdm_react.planner import ActionKind, ActionRole

    return ActionTicket(
        id=action_id,
        target_ids=target_ids,
        role=ActionRole(role),
        kind=ActionKind(kind),
        uncertainty_representation="scenario",
    )
