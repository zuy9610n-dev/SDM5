"""§6 全体連鎖と §21・§25 検証管理の試験。"""
import pytest

from sdm_react.goals import Criterion, InformationType, TargetSpec
from sdm_react.pipeline import Pipeline
from sdm_react.validation import (
    ValidationRecord,
    ValidationStage,
    forward_validation_protocol,
)


def _target():
    return TargetSpec(
        id="T1", information_type=InformationType.Q1,
        success_criteria=[Criterion("c1", "d", "m", 1.0, ">=")],
    )


def test_pipeline_runs_registered_stages_only():
    p = Pipeline()
    p.on("s01", lambda ctx: p.compile_goals([_target()]))
    p.on("s05", lambda ctx: {"verdict": "unresolved"})
    res = p.run({})
    summ = Pipeline.summarize(res)
    assert summ["s01"]["execution"] == "success"
    assert summ["s02"]["execution"] == "insufficient_input"  # 未登録は skip
    assert summ["s05"]["execution"] == "success"
    assert res.stages[0].output["compiled"][0].target.id == "T1"


def test_pipeline_records_stage_failure():
    p = Pipeline()

    def boom(ctx):
        raise RuntimeError("model failed")

    p.on("s01", boom)
    res = p.run({})
    assert res.stages[0].status.execution.value == "numerical_failure"
    assert len(res.stages) == 1  # 失敗で停止


def test_forward_protocol_rejects_overlap():
    with pytest.raises(ValueError):
        forward_validation_protocol(["t=1h"], ["t=1h"])  # 校正と検証の重複禁止 (§25.5)
    proto = forward_validation_protocol(["t=1h"], ["t=10h"])
    assert proto["validation"] == ["t=10h"]


def test_validation_record_distinguishes_prediction_kind():
    r = ValidationRecord(id="v1", stage=ValidationStage.V2, target="obs Al",
                         prediction_kind="pre_fab")
    assert r.prediction_kind == "pre_fab"  # 作製前/条件付きを別評価 (§8.5)
    assert r.stage == ValidationStage.V2
