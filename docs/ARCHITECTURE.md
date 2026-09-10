# SDM-REACT 初期実装 — 構成対応表

設計書 `SDM-REACT_設計書.md` v1.0 に対する初期実装 (§26 Stage 0–2, §30)。
方針: 一つの作製介入について、作製条件→反応予測→冷却・測定→検証→実験選択を
縦断する小さな一連の系を先に成立させる。

> 研究開発用の設計仕様の実装であり、実験検証済みではない (設計書 §0)。
> 実験温度・保持時間・粒径・配合・物性値の既定値は含まない。

## モジュール対応 (§22.1)

| モジュール | ファイル | 設計書 | 主責務 |
|---|---|---|---|
| GoalCompiler | `src/sdm_react/goals.py` | §5 | Q1–Q8 目標の分解・三値判定 |
| ProcessState | `src/sdm_react/process.py` | §8 | 作製条件→初期状態候補 (作製前/条件付きの分離) |
| ScopeAudit | `src/sdm_react/scope.py` | §7 | 支配機構の監査・モデル分岐 |
| Reachability | `src/sdm_react/reachability.py` | §9 | 原料量・輸送量の外側評価 (最大フロー) |
| ReactionSolver | `src/sdm_react/reaction.py` + `inventory.py` | §10–§13 | 在庫台帳・M0–M4・経路・参加時刻・時間窓 |
| ObservationSolver | `src/sdm_react/observation.py` | §14 | 冷却 C1–C4・観測式・観測可能性 |
| AlternativeExplorer | `src/sdm_react/alternatives.py` | §15 | 対立説明10種・整合集合・識別評価 |
| EvidenceStore | `src/sdm_react/evidence.py` | §18–§19 | 情報単位・点密度・面積割合・独立性 |
| ExperimentPlanner | `src/sdm_react/planner.py` | §16–§17, §20, §23.3 | 配置検査・操作可能性・多軸選択・行動票 |
| Validation | `src/sdm_react/validation.py` | §21, §25 | V1–V4・N01–N12・前向き検証手順 |
| 全体連鎖 | `src/sdm_react/pipeline.py` | §6, §29 | 12段階パイプライン・最終成果物の器 |
| 共通基盤 | `src/sdm_react/core.py` | §4, §22.2–22.3 | 数量・状態・単位・共通IF・結果4軸 |

## 禁止事項の実装 (§2.2)

- M0→組成場の昇格は `NotImplementedError` で拒否
- 相モデルなしの相予測: M1/M2 の `unpredictable` に明示
- P3 供給源由来の単純按分は拒否 (P1/P2 のみ提供)
- C1 は確認根拠なしに設定不可、C4 は高温主張に制限を付与
- τ の事後変更は `ConsistencySpec.registered_before_observation` で防止
- 画素≠独立試料 (`count_independent`)、未分類は分母に保持 (`AreaFraction`)
- 確率根拠なしの期待値推奨なし (`ExperimentPlanner.select`)

## 使い方

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
python examples/stage0_first_question.py
```

## 検証対応 (§25)

- `tests/test_inventory.py`: N01–N04, N07–N09
- `tests/test_reaction.py`: N06 (erf収束), N10 (散逸), N11 (刻み依存), N12 (境界宣言)
- `tests/test_core.py`: N05 (換算往復)
- `tests/test_reachability.py`: §25.2 外側評価の性質
- `tests/test_obs_alt.py`: §25.3 偽中間組成・識別限界
- `tests/test_evidence.py`: §25.4 画素/試料・選択測定・未分類・ノイズ
- `tests/test_pipeline_validation.py`: §25.5 前向き検証手順
