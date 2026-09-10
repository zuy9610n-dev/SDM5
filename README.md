私の研究目的および求めることを以下に示す
私は⾼効率な材料探索⼿法である焼結拡散マルチプル法について研究しています。
材料を構成する元素数が増加するとそれに伴って調べる必要のある組成は急激に増加します。
例えば、5元素材料では元素濃度を1%ずつ変化させると約460万通りの組成となり、すべてを作製・評価することは困難です。
そこで、1つの試料から多数の組成を取得できる本⼿法を開発しています。
焼結拡散マルチプル法は、異なる組成の材料を焼結・焼鈍し、組成分析と点密度解析から状態図の概観を効率的に調べる手法です。
一つの試料から多数の組成情報を取得できるため、新規材料開発の高速化が期待されます。
しかし、適用は現在4元系までに限られているため、Al、Si、Ti、Fe、Cuからなる5元系を対象に、5元素粉末を焼結・焼鈍し、拡散を利⽤して多数の濃度の組合せを作り、組織観察・分析までを行います。
それにより、焼結拡散マルチプル法の5元系への適用条件の解明と新規物質探索を目指し研究しています。

/deep-thinking

設計書をもとにコーディングしてください。

---

## 実装: SDM-REACT 初期版 (設計書 v1.0, §26 Stage 0–2, §30)

`SDM-REACT_設計書.md` に従い、一つの作製介入について
作製条件→反応予測→冷却・測定→検証→実験選択を縦断する小さな一連の系を実装した。
研究開発用の設計仕様の実装であり、実験検証済みではない (設計書 §0)。
実験温度・保持時間・粒径・配合・物性値の既定値は含まない。

### 構成

| モジュール | ファイル | 設計書 |
|---|---|---|
| GoalCompiler | `src/sdm_react/goals.py` | §5 |
| ProcessState | `src/sdm_react/process.py` | §8 |
| ScopeAudit | `src/sdm_react/scope.py` | §7 |
| Reachability | `src/sdm_react/reachability.py` | §9 |
| ReactionSolver | `src/sdm_react/reaction.py`, `inventory.py` | §10–§13 |
| ObservationSolver | `src/sdm_react/observation.py` | §14 |
| AlternativeExplorer | `src/sdm_react/alternatives.py` | §15 |
| EvidenceStore | `src/sdm_react/evidence.py` | §18–§19 |
| ExperimentPlanner | `src/sdm_react/planner.py` | §16–§17, §20, §23.3 |
| Validation | `src/sdm_react/validation.py` | §21, §25 |
| 全体連鎖 | `src/sdm_react/pipeline.py` | §6, §29 |

詳細は `docs/ARCHITECTURE.md` を参照。

### 実行

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
python examples/stage0_first_question.py  # §27 第一版の研究課題のデモ
```
