# 前三周 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在空仓库中完成课程方案前 3 周的工程化实现，做到离线可跑通、联网后可扩展到真实数据。

**Architecture:** 使用 YAML 固化 schema，用 JSONL 作为语料与标注交换格式，用 gazetteer 弱标注衔接第 2 周与第 3 周。NER 侧统一为“读同一种标注格式 -> 训练不同模型 -> 输出同一种评估报告”的结构。

**Tech Stack:** Python 3.12, YAML, JSONL, scikit-learn, sklearn-crfsuite, PyTorch, Transformers, Typer, pytest

---

### Task 1: 建立仓库与 schema

**Files:**
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `config/ontology.yaml`
- Create: `docs/plans/2026-04-20-first-three-weeks-design.md`

**Step 1: 定义 schema 文件**

在 `config/ontology.yaml` 中写入 7 类实体、8 类关系和边属性。

**Step 2: 建立工程入口**

在 `pyproject.toml` 中声明基础依赖和 `kg-pipeline` CLI 入口。

**Step 3: 写运行说明**

在 `README.md` 中说明第 1-3 周与代码目录的映射关系。

**Step 4: 验证**

Run: `python3.12 - <<'PY'\nimport yaml\nprint(len(yaml.safe_load(open('config/ontology.yaml'))['entities']))\nPY`

Expected: 输出 `7`

### Task 2: 实现第 2 周数据管线

**Files:**
- Create: `src/kg_project/schema.py`
- Create: `src/kg_project/data_types.py`
- Create: `src/kg_project/text.py`
- Create: `src/kg_project/gazetteer.py`
- Create: `src/kg_project/labeling.py`
- Create: `data/fixtures/esco_skills.csv`
- Create: `data/fixtures/onet_reference.tsv`
- Create: `data/fixtures/jd_corpus.jsonl`

**Step 1: 先写数据结构与 schema 加载**

让 schema、JD、gazetteer、NER 样本都可序列化。

**Step 2: 实现 gazetteer 构建**

支持从 ESCO/O*NET fixture 构建统一词典，并输出 JSONL。

**Step 3: 实现 BIO 弱标注与 gold 抽样**

将 JD 文本切句、切 token、最长匹配打标，并生成 doccano 风格的抽样文件。

**Step 4: 验证**

Run: `kg-pipeline build-gazetteer --esco data/fixtures/esco_skills.csv --onet data/fixtures/onet_reference.tsv --output data/interim/gazetteer.jsonl`

Expected: 成功生成 gazetteer 文件

### Task 3: 实现第 3 周三套 NER 模型

**Files:**
- Create: `src/kg_project/ner/__init__.py`
- Create: `src/kg_project/ner/constants.py`
- Create: `src/kg_project/ner/data.py`
- Create: `src/kg_project/ner/features.py`
- Create: `src/kg_project/ner/crf_baseline.py`
- Create: `src/kg_project/ner/neural.py`
- Create: `src/kg_project/ner/experiments.py`
- Create: `data/fixtures/gold_ner.jsonl`

**Step 1: 先写统一数据读取和 CRF 特征**

保证 CRF 和神经网络都读同一种标注样本。

**Step 2: 写 CRF 基线**

先让最轻量的 `CRF` 训练与评估跑通。

**Step 3: 写 BiLSTM-CRF 与 BERT-CRF**

共用一套 `LinearChainCRF` 层与训练循环。

**Step 4: 写统一实验报告**

输出每个模型的 precision / recall / f1 / support 到 JSON。

**Step 5: 验证**

Run: `kg-pipeline compare-ner --dataset data/fixtures/gold_ner.jsonl --output data/reports/ner_report.json`

Expected: 成功生成 NER 报告；若缺少 `torch/transformers`，至少 CRF 路径可运行并在报告中标记跳过原因

### Task 4: 补测试与完成验收

**Files:**
- Create: `tests/test_schema.py`
- Create: `tests/test_labeling.py`
- Create: `tests/test_crf_pipeline.py`

**Step 1: 写失败测试**

先断言 schema 数量、弱标注输出、CRF 训练流程。

**Step 2: 运行并看它失败**

Run: `pytest`

Expected: 初次失败，提示缺少实现

**Step 3: 补最小实现直到通过**

按测试补齐对应模块。

**Step 4: 全量验证**

Run: `pytest`

Expected: 全绿
