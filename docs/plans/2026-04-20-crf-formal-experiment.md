# CRF Formal Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将当前 toy 级 CRF 基线升级为基于真实公开数据、可复现、可生成正式报告的英文 CRF 实验模块。

**Architecture:** 使用 `SkillSpan` 作为正式主评测集，保留其官方 `train/dev/test` 划分；使用 `ESCO + O*NET` 构建英文 gazetteer；使用公开英文 JD 语料做词典覆盖统计和辅助分析。实验层只保留 `CRF-base` 与 `CRF+gazetteer` 两组，并统一输出 JSON/Markdown 报告。

**Tech Stack:** Python 3.12, Typer, sklearn-crfsuite, seqeval, pandas, pyyaml, curl-based downloads

---

### Task 1: 固化实验配置与数据约定

**Files:**
- Create: `config/crf_experiment.yaml`
- Modify: `README.md`

**Step 1: 写实验配置文件**

在 `config/crf_experiment.yaml` 中写入：

- 数据源 URL
- 外部文件保存路径
- 官方/默认划分路径
- CRF 特征开关
- 随机种子
- 报告输出路径

**Step 2: 更新 README**

补充正式 CRF 实验命令和目录说明。

**Step 3: Run test to verify file loads**

Run: `.venv/bin/python - <<'PY'\nimport yaml\nprint(sorted(yaml.safe_load(open('config/crf_experiment.yaml'))['experiments'].keys()))\nPY`

Expected: 输出 `['crf_base', 'crf_gazetteer']`

### Task 2: 先写失败测试

**Files:**
- Create: `tests/test_data_sources.py`
- Create: `tests/test_preprocess.py`
- Create: `tests/test_crf_formal_experiment.py`

**Step 1: 写下载元数据测试**

断言数据源配置和路径解析正确。

**Step 2: 写 SkillSpan 转换测试**

断言原始 SkillSpan 样本可转成统一 JSONL。

**Step 3: 写正式 CRF 报告测试**

断言 `CRF-base` / `CRF+gazetteer` 都出现在报告中。

**Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_data_sources.py tests/test_preprocess.py tests/test_crf_formal_experiment.py -vv`

Expected: FAIL，提示缺少实现模块或新接口

### Task 3: 实现数据源下载与校验

**Files:**
- Create: `src/kg_project/data_sources.py`
- Modify: `src/kg_project/cli.py`

**Step 1: 实现数据源配置加载**

让代码能从 `config/crf_experiment.yaml` 解析公开数据 URL 和目标路径。

**Step 2: 实现下载/校验逻辑**

支持以下来源：

- SkillSpan GitHub 仓库压缩包或原始文件
- O*NET 官方 text zip
- 英文 JD fallback 数据

ESCO 若无法无认证直链下载，则先支持“已有文件校验 + 手工放置”，并在 CLI 中给出明确提示。

**Step 3: 暴露 CLI**

新增 `kg-pipeline fetch-crf-data`。

**Step 4: Run targeted tests**

Run: `.venv/bin/pytest tests/test_data_sources.py -vv`

Expected: PASS

### Task 4: 实现主数据预处理

**Files:**
- Create: `src/kg_project/preprocess.py`
- Modify: `src/kg_project/data_types.py`
- Modify: `src/kg_project/ner/data.py`

**Step 1: 转换 SkillSpan**

读取 SkillSpan `json` 或 `conll`，转换到统一 `NerExample` JSONL。

**Step 2: 解析 gazetteer 所需资源**

将 ESCO/O*NET 英文文件转换为正式实验专用 gazetteer。

**Step 3: 清洗 JD**

保留英文、非空、描述足够长、职位相关样本，并输出统计。

**Step 4: 暴露 CLI**

新增 `kg-pipeline prepare-crf-data`。

**Step 5: Run targeted tests**

Run: `.venv/bin/pytest tests/test_preprocess.py -vv`

Expected: PASS

### Task 5: 扩展 CRF 特征与正式实验入口

**Files:**
- Modify: `src/kg_project/ner/features.py`
- Modify: `src/kg_project/ner/crf_baseline.py`
- Modify: `src/kg_project/ner/experiments.py`

**Step 1: 写 `CRF-base` 特征开关**

确保传统 token 特征单独可跑。

**Step 2: 写 `CRF+gazetteer` 特征**

将 gazetteer 匹配结果接入 token 特征。

**Step 3: 正式 train/dev/test 流程**

支持：

- 训练集训练
- 开发集调参/选模型
- 测试集单次评测

**Step 4: 生成 JSON/Markdown 报告**

包含：

- overall 指标
- per-label 指标
- base vs gazetteer 对比
- 错误样例
- JD 覆盖统计

**Step 5: 暴露 CLI**

新增 `kg-pipeline run-crf-experiment`。

**Step 6: Run targeted tests**

Run: `.venv/bin/pytest tests/test_crf_formal_experiment.py -vv`

Expected: PASS

### Task 6: 端到端验证与文档收口

**Files:**
- Modify: `README.md`
- Create: `data/reports/crf_experiment.json`
- Create: `data/reports/crf_experiment.md`

**Step 1: 运行完整实验命令**

Run:

```bash
.venv/bin/kg-pipeline fetch-crf-data
.venv/bin/kg-pipeline prepare-crf-data
.venv/bin/kg-pipeline run-crf-experiment
```

Expected: 成功生成正式实验产物

**Step 2: 运行全量测试**

Run: `.venv/bin/pytest`

Expected: 全绿

**Step 3: 收口文档**

README 中补上正式实验步骤、依赖和限制说明。
