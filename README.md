# 面向职位推荐与就业规划的职业—技能知识图谱

本仓库按 `knowledge_graph_proposal.docx` 的前 3 周计划落地了一个可运行的最小实现：

- 第 1 周：仓库初始化、schema 设计、数据目录、语料与实验脚手架
- 第 2 周：ESCO/O*NET 解析、gazetteer 构建、远程监督打标、gold 集抽样
- 第 3 周：CRF、BiLSTM-CRF、BERT-CRF 三套 NER 管线与对比脚本

当前仓库同时提供两类数据：

- `data/fixtures/`：离线可直接运行的示例数据，用于本地测试和演示
- `data/external/`：真实 ESCO / O*NET / JD 语料的下载位置，占位但不提交大文件

## 技术路线

- `schema`：7 类实体、8 类关系、边属性设计
- `data pipeline`：原始 JD -> gazetteer -> 弱标注 BIO -> gold 抽样
- `NER baseline`：`CRF`、`BiLSTM-CRF`、`BERT-CRF`
- `experiments`：统一训练入口、统一评估与 JSON 报告

## 快速开始

推荐使用 Python 3.12。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,training]"
pytest
```

用离线 fixture 跑通第 2-3 周流程：

```bash
kg-pipeline build-gazetteer \
  --esco data/fixtures/esco_skills.csv \
  --onet data/fixtures/onet_reference.tsv \
  --output data/interim/gazetteer.jsonl

kg-pipeline weak-label \
  --jobs data/fixtures/jd_corpus.jsonl \
  --gazetteer data/interim/gazetteer.jsonl \
  --output data/annotations/weak_labels.jsonl

kg-pipeline sample-gold \
  --input data/annotations/weak_labels.jsonl \
  --output data/annotations/gold_batch.jsonl \
  --sample-size 8

kg-pipeline compare-ner \
  --dataset data/fixtures/gold_ner.jsonl \
  --output data/reports/ner_report.json
```

## 目录结构

```text
config/                 schema 与默认实验配置
data/                   原始、弱标注、gold、模型和报告目录
docs/plans/             设计说明与实现计划
src/kg_project/         业务代码
tests/                  单元测试
```

## 与原方案的映射

- `config/ontology.yaml` 对应第 1 周 schema 文档
- `src/kg_project/gazetteer.py`、`labeling.py` 对应第 2 周远程监督打标
- `src/kg_project/ner/` 对应第 3 周三套 NER 方法
- `docs/plans/` 记录了基于课程方案做的工程化裁剪与默认假设

## 说明

- 仓库内置了小规模 fixture，保证离线环境可跑通。
- 真实 5k+ JD 语料、完整 ESCO / O*NET 数据建议通过公开源下载到 `data/external/` 后再运行全量实验。
- `BERT-CRF` 默认按 Hugging Face 模型接口实现；首次训练需要联网下载预训练权重。
