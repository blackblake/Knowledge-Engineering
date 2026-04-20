# 前三周实现设计

**Goal:** 在空仓库中落地课程方案前 3 周的最小可运行版本，覆盖 schema、语料处理、远程监督和三套 NER 基线。

## 范围

本次实现只覆盖 `knowledge_graph_proposal.docx` 中 11.2 的前 3 周：

1. 第 1 周：选题定稿、schema 设计、数据源调研、JD 语料采集、仓库初始化
2. 第 2 周：ESCO/O*NET 解析、远程监督打标、人工校验测试集抽样
3. 第 3 周：CRF、BiLSTM-CRF、BERT-CRF 三方法实现与对比

## 默认假设

- 当前目录为从零开始的空仓库，除方案文档外没有现成代码。
- 本地离线可运行优先，因此仓库内置 fixture 数据和 toy 数据集。
- 真实 ESCO/O*NET/JD 大文件不直接提交到仓库，只保留下载位置与解析能力。
- 第 2 周中的“人工校验测试集”在代码层面实现为抽样与 doccano 兼容导出；真实人工校验需在外部完成。

## 设计取舍

### 1. schema 先文件化，再代码化

`config/ontology.yaml` 作为单一事实源，代码从 YAML 加载实体、关系和边属性。这让后续 RE、图谱入库、前端可视化都能复用同一套定义。

### 2. 远程监督以可跑通为先

方案正文推荐 Aho-Corasick。当前实现先采用“归一化 + 长词优先 + 非重叠 span 匹配”的轻量版 gazetteer 匹配器，保证没有额外 C 扩展依赖时也能运行。接口层保持独立，后续可以直接替换为真正的 Aho-Corasick 自动机。

### 3. 第 3 周统一实验入口

三种 NER 方法通过统一的数据格式和评估函数接入：

- CRF：`sklearn-crfsuite`
- BiLSTM-CRF：PyTorch + 自实现线性链 CRF
- BERT-CRF：Transformers 编码器 + 同一套 CRF 层

这样可以保证对比实验的输入、输出和报告结构一致。

## 目录与职责

- `config/ontology.yaml`：课程方案对应的 schema
- `src/kg_project/schema.py`：schema 加载与校验
- `src/kg_project/gazetteer.py`：ESCO/O*NET 解析与 gazetteer
- `src/kg_project/labeling.py`：远程监督 BIO 打标、gold 抽样
- `src/kg_project/ner/`：三种 NER 方法与统一实验入口
- `data/fixtures/`：离线示例数据
- `tests/`：保证本地最小流程可回归

## 风险

- 真实公开数据下载受网络与账号限制影响，因此本次实现同时提供 fixture 和真实数据入口。
- BERT 训练首次运行需要下载预训练权重；离线测试仅验证接口和轻量流程。
