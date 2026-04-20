# CRF 正式实验设计

**日期：** 2026-04-20  
**主题：** 面向课程作业的英文优先 CRF 正式实验

## 1. 背景

当前仓库已经完成前三周的最小可运行版本，但第 3 周中的 `CRF` 仍停留在 toy 级别：

- 数据来自仓库内置 fixture
- 训练与评测使用同一份小数据
- 报告只有整体指标
- 没有真实数据下载、标准划分、正式实验配置与错误分析

本次工作要把 `CRF` 做成课程作业可直接引用的正式实验模块。

## 2. 目标

本次只完成 `CRF` 的正式实验，不扩展到 BiLSTM-CRF、BERT-CRF、Neo4j 或前端。

完成标准：

- 可下载或校验真实公开数据
- 可把主标注集整理成正式 `train/dev/test`
- 可运行两组 CRF 实验：
  - `CRF-base`
  - `CRF+gazetteer`
- 可生成机器可读和人可读两类报告
- 报告中包含数据来源、划分方式、特征说明、主指标、分标签指标、错误分析和 JD 词典覆盖统计

## 3. 数据路线

采用“混合路线”，其中主评测和辅助语料分工明确。

### 3.1 主评测标注集

使用 `SkillSpan` 作为正式主评测数据。

用途：

- 提供公开、规范、可复现的英文 job posting 标注集
- 作为正式 `train/dev/test`
- 生成课程报告中的主 `precision / recall / F1`

选择理由：

- 任务定义与本项目高度一致，聚焦 job postings 中的 skill span extraction
- 数据和说明在官方 GitHub 仓库公开可得
- 官方仓库已提供 `conll` 和 `json` 格式，并含 `train/dev/test` 划分

来源：

- SkillSpan GitHub: <https://github.com/kris927b/SkillSpan>

### 3.2 辅助 JD 语料

使用公开英文 JD 数据集作为辅助语料，不作为主测试集。

用途：

- 做 `ESCO/O*NET` 词典覆盖统计
- 做弱监督扩充与错误分析
- 证明方案中的“真实 JD 工程主线”已经接入

默认来源优先级：

1. Kaggle LinkedIn Job Postings（若可直接下载）
2. Hugging Face 上公开英文 JD 数据集（作为 fallback）

为控制本地资源消耗，只抽取和数据/算法/软件工程相关的英文岗位，目标规模控制在 `5k–20k` 条。

候选来源：

- Hugging Face Djinni English JD: <https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english>

### 3.3 外部本体与词典

使用 `ESCO + O*NET` 作为 gazetteer 和知识工程增强来源。

用途：

- 提供英文 skill / knowledge / qualification / tool 术语
- 构建 `CRF+gazetteer` 的词典特征
- 为辅助 JD 语料生成词典覆盖统计与弱监督标签

来源：

- ESCO 下载页: <https://esco.ec.europa.eu/en/use-esco/download>
- O*NET Database: <https://www.onetcenter.org/database.html>

## 4. 实验设计

### 4.1 实验组

仅做两组，控制范围：

- `CRF-base`：纯传统 token 特征
- `CRF+gazetteer`：在 base 上增加 ESCO/O*NET 词典特征

这样既有可解释 baseline，也有知识工程增强项，适合写课程报告。

### 4.2 特征

`CRF-base` 默认包含：

- 当前词、小写词
- 前词、后词
- 前后缀
- 是否首字母大写
- 是否全大写
- 是否含数字
- 词长

`CRF+gazetteer` 在以上基础上增加：

- 是否命中 gazetteer
- 命中实体类型
- 最长匹配 span 类型
- 是否命中 occupation / skill / knowledge / qualification / tool 词典

`POS` 先不作为首批强制项，避免引入额外模型依赖。

### 4.3 数据划分

对 `SkillSpan`：

- 若官方仓库自带 `train/dev/test`，优先沿用
- 不自行重切

对辅助 JD：

- 不进入正式 test
- 只用于统计和辅助分析

### 4.4 评估

主指标：

- overall `precision`
- overall `recall`
- overall `f1`

补充指标：

- per-label `precision / recall / f1 / support`
- confusion / 常见错例
- JD 上的 gazetteer 覆盖率统计

## 5. 交付物

代码交付物：

- 数据下载入口
- 数据预处理脚本
- 正式 CRF 实验入口
- 报告生成器

文件交付物：

- `config/crf_experiment.yaml`
- `data/reports/crf_experiment.json`
- `data/reports/crf_experiment.md`

## 6. 风险与控制

### 6.1 数据下载限制

风险：部分 JD 数据源可能有登录限制。  
控制：主评测不依赖 JD，必要时切换到 Hugging Face fallback。

### 6.2 标签体系不完全一致

风险：SkillSpan 更偏 `skill / knowledge` 任务，而当前项目 schema 有 7 类。  
控制：本次正式 CRF 主实验明确定位为“英文技能抽取子任务”，不强行覆盖 7 类全实体。

### 6.3 本地资源受限

风险：MacBook Air 不适合同时做所有模型的大实验。  
控制：本次只做 `CRF`，将神经模型正式实验留给后续。

## 7. 本次不做的内容

- BiLSTM-CRF 正式实验
- BERT-CRF 正式实验
- 关系抽取
- 实体对齐
- Neo4j 入库
- 前端可视化

## 8. 结论

本次实现将把当前 toy 级 `CRF` 提升为“可下载真实公开数据、可复现、可出正式报告”的课程级实验模块。只要这部分落稳，后续神经模型和图谱阶段都能沿同一数据与报告接口继续扩展。
