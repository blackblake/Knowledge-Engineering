# 职业技能知识图谱构建与应用系统结课报告


## 封面

小组项目名称：职业技能知识图谱构建与应用系统

小组成员：吕天白（学号2023302111010）

报告撰写时间：2026年6月12日

仓库地址：git@github.com:blackblake/Knowledge-Engineering.git


## 1 小组分工

[表格见PDF排版]

[表格见PDF排版]

说明：课程要求贡献占比总和为100%。已知吕天白贡献80%，赵金波参与可视化工作，因此报告草稿暂列赵金波20%；若实际比例不同，应在提交前修改。


## 2 知识图谱构建


## 2.1 数据源、获取方法和原始数据统计信息

本项目面向招聘岗位场景构建职业-技能知识图谱。主要数据源为本地处理后的岗位描述样本 jd_sample.jsonl，以及由 O*NET/本地规则整理得到的 formal_gazetteer.jsonl。岗位数据包含岗位标题、语言和岗位描述文本；词表覆盖职业、技能、工具、知识等实体类型。

[表格见PDF排版]

![图2 知识图谱构建流程](C:/Users/Administrator/Downloads/知识工程大作业/知识工程大作业/data/reports/report_assets/pipeline.png)


## 2.2 实体抽取的输入、方法和结果

实体抽取输入为岗位标题与岗位描述句子。系统先清洗岗位标题作为 OCC 职业实体，再用 gazetteer 建立 n-gram 索引，对岗位描述进行最长匹配，识别 SKL 技能、TOL 工具、KNW 知识等实体。该方法可解释、可复现，并能保证 Java、Python、React、Spring 等关键技术词进入图谱。

```python
def _find_term_mentions(text, term_index):
    tokens, token_spans = tokenize_with_spans(text)
    for start in range(len(tokens)):
        for end in range(min(len(tokens), start + MAX_TERM_TOKENS), start, -1):
            entry = term_index.get(tuple(t.casefold() for t in tokens[start:end]))
            if entry:
                # 输出实体 mention: text/start/end/type/canonical_id
                ...
```

实体抽取结果：生成35962条弱监督NER样本，覆盖9168条gazetteer词条中的4176个可索引术语。职业标题经过编号、奖金信息、括号信息等清洗，减少了脏职业节点。


## 2.3 关系抽取的输入、方法和结果

关系抽取输入为已标注实体的句子。每条样本至少包含一个 OCC 职业实体，系统将句中其他实体与最近的职业实体建立关系。不同实体类型映射到不同关系：SKL -> REQUIRES_SKILL，TOL -> USES_TOOL，KNW -> REQUIRES_KNOWLEDGE。

```python
_RELATION_MAP = {'SKL':'REQUIRES_SKILL','KNW':'REQUIRES_KNOWLEDGE','TOL':'USES_TOOL'}
def extract_relations(example):
    spans = extract_entity_spans(example)
    occupations = [s for s in spans if s.entity_type == 'OCC']
    for span in spans:
        relation = _RELATION_MAP.get(span.entity_type)
        closest = min(occupations, key=lambda occ: abs(occ.start_token - span.start_token))
```

[表格见PDF排版]


## 2.4 实体消岐的输入、方法和结果

实体消岐输入为关系抽取阶段产生的实体文本与gazetteer规范词条。系统先使用规范化精确匹配，再使用fuzzy matching进行相似度对齐，并在融合阶段合并同一canonical_id下的别名和重复关系。对齐结果：head coverage=0.9307，tail coverage=0.9932；融合后实体2630个，融合关系11606条，合并别名3206个。


## 2.5 事件抽取

本项目未单独实现事件抽取模块。原因是课程项目核心目标是职业-技能知识图谱及其下游应用，岗位文本中的事件结构较弱，主要信息以实体和实体关系形式表达。


## 2.6 其他：知识推理与图谱增强

系统在基础抽取关系之外构建 SIMILAR_TO 职业相似关系和 PREREQUISITE_OF 技能先修关系。SIMILAR_TO 根据职业技能集合重叠度生成，用于成长路径规划；PREREQUISITE_OF 根据技能全局出现频次构造启发式先修链，用于展示知识推理能力。


## 2.7 知识图谱的规模、存储和可视化结果

[表格见PDF排版]

![图1 系统架构与存储/可视化方式](C:/Users/Administrator/Downloads/知识工程大作业/知识工程大作业/data/reports/report_assets/architecture.png)


## 3 基于知识图谱的应用


## 3.1 系统基本功能

系统提供一个本地网页应用，包含图谱可视化和四类下游任务：岗位推荐、技能差距分析、职业成长路径规划和岗位画像查询。用户输入技能列表或目标职业后，前端调用FastAPI接口，后端基于GraphStore从完整图谱中检索与计算结果。

![图3 下游任务运行结果示例](C:/Users/Administrator/Downloads/知识工程大作业/知识工程大作业/data/reports/report_assets/app_result.png)

[表格见PDF排版]


## 3.2 如何使用知识图谱的实现细节

网页应用不是静态写死结果，而是通过统一GraphStore接口使用知识图谱。CSVGraphStore和Neo4jGraphStore暴露相同方法：summary、graph_raw、occupation_profile、recommend、skill_gap和growth_path。因此系统可以在CSV后端与Neo4j后端之间切换。

```python
def recommend(self, skills, top_k):
    skill_set = {_normalize(skill) for skill in skills}
    for edge in self.edges:
        if edge['relation'] in {'REQUIRES_SKILL','USES_TOOL','REQUIRES_KNOWLEDGE'}:
            # 对 OCC -> 技能/工具/知识 边累积命中权重
            ...
    return ranked[:top_k]
```

以岗位推荐为例，系统只统计OCC职业节点指向SKL/TOL/KNW节点的关系边。输入技能被规范化后与图谱节点名称匹配，命中边按evidence_count的log权重累积，并综合命中技能数、标题命中和证据强度排序。

以技能差距为例，系统查询目标职业的REQUIRES_SKILL边，与用户已掌握技能集合做差集，输出缺失技能、是否核心要求、证据数和熟练度信息。


## 提交说明

本报告已按课程要求组织为：封面、小组分工、知识图谱构建、基于知识图谱的应用。最终提交形式为PDF。当前草稿中仍需人工确认的字段包括：项目正式名称、成员学号、赵金波贡献占比/电子签，以及是否需要列出除吕天白和赵金波以外的全部成员。

GitHub仓库地址要求为git@github.com:blackblake/Knowledge-Engineering.git。由于覆盖远程仓库属于有风险操作，需要在提交前确认当前工作区内容、远程权限和是否允许force push。
