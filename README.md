# 职业-技能知识图谱（最终版）

这是一个可直接演示的最终版项目，核心交付是：
- 图谱构建与存储（CSV + 可选 Neo4j）
- Web 应用展示图谱与下游任务
- 一键启动命令

## 1. 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,training]"
```

## 2. 一键启动（推荐）

在项目根目录执行：

```powershell
$env:PYTHONPATH='src'
python -m kg_project.cli run-final-demo --host 127.0.0.1 --port 8000
```

打开：[http://127.0.0.1:8000](http://127.0.0.1:8000)

页面包含：
- 图谱可视化
- 岗位推荐
- 技能差距分析
- 成长路径规划
- 职业画像查询

如果完整图谱已经构建过，并且只想快速启动网页：

```powershell
$env:PYTHONPATH='src'
python -m kg_project.cli run-api `
  --nodes data/reports/graph_nodes_final.csv `
  --edges data/reports/graph_edges_final.csv `
  --host 127.0.0.1 `
  --port 8000
```

## 3. 分步运行（可选）

### 3.1 构建完整弱标注输入

```powershell
$env:PYTHONPATH='src'
python -m kg_project.cli build-full-input `
  --jobs data/processed/crf/jd_sample.jsonl `
  --base-gazetteer data/processed/crf/formal_gazetteer.jsonl `
  --output-ner data/processed/full_weak_ner.jsonl `
  --output-gazetteer data/processed/full_gazetteer.jsonl `
  --max-jobs 10000 `
  --max-sentences-per-job 6
```

### 3.2 构建最终图谱

```powershell
$env:PYTHONPATH='src'
python -m kg_project.cli build-graph `
  --ner data/processed/full_weak_ner.jsonl `
  --gazetteer data/processed/full_gazetteer.jsonl `
  --nodes data/reports/graph_nodes_final.csv `
  --edges data/reports/graph_edges_final.csv `
  --relations data/reports/graph_relations_final.jsonl `
  --report data/reports/graph_report_final.json
```

### 3.3 再启动 API + Web

```powershell
$env:PYTHONPATH='src'
python -m kg_project.cli run-api `
  --nodes data/reports/graph_nodes_final.csv `
  --edges data/reports/graph_edges_final.csv `
  --host 127.0.0.1 `
  --port 8000
```

## 4. Neo4j 后端（可选）

如果你已经将 CSV 导入 Neo4j，可使用 Neo4j 作为后端：

```powershell
$env:PYTHONPATH='src'
python -m kg_project.cli run-api `
  --nodes data/reports/graph_nodes_final.csv `
  --edges data/reports/graph_edges_final.csv `
  --use-neo4j `
  --neo4j-uri bolt://localhost:7687 `
  --neo4j-user neo4j `
  --neo4j-password 你的密码 `
  --neo4j-database neo4j `
  --host 127.0.0.1 `
  --port 8000
```

Neo4j 导入脚本：
- `scripts/neo4j_import.cypher`
- `scripts/neo4j_queries.cypher`

## 5. 主要命令

- `run-final-demo`: 一键构图并启动网页
- `build-graph`: 构建图谱 CSV/JSONL
- `run-api`: 启动 API 与网页
- `recommend-roles`: 岗位推荐
- `skill-gap`: 技能差距
- `growth-path`: 成长路径

## 6. 目录

- `src/kg_project/`: 主代码
- `data/reports/`: 图谱输出与报告
- `scripts/`: Neo4j 脚本
- `config/`: schema 与配置
- `tests/`: 测试

## 7. 结课报告

报告输出位于：

- `data/reports/knowledge_graph_final_report.pdf`
- `data/reports/knowledge_graph_final_report.md`

完整图谱边文件较大，默认不提交到 Git；可按照上面的命令在本地重新生成。
