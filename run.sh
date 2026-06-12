$env:PYTHONPATH='src'
..\ .venv\Scripts\python.exe -m kg_project.cli run-api `
  --nodes data/reports/graph_nodes_demo.csv `
  --edges data/reports/graph_edges_demo.csv `
  --host 127.0.0.1 `
  --port 8000