@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
set PYTHONPATH=src
echo Starting KG visualization at http://127.0.0.1:8000/
echo Using Python: ..\.venv\Scripts\python.exe
echo Logs: logs\visualization.out.log and logs\visualization.err.log
..\.venv\Scripts\python.exe -m kg_project.cli run-api --nodes data\reports\graph_nodes_final.csv --edges data\reports\graph_edges_final.csv --host 127.0.0.1 --port 8000 1>>logs\visualization.out.log 2>>logs\visualization.err.log
