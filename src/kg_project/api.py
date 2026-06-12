from __future__ import annotations

from pathlib import Path

import typer

from kg_project.storage import build_store


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Career-Skill Knowledge Graph Final</title>
  <script src=\"https://unpkg.com/vis-network/standalone/umd/vis-network.min.js\"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f7f8fb; color: #1f2937; }
    header { padding: 16px 20px; background: #0f766e; color: #fff; }
    main { padding: 16px 20px; display: grid; gap: 16px; }
    .card { background: #fff; border-radius: 10px; padding: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
    #graph { width: 100%; height: 460px; border: 1px solid #e5e7eb; border-radius: 8px; }
    .row { display: grid; gap: 12px; grid-template-columns: 1fr 1fr; }
    input, button, select { padding: 8px; border-radius: 6px; border: 1px solid #d1d5db; }
    button { background: #0f766e; color: #fff; border: none; cursor: pointer; }
    pre { margin: 0; white-space: pre-wrap; background: #111827; color: #e5e7eb; padding: 10px; border-radius: 8px; min-height: 100px; }
    .muted { color: #4b5563; font-size: 13px; }
    @media (max-width: 900px) { .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h2 style=\"margin:0;\">Career-Skill Knowledge Graph (Final)</h2>
    <div id=\"summary\" style=\"margin-top:6px;font-size:14px;\"></div>
  </header>
  <main>
    <section class=\"card\">
      <h3 style=\"margin-top:0;\">Graph Visualization</h3>
      <div class=\"muted\">You can drag nodes, zoom, and inspect downstream outputs below.</div>
      <div id=\"graph\" style=\"margin-top:8px;\"></div>
    </section>
    <section class=\"card\">
      <h3 style=\"margin-top:0;\">Downstream Applications</h3>
      <div class=\"row\">
        <div>
          <h4>1) Role Recommendation</h4>
          <input id=\"rec-skills\" value=\"Python,PyTorch\" />
          <button onclick=\"runRecommend()\" style=\"margin-top:8px;\">Run</button>
          <pre id=\"rec-out\"></pre>
        </div>
        <div>
          <h4>2) Skill Gap Analysis</h4>
          <input id=\"gap-role\" value=\"Data Scientist\" />
          <input id=\"gap-skills\" value=\"Python\" style=\"margin-top:8px;\" />
          <button onclick=\"runGap()\" style=\"margin-top:8px;\">Run</button>
          <pre id=\"gap-out\"></pre>
        </div>
      </div>
      <div class=\"row\" style=\"margin-top:12px;\">
        <div>
          <h4>3) Growth Path Planning</h4>
          <input id=\"path-source\" value=\"Data Scientist\" />
          <input id=\"path-target\" value=\"Machine Learning Engineer\" style=\"margin-top:8px;\" />
          <button onclick=\"runPath()\" style=\"margin-top:8px;\">Run</button>
          <pre id=\"path-out\"></pre>
        </div>
        <div>
          <h4>4) Occupation Profile</h4>
          <input id=\"occ-name\" value=\"Data Scientist\" />
          <button onclick=\"runProfile()\" style=\"margin-top:8px;\">Run</button>
          <pre id=\"occ-out\"></pre>
        </div>
      </div>
    </section>
  </main>
  <script>
    async function loadSummary() {
      const res = await fetch('/graph/summary');
      const data = await res.json();
      document.getElementById('summary').textContent =
        `Backend: ${data.backend} | Nodes: ${data.num_nodes} | Edges: ${data.num_edges} | Relations: ${data.relations.join(', ')}`;
    }

    async function loadGraph() {
      const res = await fetch('/graph/raw?limit=1200');
      const data = await res.json();
      document.querySelector('#graph').title =
        data.is_limited ? `Showing ${data.visualized_edges} of ${data.total_edges} stored edges` : 'Showing full graph';
      const nodes = new vis.DataSet(data.nodes.map(n => ({ id: n.id, label: n.name, group: n.entity_type })));
      const edges = new vis.DataSet(data.edges.map(e => ({ from: e.source, to: e.target, label: e.relation, arrows: 'to' })));
      new vis.Network(document.getElementById('graph'), { nodes, edges }, {
        physics: { stabilization: false },
        edges: { font: { size: 10 }, color: '#9ca3af' },
        groups: {
          OCC: { color: '#ef4444' }, SKL: { color: '#3b82f6' }, KNW: { color: '#10b981' },
          TOL: { color: '#8b5cf6' }, QLF: { color: '#f59e0b' }, TSK: { color: '#06b6d4' }, ABL: { color: '#22c55e' }
        }
      });
    }

    const toList = (v) => v.split(',').map(x => x.trim()).filter(Boolean);

    async function runRecommend() {
      const skills = toList(document.getElementById('rec-skills').value);
      const qs = skills.map(s => `skills=${encodeURIComponent(s)}`).join('&');
      const res = await fetch(`/recommend?${qs}&top_k=5`);
      document.getElementById('rec-out').textContent = JSON.stringify(await res.json(), null, 2);
    }

    async function runGap() {
      const role = document.getElementById('gap-role').value;
      const skills = toList(document.getElementById('gap-skills').value);
      const qs = skills.map(s => `skills=${encodeURIComponent(s)}`).join('&');
      const res = await fetch(`/skill-gap?occupation=${encodeURIComponent(role)}&${qs}`);
      document.getElementById('gap-out').textContent = JSON.stringify(await res.json(), null, 2);
    }

    async function runPath() {
      const source = document.getElementById('path-source').value;
      const target = document.getElementById('path-target').value;
      const res = await fetch(`/growth-path?source=${encodeURIComponent(source)}&target=${encodeURIComponent(target)}`);
      document.getElementById('path-out').textContent = JSON.stringify(await res.json(), null, 2);
    }

    async function runProfile() {
      const name = document.getElementById('occ-name').value;
      const res = await fetch(`/occupation/${encodeURIComponent(name)}`);
      document.getElementById('occ-out').textContent = JSON.stringify(await res.json(), null, 2);
    }

    loadSummary();
    loadGraph();
  </script>
</body>
</html>"""


def create_app(
    nodes_path: Path,
    edges_path: Path,
    use_neo4j: bool = False,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "neo4j",
    neo4j_database: str = "neo4j",
):
    try:
        from fastapi import FastAPI, Query
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise RuntimeError("fastapi is required. install with pip install fastapi uvicorn") from exc

    store = build_store(
        nodes_path=nodes_path,
        edges_path=edges_path,
        use_neo4j=use_neo4j,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
    )

    app = FastAPI(title="Career-Skill KG API", version="2.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return _dashboard_html()

    @app.get("/graph/summary")
    def graph_summary():
        return store.summary()

    @app.get("/graph/raw")
    def graph_raw(limit: int = 1200, include_similarity: bool = False):
        return store.graph_raw(limit=limit, include_similarity=include_similarity)

    @app.get("/occupation/{name}")
    def occupation_profile(name: str):
        return store.occupation_profile(name)

    @app.get("/recommend")
    def recommend(skills: list[str] = Query(default=[]), top_k: int = 5):
        return store.recommend(skills, top_k=top_k)

    @app.get("/skill-gap")
    def gap(occupation: str, skills: list[str] = Query(default=[])):
        return store.skill_gap(occupation, skills)

    @app.get("/growth-path")
    def path(source: str, target: str):
        return store.growth_path(source, target)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        close_fn = getattr(store, "close", None)
        if callable(close_fn):
            close_fn()

    return app


def run_api_command(
    nodes: Path = typer.Option(..., help="Graph nodes CSV."),
    edges: Path = typer.Option(..., help="Graph edges CSV."),
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    use_neo4j: bool = typer.Option(False, help="Use Neo4j backend instead of CSV backend."),
    neo4j_uri: str = typer.Option("bolt://localhost:7687", help="Neo4j bolt URI."),
    neo4j_user: str = typer.Option("neo4j", help="Neo4j username."),
    neo4j_password: str = typer.Option("neo4j", help="Neo4j password."),
    neo4j_database: str = typer.Option("neo4j", help="Neo4j database name."),
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required. install with pip install uvicorn") from exc

    app = create_app(
        nodes_path=nodes,
        edges_path=edges,
        use_neo4j=use_neo4j,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
    )
    uvicorn.run(app, host=host, port=port)
