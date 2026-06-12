from __future__ import annotations

import csv
import json
from pathlib import Path

import typer


def _load_nodes(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_edges(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_visualization(nodes_path: Path, edges_path: Path, output: Path) -> None:
    nodes = _load_nodes(nodes_path)
    edges = _load_edges(edges_path)

    payload = {
        "nodes": [{"id": row["id"], "label": row["label"], "name": row["name"], "entity_type": row["entity_type"]} for row in nodes],
        "edges": [{"source": row["source"], "target": row["target"], "relation": row["relation"], "evidence": int(row.get("evidence_count") or 0)} for row in edges],
    }

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Career-Skill Graph Demo</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    :root {{
      --bg: #f6f3ef;
      --panel: #fffdf9;
      --ink: #17212c;
      --muted: #6d7783;
      --brand: #f05d23;
      --brand2: #1f9d8f;
    }}
    body {{ margin:0; font-family:'Space Grotesk',sans-serif; color:var(--ink); background:linear-gradient(135deg,#f9efe2,#e9f5f2); min-height:100vh; }}
    header {{ padding:18px 24px; display:flex; justify-content:space-between; align-items:center; }}
    .tabs {{ display:flex; gap:8px; }}
    .tab {{ border:none; border-radius:999px; padding:8px 14px; cursor:pointer; background:#fff1e8; color:#8a3d1c; font-weight:600; }}
    .tab.active {{ background:var(--brand); color:white; }}
    .panel {{ margin:0 24px 24px; border-radius:18px; background:var(--panel); box-shadow:0 10px 30px rgba(0,0,0,.08); padding:16px; min-height:72vh; }}
    canvas {{ width:100%; height:58vh; border-radius:12px; background:#fbfaf8; }}
    .hidden {{ display:none; }}
    .grid {{ display:grid; gap:12px; grid-template-columns:1fr 1fr; }}
    .card {{ border:1px solid #f1e5da; border-radius:12px; padding:12px; background:white; }}
    input, select {{ width:100%; padding:8px; border-radius:10px; border:1px solid #d9dfe6; }}
    button.action {{ margin-top:8px; background:var(--brand2); color:white; border:none; border-radius:10px; padding:8px 12px; cursor:pointer; }}
    pre {{ background:#0d1720; color:#ebf4ff; border-radius:10px; padding:10px; overflow:auto; }}
    @media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} header {{ flex-direction:column; align-items:flex-start; gap:8px; }} }}
  </style>
</head>
<body>
  <header>
    <div><h2 style=\"margin:0\">Career-Skill Knowledge Graph</h2><div style=\"color:var(--muted)\">Occupation profile / Recommendation / Growth path</div></div>
    <div class=\"tabs\">
      <button class=\"tab active\" data-tab=\"graph\">Graph</button>
      <button class=\"tab\" data-tab=\"recommend\">Recommend</button>
      <button class=\"tab\" data-tab=\"growth\">Growth</button>
    </div>
  </header>
  <section class=\"panel\" id=\"graph\"><canvas id=\"canvas\"></canvas></section>
  <section class=\"panel hidden\" id=\"recommend\">
    <div class=\"grid\">
      <div class=\"card\"><label>Skills (comma separated)</label><input id=\"skillInput\" value=\"Python,PyTorch\"/><button class=\"action\" id=\"runRec\">Run Recommendation</button></div>
      <div class=\"card\"><label>Output</label><pre id=\"recOut\"></pre></div>
    </div>
  </section>
  <section class=\"panel hidden\" id=\"growth\">
    <div class=\"grid\">
      <div class=\"card\"><label>From</label><input id=\"fromRole\" value=\"Data Scientist\"/><label>To</label><input id=\"toRole\" value=\"Machine Learning Engineer\"/><button class=\"action\" id=\"runPath\">Run Growth Path</button></div>
      <div class=\"card\"><label>Output</label><pre id=\"pathOut\"></pre></div>
    </div>
  </section>
  <script>
    const data = {json.dumps(payload)};
    const tabs = [...document.querySelectorAll('.tab')];
    tabs.forEach(btn => btn.onclick = () => {{
      tabs.forEach(x => x.classList.remove('active')); btn.classList.add('active');
      ['graph','recommend','growth'].forEach(id => document.getElementById(id).classList.add('hidden'));
      document.getElementById(btn.dataset.tab).classList.remove('hidden');
    }});

    const nodes = data.nodes.map((n,i)=>({{...n,x:Math.cos(i*0.4)*220+360,y:Math.sin(i*0.4)*160+260}}));
    const byId = Object.fromEntries(nodes.map(n=>[n.id,n]));
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const state = {{scale:1,ox:0,oy:0,drag:false,lx:0,ly:0}};
    function resize(){{const r=canvas.getBoundingClientRect(); canvas.width=r.width*devicePixelRatio; canvas.height=r.height*devicePixelRatio; ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0); draw();}}
    function draw(){{ctx.clearRect(0,0,canvas.width,canvas.height); ctx.save(); ctx.translate(state.ox,state.oy); ctx.scale(state.scale,state.scale);
      data.edges.forEach(e=>{{const s=byId[e.source],t=byId[e.target]; if(!s||!t) return; ctx.strokeStyle='rgba(23,33,44,.15)'; ctx.beginPath(); ctx.moveTo(s.x,s.y); ctx.lineTo(t.x,t.y); ctx.stroke(); }});
      nodes.forEach(n=>{{const color=n.entity_type==='OCC'?'#f05d23':'#1f9d8f'; ctx.fillStyle=color; ctx.beginPath(); ctx.arc(n.x,n.y,8,0,Math.PI*2); ctx.fill(); ctx.fillStyle='#17212c'; ctx.font='12px Space Grotesk'; ctx.fillText(n.name,n.x+11,n.y+4);}});
      ctx.restore();
    }}
    canvas.onwheel=(e)=>{{e.preventDefault(); state.scale=Math.max(.5,Math.min(3,state.scale*(e.deltaY>0?.95:1.05))); draw();}};
    canvas.onmousedown=(e)=>{{state.drag=true; state.lx=e.clientX; state.ly=e.clientY;}};
    canvas.onmousemove=(e)=>{{if(!state.drag) return; state.ox+=e.clientX-state.lx; state.oy+=e.clientY-state.ly; state.lx=e.clientX; state.ly=e.clientY; draw();}};
    window.onmouseup=()=>state.drag=false;
    window.onresize=resize; resize();

    function norm(s){{return s.toLowerCase().replace(/\s+/g,'');}}
    function recommend(skills){{
      const skillSet=new Set(skills.map(norm));
      const edges=data.edges.filter(e=>e.relation==='REQUIRES_SKILL');
      const out={{}};
      edges.forEach(e=>{{const t=byId[e.target], s=byId[e.source]; if(!s||!t||s.entity_type!=='OCC') return; const hit=skillSet.has(norm(t.name)); if(!out[s.id]) out[s.id]={{name:s.name,n:0,d:0}}; out[s.id].d+=Math.log(1+e.evidence); if(hit){{out[s.id].n+=Math.log(1+e.evidence);}} }});
      return Object.values(out).map(r=>({{occupation:r.name,score:r.d?+(r.n/r.d).toFixed(4):0}})).sort((a,b)=>b.score-a.score).slice(0,5);
    }}
    document.getElementById('runRec').onclick=()=>{{const skills=document.getElementById('skillInput').value.split(',').map(s=>s.trim()).filter(Boolean); document.getElementById('recOut').textContent=JSON.stringify(recommend(skills),null,2);}};

    function growth(from,to){{
      const occ=nodes.filter(n=>n.entity_type==='OCC');
      const fid=occ.find(n=>n.name===from)?.id, tid=occ.find(n=>n.name===to)?.id; if(!fid||!tid) return {{error:'occupation_not_found'}};
      const sim=data.edges.filter(e=>e.relation==='SIMILAR_TO');
      const g={{}}; sim.forEach(e=>{{(g[e.source]??=[]).push(e.target)}});
      const q=[[fid,[fid]]], v=new Set([fid]);
      while(q.length){{const [x,p]=q.shift(); if(x===tid) return {{path:p.map(id=>byId[id].name)}}; for(const y of (g[x]||[])) if(!v.has(y)){{v.add(y); q.push([y,[...p,y]])}} }}
      return {{path:[from], note:'path_not_found'}};
    }}
    document.getElementById('runPath').onclick=()=>{{document.getElementById('pathOut').textContent=JSON.stringify(growth(document.getElementById('fromRole').value,document.getElementById('toRole').value),null,2);}};
  </script>
</body>
</html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def visualize_graph_command(
    nodes: Path = typer.Option(..., help="Graph nodes CSV."),
    edges: Path = typer.Option(..., help="Graph edges CSV."),
    output: Path = typer.Option(..., help="Output HTML path."),
) -> None:
    build_visualization(nodes, edges, output)
