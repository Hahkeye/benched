"""FastAPI + Plotly dashboard for benched results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from jinja2 import Environment, FileSystemLoader
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONEncoder
    DASHBOARD_AVAILABLE = True
except ModuleNotFoundError as _import_err:
    DASHBOARD_AVAILABLE = False
    _IMPORT_ERROR = _import_err

from benched.db import Database
from benched.objectives import recommend


def _to_json(fig: Any) -> str:
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _runs_with_summaries(db: Database, **filters: Any) -> list[dict[str, Any]]:
    runs = db.list_runs(**filters, limit=10000)
    for run in runs:
        run["summary"] = db.get_run_summary(run["id"])
        try:
            run["args"] = json.loads(run["args_json"])
        except json.JSONDecodeError:
            run["args"] = []
    return runs


def create_app() -> Any:
    if not DASHBOARD_AVAILABLE:
        raise RuntimeError(
            "dashboard dependencies not installed; install with: pip install 'benched[dashboard]'"
        ) from _IMPORT_ERROR

    app = FastAPI(title="benched dashboard")
    here = Path(__file__).parent
    app.mount("/static", StaticFiles(directory=str(here / "static")), name="static")
    env = Environment(loader=FileSystemLoader(str(here / "templates")), auto_reload=True)
    env.cache = None  # Disable caching to avoid unhashable dict bug in Jinja2 3.1.6+
    templates = Jinja2Templates(env=env)
    db = Database()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        runs = _runs_with_summaries(db)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"runs": runs},
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: int) -> HTMLResponse:
        run = db.run_with_summary(run_id)
        if run is None:
            return templates.TemplateResponse(
                request=request,
                name="error.html",
                context={"message": f"Run {run_id} not found"},
                status_code=404,
            )
        samples = db.get_samples_for_run(run_id)
        ok_samples = [s for s in samples if not s.get("error")]

        def _hist(values: list[float], title: str, xlabel: str) -> str:
            fig = go.Figure(data=[go.Histogram(x=values, nbinsx=20)])
            fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="count")
            return _to_json(fig)

        ttft_values = [s["ttft_ms"] for s in ok_samples if s.get("ttft_ms") is not None]
        tpot_values = [s["tpot_ms"] for s in ok_samples if s.get("tpot_ms") is not None]
        throughput_values = [
            s["throughput_tok_per_sec"]
            for s in ok_samples
            if s.get("throughput_tok_per_sec") is not None
        ]

        charts = {
            "ttft": _hist(ttft_values, "TTFT distribution", "ms") if ttft_values else None,
            "tpot": _hist(tpot_values, "TPOT distribution", "ms") if tpot_values else None,
            "throughput": _hist(throughput_values, "Throughput distribution", "tok/s") if throughput_values else None,
        }

        return templates.TemplateResponse(
            request=request,
            name="run.html",
            context={"run": run, "samples": samples, "charts": charts},
        )

    @app.get("/recommend", response_class=HTMLResponse)
    async def recommend_page(request: Request) -> HTMLResponse:
        # Aggregate across all backend/model pairs.
        runs = _runs_with_summaries(db)
        by_backend_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for run in runs:
            by_backend_model.setdefault((run["backend"], run["model_path"]), []).append(run)

        all_ranked: list[dict[str, Any]] = []
        for (backend, model), group in by_backend_model.items():
            ranked = recommend(db, backend, model, "maximize throughput_tok_per_sec", top=100)
            all_ranked.extend(ranked)

        all_ranked.sort(key=lambda x: x["score"], reverse=True)

        scatter_data = [
            {
                "backend": r["run"]["backend"],
                "throughput": r["summary"].get("median_throughput_tok_per_sec", 0.0),
                "latency": r["summary"].get("median_total_latency_ms", 0.0),
                "args": " ".join(r["run"].get("args", [])),
            }
            for r in all_ranked
            if r["summary"].get("median_throughput_tok_per_sec") is not None
        ]
        scatter_json = None
        if scatter_data:
            fig = go.Figure()
            backends = sorted(set(d["backend"] for d in scatter_data))
            for backend in backends:
                bd = [d for d in scatter_data if d["backend"] == backend]
                fig.add_trace(go.Scatter(
                    x=[d["latency"] for d in bd],
                    y=[d["throughput"] for d in bd],
                    mode="markers",
                    name=backend,
                    text=[d["args"] for d in bd],
                    hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                ))
            fig.update_layout(
                title="Throughput vs. Latency",
                xaxis_title="Latency (ms)",
                yaxis_title="Throughput (tok/s)",
            )
            scatter_json = _to_json(fig)

        return templates.TemplateResponse(
            request=request,
            name="recommend.html",
            context={
                "ranked": all_ranked[:20],
                "scatter": scatter_json,
            },
        )

    return app


def run_dashboard(port: int = 8080) -> None:
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=port)
