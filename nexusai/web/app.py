"""NexusAI Web Dashboard — FastAPI application."""
from __future__ import annotations
import json, os
from pathlib import Path

def create_app():
    try:
        from fastapi import FastAPI, WebSocket, Request
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError:
        print("FastAPI not installed. Run: pip install nexusai[web]")
        return None

    from nexusai.core.config import NexusConfig
    from nexusai.agents.orchestrator import AgentOrchestrator
    from nexusai.core.registry import ToolRegistry

    app = FastAPI(title="NexusAI Dashboard", version="1.0.0")
    config = NexusConfig.from_env()
    orchestrator = AgentOrchestrator(config=config, verbose=False)

    static_dir = Path(__file__).parent / "static"

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = static_dir / "index.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse("<h1>NexusAI Dashboard</h1><p>Static files not found.</p>")

    @app.get("/api/config")
    async def get_config():
        return JSONResponse(config.to_dict())

    @app.get("/api/providers")
    async def list_providers():
        return JSONResponse({"providers": config.available_providers})

    @app.get("/api/tools")
    async def list_tools():
        return JSONResponse({"tools": ToolRegistry.available()[:100], "total": ToolRegistry.count()})

    @app.get("/api/tools/categories")
    async def tool_categories():
        return JSONResponse(ToolRegistry.by_category())

    @app.post("/api/run")
    async def run_task(request: Request):
        data = await request.json()
        task = data.get("task", "")
        if not task:
            return JSONResponse({"error": "No task provided"}, status_code=400)
        result = orchestrator.run(task)
        return JSONResponse(result.to_dict())

    @app.post("/api/chat")
    async def chat(request: Request):
        data = await request.json()
        message = data.get("message", "")
        response = orchestrator.chat(message)
        return JSONResponse({"response": response})

    @app.post("/api/rag/ingest")
    async def rag_ingest(request: Request):
        data = await request.json()
        path = data.get("path", "")
        from nexusai.rag.pipeline import RAGPipeline
        rag = RAGPipeline(config=config)
        result = rag.ingest(path)
        return JSONResponse(result)

    @app.post("/api/rag/query")
    async def rag_query(request: Request):
        data = await request.json()
        question = data.get("question", "")
        from nexusai.rag.pipeline import RAGPipeline
        rag = RAGPipeline(config=config)
        result = rag.query(question)
        return JSONResponse(result)

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                response = orchestrator.chat(msg.get("message", ""))
                await websocket.send_json({"response": response})
        except Exception:
            pass

    return app

def run_server(host: str = "0.0.0.0", port: int = 8080):
    app = create_app()
    if app:
        import uvicorn
        uvicorn.run(app, host=host, port=port)
