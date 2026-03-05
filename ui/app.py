import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

LOG_FILE = Path("docs/plans/agent_stream.jsonl")

@app.get("/")
async def get():
    with open("ui/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Read existing lines
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    await websocket.send_text(line)
                    
    # Tail the file for new lines
    last_pos = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0
    
    try:
        while True:
            if not LOG_FILE.exists():
                await asyncio.sleep(0.5)
                continue
                
            current_size = LOG_FILE.stat().st_size
            if current_size < last_pos:
                # File was truncated/recreated
                last_pos = 0
                
            if current_size > last_pos:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    f.seek(last_pos)
                    for line in f:
                        if line.strip():
                            await websocket.send_text(line)
                    last_pos = f.tell()
                    
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"WebSocket closed: {e}")

if __name__ == "__main__":
    print("🚀 Starting YGN-SAGE Dashboard on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
