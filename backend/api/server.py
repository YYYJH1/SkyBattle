"""FastAPI Server for SkyBattle."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import uuid
from pathlib import Path

from ..envs import CombatEnv, CombatConfig
from ..agents import MAPPOAgent


# ============== Models ==============

class GameCreateRequest(BaseModel):
    mode: str = "ai_vs_ai"
    team_size: int = 3
    time_limit: int = 300


class GameResponse(BaseModel):
    game_id: str
    status: str
    websocket_url: str


class ControlRequest(BaseModel):
    action: str  # start, pause, resume, stop


# ============== Game Session ==============

class GameSession:
    def __init__(self, game_id: str, config: GameCreateRequest):
        self.game_id = game_id
        self.config = config
        self.status = "created"
        
        env_config = CombatConfig(team_size=config.team_size, max_steps=config.time_limit * 10)
        self.env = CombatEnv(config=env_config)
        
        obs_dim = self.env.obs_dim
        state_dim = obs_dim * config.team_size * 2
        
        self.agent = MAPPOAgent(obs_dim=obs_dim, state_dim=state_dim, n_agents=config.team_size * 2)
        
        self.observations = None
        self.connections: List[WebSocket] = []
        self.running = False
        self.paused = False
    
    def reset(self):
        self.observations, _ = self.env.reset()
        self.status = "ready"
    
    async def run_step(self) -> dict:
        if self.observations is None:
            self.reset()
        
        actions, _ = self.agent.act(self.observations, deterministic=True)
        self.observations, _, terminated, truncated, info = self.env.step(actions)
        
        if all(terminated.values()) or all(truncated.values()):
            self.status = "finished"
            self.running = False
        
        return self.env.get_state_for_render()
    
    async def broadcast(self, message: dict):
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except:
                pass


# ============== Global State ==============

games: Dict[str, GameSession] = {}


# ============== Create App ==============

def create_app() -> FastAPI:
    app = FastAPI(title="SkyBattle API", version="1.0.0")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.post("/api/v1/games", response_model=GameResponse)
    async def create_game(request: GameCreateRequest):
        game_id = f"game_{uuid.uuid4().hex[:8]}"
        session = GameSession(game_id, request)
        session.reset()
        games[game_id] = session
        return GameResponse(game_id=game_id, status="created", websocket_url=f"/ws/game/{game_id}")
    
    @app.get("/api/v1/games/{game_id}")
    async def get_game(game_id: str):
        if game_id not in games:
            raise HTTPException(status_code=404, detail="Game not found")
        session = games[game_id]
        info = session.env._get_info()
        return {"game_id": game_id, "status": session.status, **info}
    
    @app.post("/api/v1/games/{game_id}/control")
    async def control_game(game_id: str, request: ControlRequest):
        if game_id not in games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        session = games[game_id]
        if request.action == "start":
            session.running = True
            session.paused = False
            session.status = "running"
            asyncio.create_task(_run_game_loop(session))
        elif request.action == "pause":
            session.paused = True
            session.status = "paused"
        elif request.action == "resume":
            session.paused = False
            session.status = "running"
        elif request.action == "stop":
            session.running = False
            session.status = "stopped"
        
        return {"status": "ok", "game_status": session.status}
    
    @app.delete("/api/v1/games/{game_id}")
    async def delete_game(game_id: str):
        if game_id not in games:
            raise HTTPException(status_code=404, detail="Game not found")
        games[game_id].running = False
        del games[game_id]
        return {"status": "deleted"}
    
    @app.websocket("/ws/game/{game_id}")
    async def game_websocket(websocket: WebSocket, game_id: str):
        if game_id not in games:
            await websocket.close(code=4004)
            return
        
        session = games[game_id]
        await websocket.accept()
        session.connections.append(websocket)
        
        try:
            while True:
                await asyncio.sleep(0.1)
                if not session.running:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in session.connections:
                session.connections.remove(websocket)
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "games_active": len(games)}
    
    return app


async def _run_game_loop(session: GameSession):
    fps, frame_time = 10, 0.1
    
    while session.running:
        if session.paused:
            await asyncio.sleep(0.1)
            continue
        
        state = await session.run_step()
        await session.broadcast({"type": "game_state", "data": state})
        
        if session.status == "finished":
            await session.broadcast({"type": "game_end", "data": session.env._get_info()})
            break
        
        await asyncio.sleep(frame_time)


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
