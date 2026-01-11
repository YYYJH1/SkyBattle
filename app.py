#!/usr/bin/env python3
"""
SkyBattle - 一体化 Web 应用
=====================================
一键启动，浏览器直接观看无人机对战

用法:
    python app.py                    # 启动服务 (端口 8080)
    python app.py --port 9000        # 指定端口
    python app.py --host 0.0.0.0     # 允许远程访问

然后访问 http://localhost:8080
"""

import argparse
import json
import threading
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os

# 导入核心模块
from backend.envs import CombatEnv, CombatConfig
from backend.agents import MAPPOAgent

# ============================================================
#                       全局状态管理
# ============================================================

@dataclass
class GameState:
    """游戏状态"""
    game_id: str
    status: str  # waiting, running, paused, finished
    frames: List[dict]
    current_frame: int
    winner: Optional[str]
    config: dict

class GameManager:
    """游戏管理器"""
    
    def __init__(self):
        self.games: Dict[str, GameState] = {}
        self.current_game: Optional[str] = None
        self.lock = threading.Lock()
        self.game_counter = 0
    
    def create_game(self, team_size: int = 3, max_steps: int = 500) -> str:
        """创建新游戏"""
        with self.lock:
            self.game_counter += 1
            game_id = f"game_{self.game_counter:04d}"
            
            self.games[game_id] = GameState(
                game_id=game_id,
                status="waiting",
                frames=[],
                current_frame=0,
                winner=None,
                config={"team_size": team_size, "max_steps": max_steps}
            )
            self.current_game = game_id
            return game_id
    
    def run_game(self, game_id: str, speed: str = "normal"):
        """运行游戏（在后台线程中）"""
        if game_id not in self.games:
            return
        
        game = self.games[game_id]
        game.status = "running"
        game.frames = []
        
        # 创建环境
        config = CombatConfig(
            team_size=game.config["team_size"],
            max_steps=game.config["max_steps"]
        )
        env = CombatEnv(config=config)
        obs, info = env.reset(seed=int(time.time() * 1000) % 100000)
        
        # 运行战斗
        for step in range(config.max_steps):
            if game.status == "stopped":
                break
            
            state = env.get_state_for_render()
            
            # 智能追击策略
            actions = self._get_smart_actions(state, step)
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            state = env.get_state_for_render()
            
            # 保存帧（包含更多信息用于可视化）
            frame = {
                "step": step,
                "drones": state["drones"],
                "projectiles": state["projectiles"],
                "red_alive": info["red_alive"],
                "blue_alive": info["blue_alive"],
                "red_hp": sum(d["hp"] for d in state["drones"] if d["team"] == "red"),
                "blue_hp": sum(d["hp"] for d in state["drones"] if d["team"] == "blue"),
            }
            game.frames.append(frame)
            
            if all(terminated.values()):
                game.winner = info.get("winner")
                break
        
        game.status = "finished"
    
    def _get_smart_actions(self, state: dict, step: int) -> dict:
        """智能追击策略 - 让战斗更有策略性"""
        actions = {}
        drones = state["drones"]
        
        # 分队
        red_drones = [d for d in drones if d["team"] == "red" and d["is_alive"]]
        blue_drones = [d for d in drones if d["team"] == "blue" and d["is_alive"]]
        
        for drone in drones:
            drone_id = drone["id"]
            
            if not drone["is_alive"]:
                actions[drone_id] = {"discrete": 0, "continuous": [0, 0, 0, 0]}
                continue
            
            pos = np.array(drone["position"])
            vel = np.array(drone["velocity"])
            
            # 找最近的敌人
            enemies = blue_drones if drone["team"] == "red" else red_drones
            if not enemies:
                # 没有敌人，巡逻
                actions[drone_id] = {
                    "discrete": 0,
                    "continuous": [0.3, 0, np.sin(step * 0.05) * 0.3, 0]
                }
                continue
            
            # 找最近或血量最低的敌人
            target = min(enemies, key=lambda e: (
                np.linalg.norm(np.array(e["position"]) - pos) + 
                e["hp"] * 2  # 优先攻击低血量
            ))
            target_pos = np.array(target["position"])
            
            # 计算追击方向
            to_target = target_pos - pos
            dist = np.linalg.norm(to_target)
            
            if dist < 1:
                to_target = np.array([1, 0, 0])
                dist = 1
            
            direction = to_target / dist
            
            # 计算转向
            yaw_target = np.arctan2(direction[1], direction[0])
            pitch_target = np.arcsin(np.clip(direction[2], -1, 1))
            
            current_yaw = drone["orientation"][2] if len(drone["orientation"]) > 2 else 0
            current_pitch = drone["orientation"][1] if len(drone["orientation"]) > 1 else 0
            
            yaw_error = yaw_target - current_yaw
            pitch_error = pitch_target - current_pitch
            
            # 归一化角度误差
            while yaw_error > np.pi: yaw_error -= 2 * np.pi
            while yaw_error < -np.pi: yaw_error += 2 * np.pi
            
            # 控制输入
            throttle = 0.8 if dist > 100 else 0.5
            yaw_rate = np.clip(yaw_error * 0.5, -1, 1)
            pitch_rate = np.clip(pitch_error * 0.5, -1, 1)
            
            # 决定是否开火
            angle_to_target = abs(yaw_error) + abs(pitch_error)
            
            if dist < 150 and angle_to_target < 0.5:
                # 近距离且瞄准 - 开火
                discrete = 1  # 机枪
            elif dist < 300 and angle_to_target < 0.3 and np.random.random() < 0.02:
                # 中距离瞄准好 - 偶尔发射导弹
                discrete = 2  # 导弹
            else:
                discrete = 0  # 不开火
            
            # 添加一点随机性让战斗更自然
            throttle += np.random.uniform(-0.1, 0.1)
            yaw_rate += np.random.uniform(-0.1, 0.1)
            pitch_rate += np.random.uniform(-0.05, 0.05)
            
            actions[drone_id] = {
                "discrete": int(discrete),
                "continuous": [
                    float(np.clip(throttle, 0, 1)),
                    float(np.clip(pitch_rate, -1, 1)),
                    float(np.clip(yaw_rate, -1, 1)),
                    float(np.random.uniform(-0.1, 0.1))  # 少量翻滚
                ]
            }
        
        return actions
        if not game.winner:
            # 根据存活和HP判断胜负
            last = game.frames[-1] if game.frames else None
            if last:
                if last["red_alive"] > last["blue_alive"]:
                    game.winner = "red"
                elif last["blue_alive"] > last["red_alive"]:
                    game.winner = "blue"
                elif last["red_hp"] > last["blue_hp"]:
                    game.winner = "red"
                elif last["blue_hp"] > last["red_hp"]:
                    game.winner = "blue"
                else:
                    game.winner = "draw"
    
    def get_game_data(self, game_id: str) -> Optional[dict]:
        """获取游戏数据"""
        if game_id not in self.games:
            return None
        
        game = self.games[game_id]
        return {
            "game_id": game.game_id,
            "status": game.status,
            "frames": game.frames,
            "winner": game.winner,
            "config": game.config,
            "total_frames": len(game.frames)
        }

# 全局管理器
manager = GameManager()

# ============================================================
#                       HTTP 处理器
# ============================================================

class SkyBattleHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""
    
    def log_message(self, format, *args):
        """静默日志"""
        pass
    
    def send_json(self, data: dict, status: int = 200):
        """发送 JSON 响应"""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_html(self, html: str):
        """发送 HTML 响应"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        
        if path == "/" or path == "/index.html":
            self.send_html(get_main_page())
        
        elif path == "/api/new_game":
            team_size = int(params.get("team_size", [3])[0])
            max_steps = int(params.get("max_steps", [500])[0])
            game_id = manager.create_game(team_size, max_steps)
            
            # 在后台运行游戏
            thread = threading.Thread(
                target=manager.run_game, 
                args=(game_id,),
                daemon=True
            )
            thread.start()
            
            self.send_json({"game_id": game_id, "status": "started"})
        
        elif path == "/api/game_data":
            game_id = params.get("game_id", [manager.current_game])[0]
            if game_id:
                data = manager.get_game_data(game_id)
                if data:
                    self.send_json(data)
                else:
                    self.send_json({"error": "Game not found"}, 404)
            else:
                self.send_json({"error": "No active game"}, 404)
        
        elif path == "/api/status":
            self.send_json({
                "status": "running",
                "games_count": len(manager.games),
                "current_game": manager.current_game
            })
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

# ============================================================
#                       HTML 页面
# ============================================================

def get_main_page() -> str:
    return '''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SkyBattle - 无人机对战模拟器</title>
    <style>
        :root {
            --bg-primary: #0a0a1a;
            --bg-secondary: #12122a;
            --bg-card: rgba(20, 20, 50, 0.8);
            --accent-cyan: #00d4ff;
            --accent-red: #ff4466;
            --accent-blue: #4488ff;
            --text-primary: #ffffff;
            --text-secondary: #8888aa;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            min-height: 100vh;
            color: var(--text-primary);
            overflow-x: hidden;
        }
        
        /* 顶部导航 */
        .navbar {
            background: var(--bg-card);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(0, 212, 255, 0.2);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.8em;
            font-weight: bold;
            background: linear-gradient(90deg, var(--accent-cyan), #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        }
        
        .nav-status {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            background: #00ff88;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* 主容器 */
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
        }
        
        /* 控制面板 */
        .control-panel {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .control-row {
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        .control-group label {
            font-size: 0.85em;
            color: var(--text-secondary);
        }
        
        .control-group select, .control-group input {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 1em;
        }
        
        .btn {
            background: linear-gradient(135deg, var(--accent-cyan), #0088cc);
            border: none;
            color: white;
            padding: 15px 40px;
            font-size: 1.1em;
            font-weight: bold;
            border-radius: 30px;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.4);
        }
        
        .btn:disabled {
            background: #444;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, var(--accent-red), #cc0044);
        }
        
        /* 游戏区域 */
        .game-area {
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 30px;
        }
        
        @media (max-width: 1000px) {
            .game-area { grid-template-columns: 1fr; }
        }
        
        /* 战斗画布 */
        .arena-container {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .arena-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .arena-title {
            font-size: 1.3em;
            color: var(--accent-cyan);
        }
        
        #arena {
            width: 100%;
            height: 500px;
            background: radial-gradient(ellipse at center, #1a1a3a 0%, #0a0a1a 100%);
            border-radius: 15px;
            border: 2px solid rgba(0, 212, 255, 0.3);
        }
        
        /* 侧边栏 */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .card-title {
            font-size: 1.1em;
            margin-bottom: 15px;
            color: var(--accent-cyan);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* 队伍状态 */
        .team-status {
            margin-bottom: 20px;
        }
        
        .team-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .team-name {
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .team-name.red { color: var(--accent-red); }
        .team-name.blue { color: var(--accent-blue); }
        
        .hp-bar {
            height: 12px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 6px;
            overflow: hidden;
        }
        
        .hp-fill {
            height: 100%;
            transition: width 0.3s ease;
            border-radius: 6px;
        }
        
        .hp-fill.red { 
            background: linear-gradient(90deg, var(--accent-red), #ff6688); 
        }
        .hp-fill.blue { 
            background: linear-gradient(90deg, var(--accent-blue), #66aaff); 
        }
        
        /* 统计数据 */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .stat-item {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: var(--accent-cyan);
        }
        
        .stat-label {
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-top: 5px;
        }
        
        /* 进度条 */
        .progress-container {
            margin-top: 15px;
        }
        
        .progress-bar {
            height: 6px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), #00ff88);
            transition: width 0.3s;
        }
        
        /* 胜利提示 */
        .winner-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .winner-box {
            background: var(--bg-card);
            padding: 50px 80px;
            border-radius: 30px;
            text-align: center;
            border: 3px solid var(--accent-cyan);
            animation: winner-appear 0.5s ease;
        }
        
        @keyframes winner-appear {
            from { transform: scale(0.5); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        
        .winner-title {
            font-size: 3em;
            margin-bottom: 20px;
        }
        
        .winner-team {
            font-size: 2em;
            font-weight: bold;
        }
        
        .winner-team.red { color: var(--accent-red); }
        .winner-team.blue { color: var(--accent-blue); }
        .winner-team.draw { color: var(--text-secondary); }
        
        /* 加载动画 */
        .loading {
            display: none;
            text-align: center;
            padding: 50px;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(0, 212, 255, 0.2);
            border-top-color: var(--accent-cyan);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <!-- 导航栏 -->
    <nav class="navbar">
        <div class="logo">✈️ SkyBattle</div>
        <div class="nav-status">
            <div class="status-dot"></div>
            <span>系统运行中</span>
        </div>
    </nav>
    
    <div class="container">
        <!-- 控制面板 -->
        <div class="control-panel">
            <div class="control-row">
                <div class="control-group">
                    <label>队伍大小</label>
                    <select id="teamSize">
                        <option value="2">2 vs 2</option>
                        <option value="3" selected>3 vs 3</option>
                        <option value="4">4 vs 4</option>
                        <option value="5">5 vs 5</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>最大步数</label>
                    <select id="maxSteps">
                        <option value="200">200 (快速)</option>
                        <option value="500" selected>500 (标准)</option>
                        <option value="1000">1000 (持久)</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>播放速度</label>
                    <select id="playSpeed">
                        <option value="20">极快</option>
                        <option value="50" selected>快速</option>
                        <option value="100">正常</option>
                        <option value="200">慢速</option>
                    </select>
                </div>
                
                <button class="btn" id="startBtn" onclick="startNewGame()">
                    🎮 开始战斗
                </button>
            </div>
        </div>
        
        <!-- 游戏区域 -->
        <div class="game-area">
            <!-- 战斗画布 -->
            <div class="arena-container">
                <div class="arena-header">
                    <div class="arena-title">⚔️ 战斗区域</div>
                    <div id="gameStatus">等待开始...</div>
                </div>
                <canvas id="arena"></canvas>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <div>战斗模拟中...</div>
                </div>
            </div>
            
            <!-- 侧边栏 -->
            <div class="sidebar">
                <!-- 队伍状态 -->
                <div class="card">
                    <div class="card-title">📊 队伍状态</div>
                    
                    <div class="team-status">
                        <div class="team-header">
                            <span class="team-name red">🔴 红队</span>
                            <span id="redAlive">0/0</span>
                        </div>
                        <div class="hp-bar">
                            <div class="hp-fill red" id="redHp" style="width: 100%"></div>
                        </div>
                    </div>
                    
                    <div class="team-status">
                        <div class="team-header">
                            <span class="team-name blue">🔵 蓝队</span>
                            <span id="blueAlive">0/0</span>
                        </div>
                        <div class="hp-bar">
                            <div class="hp-fill blue" id="blueHp" style="width: 100%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- 统计数据 -->
                <div class="card">
                    <div class="card-title">📈 战斗数据</div>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="currentStep">0</div>
                            <div class="stat-label">当前步数</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="totalFrames">0</div>
                            <div class="stat-label">总帧数</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="projectiles">0</div>
                            <div class="stat-label">弹药数</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="gameTime">0s</div>
                            <div class="stat-label">战斗时间</div>
                        </div>
                    </div>
                    
                    <div class="progress-container">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-size: 0.85em; color: var(--text-secondary);">战斗进度</span>
                            <span id="progressText" style="font-size: 0.85em;">0%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- 说明 -->
                <div class="card">
                    <div class="card-title">ℹ️ 说明</div>
                    <ul style="font-size: 0.9em; color: var(--text-secondary); line-height: 1.8; padding-left: 20px;">
                        <li>点击"开始战斗"启动新对局</li>
                        <li>红蓝两队 AI 自动对战</li>
                        <li>可调整队伍大小和战斗长度</li>
                        <li>实时显示 HP 和存活状态</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 胜利提示 -->
    <div class="winner-overlay" id="winnerOverlay" onclick="this.style.display='none'">
        <div class="winner-box">
            <div class="winner-title">🏆 战斗结束</div>
            <div class="winner-team" id="winnerText">红队获胜！</div>
            <p style="margin-top: 20px; color: var(--text-secondary);">点击任意处关闭</p>
        </div>
    </div>
    
    <script>
        // 画布设置
        const canvas = document.getElementById('arena');
        const ctx = canvas.getContext('2d');
        
        // 状态
        let gameData = null;
        let currentFrame = 0;
        let playing = false;
        let animationId = null;
        let teamSize = 3;
        
        // 调整画布大小
        function resizeCanvas() {
            const container = canvas.parentElement;
            canvas.width = container.clientWidth - 40;
            canvas.height = 500;
            drawIdleScreen();
        }
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        
        // 空闲画面
        function drawIdleScreen() {
            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // 网格
            ctx.strokeStyle = 'rgba(0, 212, 255, 0.1)';
            ctx.lineWidth = 1;
            for (let x = 0; x < canvas.width; x += 50) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
            for (let y = 0; y < canvas.height; y += 50) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }
            
            // 提示文字
            ctx.font = 'bold 24px Segoe UI';
            ctx.fillStyle = '#444';
            ctx.textAlign = 'center';
            ctx.fillText('点击上方按钮开始战斗', canvas.width / 2, canvas.height / 2);
        }
        
        // 开始新游戏
        async function startNewGame() {
            const startBtn = document.getElementById('startBtn');
            startBtn.disabled = true;
            startBtn.textContent = '⏳ 加载中...';
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('winnerOverlay').style.display = 'none';
            
            teamSize = parseInt(document.getElementById('teamSize').value);
            const maxSteps = parseInt(document.getElementById('maxSteps').value);
            
            try {
                // 创建游戏
                const res = await fetch(`/api/new_game?team_size=${teamSize}&max_steps=${maxSteps}`);
                const data = await res.json();
                
                // 等待游戏完成
                await waitForGame(data.game_id);
                
                // 获取完整数据
                const gameRes = await fetch(`/api/game_data?game_id=${data.game_id}`);
                gameData = await gameRes.json();
                
                document.getElementById('loading').style.display = 'none';
                document.getElementById('totalFrames').textContent = gameData.total_frames;
                
                // 开始播放
                currentFrame = 0;
                playing = true;
                playAnimation();
                
            } catch (error) {
                console.error(error);
                alert('启动失败，请重试');
            }
            
            startBtn.disabled = false;
            startBtn.textContent = '🎮 开始战斗';
        }
        
        // 等待游戏完成
        async function waitForGame(gameId) {
            while (true) {
                const res = await fetch(`/api/game_data?game_id=${gameId}`);
                const data = await res.json();
                
                if (data.status === 'finished') {
                    return;
                }
                
                await new Promise(r => setTimeout(r, 200));
            }
        }
        
        // 播放动画
        function playAnimation() {
            if (!playing || !gameData || currentFrame >= gameData.frames.length) {
                playing = false;
                if (gameData && gameData.winner) {
                    showWinner(gameData.winner);
                }
                return;
            }
            
            const frame = gameData.frames[currentFrame];
            drawFrame(frame);
            updateStats(frame);
            
            currentFrame++;
            
            const speed = parseInt(document.getElementById('playSpeed').value);
            animationId = setTimeout(playAnimation, speed);
        }
        
        // 绘制帧
        function drawFrame(frame) {
            // 清空（带轨迹效果）
            ctx.fillStyle = 'rgba(10, 10, 26, 0.2)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // 网格
            ctx.strokeStyle = 'rgba(0, 212, 255, 0.05)';
            ctx.lineWidth = 1;
            for (let x = 0; x < canvas.width; x += 50) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
            for (let y = 0; y < canvas.height; y += 50) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }
            
            // 无人机
            frame.drones.forEach(drone => {
                const x = (drone.position[0] + 500) / 1000 * canvas.width;
                const y = (drone.position[1] + 500) / 1000 * canvas.height;
                const z = drone.position[2];
                
                const isRed = drone.team === 'red';
                const color = isRed ? '#ff4466' : '#4488ff';
                const glowColor = isRed ? 'rgba(255, 68, 102, 0.4)' : 'rgba(68, 136, 255, 0.4)';
                
                if (drone.is_alive) {
                    const size = 10 + z / 25;
                    
                    // 光晕
                    ctx.beginPath();
                    ctx.arc(x, y, size + 8, 0, Math.PI * 2);
                    ctx.fillStyle = glowColor;
                    ctx.fill();
                    
                    // 无人机
                    ctx.beginPath();
                    ctx.arc(x, y, size, 0, Math.PI * 2);
                    ctx.fillStyle = color;
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    // HP 条
                    const hpWidth = 30;
                    const hpHeight = 4;
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                    ctx.fillRect(x - hpWidth/2, y - size - 12, hpWidth, hpHeight);
                    ctx.fillStyle = color;
                    ctx.fillRect(x - hpWidth/2, y - size - 12, hpWidth * (drone.hp / 100), hpHeight);
                    
                    // 速度向量
                    const vx = drone.velocity[0] * 0.03;
                    const vy = drone.velocity[1] * 0.03;
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x + vx, y + vy);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                } else {
                    // 爆炸残骸
                    ctx.font = '16px Arial';
                    ctx.fillText('💥', x - 8, y + 5);
                }
            });
            
            // 弹药
            frame.projectiles.forEach(proj => {
                const x = (proj.position[0] + 500) / 1000 * canvas.width;
                const y = (proj.position[1] + 500) / 1000 * canvas.height;
                
                // 光晕
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255, 255, 0, 0.3)';
                ctx.fill();
                
                // 弹药
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fillStyle = '#ffff00';
                ctx.fill();
            });
            
            // 步数
            ctx.font = 'bold 14px Segoe UI';
            ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.textAlign = 'left';
            ctx.fillText(`Step: ${frame.step}`, 15, 25);
        }
        
        // 更新统计
        function updateStats(frame) {
            const maxHp = teamSize * 100;
            
            document.getElementById('redAlive').textContent = `${frame.red_alive}/${teamSize}`;
            document.getElementById('blueAlive').textContent = `${frame.blue_alive}/${teamSize}`;
            document.getElementById('redHp').style.width = (frame.red_hp / maxHp * 100) + '%';
            document.getElementById('blueHp').style.width = (frame.blue_hp / maxHp * 100) + '%';
            
            document.getElementById('currentStep').textContent = frame.step;
            document.getElementById('projectiles').textContent = frame.projectiles.length;
            document.getElementById('gameTime').textContent = (frame.step * 0.1).toFixed(1) + 's';
            document.getElementById('gameStatus').textContent = '战斗进行中...';
            
            const progress = (currentFrame / gameData.total_frames * 100).toFixed(0);
            document.getElementById('progressText').textContent = progress + '%';
            document.getElementById('progressFill').style.width = progress + '%';
        }
        
        // 显示胜利
        function showWinner(winner) {
            document.getElementById('gameStatus').textContent = '战斗结束';
            
            const overlay = document.getElementById('winnerOverlay');
            const text = document.getElementById('winnerText');
            
            if (winner === 'red') {
                text.textContent = '🔴 红队获胜！';
                text.className = 'winner-team red';
            } else if (winner === 'blue') {
                text.textContent = '🔵 蓝队获胜！';
                text.className = 'winner-team blue';
            } else {
                text.textContent = '⚖️ 平局';
                text.className = 'winner-team draw';
            }
            
            overlay.style.display = 'flex';
        }
        
        // 初始化
        drawIdleScreen();
    </script>
</body>
</html>
'''

# ============================================================
#                       主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SkyBattle Web 应用")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="端口号")
    args = parser.parse_args()
    
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "✈️  SkyBattle Web App  ✈️" + " " * 15 + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  🌐 本地访问: http://localhost:{args.port}" + " " * (27 - len(str(args.port))) + "║")
    if args.host == "0.0.0.0":
        print(f"║  🔗 远程访问: http://<服务器IP>:{args.port}" + " " * (18 - len(str(args.port))) + "║")
    print("║" + " " * 58 + "║")
    print("║  📌 按 Ctrl+C 停止服务器" + " " * 32 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    server = HTTPServer((args.host, args.port), SkyBattleHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
        server.shutdown()

if __name__ == "__main__":
    main()
