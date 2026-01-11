#!/usr/bin/env python3
"""
SkyBattle v2.0 - 增强版无人机对战模拟器
==========================================
特性:
- 智能追击策略
- 团队协作战术
- 增强可视化效果
- 战斗统计面板

用法: python app_v2.py --port 8088
"""

import argparse
import json
import threading
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from backend.envs import CombatEnv, CombatConfig

# ============================================================
#                     智能策略系统
# ============================================================

class SmartStrategy:
    """智能战斗策略 - 让 AI 更聪明"""
    
    def __init__(self):
        self.role_assignments: Dict[str, str] = {}
        self.targets: Dict[str, str] = {}  # drone_id -> target_id
        self.formation_center = np.zeros(3)
    
    def assign_roles(self, team: str, drones: List[dict]):
        """分配角色"""
        team_drones = [d for d in drones if d["team"] == team and d["is_alive"]]
        n = len(team_drones)
        
        # 按位置排序（前面的当突击手）
        if team == "red":
            team_drones.sort(key=lambda d: -d["position"][0])
        else:
            team_drones.sort(key=lambda d: d["position"][0])
        
        for i, drone in enumerate(team_drones):
            if i == 0:
                self.role_assignments[drone["id"]] = "leader"  # 队长
            elif i < n * 0.6:
                self.role_assignments[drone["id"]] = "attacker"  # 突击手
            else:
                self.role_assignments[drone["id"]] = "support"  # 支援
    
    def get_team_actions(self, drones: List[dict], step: int) -> Dict[str, dict]:
        """获取整个团队的行动"""
        actions = {}
        
        red_drones = [d for d in drones if d["team"] == "red" and d["is_alive"]]
        blue_drones = [d for d in drones if d["team"] == "blue" and d["is_alive"]]
        
        # 分配角色
        if step == 0 or step % 50 == 0:
            self.assign_roles("red", drones)
            self.assign_roles("blue", drones)
            self._assign_targets(red_drones, blue_drones)
            self._assign_targets(blue_drones, red_drones)
        
        # 获取每个无人机的行动
        for drone in drones:
            if not drone["is_alive"]:
                actions[drone["id"]] = {"discrete": 0, "continuous": [0, 0, 0, 0]}
                continue
            
            enemies = blue_drones if drone["team"] == "red" else red_drones
            allies = red_drones if drone["team"] == "red" else blue_drones
            
            role = self.role_assignments.get(drone["id"], "attacker")
            action = self._get_role_action(drone, allies, enemies, role, step)
            actions[drone["id"]] = action
        
        return actions
    
    def _assign_targets(self, attackers: List[dict], targets: List[dict]):
        """分配攻击目标 - 集火策略"""
        if not targets:
            return
        
        # 按血量排序，优先攻击低血量
        sorted_targets = sorted(targets, key=lambda t: t["hp"])
        
        for i, attacker in enumerate(attackers):
            # 分散攻击目标，但优先低血量
            target_idx = i % len(sorted_targets)
            if i < len(attackers) // 2:
                target_idx = 0  # 一半人集火最低血量
            self.targets[attacker["id"]] = sorted_targets[target_idx]["id"]
    
    def _get_role_action(self, drone: dict, allies: List[dict], enemies: List[dict], 
                         role: str, step: int) -> dict:
        """根据角色获取行动"""
        if not enemies:
            return self._patrol_action(drone, step)
        
        if role == "leader":
            return self._leader_action(drone, allies, enemies, step)
        elif role == "attacker":
            return self._attacker_action(drone, enemies, step)
        else:
            return self._support_action(drone, allies, enemies, step)
    
    def _leader_action(self, drone: dict, allies: List[dict], enemies: List[dict], 
                       step: int) -> dict:
        """队长行为 - 冲锋在前，选择最优目标"""
        pos = np.array(drone["position"])
        
        # 找最近的敌人
        target = min(enemies, key=lambda e: np.linalg.norm(np.array(e["position"]) - pos))
        
        return self._pursue_and_attack(drone, target, aggression=0.9)
    
    def _attacker_action(self, drone: dict, enemies: List[dict], step: int) -> dict:
        """突击手行为 - 追击分配的目标"""
        pos = np.array(drone["position"])
        
        # 使用分配的目标
        target_id = self.targets.get(drone["id"])
        target = next((e for e in enemies if e["id"] == target_id), None)
        
        if not target:
            target = min(enemies, key=lambda e: np.linalg.norm(np.array(e["position"]) - pos))
        
        return self._pursue_and_attack(drone, target, aggression=0.85)
    
    def _support_action(self, drone: dict, allies: List[dict], enemies: List[dict], 
                        step: int) -> dict:
        """支援行为 - 保持距离，支援队友"""
        pos = np.array(drone["position"])
        
        # 找被围攻的队友
        allies_in_danger = []
        for ally in allies:
            if ally["id"] == drone["id"]:
                continue
            ally_pos = np.array(ally["position"])
            nearby_enemies = sum(1 for e in enemies 
                               if np.linalg.norm(np.array(e["position"]) - ally_pos) < 150)
            if nearby_enemies >= 2:
                allies_in_danger.append(ally)
        
        if allies_in_danger:
            # 支援被围攻的队友
            ally = min(allies_in_danger, key=lambda a: a["hp"])
            target = min(enemies, 
                        key=lambda e: np.linalg.norm(np.array(e["position"]) - np.array(ally["position"])))
            return self._pursue_and_attack(drone, target, aggression=0.7)
        else:
            # 正常追击
            target = min(enemies, key=lambda e: e["hp"])  # 集火低血量
            return self._pursue_and_attack(drone, target, aggression=0.75)
    
    def _pursue_and_attack(self, drone: dict, target: dict, aggression: float = 0.8) -> dict:
        """追击并攻击目标"""
        pos = np.array(drone["position"])
        vel = np.array(drone["velocity"])
        target_pos = np.array(target["position"])
        target_vel = np.array(target["velocity"])
        
        # 预测目标位置（提前量）
        dist = np.linalg.norm(target_pos - pos)
        predict_time = dist / 500  # 假设子弹速度 500
        predicted_pos = target_pos + target_vel * predict_time * 0.5
        
        # 计算追击方向
        to_target = predicted_pos - pos
        dist = np.linalg.norm(to_target)
        
        if dist < 1:
            direction = np.array([1, 0, 0])
        else:
            direction = to_target / dist
        
        # 计算需要的偏航和俯仰
        yaw_target = np.arctan2(direction[1], direction[0])
        pitch_target = np.arcsin(np.clip(direction[2], -1, 1))
        
        ori = drone["orientation"]
        current_yaw = ori[2] if len(ori) > 2 else 0
        current_pitch = ori[1] if len(ori) > 1 else 0
        
        yaw_error = yaw_target - current_yaw
        pitch_error = pitch_target - current_pitch
        
        # 归一化
        while yaw_error > np.pi: yaw_error -= 2 * np.pi
        while yaw_error < -np.pi: yaw_error += 2 * np.pi
        
        # 控制增益
        yaw_rate = np.clip(yaw_error * 1.5, -1, 1)
        pitch_rate = np.clip(pitch_error * 1.2, -1, 1)
        
        # 速度控制
        if dist > 200:
            throttle = 1.0  # 全速追击
        elif dist > 100:
            throttle = 0.7
        else:
            throttle = 0.5  # 近距离减速
        
        # 决定开火
        angle_error = abs(yaw_error) + abs(pitch_error)
        
        if dist < 200 and angle_error < 0.4:
            discrete = 1  # 机枪
        elif dist < 350 and angle_error < 0.25 and np.random.random() < 0.03:
            discrete = 2  # 导弹
        elif dist < 120 and angle_error < 0.6:
            discrete = 1  # 近距离更容易开火
        else:
            discrete = 0
        
        # 添加微小随机性
        throttle += np.random.uniform(-0.05, 0.05)
        yaw_rate += np.random.uniform(-0.05, 0.05)
        
        return {
            "discrete": int(discrete),
            "continuous": [
                float(np.clip(throttle * aggression, 0, 1)),
                float(np.clip(pitch_rate, -1, 1)),
                float(np.clip(yaw_rate, -1, 1)),
                float(np.random.uniform(-0.1, 0.1))
            ]
        }
    
    def _patrol_action(self, drone: dict, step: int) -> dict:
        """巡逻行为"""
        return {
            "discrete": 0,
            "continuous": [
                0.3,
                np.sin(step * 0.03) * 0.2,
                np.cos(step * 0.02) * 0.3,
                0
            ]
        }


# ============================================================
#                     游戏管理器
# ============================================================

@dataclass
class GameState:
    game_id: str
    status: str
    frames: List[dict]
    winner: Optional[str]
    config: dict
    stats: dict

class GameManager:
    def __init__(self):
        self.games: Dict[str, GameState] = {}
        self.current_game: Optional[str] = None
        self.game_counter = 0
        self.strategy = SmartStrategy()
    
    def create_game(self, team_size: int = 3, max_steps: int = 400) -> str:
        self.game_counter += 1
        game_id = f"battle_{self.game_counter:04d}"
        
        self.games[game_id] = GameState(
            game_id=game_id,
            status="waiting",
            frames=[],
            winner=None,
            config={"team_size": team_size, "max_steps": max_steps},
            stats={"red_damage": 0, "blue_damage": 0, "red_kills": 0, "blue_kills": 0}
        )
        self.current_game = game_id
        return game_id
    
    def run_game(self, game_id: str):
        if game_id not in self.games:
            return
        
        game = self.games[game_id]
        game.status = "running"
        game.frames = []
        
        config = CombatConfig(
            team_size=game.config["team_size"],
            max_steps=game.config["max_steps"]
        )
        env = CombatEnv(config=config)
        obs, info = env.reset(seed=int(time.time() * 1000) % 100000)
        
        # 重置策略
        self.strategy = SmartStrategy()
        
        prev_hp = {"red": config.team_size * 100, "blue": config.team_size * 100}
        
        for step in range(config.max_steps):
            if game.status == "stopped":
                break
            
            state = env.get_state_for_render()
            
            # 使用智能策略
            actions = self.strategy.get_team_actions(state["drones"], step)
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            state = env.get_state_for_render()
            
            # 计算伤害统计
            red_hp = sum(d["hp"] for d in state["drones"] if d["team"] == "red")
            blue_hp = sum(d["hp"] for d in state["drones"] if d["team"] == "blue")
            
            game.stats["blue_damage"] += max(0, prev_hp["blue"] - blue_hp)
            game.stats["red_damage"] += max(0, prev_hp["red"] - red_hp)
            prev_hp = {"red": red_hp, "blue": blue_hp}
            
            # 保存帧
            frame = {
                "step": step,
                "drones": state["drones"],
                "projectiles": state["projectiles"],
                "red_alive": info["red_alive"],
                "blue_alive": info["blue_alive"],
                "red_hp": red_hp,
                "blue_hp": blue_hp,
            }
            game.frames.append(frame)
            
            if all(terminated.values()):
                game.winner = info.get("winner")
                break
        
        game.status = "finished"
        if not game.winner:
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
        if game_id not in self.games:
            return None
        game = self.games[game_id]
        return {
            "game_id": game.game_id,
            "status": game.status,
            "frames": game.frames,
            "winner": game.winner,
            "config": game.config,
            "stats": game.stats,
            "total_frames": len(game.frames)
        }

manager = GameManager()

# ============================================================
#                     HTTP 服务器
# ============================================================

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    
    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_html(self, html):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        
        if path == "/" or path == "/index.html":
            self.send_html(HTML_PAGE)
        elif path == "/api/new_game":
            team_size = int(params.get("team_size", [3])[0])
            max_steps = int(params.get("max_steps", [400])[0])
            game_id = manager.create_game(team_size, max_steps)
            threading.Thread(target=manager.run_game, args=(game_id,), daemon=True).start()
            self.send_json({"game_id": game_id, "status": "started"})
        elif path == "/api/game_data":
            game_id = params.get("game_id", [manager.current_game])[0]
            if game_id:
                data = manager.get_game_data(game_id)
                if data:
                    self.send_json(data)
                    return
            self.send_json({"error": "No game"}, 404)
        elif path == "/api/status":
            self.send_json({"status": "running", "version": "2.0"})
        else:
            self.send_response(404)
            self.end_headers()

# ============================================================
#                     增强版 HTML 页面
# ============================================================

HTML_PAGE = '''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SkyBattle v2.0 - 智能无人机对战</title>
    <style>
        :root {
            --bg-dark: #0a0f1a;
            --bg-card: rgba(15, 25, 45, 0.95);
            --accent: #00e5ff;
            --red: #ff3366;
            --blue: #3399ff;
            --gold: #ffd700;
            --text: #e0e0e0;
            --text-dim: #7a8599;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', sans-serif;
            background: var(--bg-dark);
            background-image: 
                radial-gradient(ellipse at 20% 80%, rgba(0, 229, 255, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(255, 51, 102, 0.08) 0%, transparent 50%);
            min-height: 100vh;
            color: var(--text);
        }
        
        .header {
            background: linear-gradient(180deg, rgba(0,0,0,0.8) 0%, transparent 100%);
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(0, 229, 255, 0.2);
        }
        
        .logo {
            font-size: 2em;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent), #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .version {
            background: var(--accent);
            color: var(--bg-dark);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }
        
        .main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 30px;
        }
        
        @media (max-width: 1100px) {
            .main { grid-template-columns: 1fr; }
        }
        
        .arena-section {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(0, 229, 255, 0.15);
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        
        select, input {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255,255,255,0.1);
            color: var(--text);
            padding: 12px 16px;
            border-radius: 10px;
            font-size: 1em;
        }
        
        .btn {
            background: linear-gradient(135deg, var(--accent), #0088aa);
            border: none;
            color: var(--bg-dark);
            padding: 14px 35px;
            font-size: 1.1em;
            font-weight: bold;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 229, 255, 0.4);
        }
        
        .btn:disabled {
            background: #444;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        #arena {
            width: 100%;
            height: 550px;
            background: radial-gradient(ellipse at center, #111827 0%, #0a0f1a 100%);
            border-radius: 15px;
            border: 2px solid rgba(0, 229, 255, 0.2);
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(0, 229, 255, 0.15);
        }
        
        .card-title {
            font-size: 1em;
            color: var(--accent);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .team-block {
            margin-bottom: 15px;
        }
        
        .team-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        
        .team-name { font-weight: bold; }
        .team-name.red { color: var(--red); }
        .team-name.blue { color: var(--blue); }
        
        .hp-bar {
            height: 10px;
            background: rgba(0,0,0,0.5);
            border-radius: 5px;
            overflow: hidden;
        }
        
        .hp-fill {
            height: 100%;
            transition: width 0.2s;
        }
        
        .hp-fill.red { background: linear-gradient(90deg, var(--red), #ff6688); }
        .hp-fill.blue { background: linear-gradient(90deg, var(--blue), #66aaff); }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .stat-item {
            background: rgba(0,0,0,0.3);
            padding: 12px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: var(--accent);
        }
        
        .stat-label {
            font-size: 0.75em;
            color: var(--text-dim);
            margin-top: 3px;
        }
        
        .legend {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 15px;
            font-size: 0.85em;
            color: var(--text-dim);
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .legend-icon {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        .legend-icon.red { background: var(--red); }
        .legend-icon.blue { background: var(--blue); }
        .legend-icon.bullet { background: #ffff00; }
        
        /* 胜利弹窗 */
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.85);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .modal-content {
            background: var(--bg-card);
            padding: 50px 70px;
            border-radius: 25px;
            text-align: center;
            border: 2px solid var(--gold);
            animation: pop 0.4s ease;
        }
        
        @keyframes pop {
            from { transform: scale(0.7); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        
        .modal-title {
            font-size: 2.5em;
            color: var(--gold);
            margin-bottom: 15px;
        }
        
        .modal-winner {
            font-size: 2em;
            font-weight: bold;
        }
        
        .modal-winner.red { color: var(--red); }
        .modal-winner.blue { color: var(--blue); }
        .modal-winner.draw { color: var(--text-dim); }
        
        .modal-stats {
            margin-top: 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            text-align: center;
        }
        
        .modal-stat {
            padding: 10px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <header class="header">
        <div>
            <span class="logo">✈️ SkyBattle</span>
            <span class="version">v2.0</span>
        </div>
        <div style="color: var(--text-dim)">智能无人机对战模拟器</div>
    </header>
    
    <main class="main">
        <section class="arena-section">
            <div class="controls">
                <select id="teamSize">
                    <option value="2">2 vs 2</option>
                    <option value="3" selected>3 vs 3</option>
                    <option value="4">4 vs 4</option>
                    <option value="5">5 vs 5</option>
                </select>
                
                <select id="maxSteps">
                    <option value="200">快速战斗</option>
                    <option value="400" selected>标准战斗</option>
                    <option value="600">持久战</option>
                </select>
                
                <select id="speed">
                    <option value="16">极快 (60fps)</option>
                    <option value="33" selected>快速 (30fps)</option>
                    <option value="66">正常 (15fps)</option>
                    <option value="100">慢速</option>
                </select>
                
                <button class="btn" id="startBtn" onclick="startGame()">
                    ⚔️ 开始战斗
                </button>
            </div>
            
            <canvas id="arena"></canvas>
            
            <div class="legend">
                <div class="legend-item"><div class="legend-icon red"></div> 红队</div>
                <div class="legend-item"><div class="legend-icon blue"></div> 蓝队</div>
                <div class="legend-item"><div class="legend-icon bullet"></div> 弹药</div>
            </div>
        </section>
        
        <aside class="sidebar">
            <div class="card">
                <div class="card-title">⚔️ 队伍状态</div>
                
                <div class="team-block">
                    <div class="team-header">
                        <span class="team-name red">🔴 红队</span>
                        <span id="redInfo">0/0 | 0 HP</span>
                    </div>
                    <div class="hp-bar">
                        <div class="hp-fill red" id="redHp" style="width:100%"></div>
                    </div>
                </div>
                
                <div class="team-block">
                    <div class="team-header">
                        <span class="team-name blue">🔵 蓝队</span>
                        <span id="blueInfo">0/0 | 0 HP</span>
                    </div>
                    <div class="hp-bar">
                        <div class="hp-fill blue" id="blueHp" style="width:100%"></div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">📊 战斗数据</div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="step">0</div>
                        <div class="stat-label">回合</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="time">0s</div>
                        <div class="stat-label">时间</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="redDmg">0</div>
                        <div class="stat-label">红队输出</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="blueDmg">0</div>
                        <div class="stat-label">蓝队输出</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">ℹ️ 策略说明</div>
                <ul style="font-size:0.85em; color:var(--text-dim); line-height:1.8; padding-left:18px;">
                    <li><b>队长</b>：冲锋在前，选择最优目标</li>
                    <li><b>突击手</b>：追击分配的目标</li>
                    <li><b>支援</b>：保护队友，集火低血量</li>
                    <li>AI 会预判敌人位置并提前瞄准</li>
                </ul>
            </div>
        </aside>
    </main>
    
    <!-- 胜利弹窗 -->
    <div class="modal" id="modal" onclick="this.style.display='none'">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-title">🏆 战斗结束</div>
            <div class="modal-winner" id="winnerText">红队获胜！</div>
            <div class="modal-stats">
                <div class="modal-stat">
                    <div style="color:var(--red);font-size:1.3em" id="modalRedDmg">0</div>
                    <div style="font-size:0.8em;color:var(--text-dim)">红队输出</div>
                </div>
                <div class="modal-stat">
                    <div style="color:var(--blue);font-size:1.3em" id="modalBlueDmg">0</div>
                    <div style="font-size:0.8em;color:var(--text-dim)">蓝队输出</div>
                </div>
            </div>
            <button class="btn" style="margin-top:25px" onclick="document.getElementById('modal').style.display='none'">
                关闭
            </button>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('arena');
        const ctx = canvas.getContext('2d');
        let gameData = null;
        let frame = 0;
        let playing = false;
        let teamSize = 3;
        let trails = {};  // 尾迹数据
        
        function resize() {
            const w = canvas.parentElement.clientWidth - 50;
            canvas.width = w;
            canvas.height = 550;
            drawIdle();
        }
        
        window.addEventListener('resize', resize);
        resize();
        
        function drawIdle() {
            ctx.fillStyle = '#0a0f1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // 网格
            ctx.strokeStyle = 'rgba(0, 229, 255, 0.06)';
            ctx.lineWidth = 1;
            for (let x = 0; x < canvas.width; x += 40) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
            for (let y = 0; y < canvas.height; y += 40) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }
            
            ctx.font = 'bold 20px Segoe UI';
            ctx.fillStyle = '#333';
            ctx.textAlign = 'center';
            ctx.fillText('点击「开始战斗」观看 AI 对战', canvas.width/2, canvas.height/2);
        }
        
        async function startGame() {
            const btn = document.getElementById('startBtn');
            btn.disabled = true;
            btn.textContent = '⏳ 生成中...';
            
            teamSize = parseInt(document.getElementById('teamSize').value);
            const maxSteps = parseInt(document.getElementById('maxSteps').value);
            trails = {};
            
            try {
                const res = await fetch(`/api/new_game?team_size=${teamSize}&max_steps=${maxSteps}`);
                const data = await res.json();
                
                // 等待完成
                while (true) {
                    const r = await fetch(`/api/game_data?game_id=${data.game_id}`);
                    const g = await r.json();
                    if (g.status === 'finished') {
                        gameData = g;
                        break;
                    }
                    await new Promise(r => setTimeout(r, 100));
                }
                
                frame = 0;
                playing = true;
                animate();
                
            } catch (e) {
                alert('错误: ' + e.message);
            }
            
            btn.disabled = false;
            btn.textContent = '⚔️ 开始战斗';
        }
        
        function animate() {
            if (!playing || !gameData || frame >= gameData.frames.length) {
                playing = false;
                if (gameData && gameData.winner) showWinner();
                return;
            }
            
            const f = gameData.frames[frame];
            draw(f);
            updateUI(f);
            
            frame++;
            const speed = parseInt(document.getElementById('speed').value);
            setTimeout(animate, speed);
        }
        
        function draw(f) {
            // 渐隐背景（产生尾迹效果）
            ctx.fillStyle = 'rgba(10, 15, 26, 0.15)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // 网格（微弱）
            ctx.strokeStyle = 'rgba(0, 229, 255, 0.03)';
            for (let x = 0; x < canvas.width; x += 40) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
            
            // 无人机
            f.drones.forEach(d => {
                const x = (d.position[0] + 500) / 1000 * canvas.width;
                const y = (d.position[1] + 500) / 1000 * canvas.height;
                
                // 保存尾迹
                if (!trails[d.id]) trails[d.id] = [];
                if (d.is_alive) {
                    trails[d.id].push({x, y});
                    if (trails[d.id].length > 15) trails[d.id].shift();
                }
                
                const color = d.team === 'red' ? '#ff3366' : '#3399ff';
                const glowColor = d.team === 'red' ? 'rgba(255,51,102,0.3)' : 'rgba(51,153,255,0.3)';
                
                // 绘制尾迹
                if (trails[d.id].length > 1) {
                    ctx.beginPath();
                    ctx.moveTo(trails[d.id][0].x, trails[d.id][0].y);
                    for (let i = 1; i < trails[d.id].length; i++) {
                        ctx.lineTo(trails[d.id][i].x, trails[d.id][i].y);
                    }
                    ctx.strokeStyle = glowColor;
                    ctx.lineWidth = 3;
                    ctx.stroke();
                }
                
                if (d.is_alive) {
                    const size = 10 + d.position[2] / 30;
                    
                    // 绘制三角形无人机
                    const angle = Math.atan2(d.velocity[1], d.velocity[0]);
                    
                    ctx.save();
                    ctx.translate(x, y);
                    ctx.rotate(angle);
                    
                    // 光晕
                    ctx.beginPath();
                    ctx.arc(0, 0, size + 6, 0, Math.PI * 2);
                    ctx.fillStyle = glowColor;
                    ctx.fill();
                    
                    // 三角形
                    ctx.beginPath();
                    ctx.moveTo(size * 1.2, 0);
                    ctx.lineTo(-size * 0.8, -size * 0.7);
                    ctx.lineTo(-size * 0.4, 0);
                    ctx.lineTo(-size * 0.8, size * 0.7);
                    ctx.closePath();
                    ctx.fillStyle = color;
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                    
                    ctx.restore();
                    
                    // HP条
                    const hpW = 28, hpH = 4;
                    ctx.fillStyle = 'rgba(0,0,0,0.6)';
                    ctx.fillRect(x - hpW/2, y - size - 14, hpW, hpH);
                    ctx.fillStyle = color;
                    ctx.fillRect(x - hpW/2, y - size - 14, hpW * d.hp / 100, hpH);
                } else {
                    // 爆炸效果
                    ctx.font = '18px Arial';
                    ctx.fillText('💥', x - 9, y + 6);
                }
            });
            
            // 弹药
            f.projectiles.forEach(p => {
                const x = (p.position[0] + 500) / 1000 * canvas.width;
                const y = (p.position[1] + 500) / 1000 * canvas.height;
                
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255,255,0,0.4)';
                ctx.fill();
                
                ctx.beginPath();
                ctx.arc(x, y, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = '#ffff00';
                ctx.fill();
            });
            
            // 步数
            ctx.font = 'bold 13px Segoe UI';
            ctx.fillStyle = 'rgba(255,255,255,0.6)';
            ctx.textAlign = 'left';
            ctx.fillText(`Step ${f.step}`, 12, 22);
        }
        
        function updateUI(f) {
            const maxHp = teamSize * 100;
            
            document.getElementById('redInfo').textContent = 
                `${f.red_alive}/${teamSize} | ${Math.round(f.red_hp)} HP`;
            document.getElementById('blueInfo').textContent = 
                `${f.blue_alive}/${teamSize} | ${Math.round(f.blue_hp)} HP`;
            
            document.getElementById('redHp').style.width = (f.red_hp / maxHp * 100) + '%';
            document.getElementById('blueHp').style.width = (f.blue_hp / maxHp * 100) + '%';
            
            document.getElementById('step').textContent = f.step;
            document.getElementById('time').textContent = (f.step * 0.1).toFixed(1) + 's';
            
            // 伤害统计
            document.getElementById('redDmg').textContent = 
                Math.round(gameData.stats.red_damage || 0);
            document.getElementById('blueDmg').textContent = 
                Math.round(gameData.stats.blue_damage || 0);
        }
        
        function showWinner() {
            const modal = document.getElementById('modal');
            const text = document.getElementById('winnerText');
            
            if (gameData.winner === 'red') {
                text.textContent = '🔴 红队获胜！';
                text.className = 'modal-winner red';
            } else if (gameData.winner === 'blue') {
                text.textContent = '🔵 蓝队获胜！';
                text.className = 'modal-winner blue';
            } else {
                text.textContent = '⚖️ 平局';
                text.className = 'modal-winner draw';
            }
            
            document.getElementById('modalRedDmg').textContent = 
                Math.round(gameData.stats.red_damage || 0);
            document.getElementById('modalBlueDmg').textContent = 
                Math.round(gameData.stats.blue_damage || 0);
            
            modal.style.display = 'flex';
        }
        
        drawIdle();
    </script>
</body>
</html>
'''

# ============================================================
#                        主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SkyBattle v2.0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    
    print()
    print("╔" + "═" * 50 + "╗")
    print("║" + " " * 12 + "✈️  SkyBattle v2.0  ✈️" + " " * 13 + "║")
    print("╠" + "═" * 50 + "╣")
    print(f"║  🌐 访问: http://localhost:{args.port}" + " " * (19 - len(str(args.port))) + "║")
    print("║  📌 Ctrl+C 停止服务" + " " * 28 + "║")
    print("╚" + "═" * 50 + "╝")
    print()
    
    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 已停止")

if __name__ == "__main__":
    main()
