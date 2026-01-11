#!/usr/bin/env python3
"""
SkyBattle Simple Viewer - 一键启动，浏览器直接观看战斗
用法: python simple_viewer.py
然后访问 http://localhost:8080
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import threading
import time
import numpy as np
from backend.envs import CombatEnv, CombatConfig

# 全局战斗数据
battle_frames = []
battle_running = False

def run_battle():
    """运行一场战斗并收集所有帧"""
    global battle_frames, battle_running
    battle_frames = []
    battle_running = True
    
    config = CombatConfig(team_size=3, max_steps=300)
    env = CombatEnv(config=config)
    obs, info = env.reset(seed=int(time.time()) % 10000)
    
    for step in range(300):
        # 简单的激进策略
        actions = {}
        for drone_id in obs.keys():
            actions[drone_id] = {
                'discrete': 1,  # 开火
                'continuous': [1.0, np.random.uniform(-0.3, 0.3), 
                              np.random.uniform(-0.3, 0.3), 0.0]
            }
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        state = env.get_state_for_render()
        
        # 保存帧数据
        frame = {
            'step': step,
            'drones': state['drones'],
            'projectiles': state['projectiles'],
            'red_alive': info['red_alive'],
            'blue_alive': info['blue_alive'],
            'winner': info.get('winner')
        }
        battle_frames.append(frame)
        
        if all(terminated.values()):
            break
    
    battle_running = False

class BattleHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == '/battle':
            # 启动新战斗
            if not battle_running:
                threading.Thread(target=run_battle, daemon=True).start()
                time.sleep(0.5)  # 等待一些帧
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'frames': battle_frames,
                'running': battle_running
            }).encode())
        else:
            super().do_GET()

HTML_PAGE = '''<!DOCTYPE html>
<html>
<head>
    <title>✈️ SkyBattle Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', sans-serif;
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0,255,255,0.5);
        }
        .container {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        canvas {
            background: radial-gradient(circle, #1a1a3e 0%, #0a0a1e 100%);
            border: 3px solid #00ffff;
            border-radius: 15px;
            box-shadow: 0 0 30px rgba(0,255,255,0.3);
        }
        .panel {
            background: rgba(0,0,0,0.5);
            border: 2px solid #444;
            border-radius: 15px;
            padding: 20px;
            min-width: 250px;
        }
        .stats { font-size: 1.2em; line-height: 2; }
        .red { color: #ff4444; }
        .blue { color: #4488ff; }
        .btn {
            background: linear-gradient(135deg, #00b4db, #0083b0);
            border: none;
            color: white;
            padding: 15px 40px;
            font-size: 1.2em;
            border-radius: 30px;
            cursor: pointer;
            margin: 20px 0;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(0,180,219,0.5);
        }
        .btn:disabled { background: #666; cursor: not-allowed; }
        .hp-bar {
            height: 20px;
            background: #333;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }
        .hp-fill {
            height: 100%;
            transition: width 0.3s;
        }
        .hp-fill.red { background: linear-gradient(90deg, #ff4444, #ff6666); }
        .hp-fill.blue { background: linear-gradient(90deg, #4488ff, #66aaff); }
        #winner {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
            display: none;
        }
    </style>
</head>
<body>
    <h1>✈️ SkyBattle Arena ✈️</h1>
    <button class="btn" onclick="startBattle()">🎮 开始新战斗</button>
    
    <div class="container">
        <canvas id="arena" width="600" height="500"></canvas>
        
        <div class="panel">
            <h2>📊 战斗状态</h2>
            <div class="stats">
                <div>⏱️ 步数: <span id="step">0</span></div>
                <br>
                <div class="red">🔴 红队</div>
                <div>存活: <span id="red-alive">3</span>/3</div>
                <div class="hp-bar"><div class="hp-fill red" id="red-hp" style="width:100%"></div></div>
                <br>
                <div class="blue">🔵 蓝队</div>
                <div>存活: <span id="blue-alive">3</span>/3</div>
                <div class="hp-bar"><div class="hp-fill blue" id="blue-hp" style="width:100%"></div></div>
            </div>
            <div id="winner"></div>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('arena');
        const ctx = canvas.getContext('2d');
        let frames = [];
        let currentFrame = 0;
        let playing = false;
        
        function startBattle() {
            document.querySelector('.btn').disabled = true;
            document.querySelector('.btn').textContent = '⏳ 战斗中...';
            document.getElementById('winner').style.display = 'none';
            
            fetch('/battle')
                .then(r => r.json())
                .then(data => {
                    frames = data.frames;
                    currentFrame = 0;
                    playing = true;
                    document.querySelector('.btn').disabled = false;
                    document.querySelector('.btn').textContent = '🎮 开始新战斗';
                    playAnimation();
                });
        }
        
        function playAnimation() {
            if (!playing || currentFrame >= frames.length) {
                playing = false;
                return;
            }
            
            const frame = frames[currentFrame];
            drawFrame(frame);
            updateStats(frame);
            
            currentFrame++;
            setTimeout(playAnimation, 50);
        }
        
        function drawFrame(frame) {
            // 清空画布
            ctx.fillStyle = 'rgba(10, 10, 30, 0.3)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // 绘制网格
            ctx.strokeStyle = 'rgba(100, 100, 150, 0.2)';
            ctx.lineWidth = 1;
            for (let i = 0; i < canvas.width; i += 50) {
                ctx.beginPath();
                ctx.moveTo(i, 0);
                ctx.lineTo(i, canvas.height);
                ctx.stroke();
            }
            for (let i = 0; i < canvas.height; i += 50) {
                ctx.beginPath();
                ctx.moveTo(0, i);
                ctx.lineTo(canvas.width, i);
                ctx.stroke();
            }
            
            // 绘制无人机
            frame.drones.forEach(drone => {
                const x = (drone.position[0] + 500) / 1000 * canvas.width;
                const y = (drone.position[1] + 500) / 1000 * canvas.height;
                const z = drone.position[2];
                
                if (drone.is_alive) {
                    // 阴影
                    ctx.beginPath();
                    ctx.ellipse(x, y + 10, 15, 5, 0, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(0,0,0,0.3)';
                    ctx.fill();
                    
                    // 无人机
                    const color = drone.team === 'red' ? '#ff4444' : '#4488ff';
                    const size = 12 + z / 30;
                    
                    ctx.beginPath();
                    ctx.arc(x, y, size, 0, Math.PI * 2);
                    ctx.fillStyle = color;
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    // HP条
                    const hpWidth = 30;
                    const hpHeight = 4;
                    ctx.fillStyle = '#333';
                    ctx.fillRect(x - hpWidth/2, y - size - 10, hpWidth, hpHeight);
                    ctx.fillStyle = color;
                    ctx.fillRect(x - hpWidth/2, y - size - 10, hpWidth * (drone.hp / 100), hpHeight);
                    
                    // 速度向量
                    const vx = drone.velocity[0] * 0.05;
                    const vy = drone.velocity[1] * 0.05;
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x + vx, y + vy);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                } else {
                    // 死亡标记
                    ctx.font = '20px Arial';
                    ctx.fillText('💀', x - 10, y + 7);
                }
            });
            
            // 绘制子弹
            frame.projectiles.forEach(proj => {
                const x = (proj.position[0] + 500) / 1000 * canvas.width;
                const y = (proj.position[1] + 500) / 1000 * canvas.height;
                
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fillStyle = '#ffff00';
                ctx.fill();
                
                // 光晕
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255, 255, 0, 0.3)';
                ctx.fill();
            });
            
            // 步数
            ctx.font = 'bold 16px Arial';
            ctx.fillStyle = 'white';
            ctx.fillText(`Step: ${frame.step}`, 10, 25);
        }
        
        function updateStats(frame) {
            document.getElementById('step').textContent = frame.step;
            document.getElementById('red-alive').textContent = frame.red_alive;
            document.getElementById('blue-alive').textContent = frame.blue_alive;
            
            // 计算总HP
            let redHP = 0, blueHP = 0;
            frame.drones.forEach(d => {
                if (d.team === 'red') redHP += d.hp;
                else blueHP += d.hp;
            });
            
            document.getElementById('red-hp').style.width = (redHP / 300 * 100) + '%';
            document.getElementById('blue-hp').style.width = (blueHP / 300 * 100) + '%';
            
            // 胜利提示
            if (frame.winner) {
                const winnerEl = document.getElementById('winner');
                winnerEl.style.display = 'block';
                winnerEl.textContent = frame.winner === 'red' ? 
                    '🏆 红队获胜!' : '🏆 蓝队获胜!';
                winnerEl.style.color = frame.winner === 'red' ? '#ff4444' : '#4488ff';
            }
        }
        
        // 初始绘制
        ctx.fillStyle = '#0a0a1e';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.font = '24px Arial';
        ctx.fillStyle = '#888';
        ctx.textAlign = 'center';
        ctx.fillText('点击上方按钮开始战斗', canvas.width/2, canvas.height/2);
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    PORT = 8080
    print()
    print('╔' + '═'*50 + '╗')
    print('║' + ' '*12 + '✈️  SkyBattle Viewer  ✈️' + ' '*12 + '║')
    print('╠' + '═'*50 + '╣')
    print(f'║  🌐 打开浏览器访问: http://localhost:{PORT}' + ' '*8 + '║')
    print('║  📌 按 Ctrl+C 停止服务器' + ' '*21 + '║')
    print('╚' + '═'*50 + '╝')
    print()
    
    server = HTTPServer(('0.0.0.0', PORT), BattleHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n👋 服务器已停止')
