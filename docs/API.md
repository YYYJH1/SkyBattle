# SkyBattle API Documentation

## REST API

### Games

#### Create Game
```http
POST /api/v1/games
Content-Type: application/json

{
  "mode": "ai_vs_ai",
  "team_size": 3,
  "time_limit": 300
}

Response: 201 Created
{
  "game_id": "game_abc123",
  "status": "created",
  "websocket_url": "/ws/game/game_abc123"
}
```

#### Get Game State
```http
GET /api/v1/games/{game_id}

Response: 200 OK
{
  "game_id": "game_abc123",
  "status": "running",
  "step": 150,
  "red_alive": 2,
  "blue_alive": 3,
  "winner": null
}
```

#### Control Game
```http
POST /api/v1/games/{game_id}/control
Content-Type: application/json

{
  "action": "start"  // start, pause, resume, stop
}
```

#### Delete Game
```http
DELETE /api/v1/games/{game_id}
```

### Training

#### Start Training
```http
POST /api/v1/training
Content-Type: application/json

{
  "algorithm": "mappo",
  "num_episodes": 1000,
  "team_size": 3,
  "lr_actor": 0.0003,
  "lr_critic": 0.0005
}
```

## WebSocket API

### Game WebSocket

Connect: `ws://localhost:8000/ws/game/{game_id}`

#### Receive: Game State
```json
{
  "type": "game_state",
  "data": {
    "step": 150,
    "drones": [
      {
        "id": "red_0",
        "team": "red",
        "position": [100.5, 50.2, 80.0],
        "velocity": [15.0, -5.0, 2.0],
        "hp": 85,
        "shield": 30,
        "is_alive": true
      }
    ],
    "projectiles": [...]
  }
}
```

#### Receive: Game End
```json
{
  "type": "game_end",
  "data": {
    "winner": "red",
    "red_alive": 2,
    "blue_alive": 0
  }
}
```

### Training WebSocket

Connect: `ws://localhost:8000/ws/training/{training_id}`

#### Receive: Training Update
```json
{
  "type": "training_update",
  "data": {
    "episode": 100,
    "step": 5000,
    "metrics": {
      "actor_loss": 0.023,
      "critic_loss": 0.156,
      "entropy": 1.45
    }
  }
}
```

#### Receive: Episode End
```json
{
  "type": "episode_end",
  "data": {
    "episode": 100,
    "reward": 85.3,
    "steps": 234,
    "winner": "red"
  }
}
```
