<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useThreeScene } from '../composables/useThreeScene'

const route = useRoute()
const router = useRouter()

interface Drone {
  id: string
  team: string
  position: number[]
  velocity: number[]
  orientation: number[]
  hp: number
  shield: number
  is_alive: boolean
}

interface GameState {
  step: number
  drones: Drone[]
  projectiles: { id: string; position: number[] }[]
}

const gameId = ref(route.params.id as string)
const gameState = ref<GameState | null>(null)
const status = ref('connecting')
const ws = ref<WebSocket | null>(null)
const canvasContainer = ref<HTMLElement | null>(null)

const { updateGameState } = useThreeScene(canvasContainer)

const redTeam = ref<Drone[]>([])
const blueTeam = ref<Drone[]>([])

function connectWebSocket() {
  if (!gameId.value) return
  
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  ws.value = new WebSocket(`${protocol}//${window.location.host}/ws/game/${gameId.value}`)
  
  ws.value.onopen = () => {
    status.value = 'connected'
    startGame()
  }
  
  ws.value.onmessage = (event) => {
    const message = JSON.parse(event.data)
    if (message.type === 'game_state') {
      gameState.value = message.data
      redTeam.value = message.data.drones.filter((d: Drone) => d.team === 'red')
      blueTeam.value = message.data.drones.filter((d: Drone) => d.team === 'blue')
      updateGameState(message.data)
    } else if (message.type === 'game_end') {
      status.value = 'finished'
    }
  }
  
  ws.value.onclose = () => {
    status.value = 'disconnected'
  }
}

async function startGame() {
  await fetch(`/api/v1/games/${gameId.value}/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'start' }),
  })
  status.value = 'running'
}

async function pauseGame() {
  const action = status.value === 'paused' ? 'resume' : 'pause'
  await fetch(`/api/v1/games/${gameId.value}/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  })
  status.value = action === 'pause' ? 'paused' : 'running'
}

async function restartGame() {
  await fetch(`/api/v1/games/${gameId.value}/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'restart' }),
  })
  status.value = 'running'
}

function goHome() {
  router.push('/')
}

onMounted(() => {
  connectWebSocket()
})

onUnmounted(() => {
  ws.value?.close()
})
</script>

<template>
  <div class="game-view">
    <!-- Header -->
    <header class="game-header">
      <div class="team red-team">
        <span class="team-icon">🔴</span>
        <span class="team-label">RED</span>
        <div class="team-drones">
          <div v-for="drone in redTeam" :key="drone.id" 
               class="drone-indicator" :class="{ dead: !drone.is_alive }">
            <div class="hp-bar">
              <div class="hp-fill" :style="{ width: `${drone.hp}%` }"></div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="game-info">
        <div class="step-counter">STEP {{ gameState?.step || 0 }}</div>
        <div class="status-badge" :class="status">{{ status.toUpperCase() }}</div>
      </div>
      
      <div class="team blue-team">
        <div class="team-drones">
          <div v-for="drone in blueTeam" :key="drone.id" 
               class="drone-indicator" :class="{ dead: !drone.is_alive }">
            <div class="hp-bar">
              <div class="hp-fill" :style="{ width: `${drone.hp}%` }"></div>
            </div>
          </div>
        </div>
        <span class="team-label">BLUE</span>
        <span class="team-icon">🔵</span>
      </div>
    </header>

    <!-- 3D Canvas -->
    <main class="game-canvas-container" ref="canvasContainer">
      <!-- Three.js canvas will be inserted here -->
    </main>

    <!-- Controls -->
    <footer class="game-controls">
      <button class="control-btn home" @click="goHome">
        <span class="icon">🏠</span>
        <span class="label">Home</span>
      </button>
      
      <div class="main-controls">
        <button class="control-btn" @click="restartGame">
          <span class="icon">🔄</span>
        </button>
        <button class="control-btn primary" @click="pauseGame">
          <span class="icon">{{ status === 'paused' ? '▶️' : '⏸️' }}</span>
        </button>
      </div>
      
      <div class="camera-controls">
        <span class="label">Camera: Free</span>
      </div>
    </footer>
  </div>
</template>

<style scoped>
.game-view {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--color-bg-primary);
  overflow: hidden;
}

.game-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-sm) var(--spacing-xl);
  background: linear-gradient(180deg, rgba(10, 14, 23, 0.95) 0%, rgba(10, 14, 23, 0.8) 100%);
  border-bottom: 1px solid rgba(0, 240, 255, 0.2);
  z-index: 10;
}

.team {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.team-icon {
  font-size: 1.5rem;
}

.team-label {
  font-family: var(--font-display);
  font-size: 1.2rem;
  font-weight: 700;
  letter-spacing: 0.1em;
}

.red-team .team-label { color: var(--color-red); }
.blue-team .team-label { color: var(--color-blue); }

.team-drones {
  display: flex;
  gap: var(--spacing-xs);
}

.drone-indicator {
  width: 40px;
  height: 24px;
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-sm);
  padding: 2px;
  transition: opacity 0.3s;
}

.drone-indicator.dead {
  opacity: 0.3;
}

.hp-bar {
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  border-radius: 2px;
  overflow: hidden;
}

.hp-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--color-green), #22c55e);
  transition: width 0.2s ease;
}

.game-info {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-xs);
}

.step-counter {
  font-family: var(--font-display);
  font-size: 1rem;
  color: var(--color-accent);
  letter-spacing: 0.2em;
}

.status-badge {
  font-family: var(--font-display);
  font-size: 0.7rem;
  padding: 2px 12px;
  border-radius: 20px;
  letter-spacing: 0.1em;
}

.status-badge.running, .status-badge.connected {
  background: var(--color-green);
  color: white;
}

.status-badge.paused {
  background: var(--color-yellow);
  color: black;
}

.status-badge.finished {
  background: var(--color-accent);
  color: black;
}

.status-badge.disconnected {
  background: var(--color-red);
  color: white;
}

.game-canvas-container {
  flex: 1;
  position: relative;
}

.game-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-sm) var(--spacing-xl);
  background: linear-gradient(0deg, rgba(10, 14, 23, 0.95) 0%, rgba(10, 14, 23, 0.8) 100%);
  border-top: 1px solid rgba(0, 240, 255, 0.2);
  z-index: 10;
}

.control-btn {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-bg-tertiary);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-md);
  color: var(--color-text-primary);
  font-family: var(--font-display);
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s;
}

.control-btn:hover {
  border-color: var(--color-accent);
  box-shadow: 0 0 15px var(--color-accent-glow);
}

.control-btn.primary {
  background: var(--color-accent);
  color: var(--color-bg-primary);
  border-color: var(--color-accent);
}

.control-btn.primary:hover {
  background: #00d4e6;
}

.control-btn .icon {
  font-size: 1.2rem;
}

.main-controls {
  display: flex;
  gap: var(--spacing-sm);
}

.camera-controls {
  color: var(--color-text-muted);
  font-size: 0.85rem;
}
</style>
