<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()

interface Drone {
  id: string
  team: string
  position: number[]
  velocity: number[]
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

const redTeam = computed(() => gameState.value?.drones.filter(d => d.team === 'red') || [])
const blueTeam = computed(() => gameState.value?.drones.filter(d => d.team === 'blue') || [])

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
}

async function pauseGame() {
  await fetch(`/api/v1/games/${gameId.value}/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: status.value === 'paused' ? 'resume' : 'pause' }),
  })
  status.value = status.value === 'paused' ? 'running' : 'paused'
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
        <span class="team-label">🔴 RED</span>
        <span class="team-count">{{ redTeam.filter(d => d.is_alive).length }}/{{ redTeam.length }}</span>
      </div>
      
      <div class="game-info">
        <span class="step">Step: {{ gameState?.step || 0 }}</span>
        <span class="status" :class="status">{{ status }}</span>
      </div>
      
      <div class="team blue-team">
        <span class="team-count">{{ blueTeam.filter(d => d.is_alive).length }}/{{ blueTeam.length }}</span>
        <span class="team-label">BLUE 🔵</span>
      </div>
    </header>

    <!-- Game Canvas -->
    <main class="game-area">
      <canvas ref="canvas" class="game-canvas"></canvas>
      
      <!-- Overlay for drone info -->
      <div class="drone-overlay" v-if="gameState">
        <div v-for="drone in gameState.drones" :key="drone.id"
             class="drone-marker" :class="[drone.team, { dead: !drone.is_alive }]"
             :style="{
               left: `${50 + drone.position[0] / 10}%`,
               top: `${50 - drone.position[1] / 10}%`
             }">
          <div class="drone-icon">✈️</div>
          <div class="drone-hp-bar">
            <div class="hp-fill" :style="{ width: `${drone.hp}%` }"></div>
          </div>
        </div>
      </div>
    </main>

    <!-- Controls -->
    <footer class="game-controls">
      <button class="btn-secondary" @click="goHome">🏠 Home</button>
      <button class="btn-primary" @click="pauseGame">
        {{ status === 'paused' ? '▶️ Resume' : '⏸️ Pause' }}
      </button>
      <button class="btn-secondary" @click="restartGame">🔄 Restart</button>
    </footer>
  </div>
</template>

<style scoped>
.game-view {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--color-bg-primary);
}

.game-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-md) var(--spacing-xl);
  background: var(--color-bg-secondary);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.team {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  font-family: var(--font-display);
  font-size: 1.2rem;
}

.red-team {
  color: var(--color-red);
}

.blue-team {
  color: var(--color-blue);
}

.team-count {
  font-weight: 700;
  font-size: 1.5rem;
}

.game-info {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-xs);
}

.step {
  font-family: var(--font-display);
  color: var(--color-text-secondary);
}

.status {
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.8rem;
  text-transform: uppercase;
}

.status.connected, .status.running {
  background: var(--color-green);
  color: white;
}

.status.paused {
  background: var(--color-yellow);
  color: black;
}

.status.finished {
  background: var(--color-accent);
  color: black;
}

.status.disconnected {
  background: var(--color-red);
  color: white;
}

.game-area {
  flex: 1;
  position: relative;
  overflow: hidden;
  background: 
    radial-gradient(circle at center, var(--color-bg-secondary) 0%, var(--color-bg-primary) 100%);
}

.game-canvas {
  width: 100%;
  height: 100%;
}

.drone-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.drone-marker {
  position: absolute;
  transform: translate(-50%, -50%);
  transition: all 0.1s linear;
}

.drone-marker.dead {
  opacity: 0.3;
}

.drone-icon {
  font-size: 1.5rem;
  text-shadow: 0 0 10px currentColor;
}

.drone-marker.red .drone-icon {
  filter: hue-rotate(-60deg);
}

.drone-hp-bar {
  width: 40px;
  height: 4px;
  background: rgba(0, 0, 0, 0.5);
  border-radius: 2px;
  margin-top: 4px;
}

.hp-fill {
  height: 100%;
  background: var(--color-green);
  border-radius: 2px;
  transition: width 0.2s ease;
}

.game-controls {
  display: flex;
  justify-content: center;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  background: var(--color-bg-secondary);
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}
</style>
