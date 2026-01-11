<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const config = ref({
  algorithm: 'mappo',
  episodes: 1000,
  teamSize: 3,
  lrActor: 0.0003,
  lrCritic: 0.0005,
})

const isTraining = ref(false)
const trainingId = ref<string | null>(null)
const logs = ref<string[]>([])

async function startTraining() {
  isTraining.value = true
  logs.value = ['Starting training...']
  
  try {
    const response = await fetch('/api/v1/training', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        algorithm: config.value.algorithm,
        num_episodes: config.value.episodes,
        team_size: config.value.teamSize,
        lr_actor: config.value.lrActor,
        lr_critic: config.value.lrCritic,
      }),
    })
    
    const data = await response.json()
    trainingId.value = data.training_id
    logs.value.push(`Training ID: ${data.training_id}`)
    
    // Connect to WebSocket for updates
    connectTrainingWebSocket(data.training_id)
    
  } catch (error) {
    logs.value.push(`Error: ${error}`)
    isTraining.value = false
  }
}

function connectTrainingWebSocket(id: string) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/training/${id}`)
  
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data)
    if (message.type === 'training_update') {
      const { episode, metrics } = message.data
      logs.value.push(`[Episode ${episode}] Loss: ${metrics.actor_loss.toFixed(4)}`)
    } else if (message.type === 'episode_end') {
      const { episode, reward, winner } = message.data
      logs.value.push(`[Episode ${episode}] Reward: ${reward.toFixed(1)} Winner: ${winner || 'draw'}`)
    } else if (message.type === 'training_complete') {
      logs.value.push('Training complete!')
      isTraining.value = false
    }
    
    // Keep only last 50 logs
    if (logs.value.length > 50) {
      logs.value = logs.value.slice(-50)
    }
  }
  
  ws.onclose = () => {
    if (isTraining.value) {
      logs.value.push('Connection lost')
    }
  }
}

function goHome() {
  router.push('/')
}
</script>

<template>
  <div class="training-view">
    <header class="header">
      <button class="btn-secondary" @click="goHome">← Back</button>
      <h1>🔬 Training Mode</h1>
      <div></div>
    </header>

    <main class="content">
      <!-- Config Panel -->
      <section class="config-panel card">
        <h2>Configuration</h2>
        
        <div class="form-group">
          <label>Algorithm</label>
          <select v-model="config.algorithm" :disabled="isTraining">
            <option value="mappo">MAPPO</option>
          </select>
        </div>
        
        <div class="form-group">
          <label>Episodes</label>
          <input type="number" v-model="config.episodes" :disabled="isTraining" />
        </div>
        
        <div class="form-group">
          <label>Team Size</label>
          <input type="number" v-model="config.teamSize" min="1" max="5" :disabled="isTraining" />
        </div>
        
        <div class="form-group">
          <label>Actor Learning Rate</label>
          <input type="number" v-model="config.lrActor" step="0.0001" :disabled="isTraining" />
        </div>
        
        <div class="form-group">
          <label>Critic Learning Rate</label>
          <input type="number" v-model="config.lrCritic" step="0.0001" :disabled="isTraining" />
        </div>
        
        <button 
          class="btn-primary start-btn" 
          @click="startTraining" 
          :disabled="isTraining"
        >
          {{ isTraining ? '🔄 Training...' : '▶️ Start Training' }}
        </button>
      </section>

      <!-- Logs Panel -->
      <section class="logs-panel card">
        <h2>Training Logs</h2>
        <div class="logs-container">
          <div v-for="(log, index) in logs" :key="index" class="log-entry">
            {{ log }}
          </div>
          <div v-if="logs.length === 0" class="log-empty">
            No logs yet. Start training to see output.
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<style scoped>
.training-view {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-md) var(--spacing-xl);
  background: var(--color-bg-secondary);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.header h1 {
  font-size: 1.5rem;
  color: var(--color-accent);
}

.content {
  flex: 1;
  display: grid;
  grid-template-columns: 350px 1fr;
  gap: var(--spacing-lg);
  padding: var(--spacing-lg);
}

.config-panel {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.config-panel h2, .logs-panel h2 {
  font-size: 1.2rem;
  color: var(--color-accent);
  margin-bottom: var(--spacing-sm);
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.form-group label {
  font-size: 0.9rem;
  color: var(--color-text-secondary);
}

.form-group input, .form-group select {
  padding: var(--spacing-sm);
  background: var(--color-bg-tertiary);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-sm);
  color: var(--color-text-primary);
  font-family: var(--font-body);
}

.form-group input:focus, .form-group select:focus {
  outline: none;
  border-color: var(--color-accent);
}

.start-btn {
  margin-top: var(--spacing-md);
  padding: var(--spacing-md);
  font-size: 1rem;
}

.logs-panel {
  display: flex;
  flex-direction: column;
}

.logs-container {
  flex: 1;
  background: var(--color-bg-tertiary);
  border-radius: var(--radius-sm);
  padding: var(--spacing-md);
  font-family: monospace;
  font-size: 0.85rem;
  overflow-y: auto;
  max-height: 500px;
}

.log-entry {
  padding: var(--spacing-xs) 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  color: var(--color-text-secondary);
}

.log-empty {
  color: var(--color-text-muted);
  font-style: italic;
}

@media (max-width: 900px) {
  .content {
    grid-template-columns: 1fr;
  }
}
</style>
