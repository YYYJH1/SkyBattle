<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const isLoading = ref(false)

async function startGame(mode: string) {
  isLoading.value = true
  
  try {
    const response = await fetch('/api/v1/games', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode, team_size: 3 }),
    })
    
    const data = await response.json()
    router.push(`/game/${data.game_id}`)
  } catch (error) {
    console.error('Failed to create game:', error)
  } finally {
    isLoading.value = false
  }
}
</script>

<template>
  <div class="home">
    <header class="hero">
      <h1 class="title">
        <span class="icon">✈️</span>
        SKYBATTLE
        <span class="icon">✈️</span>
      </h1>
      <p class="subtitle">Multi-Agent Drone Combat Simulator</p>
    </header>

    <main class="menu">
      <div class="menu-grid">
        <button class="menu-card" @click="startGame('ai_vs_ai')" :disabled="isLoading">
          <div class="card-icon">👁️</div>
          <h3>AI vs AI</h3>
          <p>Watch AI drones battle each other</p>
        </button>

        <button class="menu-card" @click="startGame('human_vs_ai')" :disabled="isLoading">
          <div class="card-icon">🎮</div>
          <h3>Human vs AI</h3>
          <p>Challenge the AI yourself</p>
        </button>

        <button class="menu-card" @click="router.push('/training')" :disabled="isLoading">
          <div class="card-icon">🔬</div>
          <h3>Training</h3>
          <p>Train your own AI agent</p>
        </button>

        <button class="menu-card" disabled>
          <div class="card-icon">🏆</div>
          <h3>Tournament</h3>
          <p>Coming soon...</p>
        </button>
      </div>
    </main>

    <footer class="footer">
      <p>Made with ❤️ by <a href="https://github.com/YYYJH1" target="_blank">YYYJH1</a></p>
    </footer>
  </div>
</template>

<style scoped>
.home {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: var(--spacing-xl);
}

.hero {
  text-align: center;
  margin-bottom: var(--spacing-xl);
  padding-top: 4rem;
}

.title {
  font-size: 4rem;
  font-weight: 900;
  background: linear-gradient(135deg, var(--color-accent), var(--color-blue));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: 0 0 60px var(--color-accent-glow);
  margin-bottom: var(--spacing-md);
}

.title .icon {
  -webkit-text-fill-color: initial;
  animation: float 3s ease-in-out infinite;
}

.subtitle {
  font-size: 1.25rem;
  color: var(--color-text-secondary);
  letter-spacing: 0.2em;
  text-transform: uppercase;
}

.menu {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  max-width: 900px;
}

.menu-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--spacing-lg);
  width: 100%;
}

.menu-card {
  background: var(--color-bg-secondary);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-lg);
  padding: var(--spacing-xl);
  text-align: center;
  transition: all 0.3s ease;
  cursor: pointer;
}

.menu-card:hover:not(:disabled) {
  transform: translateY(-5px);
  border-color: var(--color-accent);
  box-shadow: 0 10px 40px var(--color-accent-glow);
}

.menu-card:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.card-icon {
  font-size: 3rem;
  margin-bottom: var(--spacing-md);
}

.menu-card h3 {
  font-size: 1.5rem;
  color: var(--color-text-primary);
  margin-bottom: var(--spacing-sm);
}

.menu-card p {
  color: var(--color-text-secondary);
  font-size: 0.9rem;
}

.footer {
  padding: var(--spacing-lg);
  color: var(--color-text-muted);
}

@media (max-width: 768px) {
  .title {
    font-size: 2.5rem;
  }
  
  .menu-grid {
    grid-template-columns: 1fr;
  }
}
</style>
