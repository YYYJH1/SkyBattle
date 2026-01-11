import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('./views/HomeView.vue'),
  },
  {
    path: '/game/:id?',
    name: 'Game',
    component: () => import('./views/GameView.vue'),
  },
  {
    path: '/training',
    name: 'Training',
    component: () => import('./views/TrainingView.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
