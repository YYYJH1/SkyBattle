import { ref, onMounted, onUnmounted, Ref } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

interface Drone {
  id: string
  team: string
  position: number[]
  velocity: number[]
  orientation: number[]
  hp: number
  is_alive: boolean
}

interface Projectile {
  id: string
  position: number[]
}

interface GameState {
  step: number
  drones: Drone[]
  projectiles: Projectile[]
}

export function useThreeScene(containerRef: Ref<HTMLElement | null>) {
  const scene = ref<THREE.Scene | null>(null)
  const camera = ref<THREE.PerspectiveCamera | null>(null)
  const renderer = ref<THREE.WebGLRenderer | null>(null)
  const controls = ref<OrbitControls | null>(null)
  
  const droneObjects = new Map<string, THREE.Group>()
  const projectileObjects = new Map<string, THREE.Mesh>()
  
  let animationId: number
  
  function init() {
    if (!containerRef.value) return
    
    const container = containerRef.value
    const width = container.clientWidth
    const height = container.clientHeight
    
    // Scene
    scene.value = new THREE.Scene()
    scene.value.background = new THREE.Color(0x0a0e17)
    scene.value.fog = new THREE.Fog(0x0a0e17, 500, 1500)
    
    // Camera
    camera.value = new THREE.PerspectiveCamera(60, width / height, 0.1, 2000)
    camera.value.position.set(0, 300, 500)
    camera.value.lookAt(0, 0, 0)
    
    // Renderer
    renderer.value = new THREE.WebGLRenderer({ antialias: true })
    renderer.value.setSize(width, height)
    renderer.value.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.value.shadowMap.enabled = true
    container.appendChild(renderer.value.domElement)
    
    // Controls
    controls.value = new OrbitControls(camera.value, renderer.value.domElement)
    controls.value.enableDamping = true
    controls.value.dampingFactor = 0.05
    controls.value.maxPolarAngle = Math.PI / 2.1
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0x404060, 0.5)
    scene.value.add(ambientLight)
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
    directionalLight.position.set(100, 200, 100)
    directionalLight.castShadow = true
    scene.value.add(directionalLight)
    
    // Add atmosphere light
    const pointLight1 = new THREE.PointLight(0x00f0ff, 0.5, 1000)
    pointLight1.position.set(-300, 100, -300)
    scene.value.add(pointLight1)
    
    const pointLight2 = new THREE.PointLight(0xff4757, 0.5, 1000)
    pointLight2.position.set(300, 100, 300)
    scene.value.add(pointLight2)
    
    // Ground (battlefield)
    createBattlefield()
    
    // Grid helper
    const gridHelper = new THREE.GridHelper(1000, 50, 0x00f0ff, 0x1a1a2e)
    gridHelper.position.y = -0.1
    scene.value.add(gridHelper)
    
    // Start animation loop
    animate()
    
    // Handle resize
    window.addEventListener('resize', onResize)
  }
  
  function createBattlefield() {
    if (!scene.value) return
    
    // Ground plane
    const groundGeometry = new THREE.PlaneGeometry(1000, 1000)
    const groundMaterial = new THREE.MeshStandardMaterial({
      color: 0x111827,
      metalness: 0.8,
      roughness: 0.4,
    })
    const ground = new THREE.Mesh(groundGeometry, groundMaterial)
    ground.rotation.x = -Math.PI / 2
    ground.receiveShadow = true
    scene.value.add(ground)
    
    // Boundary walls (visual)
    const wallMaterial = new THREE.MeshStandardMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.1,
      side: THREE.DoubleSide,
    })
    
    const wallGeometry = new THREE.PlaneGeometry(1000, 300)
    
    // Create 4 walls
    const positions = [
      { pos: [0, 150, -500], rot: [0, 0, 0] },
      { pos: [0, 150, 500], rot: [0, Math.PI, 0] },
      { pos: [-500, 150, 0], rot: [0, Math.PI / 2, 0] },
      { pos: [500, 150, 0], rot: [0, -Math.PI / 2, 0] },
    ]
    
    positions.forEach(({ pos, rot }) => {
      const wall = new THREE.Mesh(wallGeometry, wallMaterial)
      wall.position.set(pos[0], pos[1], pos[2])
      wall.rotation.set(rot[0], rot[1], rot[2])
      scene.value!.add(wall)
    })
  }
  
  function createDrone(drone: Drone): THREE.Group {
    const group = new THREE.Group()
    
    // Main body
    const bodyGeometry = new THREE.ConeGeometry(3, 12, 8)
    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: drone.team === 'red' ? 0xff4757 : 0x3b82f6,
      metalness: 0.7,
      roughness: 0.3,
    })
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial)
    body.rotation.x = Math.PI / 2
    group.add(body)
    
    // Wings
    const wingGeometry = new THREE.BoxGeometry(20, 0.5, 4)
    const wingMaterial = new THREE.MeshStandardMaterial({
      color: drone.team === 'red' ? 0xcc3344 : 0x2563eb,
      metalness: 0.6,
      roughness: 0.4,
    })
    const wings = new THREE.Mesh(wingGeometry, wingMaterial)
    wings.position.z = 2
    group.add(wings)
    
    // Engine glow
    const glowGeometry = new THREE.SphereGeometry(2, 16, 16)
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: drone.team === 'red' ? 0xff6b6b : 0x60a5fa,
      transparent: true,
      opacity: 0.8,
    })
    const glow = new THREE.Mesh(glowGeometry, glowMaterial)
    glow.position.z = 6
    group.add(glow)
    
    // Trail (particle system simulation with line)
    const trailGeometry = new THREE.BufferGeometry()
    const trailPositions = new Float32Array(30 * 3)
    trailGeometry.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3))
    
    const trailMaterial = new THREE.LineBasicMaterial({
      color: drone.team === 'red' ? 0xff4757 : 0x3b82f6,
      transparent: true,
      opacity: 0.5,
    })
    const trail = new THREE.Line(trailGeometry, trailMaterial)
    trail.name = 'trail'
    group.add(trail)
    
    // HP bar
    const hpBarGroup = new THREE.Group()
    hpBarGroup.name = 'hpBar'
    
    const hpBgGeometry = new THREE.PlaneGeometry(10, 1)
    const hpBgMaterial = new THREE.MeshBasicMaterial({ color: 0x333333 })
    const hpBg = new THREE.Mesh(hpBgGeometry, hpBgMaterial)
    hpBarGroup.add(hpBg)
    
    const hpFillGeometry = new THREE.PlaneGeometry(10, 1)
    const hpFillMaterial = new THREE.MeshBasicMaterial({ color: 0x10b981 })
    const hpFill = new THREE.Mesh(hpFillGeometry, hpFillMaterial)
    hpFill.name = 'hpFill'
    hpBarGroup.add(hpFill)
    
    hpBarGroup.position.y = 10
    group.add(hpBarGroup)
    
    group.castShadow = true
    
    return group
  }
  
  function createProjectile(): THREE.Mesh {
    const geometry = new THREE.SphereGeometry(1, 8, 8)
    const material = new THREE.MeshBasicMaterial({
      color: 0xffff00,
      transparent: true,
      opacity: 0.8,
    })
    return new THREE.Mesh(geometry, material)
  }
  
  function updateGameState(state: GameState) {
    if (!scene.value) return
    
    // Update drones
    state.drones.forEach(drone => {
      let droneObj = droneObjects.get(drone.id)
      
      if (!droneObj) {
        droneObj = createDrone(drone)
        scene.value!.add(droneObj)
        droneObjects.set(drone.id, droneObj)
      }
      
      // Update position
      droneObj.position.set(
        drone.position[0],
        drone.position[2] + 10, // Y is up in Three.js
        drone.position[1]
      )
      
      // Update rotation
      droneObj.rotation.set(
        drone.orientation[1], // pitch
        -drone.orientation[2], // yaw
        drone.orientation[0] // roll
      )
      
      // Update visibility
      droneObj.visible = drone.is_alive
      
      // Update HP bar
      const hpBar = droneObj.getObjectByName('hpBar') as THREE.Group
      if (hpBar) {
        const hpFill = hpBar.getObjectByName('hpFill') as THREE.Mesh
        if (hpFill) {
          hpFill.scale.x = drone.hp / 100
          hpFill.position.x = -(1 - drone.hp / 100) * 5
        }
        // Billboard - face camera
        if (camera.value) {
          hpBar.lookAt(camera.value.position)
        }
      }
    })
    
    // Update projectiles
    const activeProjectileIds = new Set(state.projectiles.map(p => p.id))
    
    // Remove old projectiles
    projectileObjects.forEach((obj, id) => {
      if (!activeProjectileIds.has(id)) {
        scene.value!.remove(obj)
        projectileObjects.delete(id)
      }
    })
    
    // Add/update projectiles
    state.projectiles.forEach(proj => {
      let projObj = projectileObjects.get(proj.id)
      
      if (!projObj) {
        projObj = createProjectile()
        scene.value!.add(projObj)
        projectileObjects.set(proj.id, projObj)
      }
      
      projObj.position.set(
        proj.position[0],
        proj.position[2] + 10,
        proj.position[1]
      )
    })
  }
  
  function animate() {
    animationId = requestAnimationFrame(animate)
    
    if (controls.value) {
      controls.value.update()
    }
    
    if (renderer.value && scene.value && camera.value) {
      renderer.value.render(scene.value, camera.value)
    }
  }
  
  function onResize() {
    if (!containerRef.value || !camera.value || !renderer.value) return
    
    const width = containerRef.value.clientWidth
    const height = containerRef.value.clientHeight
    
    camera.value.aspect = width / height
    camera.value.updateProjectionMatrix()
    renderer.value.setSize(width, height)
  }
  
  function cleanup() {
    window.removeEventListener('resize', onResize)
    cancelAnimationFrame(animationId)
    
    if (renderer.value && containerRef.value) {
      containerRef.value.removeChild(renderer.value.domElement)
      renderer.value.dispose()
    }
    
    droneObjects.clear()
    projectileObjects.clear()
  }
  
  onMounted(() => {
    init()
  })
  
  onUnmounted(() => {
    cleanup()
  })
  
  return {
    scene,
    camera,
    updateGameState,
  }
}
