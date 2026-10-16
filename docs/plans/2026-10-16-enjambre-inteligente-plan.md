# El Enjambre Inteligente - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir e implementar el juego interactivo "El Enjambre Inteligente" para tótems de exhibición en AudacIA, permitiendo al público controlar un enjambre de cientos de microrobots mediante gestos de visión artificial (palma, puño, bimanual) a 30-60 FPS sobre Raspberry Pi 4/5.

**Architecture:** Arquitectura desacoplada en 4 capas: 
1) Módulo de percepción de visión (`vision/`) que procesa la cámara web y clasifica gestos (atracción, dispersión, condensación) con soporte de entrada simulada; 
2) Motor de física de enjambre (`swarm/`) que ejecuta simulación vectorial de Boids y campos de potencial artificial en NumPy; 
3) Director de juego (`game/`) que administra la máquina de estados finitos de las 3 fases de la misión ("Reactivación del Núcleo"); 
4) Renderizador acelerado en 2.5D (`render/`) con Pygame-CE/SDL2 para dibujar partículas con estela bioluminiscente, HUD interactivo y modo reposo (*Attract Mode*).

**Tech Stack:** Python 3.10+, NumPy (cálculo vectorial optimizado), OpenCV-Python (`cv2`), MediaPipe Hands Lite (`mediapipe`), Pygame-CE (`pygame`), Pytest.

**Spec:** [docs/2026-10-15-juegos-interactivos-audacia-design.md](file:///C:/Users/jchrn/Documents/Trabajos/U/juegos-audacia/docs/2026-10-15-juegos-interactivos-audacia-design.md)

## Global Constraints

- **Platform:** Raspberry Pi 4 Model B (4GB) baseline (64-bit OS), desarrollada y probada primero en laptop.
- **Display Target:** 1920x1080 (16:9) horizontal fullscreen.
- **Camera Resolution:** Captura a 640x480, preprocesamiento de inferencia a 320x240 para mantener tiempo de detección por cuadro < 20 ms.
- **FPS Target:** Mínimo sostenido de 30 FPS continuos en Raspberry Pi 4; 60 FPS en laptop / Raspberry Pi 5.
- **Latency Target:** Latencia entre movimiento de mano y respuesta visual menor a 80 ms.
- **Camera Invisibility:** El sistema debe operar con fallo elegante si no hay cámara disponible (conmutando a modo simulación/mouse para desarrollo y pruebas automatizadas).

## Review Focus

1. **Pérdida súbita de seguimiento de manos:** Cuando la mano sale del campo de visión de la cámara o sufre oclusión temporal, el enjambre no debe congelarse ni teletransportarse bruscamente, sino desacelerar suavemente a velocidad de crucero autónoma (inercia).
2. **Ruido de detección y parpadeo de gestos (Jitter):** Pequeñas fluctuaciones entre fotogramas no deben causar alternancia descontrolada entre dispersión y condensación; se requiere un filtro de histeresis temporal de 3 fotogramas.
3. **Escalamiento de límites de pantalla (Resolution Boundary Clamping):** Ninguna partícula del enjambre debe salir o quedar atrapada fuera del viewport 1920x1080 independientemente de la fuerza del atractor humano.
4. **Desconexión física de la cámara USB durante ejecución:** El hilo de captura debe reconectar automáticamente o pasar a estado de espera sin bloquear el bucle de renderizado ni causar crash por excepción no controlada.
5. **Transición a modo reposo (Attract Mode):** Si transcurren más de 15 segundos sin presencia de manos, el juego debe reiniciar su estado, mostrar un baile de partículas autónomo hipnótico y reactivarse instantáneamente en cuanto una mano entre en cuadro.

---

### Task 1: Entorno, Dependencias y Módulo de Gestos de Visión Artificial

**Files:**
- Create: `src/vision/gesture_tracker.py`
- Create: `requirements.txt`
- Test: `tests/test_gesture_tracker.py`

**Interfaces:**
- Consumes: `cv2.VideoCapture` frames or synthetic RGB numpy arrays `(H, W, 3)`.
- Produces: `HandGestureData(is_detected: bool, x: float, y: float, gesture_type: str, hand_count: int, hands: list[dict])`.
  - `gesture_type`: `'ATTRACT'` (neutral palm), `'DISPERSE'` (open fingers wide), `'CONDENSE'` (closed fist), o `'NONE'`.
  - Coordenadas normalizadas $(x, y) \in [0.0, 1.0]$.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gesture_tracker.py
import numpy as np
from src.vision.gesture_tracker import GestureTracker, HandGestureData

def test_gesture_tracker_initialization():
    tracker = GestureTracker(use_mock=True)
    state = tracker.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert isinstance(state, HandGestureData)
    assert state.is_detected is False
    assert state.gesture_type == 'NONE'

def test_gesture_classification_logic():
    tracker = GestureTracker(use_mock=True)
    # Simulate open palm landmarks
    tracker.set_mock_landmarks(gesture='DISPERSE', x=0.5, y=0.5)
    state = tracker.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert state.is_detected is True
    assert state.gesture_type == 'DISPERSE'
    assert state.x == 0.5 and state.y == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gesture_tracker.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.vision.gesture_tracker'`

- [ ] **Step 3: Implement `GestureTracker` and data classes in `src/vision/gesture_tracker.py`**

Crear `requirements.txt` con `numpy>=1.23.0`, `opencv-python>=4.8.0`, `mediapipe>=0.10.0`, `pygame-ce>=2.3.0`, `pytest>=7.0.0`.  
Implementar `GestureTracker`:
- Carga de MediaPipe Hands con `model_complexity=0` (Lite), `min_detection_confidence=0.6`, `min_tracking_confidence=0.5`.
- Cálculo de distancias euclidianas entre punta de dedos y articulaciones MCP para discriminar puño cerrado (`CONDENSE`), palma abierta (`DISPERSE`) y posición neutra (`ATTRACT`).
- Histeresis temporal de 3 fotogramas para estabilidad de clasificación.
- Mapeo horizontal espejo (flip horizontal) para interacción tipo espejo natural.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gesture_tracker.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/vision/gesture_tracker.py tests/test_gesture_tracker.py
git commit -m "feat: implement vision gesture tracker with MediaPipe and mock mode"
```

---

### Task 2: Motor Vectorial de Enjambre (Boids + Campos de Potencial)

**Files:**
- Create: `src/swarm/boids_engine.py`
- Test: `tests/test_boids_engine.py`

**Interfaces:**
- Consumes: Parámetros de simulación (`num_boids`, `bounds=(width, height)`), comandos de control de manos (`targets: list[tuple[float, float, str]]`).
- Produces: `BoidsEngine.step(dt: float, hand_targets: list) -> np.ndarray` (posiciones shape `(N, 2)` y velocidades shape `(N, 2)`).
- Métodos auxiliares: `set_obstacles(obstacles: list[tuple[x, y, radius]])`, `get_state() -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_boids_engine.py
import numpy as np
from src.swarm/boids_engine import BoidsEngine

def test_boids_engine_initialization():
    engine = BoidsEngine(num_boids=300, bounds=(1920, 1080))
    positions, velocities = engine.get_positions(), engine.get_velocities()
    assert positions.shape == (300, 2)
    assert velocities.shape == (300, 2)
    assert np.all(positions >= 0) and np.all(positions[:, 0] <= 1920)

def test_boids_attractor_force():
    engine = BoidsEngine(num_boids=100, bounds=(1000, 1000))
    # Place all boids at origin and attractor at (500, 500)
    engine.positions[:, :] = 100.0
    engine.velocities[:, :] = 0.0
    engine.step(dt=0.1, hand_targets=[(500.0, 500.0, 'ATTRACT')])
    # Boids must have moved towards (500, 500)
    assert np.mean(engine.velocities[:, 0]) > 0
    assert np.mean(engine.velocities[:, 1]) > 0

def test_boids_boundary_clamping():
    engine = BoidsEngine(num_boids=50, bounds=(1920, 1080))
    # Place boid beyond bounds
    engine.positions[0] = [2000.0, 1100.0]
    engine.step(dt=0.016, hand_targets=[])
    assert 0 <= engine.positions[0, 0] <= 1920
    assert 0 <= engine.positions[0, 1] <= 1080
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_boids_engine.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.swarm.boids_engine'`

- [ ] **Step 3: Implement `BoidsEngine` in `src/swarm/boids_engine.py`**

- Vectorización pura en NumPy (sin bucles anidados en Python) para cálculo de distancias por pares con sub-muestreo de vecindario espacial (grid hashing o subgrupos de 40 vecinos para mantener la complejidad en $\mathcal{O}(N)$).
- Reglas de Reynolds:
  - Cohesión ($w_{\text{coh}} = 1.0$), Alineación ($w_{\text{ali}} = 0.8$), Separación ($w_{\text{sep}} = 1.5$).
- Reglas de interacción gestual:
  - `'ATTRACT'`: fuerza proporcional a distancia hacia la palma con amortiguamiento a radio de proximidad.
  - `'DISPERSE'`: fuerza repulsiva inversa al cuadrado de la distancia, abriendo espacio radial.
  - `'CONDENSE'`: atracción ultradensa con radio de frenado estrecho para simular masa compacta.
- Evasión de obstáculos circulares mediante campos de potencial artificial repulsivos.
- Limitador estricto de velocidad máxima (`max_speed = 400.0 px/s`) y envoltura/rebote suave en los bordes de la pantalla.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_boids_engine.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/swarm/boids_engine.py tests/test_boids_engine.py
git commit -m "feat: implement high-performance vectorized boids simulation engine"
```

---

### Task 3: Director de Misiones y Progresión de Juego ("Reactivación del Núcleo")

**Files:**
- Create: `src/game/mission_director.py`
- Test: `tests/test_mission_director.py`

**Interfaces:**
- Consumes: `BoidsEngine` positions/velocities, `HandGestureData`.
- Produces: `MissionState(phase: int, time_remaining: float, score: int, is_completed: bool, is_game_over: bool, active_obstacles: list, target_nodes: list, core_status: float)`.
- Métodos: `update(dt: float, boid_positions: np.ndarray, gesture: HandGestureData)`, `reset()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mission_director.py
import numpy as np
from src.game.mission_director import MissionDirector

def test_mission_progression_phase1_to_phase2():
    director = MissionDirector(total_time=180.0)
    assert director.phase == 1
    # Simulate swarm reaching Phase 1 checkpoint
    checkpoint = director.get_phase1_checkpoint()
    fake_positions = np.tile(checkpoint, (300, 1))
    director.update(dt=0.1, boid_positions=fake_positions)
    assert director.phase == 2

def test_mission_timeout_game_over():
    director = MissionDirector(total_time=2.0)
    director.update(dt=2.5, boid_positions=np.zeros((10, 2)))
    assert director.is_game_over is True
    assert director.is_completed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mission_director.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.game.mission_director'`

- [ ] **Step 3: Implement `MissionDirector` in `src/game/mission_director.py`**

- Máquina de estados finitos (FSM):
  - `PHASE_1_NAVIGATION`: Obstáculos de calor electromagnético móviles; objetivo: llevar al menos el 70% del enjambre a la compuerta del reactor.
  - `PHASE_2_DISTRIBUTED`: Dos nodos de carga en extremos opuestos de la pantalla; requiere dividir el enjambre usando ambas manos (`hand_count == 2`) para energizar ambos nodos simultáneamente durante 3 segundos.
  - `PHASE_3_CORE_INSERTION`: Núcleo central con un fusible de alta inercia; requiere usar puño cerrado (`CONDENSE`) para aplicar suficiente densidad de fuerza y empujar el fusible a su zócalo.
  - `MISSION_SUCCESS` & `MISSION_FAILURE`.
- Sistema de puntuación basado en tiempo restante y porcentaje de microrobots supervivientes.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mission_director.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/game/mission_director.py tests/test_mission_director.py
git commit -m "feat: implement mission director with 3 game phases and state machine"
```

---

### Task 4: Renderizador 2.5D, HUD y Partículas Bioluminiscentes

**Files:**
- Create: `src/render/viewport_renderer.py`
- Test: `tests/test_viewport_renderer.py`

**Interfaces:**
- Consumes: Dimensiones de pantalla `(1920, 1080)`, posiciones de boids `np.ndarray`, estado de misión `MissionState`, estado de gestos `HandGestureData`.
- Produces: `render_frame(screen: pygame.Surface) -> None`.
- Modos: `RENDER_GAME`, `RENDER_ATTRACT_MODE`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_viewport_renderer.py
import pygame
import numpy as np
from src.render.viewport_renderer import ViewportRenderer

def test_renderer_initialization_headless():
    pygame.init()
    surface = pygame.Surface((1920, 1080))
    renderer = ViewportRenderer(resolution=(1920, 1080))
    boid_pos = np.random.uniform(0, 1000, (100, 2))
    # Render without raising exceptions
    renderer.render_boids(surface, boid_pos)
    assert surface.get_at((0, 0)) is not None
    pygame.quit()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_viewport_renderer.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.render.viewport_renderer'`

- [ ] **Step 3: Implement `ViewportRenderer` in `src/render/viewport_renderer.py`**

- Renderizado de partículas con estela bioluminiscente usando surface alpha blending (técnica de desvanecimiento con acumulación de buffer para lograr efecto *glow* de 60 FPS sin shaders pesados de GPU).
- Colores reactivos a gestos:
  - Cyan neón (`#00F0FF`) en modo normal / atracción.
  - Ámbar dorado (`#FFAA00`) en modo dispersión.
  - Magenta / Violeta de alta energía (`#BD00FF`) en modo condensación.
- HUD superior e inferior:
  - Barra de integridad del enjambre (% de microrobots activos).
  - Cronómetro digital y fase actual ("FASE 2: SINCRONIZACIÓN BIMANUAL").
  - Iconos de retroalimentación gestual en tiempo real en la esquina inferior (indicando si la cámara detecta mano izquierda, derecha, o puño cerrado).
- Dibujado de obstáculos térmicos y núcleo con pulsación senoidal.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_viewport_renderer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/render/viewport_renderer.py tests/test_viewport_renderer.py
git commit -m "feat: implement bioluminescent particle renderer and game HUD"
```

---

### Task 5: Bucle Principal, Modo Kiosk y Recuperación de Cámara

**Files:**
- Create: `src/app/main.py`
- Test: `tests/test_kiosk_lifecycle.py`

**Interfaces:**
- CLI Arguments: `--mock-camera`, `--fullscreen`, `--fps-cap [30|60]`, `--resolution 1920x1080`.
- Controla el bucle a 60 FPS coordinando captura en hilo secundario para evitar que el I/O de la cámara bloquee los fotogramas del juego.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kiosk_lifecycle.py
from src.app.main import KioskApp

def test_kiosk_idle_timeout_triggers_attract_mode():
    app = KioskApp(headless=True, mock_camera=True)
    app.idle_timer = 16.0  # simulate >15 seconds of no hands
    app.update_lifecycle(dt=0.1, hands_detected=False)
    assert app.is_attract_mode is True

def test_kiosk_hand_presence_exits_attract_mode():
    app = KioskApp(headless=True, mock_camera=True)
    app.is_attract_mode = True
    app.update_lifecycle(dt=0.1, hands_detected=True)
    assert app.is_attract_mode is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kiosk_lifecycle.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.app.main'`

- [ ] **Step 3: Implement `KioskApp` in `src/app/main.py`**

- Hilo independiente para captura de cámara (`CameraThread`) que lee a 640x480 con `cv2.VideoCapture` y previene cuellos de botella por V4L2 en Raspberry Pi.
- Conmutador de tolerancia a fallos: si la cámara se desconecta, mantiene el juego en ejecución registrando el error e intentando reconectar cada 2 segundos.
- Temporizador de inactividad de 15 segundos para *Attract Mode* (coreografía matemática autónoma en bucle de Lissajous para atraer al público que camina cerca del tótem).
- Detección de teclas de escape / salida para mantenimiento del centro (`ESC` o `Ctrl+C`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kiosk_lifecycle.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/main.py tests/test_kiosk_lifecycle.py
git commit -m "feat: implement kiosk lifecycle, attract mode and camera recovery thread"
```

---

### Task 6: Pruebas Integradas de Estrés y Medición de Rendimiento (Benchmark de Raspberry Pi)

**Files:**
- Create: `tests/test_benchmark_performance.py`
- Create: `scripts/run_kiosk.sh`

**Interfaces:**
- Mide el tiempo de ejecución promedio por fotograma bajo carga de 350 boids y pipeline de visión activo para verificar el cumplimiento del umbral de < 33.3 ms (30 FPS) en Raspberry Pi 4.

- [ ] **Step 1: Write benchmark test**

```python
# tests/test_benchmark_performance.py
import time
import numpy as np
from src.swarm.boids_engine import BoidsEngine
from src.vision.gesture_tracker import GestureTracker

def test_swarm_performance_budget():
    engine = BoidsEngine(num_boids=350, bounds=(1920, 1080))
    targets = [(960.0, 540.0, 'ATTRACT')]
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        engine.step(0.016, targets)
        times.append(time.perf_counter() - t0)
    avg_step_ms = np.mean(times) * 1000.0
    # Engine step must execute in under 8 ms on CPU
    assert avg_step_ms < 8.0, f"Boids step too slow: {avg_step_ms:.2f} ms"
```

- [ ] **Step 2: Run benchmark test**

Run: `pytest tests/test_benchmark_performance.py -v`  
Expected: PASS with execution time well below 8.0 ms.

- [ ] **Step 3: Create launch script `scripts/run_kiosk.sh`**

Script Bash para auto-arranque en Raspberry Pi OS:
- Activación de entorno virtual Python.
- Configuración de variables de entorno para SDL/Pygame (`SDL_VIDEODRIVER=x11` o `wayland`).
- Ejecución en pantalla completa con argumentos `--fullscreen`.

- [ ] **Step 4: Verify test suite completa**

Run: `pytest tests/ -v`  
Expected: Todos los tests pasando (100% PASS).

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmark_performance.py scripts/run_kiosk.sh
git commit -m "feat: add performance benchmark tests and raspberry pi kiosk launch script"
```
