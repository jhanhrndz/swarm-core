import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import time
import pytest
import numpy as np
from src.swarm.boids_engine import BoidsEngine
from src.game.mission_director import MissionDirector
from src.vision.gesture_tracker import GestureTracker, HandGestureData

def test_boids_physics_step_benchmark():
    """
    Verifica que el cálculo vectorial de 350 boids se mantenga
    por debajo del presupuesto de latencia de 16 ms para garantizar 60 FPS.
    """
    engine = BoidsEngine(num_boids=350, bounds=(1920.0, 1080.0))
    targets = [(960.0, 540.0, 'ATTRACT')]

    # Warmup
    for _ in range(10):
        engine.step(0.016, targets)

    # 100 benchmark iterations
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        engine.step(0.016, targets)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # El cálculo físico debe tomar menos de 16 ms (incluso en CPUs moderadas)
    assert avg_latency < 16.0, f"Boids step too slow: {avg_latency:.2f} ms (p95: {p95_latency:.2f} ms)"

def test_integrated_pipeline_throughput():
    """
    Prueba de estrés integrada: procesa 100 ciclos continuos de física de enjambre
    más lógica de la máquina de estados de misión y clasificación gestual simulada.
    """
    engine = BoidsEngine(num_boids=350, bounds=(1920.0, 1080.0))
    director = MissionDirector(total_time=180.0, bounds=(1920.0, 1080.0))
    tracker = GestureTracker(use_mock=True)

    latencies = []
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(100):
        t0 = time.perf_counter()

        # 1. Visión
        tracker.set_mock_state(
            is_detected=True,
            gesture_type='ATTRACT' if i % 2 == 0 else 'DISPERSE',
            x=0.5 + 0.2 * np.sin(i * 0.1),
            y=0.5 + 0.2 * np.cos(i * 0.1),
            hand_count=1
        )
        gesture = tracker.process_frame(fake_frame)

        # 2. Física
        targets = [(gesture.x * 1920.0, gesture.y * 1080.0, gesture.gesture_type)]
        pos = engine.step(0.016, targets)

        # 3. Lógica de Misión
        director.update(0.016, pos, gesture)

        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_total_ms = float(np.mean(latencies))
    # Todo el pipeline de CPU por cuadro debe completarse con holgura para 60 FPS
    assert avg_total_ms < 20.0, f"Full pipeline too slow: {avg_total_ms:.2f} ms"
