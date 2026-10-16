import os
# Configurar SDL en modo dummy para pruebas headless sin monitor físico
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pytest
import pygame
import numpy as np
from src.render.viewport_renderer import ViewportRenderer
from src.game.mission_director import MissionState
from src.vision.gesture_tracker import HandGestureData

@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    pygame.init()
    yield
    pygame.quit()

def test_renderer_initialization():
    renderer = ViewportRenderer(resolution=(1920, 1080))
    assert renderer.width == 1920
    assert renderer.height == 1080
    assert renderer.glow_trail_surface is not None

def test_render_gesture_colors():
    renderer = ViewportRenderer(resolution=(1920, 1080))
    color_attract = renderer.get_gesture_color('ATTRACT')
    color_disperse = renderer.get_gesture_color('DISPERSE')
    color_condense = renderer.get_gesture_color('CONDENSE')

    # Deben ser tuplas RGB distintas y válidas
    assert len(color_attract) == 3
    assert len(color_disperse) == 3
    assert len(color_condense) == 3
    assert color_attract != color_disperse
    assert color_attract != color_condense

def test_render_frame_headless():
    renderer = ViewportRenderer(resolution=(1920, 1080))
    surface = pygame.Surface((1920, 1080))

    positions = np.random.uniform(200.0, 800.0, size=(100, 2))
    velocities = np.random.uniform(-50.0, 50.0, size=(100, 2))
    mission_state = MissionState(phase=1, time_remaining=150.0)
    gesture = HandGestureData(is_detected=True, gesture_type='ATTRACT', x=0.5, y=0.5)

    # Renderizar cuadro completo
    renderer.render_frame(
        surface=surface,
        boid_positions=positions,
        boid_velocities=velocities,
        mission_state=mission_state,
        gesture_data=gesture,
        fps=60.0
    )

    # Verificar que el canvas se modificó (no permanece completamente negro)
    arr = pygame.surfarray.array3d(surface)
    assert np.any(arr > 0)

def test_render_attract_mode_headless():
    renderer = ViewportRenderer(resolution=(1920, 1080))
    surface = pygame.Surface((1920, 1080))
    positions = np.random.uniform(100.0, 900.0, size=(150, 2))

    renderer.render_attract_mode(surface, positions, time_elapsed=10.0)
    arr = pygame.surfarray.array3d(surface)
    assert np.any(arr > 0)
