import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pytest
import numpy as np
from src.app.main import KioskApp
from src.vision.gesture_tracker import HandGestureData

def test_kiosk_app_initialization():
    app = KioskApp(headless=True, mock_camera=True, idle_timeout=15.0)
    assert app.is_running is True
    assert app.is_attract_mode is True
    assert app.idle_timer == 0.0
    assert app.boids_engine.num_boids > 0

def test_kiosk_hand_presence_exits_attract_mode():
    app = KioskApp(headless=True, mock_camera=True, idle_timeout=15.0)
    assert app.is_attract_mode is True

    # User enters frame
    app.update_lifecycle(dt=0.1, hands_detected=True)
    assert app.is_attract_mode is False
    assert app.idle_timer == 0.0

def test_kiosk_idle_timeout_triggers_attract_mode():
    app = KioskApp(headless=True, mock_camera=True, idle_timeout=15.0)
    app.is_attract_mode = False

    # Simulate 10 seconds of no hands -> should remain in game
    app.update_lifecycle(dt=10.0, hands_detected=False)
    assert app.is_attract_mode is False
    assert app.idle_timer == 10.0

    # Simulate another 6 seconds (total 16s > 15s) -> should enter attract mode
    app.update_lifecycle(dt=6.0, hands_detected=False)
    assert app.is_attract_mode is True

def test_kiosk_step_frame_in_game_mode():
    app = KioskApp(headless=True, mock_camera=True)
    app.is_attract_mode = False

    # Set mock gesture to ATTRACT at (960, 540)
    app.gesture_tracker.set_mock_state(
        is_detected=True,
        gesture_type='ATTRACT',
        x=0.5,
        y=0.5,
        hand_count=1
    )

    p0 = app.boids_engine.get_positions().copy()
    app.step_frame(dt=0.016)
    p1 = app.boids_engine.get_positions()

    # Boids should have moved
    assert not np.array_equal(p0, p1)
    assert app.mission_director.time_remaining < 180.0

def test_kiosk_step_frame_in_attract_mode():
    app = KioskApp(headless=True, mock_camera=True)
    app.is_attract_mode = True

    p0 = app.boids_engine.get_positions().copy()
    app.step_frame(dt=0.016)
    p1 = app.boids_engine.get_positions()

    # Boids move in autonomous attract choreography
    assert not np.array_equal(p0, p1)
