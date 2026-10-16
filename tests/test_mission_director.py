import numpy as np
import pytest
from src.game.mission_director import MissionDirector, MissionState
from src.vision.gesture_tracker import HandGestureData

def test_mission_director_initialization():
    director = MissionDirector(total_time=180.0, bounds=(1920.0, 1080.0))
    state = director.get_state()
    assert isinstance(state, MissionState)
    assert state.phase == 1
    assert state.time_remaining == 180.0
    assert state.is_completed is False
    assert state.is_game_over is False
    assert state.swarm_integrity == 1.0
    assert len(director.get_obstacles()) > 0

def test_phase1_navigation_progression():
    director = MissionDirector(total_time=180.0, bounds=(1920.0, 1080.0))
    # Checkpoint is on the right side of the screen
    checkpoint_x, checkpoint_y, checkpoint_r = director.phase1_checkpoint
    # Place 300 boids inside the checkpoint radius
    positions = np.zeros((300, 2), dtype=np.float64)
    positions[:, 0] = checkpoint_x
    positions[:, 1] = checkpoint_y

    gesture = HandGestureData(is_detected=True, gesture_type='ATTRACT')
    state = director.update(dt=0.1, boid_positions=positions, gesture_data=gesture)
    assert state.phase == 2

def test_phase1_hazard_collision_reduces_integrity():
    director = MissionDirector(total_time=180.0, bounds=(1920.0, 1080.0))
    hazard = director.get_obstacles()[0]
    hx, hy, hr = hazard

    # Place 100 boids directly inside hazard
    positions = np.zeros((100, 2), dtype=np.float64)
    positions[:, 0] = hx
    positions[:, 1] = hy

    gesture = HandGestureData(is_detected=True, gesture_type='ATTRACT')
    state = director.update(dt=0.5, boid_positions=positions, gesture_data=gesture)
    assert state.swarm_integrity < 1.0

def test_phase2_bimanual_node_activation():
    director = MissionDirector(total_time=180.0, bounds=(1920.0, 1080.0))
    director.phase = 2  # jump directly to Phase 2

    node_a = director.phase2_nodes[0]
    node_b = director.phase2_nodes[1]

    # Split 200 boids: 100 in Node A, 100 in Node B
    positions = np.zeros((200, 2), dtype=np.float64)
    positions[:100, 0] = node_a['x']
    positions[:100, 1] = node_a['y']
    positions[100:, 0] = node_b['x']
    positions[100:, 1] = node_b['y']

    gesture = HandGestureData(is_detected=True, gesture_type='ATTRACT', hand_count=2)
    # Sustain presence for the activation duration (e.g. 2.5s)
    for _ in range(30):
        state = director.update(dt=0.1, boid_positions=positions, gesture_data=gesture)

    assert state.phase == 3

def test_phase3_core_insertion_requires_condense_gesture():
    director = MissionDirector(total_time=180.0, bounds=(1920.0, 1080.0))
    director.phase = 3  # jump to Phase 3

    fuse_pos = director.fuse_position.copy()
    # 1. Place boids near fuse with ATTRACT (should not move the heavy fuse)
    positions = np.zeros((100, 2), dtype=np.float64)
    positions[:, 0] = fuse_pos[0]
    positions[:, 1] = fuse_pos[1]

    gesture_attract = HandGestureData(is_detected=True, gesture_type='ATTRACT')
    director.update(dt=0.2, boid_positions=positions, gesture_data=gesture_attract)
    assert np.allclose(director.fuse_position, fuse_pos)

    # 2. Place boids near fuse with CONDENSE (fist) -> moves the fuse towards core
    gesture_condense = HandGestureData(is_detected=True, gesture_type='CONDENSE')
    for _ in range(50):
        state = director.update(dt=0.1, boid_positions=positions, gesture_data=gesture_condense)
        positions[:, 0] = director.fuse_position[0]
        positions[:, 1] = director.fuse_position[1]

    assert state.core_status > 0.0
    assert state.is_completed is True

def test_mission_timeout_triggers_game_over():
    director = MissionDirector(total_time=1.0, bounds=(1920.0, 1080.0))
    positions = np.zeros((50, 2), dtype=np.float64)
    gesture = HandGestureData()

    state = director.update(dt=1.5, boid_positions=positions, gesture_data=gesture)
    assert state.is_game_over is True
    assert state.is_completed is False
