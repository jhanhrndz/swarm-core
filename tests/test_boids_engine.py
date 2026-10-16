import numpy as np
import pytest
from src.swarm.boids_engine import BoidsEngine

def test_boids_engine_initialization():
    engine = BoidsEngine(num_boids=300, bounds=(1920.0, 1080.0))
    positions = engine.get_positions()
    velocities = engine.get_velocities()
    assert positions.shape == (300, 2)
    assert velocities.shape == (300, 2)
    assert np.all(positions >= 0.0)
    assert np.all(positions[:, 0] <= 1920.0)
    assert np.all(positions[:, 1] <= 1080.0)

def test_boids_attractor_force():
    engine = BoidsEngine(num_boids=100, bounds=(1000.0, 1000.0))
    # Place all boids at (100, 100) with zero velocity
    engine.positions[:, 0] = 100.0
    engine.positions[:, 1] = 100.0
    engine.velocities[:, :] = 0.0

    # Hand at (500, 500) with ATTRACT gesture
    engine.step(dt=0.1, hand_targets=[(500.0, 500.0, 'ATTRACT')])

    # Velocities should be oriented towards +X and +Y
    assert np.mean(engine.velocities[:, 0]) > 0
    assert np.mean(engine.velocities[:, 1]) > 0

def test_boids_disperse_force():
    engine = BoidsEngine(num_boids=100, bounds=(1000.0, 1000.0))
    # Place boids in a ring around (500, 500)
    engine.positions[:, 0] = 510.0
    engine.positions[:, 1] = 510.0
    engine.velocities[:, :] = 0.0

    # Hand at (500, 500) with DISPERSE gesture
    engine.step(dt=0.1, hand_targets=[(500.0, 500.0, 'DISPERSE')])

    # Since boids are at (510, 510), repulsion must push them further away (+X, +Y)
    assert np.mean(engine.velocities[:, 0]) > 0
    assert np.mean(engine.velocities[:, 1]) > 0

def test_boids_boundary_clamping():
    engine = BoidsEngine(num_boids=50, bounds=(1920.0, 1080.0))
    # Manually place boids beyond boundaries
    engine.positions[0] = [2500.0, 1500.0]
    engine.positions[1] = [-100.0, -50.0]
    engine.step(dt=0.016, hand_targets=[])

    assert 0.0 <= engine.positions[0, 0] <= 1920.0
    assert 0.0 <= engine.positions[0, 1] <= 1080.0
    assert 0.0 <= engine.positions[1, 0] <= 1920.0
    assert 0.0 <= engine.positions[1, 1] <= 1080.0

def test_boids_obstacle_avoidance():
    engine = BoidsEngine(num_boids=100, bounds=(1000.0, 1000.0))
    # Place boid near an obstacle
    engine.positions[0] = [520.0, 500.0]
    engine.velocities[0] = [-50.0, 0.0]  # moving directly towards obstacle center
    engine.set_obstacles([(500.0, 500.0, 30.0)])  # center (500, 500), radius 30

    engine.step(dt=0.1, hand_targets=[])
    # The obstacle repulsive force must have pushed the boid in +X direction
    assert engine.velocities[0, 0] > 0

def test_boids_inertia_when_no_hands():
    engine = BoidsEngine(num_boids=100, bounds=(1920.0, 1080.0))
    # Give initial movement
    engine.velocities[:, 0] = 100.0
    engine.velocities[:, 1] = 50.0
    # Step with no hands
    engine.step(dt=0.016, hand_targets=[])
    speeds = np.linalg.norm(engine.velocities, axis=1)
    # Speeds should be positive and bounded by max_speed
    assert np.all(speeds > 0.0)
    assert np.all(speeds <= engine.max_speed)
