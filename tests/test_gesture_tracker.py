import numpy as np
import pytest
from src.vision.gesture_tracker import GestureTracker, HandGestureData

def test_gesture_tracker_initialization_mock():
    tracker = GestureTracker(use_mock=True)
    state = tracker.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert isinstance(state, HandGestureData)
    assert state.is_detected is False
    assert state.gesture_type == 'NONE'
    assert state.hand_count == 0

def test_gesture_classification_mock_disperse():
    tracker = GestureTracker(use_mock=True)
    tracker.set_mock_state(is_detected=True, gesture_type='DISPERSE', x=0.5, y=0.5, hand_count=1)
    state = tracker.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert state.is_detected is True
    assert state.gesture_type == 'DISPERSE'
    assert state.x == 0.5
    assert state.y == 0.5

def test_gesture_classification_mock_condense():
    tracker = GestureTracker(use_mock=True)
    tracker.set_mock_state(is_detected=True, gesture_type='CONDENSE', x=0.3, y=0.7, hand_count=1)
    state = tracker.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert state.is_detected is True
    assert state.gesture_type == 'CONDENSE'
    assert state.x == 0.3
    assert state.y == 0.7

def test_bimanual_tracking_mock():
    tracker = GestureTracker(use_mock=True)
    hands = [
        {'x': 0.2, 'y': 0.5, 'gesture': 'ATTRACT'},
        {'x': 0.8, 'y': 0.5, 'gesture': 'ATTRACT'}
    ]
    tracker.set_mock_state(is_detected=True, gesture_type='ATTRACT', x=0.5, y=0.5, hand_count=2, hands=hands)
    state = tracker.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert state.hand_count == 2
    assert len(state.hands) == 2
    assert state.hands[0]['x'] == 0.2
    assert state.hands[1]['x'] == 0.8

def test_landmark_geometric_gesture_calculation():
    tracker = GestureTracker(use_mock=False)
    # Simulate synthetic 21 landmarks for an open hand (fingertips far from wrist/MCP)
    # MediaPipe landmarks: 0 = wrist, 5 = index_mcp, 8 = index_tip, 9 = middle_mcp, 12 = middle_tip, etc.
    open_hand = [(0.5, 0.9, 0.0)] * 21
    # Wrist at (0.5, 0.9), fingertips stretched upwards (0.5, 0.1)
    for tip_idx in [4, 8, 12, 16, 20]:
        open_hand[tip_idx] = (0.5, 0.1, 0.0)
    for mcp_idx in [2, 5, 9, 13, 17]:
        open_hand[mcp_idx] = (0.5, 0.6, 0.0)
    gesture = tracker.classify_landmarks(open_hand)
    assert gesture == 'DISPERSE'

    # Simulate fist (fingertips curled near wrist/MCP)
    fist_hand = [(0.5, 0.9, 0.0)] * 21
    for tip_idx in [4, 8, 12, 16, 20]:
        fist_hand[tip_idx] = (0.5, 0.75, 0.0)
    for mcp_idx in [2, 5, 9, 13, 17]:
        fist_hand[mcp_idx] = (0.5, 0.6, 0.0)
    gesture_fist = tracker.classify_landmarks(fist_hand)
    assert gesture_fist == 'CONDENSE'
