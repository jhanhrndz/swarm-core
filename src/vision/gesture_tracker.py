from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np

try:
    import cv2
    import mediapipe as mp
except ImportError:
    cv2 = None
    mp = None

@dataclass
class HandGestureData:
    is_detected: bool = False
    x: float = 0.5
    y: float = 0.5
    gesture_type: str = 'NONE'  # 'ATTRACT', 'DISPERSE', 'CONDENSE', 'NONE'
    hand_count: int = 0
    hands: List[Dict[str, Any]] = field(default_factory=list)

class GestureTracker:
    """
    Submódulo de percepción de visión por computadora.
    Detecta manos, extrae coordenadas clave normalizadas (0.0 a 1.0)
    y clasifica el gesto en:
      - 'ATTRACT' (palma neutra/relajada)
      - 'DISPERSE' (palma abierta con dedos extendidos)
      - 'CONDENSE' (puño cerrado)
      - 'NONE' (sin mano en cuadro)
    """

    def __init__(
        self,
        use_mock: bool = False,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        history_len: int = 3
    ):
        self.use_mock = use_mock
        self.history_len = history_len
        self._history: List[str] = []
        self._mock_data = HandGestureData()

        self.mp_hands = None
        self.hands_detector = None

        if not self.use_mock:
            if mp is not None and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                self.mp_hands = mp.solutions.hands
                self.hands_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=0,  # MediaPipe Lite (CPU optimized for Raspberry Pi)
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )

    def set_mock_state(
        self,
        is_detected: bool,
        gesture_type: str = 'NONE',
        x: float = 0.5,
        y: float = 0.5,
        hand_count: int = 0,
        hands: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Configura estado simulado para pruebas unitarias y modo sin cámara."""
        self._mock_data = HandGestureData(
            is_detected=is_detected,
            x=x,
            y=y,
            gesture_type=gesture_type,
            hand_count=hand_count,
            hands=hands if hands is not None else []
        )

    def classify_landmarks(self, landmarks: List[Tuple[float, float, float]]) -> str:
        """
        Determina el tipo de gesto a partir de los 21 landmarks normalizados.
        """
        if len(landmarks) < 21:
            return 'NONE'

        wrist = landmarks[0]

        # Pares (MCP, TIP) para los cuatro dedos principales
        finger_pairs = [
            (5, 8),    # Índice
            (9, 12),   # Medio
            (13, 16),  # Anular
            (17, 20)   # Meñique
        ]

        extended_fingers = 0
        curled_fingers = 0

        for mcp_idx, tip_idx in finger_pairs:
            mcp = landmarks[mcp_idx]
            tip = landmarks[tip_idx]

            d_wrist_tip = math.dist((wrist[0], wrist[1]), (tip[0], tip[1]))
            d_wrist_mcp = math.dist((wrist[0], wrist[1]), (mcp[0], mcp[1]))

            if d_wrist_mcp > 1e-4:
                ratio = d_wrist_tip / d_wrist_mcp
                if ratio > 1.25:
                    extended_fingers += 1
                elif ratio < 1.05:
                    curled_fingers += 1

        # Detección del pulgar (landmarks 2 MCP y 4 TIP)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        d_wrist_thumb_tip = math.dist((wrist[0], wrist[1]), (thumb_tip[0], thumb_tip[1]))
        d_wrist_thumb_mcp = math.dist((wrist[0], wrist[1]), (thumb_mcp[0], thumb_mcp[1]))
        if d_wrist_thumb_mcp > 1e-4:
            thumb_ratio = d_wrist_thumb_tip / d_wrist_thumb_mcp
            if thumb_ratio > 1.2:
                extended_fingers += 1
            elif thumb_ratio < 1.05:
                curled_fingers += 1

        if extended_fingers >= 4:
            return 'DISPERSE'
        elif curled_fingers >= 4:
            return 'CONDENSE'
        else:
            return 'ATTRACT'

    def _apply_hysteresis(self, raw_gesture: str) -> str:
        """Filtro temporal para eliminar jitter entre fotogramas sucesivos."""
        self._history.append(raw_gesture)
        if len(self._history) > self.history_len:
            self._history.pop(0)

        # Si el gesto más reciente domina en la ventana histórica, se confirma
        counts: Dict[str, int] = {}
        for g in self._history:
            counts[g] = counts.get(g, 0) + 1

        # Retorna el gesto con mayor frecuencia en la ventana
        return max(counts, key=counts.get)

    def process_frame(self, frame: np.ndarray) -> HandGestureData:
        """
        Procesa un fotograma BGR proveniente de la cámara web.
        """
        if self.use_mock or self.hands_detector is None or cv2 is None:
            return self._mock_data

        if frame is None or frame.size == 0:
            return HandGestureData(is_detected=False, gesture_type='NONE')

        # 1. Flip horizontal para interacción tipo espejo
        flipped = cv2.flip(frame, 1)

        # 2. Downsampling a 320x240 para acelerar la inferencia en Raspberry Pi
        small_frame = cv2.resize(flipped, (320, 240), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 3. Inferencia de MediaPipe
        results = self.hands_detector.process(rgb_frame)

        if not results.multi_hand_landmarks:
            self._history.clear()
            return HandGestureData(is_detected=False, gesture_type='NONE', hand_count=0)

        hands_list: List[Dict[str, Any]] = []
        center_x_accum = 0.0
        center_y_accum = 0.0

        for hand_landmarks in results.multi_hand_landmarks:
            points: List[Tuple[float, float, float]] = [
                (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
            ]
            gesture = self.classify_landmarks(points)
            palm_center_x = (points[0][0] + points[9][0]) / 2.0
            palm_center_y = (points[0][1] + points[9][1]) / 2.0

            hands_list.append({
                'x': palm_center_x,
                'y': palm_center_y,
                'gesture': gesture
            })
            center_x_accum += palm_center_x
            center_y_accum += palm_center_y

        num_hands = len(hands_list)
        avg_x = center_x_accum / num_hands
        avg_y = center_y_accum / num_hands

        # Si hay una sola mano, su gesto define el estado principal
        # Si hay dos manos, se evalúa si alguna activa dispersión o condensación
        raw_gesture = hands_list[0]['gesture']
        if num_hands > 1:
            gestures = [h['gesture'] for h in hands_list]
            if 'CONDENSE' in gestures:
                raw_gesture = 'CONDENSE'
            elif 'DISPERSE' in gestures:
                raw_gesture = 'DISPERSE'
            else:
                raw_gesture = 'ATTRACT'

        filtered_gesture = self._apply_hysteresis(raw_gesture)

        return HandGestureData(
            is_detected=True,
            x=avg_x,
            y=avg_y,
            gesture_type=filtered_gesture,
            hand_count=num_hands,
            hands=hands_list
        )

    def close(self) -> None:
        """Libera recursos del detector."""
        if self.hands_detector is not None:
            self.hands_detector.close()
