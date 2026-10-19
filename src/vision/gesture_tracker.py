from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import os
import sys
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
    preview_frame: Optional[np.ndarray] = None  # RGB (180, 240, 3) con esqueleto anotado para PiP

class GestureTracker:
    """
    Submódulo de percepción de visión por computadora.
    Soporta tanto MediaPipe Tasks (Python 3.13 / moderna) como MediaPipe Solutions (legacy 3.10/3.11).
    Detecta manos, extrae coordenadas clave normalizadas (0.0 a 1.0)
    y clasifica el gesto en:
      - 'ATTRACT' (palma neutra/relajada)
      - 'DISPERSE' (palma abierta con dedos extendidos)
      - 'CONDENSE' (puño cerrado)
      - 'NONE' (sin mano en cuadro)
    Genera además un cuadro anotado 'preview_frame' para retroalimentación visual en pantalla.
    """

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Pulgar
        (0, 5), (5, 6), (6, 7), (7, 8),        # Índice
        (9, 10), (10, 11), (11, 12), (0, 9),   # Medio
        (13, 14), (14, 15), (15, 16), (0, 13), # Anular
        (0, 17), (17, 18), (18, 19), (19, 20), # Meñique
        (5, 9), (9, 13), (13, 17)              # Palma
    ]

    def __init__(
        self,
        use_mock: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        history_len: int = 3,
        model_path: Optional[str] = None
    ):
        self.use_mock = use_mock
        self.history_len = history_len
        self._history: List[str] = []
        self._mock_data = HandGestureData()

        self.backend = "none"
        self.solutions_detector = None
        self.tasks_landmarker = None

        if not self.use_mock and mp is not None:
            # 1. Intentar backend moderno MediaPipe Tasks (HandLandmarker)
            if hasattr(mp, 'tasks') and hasattr(mp.tasks, 'python') and hasattr(mp.tasks.python, 'vision'):
                vision = mp.tasks.python.vision
                BaseOptions = mp.tasks.BaseOptions

                # Resolver ruta del modelo
                candidates = [
                    model_path,
                    os.path.join(os.path.dirname(__file__), "../../models/hand_landmarker.task"),
                    os.path.join(os.getcwd(), "models/hand_landmarker.task"),
                    "models/hand_landmarker.task"
                ]
                resolved_model = None
                for c in candidates:
                    if c and os.path.isfile(c):
                        resolved_model = os.path.abspath(c)
                        break

                if resolved_model:
                    try:
                        options = vision.HandLandmarkerOptions(
                            base_options=BaseOptions(model_asset_path=resolved_model),
                            running_mode=vision.RunningMode.IMAGE,
                            num_hands=2,
                            min_hand_detection_confidence=min_detection_confidence,
                            min_hand_presence_confidence=min_detection_confidence,
                            min_tracking_confidence=min_tracking_confidence
                        )
                        self.tasks_landmarker = vision.HandLandmarker.create_from_options(options)
                        self.backend = "tasks"
                    except Exception:
                        pass

            # 2. Fallback a MediaPipe Solutions legacy si está disponible
            if self.backend == "none" and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                try:
                    self.solutions_detector = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        model_complexity=0,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence
                    )
                    self.backend = "solutions"
                except Exception:
                    pass

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
            hands=hands if hands is not None else [],
            preview_frame=np.zeros((180, 240, 3), dtype=np.uint8) if is_detected else None
        )

    def classify_landmarks(self, landmarks: List[Tuple[float, float, float]]) -> str:
        """
        Determina el tipo de gesto a partir de los 21 landmarks normalizados.
        """
        if len(landmarks) < 21:
            return 'NONE'

        wrist = landmarks[0]

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

        counts: Dict[str, int] = {}
        for g in self._history:
            counts[g] = counts.get(g, 0) + 1

        return max(counts, key=counts.get)

    def _draw_skeleton_on_preview(
        self,
        preview_bgr: np.ndarray,
        all_landmarks: List[List[Tuple[float, float, float]]],
        gesture: str
    ) -> np.ndarray:
        """Dibuja las conexiones y puntos clave de las manos sobre la previsualización."""
        h, w, _ = preview_bgr.shape
        color_bgr = (255, 240, 0)  # Cyan BGR (0, 240, 255) -> (255, 240, 0)
        if gesture == 'DISPERSE':
            color_bgr = (0, 170, 255)  # Ámbar BGR
        elif gesture == 'CONDENSE':
            color_bgr = (255, 0, 190)  # Magenta BGR

        for landmarks in all_landmarks:
            pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]

            # Dibujar huesos
            for i1, i2 in self.HAND_CONNECTIONS:
                if i1 < len(pts) and i2 < len(pts):
                    cv2.line(preview_bgr, pts[i1], pts[i2], (40, 200, 255), 2)

            # Dibujar nodos articulares
            for pt in pts:
                cv2.circle(preview_bgr, pt, 3, color_bgr, -1)

        # Convertir a RGB para Pygame
        return cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)

    def process_frame(self, frame: Optional[np.ndarray]) -> HandGestureData:
        """
        Procesa un fotograma BGR proveniente de la cámara web.
        Genera coordenadas, tipo de gesto y el cuadro de previsualización (preview_frame).
        """
        if self.use_mock or cv2 is None or frame is None or frame.size == 0:
            return self._mock_data

        # 1. Flip horizontal para interacción tipo espejo
        flipped = cv2.flip(frame, 1)

        # 2. Generar miniatura base para el cuadro de retroalimentación (240x180)
        preview_base = cv2.resize(flipped, (240, 180), interpolation=cv2.INTER_LINEAR)

        # 3. Downsampling a 320x240 para inferencia ultra-rápida en Raspberry Pi
        infer_frame = cv2.resize(flipped, (320, 240), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)

        all_hands_landmarks: List[List[Tuple[float, float, float]]] = []

        # 4. Inferencia según backend activo
        if self.backend == "tasks" and self.tasks_landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.tasks_landmarker.detect(mp_image)
            if detection_result.hand_landmarks:
                for hand_lms in detection_result.hand_landmarks:
                    all_hands_landmarks.append([(lm.x, lm.y, lm.z) for lm in hand_lms])

        elif self.backend == "solutions" and self.solutions_detector is not None:
            results = self.solutions_detector.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    all_hands_landmarks.append([(lm.x, lm.y, lm.z) for lm in hand_lms.landmark])

        # Si no se detectan manos
        if not all_hands_landmarks:
            self._history.clear()
            rgb_preview = cv2.cvtColor(preview_base, cv2.COLOR_BGR2RGB)
            return HandGestureData(
                is_detected=False,
                gesture_type='NONE',
                hand_count=0,
                preview_frame=rgb_preview
            )

        # 5. Clasificación geométrica de cada mano
        hands_list: List[Dict[str, Any]] = []
        center_x_accum = 0.0
        center_y_accum = 0.0

        for points in all_hands_landmarks:
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

        # 6. Dibujar esqueleto de manos sobre el preview
        annotated_preview = self._draw_skeleton_on_preview(
            preview_base,
            all_hands_landmarks,
            filtered_gesture
        )

        return HandGestureData(
            is_detected=True,
            x=avg_x,
            y=avg_y,
            gesture_type=filtered_gesture,
            hand_count=num_hands,
            hands=hands_list,
            preview_frame=annotated_preview
        )

    def close(self) -> None:
        """Libera recursos del detector."""
        if self.solutions_detector is not None:
            self.solutions_detector.close()
        if self.tasks_landmarker is not None:
            self.tasks_landmarker.close()
