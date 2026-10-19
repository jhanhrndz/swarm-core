import sys
import os
import time
import math
import argparse
import threading
from typing import Tuple, Optional, List
import numpy as np
import pygame

try:
    import cv2
except ImportError:
    cv2 = None

from src.vision.gesture_tracker import GestureTracker, HandGestureData
from src.swarm.boids_engine import BoidsEngine
from src.game.mission_director import MissionDirector, MissionState
from src.render.viewport_renderer import ViewportRenderer

class CameraWorker(threading.Thread):
    """
    Hilo asíncrono para captura continua de la cámara web.
    Desacopla la latencia de E/S de la cámara del bucle de renderizado a 60 FPS
    e implementa reconexión automática ante desconexión física de la cámara.
    """

    def __init__(self, camera_index: int = 0, capture_size: Tuple[int, int] = (640, 480)):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.capture_size = capture_size
        self.running = True
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._cap: Optional[Any] = None

    def _open_camera(self) -> bool:
        if cv2 is None:
            return False
        try:
            if sys.platform == "win32":
                self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.camera_index)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_size[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_size[1])
                return True
        except Exception:
            pass
        return False

    def run(self) -> None:
        self._open_camera()
        while self.running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(1.0)
                self._open_camera()
                continue

            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
            else:
                # Cámara desconectada o error de lectura; reintentar
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                time.sleep(1.0)

        if self._cap is not None:
            self._cap.release()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def stop(self) -> None:
        self.running = False


class KioskApp:
    """
    Controlador principal del tótem interactivo 'SwarmCore'.
    Coordina percepción por visión, física de boids, máquina de estados,
    renderizado y ciclo de vida de exhibición (Attract Mode).
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        fullscreen: bool = False,
        fps_cap: int = 60,
        mock_camera: bool = False,
        headless: bool = False,
        num_boids: int = 350,
        idle_timeout: float = 15.0,
        camera_index: int = 0
    ):
        self.width, self.height = resolution
        self.fullscreen = fullscreen
        self.fps_cap = fps_cap
        self.mock_camera = mock_camera
        self.headless = headless
        self.idle_timeout = idle_timeout

        # Estado del ciclo de vida
        self.is_running = True
        self.is_attract_mode = True
        self.idle_timer = 0.0
        self.attract_elapsed = 0.0

        # Submódulos del sistema
        self.gesture_tracker = GestureTracker(use_mock=mock_camera)
        self.boids_engine = BoidsEngine(num_boids=num_boids, bounds=resolution)
        self.mission_director = MissionDirector(total_time=180.0, bounds=resolution)

        # Pygame y Renderizador
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        if not self.headless:
            if not pygame.get_init():
                pygame.init()
            flags = pygame.FULLSCREEN if self.fullscreen else 0
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
            pygame.display.set_caption("SwarmCore // AudacIA AI & Robotics")
            self.clock = pygame.time.Clock()

        self.renderer = ViewportRenderer(resolution=resolution)

        # Hilo de cámara
        self.camera_worker: Optional[CameraWorker] = None
        if not self.mock_camera and not self.headless:
            self.camera_worker = CameraWorker(camera_index=camera_index)
            self.camera_worker.start()

    def update_lifecycle(self, dt: float, hands_detected: bool) -> None:
        """
        Gestiona la transición fluida entre Modo Misión y Attract Mode (Reposo).
        """
        if hands_detected:
            self.idle_timer = 0.0
            if self.is_attract_mode:
                self.is_attract_mode = False
                self.mission_director.reset()
                self.boids_engine.reset()
        else:
            self.idle_timer += dt
            if self.idle_timer >= self.idle_timeout:
                if not self.is_attract_mode:
                    self.is_attract_mode = True
                    self.mission_director.reset()
                    self.boids_engine.reset()

    def step_frame(self, dt: float = 0.016) -> None:
        """
        Ejecuta un ciclo completo de lógica y renderizado.
        """
        # 1. Obtener cuadro de visión
        frame = None
        if self.camera_worker is not None:
            frame = self.camera_worker.get_latest_frame()

        if frame is None and not self.mock_camera:
            # Fallback seguro con cuadro sintético en ausencia temporal de cámara
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 2. Procesar gestos
        gesture_data = self.gesture_tracker.process_frame(frame)

        # 3. Actualizar temporizador de inactividad
        self.update_lifecycle(dt, gesture_data.is_detected)

        # 4. Modo Reposo (Attract Mode)
        if self.is_attract_mode:
            self.attract_elapsed += dt
            # Coreografía matemática autónoma en curva de Lissajous
            lx = self.width * 0.5 + math.sin(self.attract_elapsed * 0.8) * (self.width * 0.35)
            ly = self.height * 0.5 + math.cos(self.attract_elapsed * 1.3) * (self.height * 0.3)
            self.boids_engine.step(dt, hand_targets=[(lx, ly, 'ATTRACT')])

            if self.screen is not None:
                self.renderer.render_attract_mode(
                    surface=self.screen,
                    boid_positions=self.boids_engine.get_positions(),
                    time_elapsed=self.attract_elapsed,
                    preview_frame=gesture_data.preview_frame,
                    is_detected=gesture_data.is_detected,
                    gesture_type=gesture_data.gesture_type
                )
        else:
            # 5. Modo Misión Activa
            self.boids_engine.set_obstacles(self.mission_director.get_obstacles())

            hand_targets: List[Tuple[float, float, str]] = []
            if gesture_data.hands:
                for h in gesture_data.hands:
                    hand_targets.append((h['x'] * self.width, h['y'] * self.height, h['gesture']))
            elif gesture_data.is_detected:
                hand_targets.append((gesture_data.x * self.width, gesture_data.y * self.height, gesture_data.gesture_type))

            boid_pos = self.boids_engine.step(dt, hand_targets)
            mission_state = self.mission_director.update(dt, boid_pos, gesture_data)

            if self.screen is not None:
                fps_val = self.clock.get_fps() if self.clock is not None else 60.0
                self.renderer.render_frame(
                    self.screen,
                    boid_pos,
                    self.boids_engine.get_velocities(),
                    mission_state,
                    gesture_data,
                    fps=fps_val,
                    dt=dt
                )

        if self.screen is not None:
            pygame.display.flip()

    def handle_events(self) -> None:
        """Procesa eventos de teclado y ventana para mantenimiento del tótem."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.is_running = False
                elif event.key == pygame.K_r:
                    # Reinicio rápido manual
                    self.mission_director.reset()
                    self.boids_engine.reset()
                elif event.key == pygame.K_SPACE:
                    # Conmutar modo reposo
                    self.is_attract_mode = not self.is_attract_mode
                elif event.key == pygame.K_c:
                    # Conmutar visualización del cuadro de cámara PiP
                    self.renderer.show_camera_pip = not self.renderer.show_camera_pip

    def run(self) -> None:
        """Bucle principal de ejecución del kiosco."""
        while self.is_running:
            dt = 0.016
            if self.clock is not None:
                dt = self.clock.tick(self.fps_cap) / 1000.0
                dt = min(dt, 0.05)  # Prevenir salto de física ante lag temporal

            self.handle_events()
            self.step_frame(dt)

        self.cleanup()

    def cleanup(self) -> None:
        """Cierre ordenado y liberación de recursos."""
        if self.camera_worker is not None:
            self.camera_worker.stop()
        self.gesture_tracker.close()
        if pygame.get_init():
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="SwarmCore - Juego Interactivo de Visión Artificial para AudacIA")
    parser.add_argument("--fullscreen", action="store_true", help="Ejecutar en pantalla completa 1080p")
    parser.add_argument("--mock-camera", action="store_true", help="Ejecutar sin cámara física (modo simulación)")
    parser.add_argument("--fps", type=int, default=60, help="Límite de cuadros por segundo (default: 60)")
    parser.add_argument("--boids", type=int, default=350, help="Número de microrobots en el enjambre (default: 350)")
    parser.add_argument("--camera-idx", type=int, default=0, help="Índice de la cámara USB (default: 0)")
    parser.add_argument("--width", type=int, default=1920, help="Ancho de resolución")
    parser.add_argument("--height", type=int, default=1080, help="Alto de resolución")
    args = parser.parse_args()

    app = KioskApp(
        resolution=(args.width, args.height),
        fullscreen=args.fullscreen,
        fps_cap=args.fps,
        mock_camera=args.mock_camera,
        num_boids=args.boids,
        camera_index=args.camera_idx
    )
    app.run()


if __name__ == "__main__":
    main()
