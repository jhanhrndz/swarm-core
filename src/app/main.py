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
        resolution: Optional[Tuple[int, int]] = None,
        fullscreen: bool = False,
        fps_cap: int = 60,
        mock_camera: bool = False,
        headless: bool = False,
        num_boids: int = 350,
        idle_timeout: float = 15.0,
        camera_index: int = 0
    ):
        self.virtual_width = 1920
        self.virtual_height = 1080
        self.width = self.virtual_width
        self.height = self.virtual_height
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

        # Submódulos del sistema (operan en el espacio virtual 1920x1080)
        self.gesture_tracker = GestureTracker(use_mock=mock_camera)
        self.boids_engine = BoidsEngine(num_boids=num_boids, bounds=(self.width, self.height))
        self.mission_director = MissionDirector(total_time=180.0, bounds=(self.width, self.height))

        # Pygame, Lienzo Virtual y Renderizador
        self.screen: Optional[pygame.Surface] = None
        self.virtual_surface: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.window_size: Tuple[int, int] = (self.width, self.height)

        if not self.headless:
            if not pygame.get_init():
                pygame.init()

            display_info = pygame.display.Info()
            disp_w = display_info.current_w if display_info.current_w > 0 else 1920
            disp_h = display_info.current_h if display_info.current_h > 0 else 1080

            if self.fullscreen:
                win_w = disp_w
                win_h = disp_h
                flags = pygame.FULLSCREEN
            else:
                if resolution is not None:
                    req_w, req_h = resolution
                    # Si la resolución solicitada excede la pantalla del portátil, ajustar para evitar desborde
                    if req_w >= disp_w or req_h >= disp_h:
                        if disp_w <= 1600 or disp_h <= 900:
                            win_w, win_h = 1280, 720
                        else:
                            win_w, win_h = 1600, 900
                    else:
                        win_w, win_h = req_w, req_h
                else:
                    # En modo ventana por defecto: resolución 720p 16:9 para asegurar que quepa 100% en portátiles
                    if disp_w <= 1600 or disp_h <= 900:
                        win_w, win_h = 1280, 720
                    else:
                        win_w, win_h = 1600, 900
                flags = 0

            self.window_size = (win_w, win_h)
            self.screen = pygame.display.set_mode(self.window_size, flags)
            pygame.display.set_caption("SwarmCore // AudacIA AI & Robotics")
            self.clock = pygame.time.Clock()
            self.virtual_surface = pygame.Surface((self.virtual_width, self.virtual_height))

        self.renderer = ViewportRenderer(resolution=(self.width, self.height))

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

        render_target = self.virtual_surface if self.screen is not None else None

        # 4. Modo Reposo (Attract Mode)
        if self.is_attract_mode:
            self.attract_elapsed += dt
            # Coreografía matemática autónoma en curva de Lissajous
            lx = self.width * 0.5 + math.sin(self.attract_elapsed * 0.8) * (self.width * 0.35)
            ly = self.height * 0.5 + math.cos(self.attract_elapsed * 1.3) * (self.height * 0.3)
            self.boids_engine.step(dt, hand_targets=[(lx, ly, 'ATTRACT')])

            if render_target is not None:
                self.renderer.render_attract_mode(
                    surface=render_target,
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

            if render_target is not None:
                fps_val = self.clock.get_fps() if self.clock is not None else 60.0
                self.renderer.render_frame(
                    surface=render_target,
                    boid_positions=boid_pos,
                    boid_velocities=self.boids_engine.get_velocities(),
                    mission_state=mission_state,
                    gesture_data=gesture_data,
                    fps=fps_val,
                    dt=dt
                )

        if self.screen is not None and self.virtual_surface is not None:
            if self.screen.get_size() == (self.virtual_width, self.virtual_height):
                self.screen.blit(self.virtual_surface, (0, 0))
            else:
                pygame.transform.scale(self.virtual_surface, self.screen.get_size(), self.screen)
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
                elif event.key in (pygame.K_f, pygame.K_F11):
                    # Alternar pantalla completa
                    self.fullscreen = not self.fullscreen
                    if not self.headless and pygame.display.get_init():
                        display_info = pygame.display.Info()
                        disp_w = display_info.current_w if display_info.current_w > 0 else 1920
                        disp_h = display_info.current_h if display_info.current_h > 0 else 1080
                        if self.fullscreen:
                            self.window_size = (disp_w, disp_h)
                            self.screen = pygame.display.set_mode(self.window_size, pygame.FULLSCREEN)
                        else:
                            if disp_w <= 1600 or disp_h <= 900:
                                self.window_size = (1280, 720)
                            else:
                                self.window_size = (1600, 900)
                            self.screen = pygame.display.set_mode(self.window_size, 0)

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
    parser.add_argument("--fullscreen", action="store_true", help="Ejecutar en pantalla completa nativa")
    parser.add_argument("--mock-camera", action="store_true", help="Ejecutar sin cámara física (modo simulación)")
    parser.add_argument("--fps", type=int, default=60, help="Límite de cuadros por segundo (default: 60)")
    parser.add_argument("--boids", type=int, default=350, help="Número de microrobots en el enjambre (default: 350)")
    parser.add_argument("--camera-idx", type=int, default=0, help="Índice de la cámara USB (default: 0)")
    parser.add_argument("--width", type=int, default=None, help="Ancho de ventana personalizado")
    parser.add_argument("--height", type=int, default=None, help="Alto de ventana personalizado")
    args = parser.parse_args()

    custom_res = None
    if args.width is not None and args.height is not None:
        custom_res = (args.width, args.height)

    app = KioskApp(
        resolution=custom_res,
        fullscreen=args.fullscreen,
        fps_cap=args.fps,
        mock_camera=args.mock_camera,
        num_boids=args.boids,
        camera_index=args.camera_idx
    )
    app.run()


if __name__ == "__main__":
    main()
