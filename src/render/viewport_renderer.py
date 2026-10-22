from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
import pygame
from src.game.mission_director import MissionState
from src.vision.gesture_tracker import HandGestureData

class ViewportRenderer:
    """
    Renderizador 2.5D acelerado por hardware (SDL2/Pygame-CE).
    Dibuja el enjambre de microrobots con estelas bioluminiscentes reactivas,
    obstáculos de calor, nodos de recarga, núcleo del reactor y HUD cyberpunk.
    Optimizado para sostener 60 FPS en laptop y 30-45 FPS en Raspberry Pi 4.
    """

    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)):
        self.width, self.height = resolution
        self.fade_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.glow_trail_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.fade_surface.fill((10, 14, 23, 45))  # R, G, B, Alpha para estela bioluminiscente

        # Inicialización de fuentes (fallback seguro a fuente del sistema si no hay TTF)
        if not pygame.font.get_init():
            pygame.font.init()

        self.font_large = pygame.font.SysFont("consolas,monospace", 36, bold=True)
        self.font_medium = pygame.font.SysFont("consolas,monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("consolas,monospace", 18, bold=True)

        # Paleta de colores Neón Cyberpunk
        self.BG_COLOR = (10, 14, 23)
        self.COLOR_ATTRACT = (0, 240, 255)     # Cyan Neón
        self.COLOR_DISPERSE = (255, 170, 0)    # Ámbar Dorado
        self.COLOR_CONDENSE = (190, 0, 255)    # Magenta Plasma
        self.COLOR_NEUTRAL = (0, 200, 220)     # Teal Eléctrico

        self.COLOR_HAZARD = (255, 60, 40)      # Rojo-Naranja térmico
        self.COLOR_PORTAL = (0, 255, 160)      # Verde Esmeralda
        self.COLOR_CORE = (0, 230, 255)        # Núcleo Cyan Brillante
        self.COLOR_FUSE = (255, 215, 0)        # Fusible Dorado
        self.COLOR_TEXT_PRIMARY = (235, 245, 255)
        self.COLOR_TEXT_MUTED = (120, 150, 180)

        self._pulse_timer = 0.0
        self.show_camera_pip = True  # Control de visibilidad del cuadro de cámara en vivo

    def get_gesture_color(self, gesture_type: str) -> Tuple[int, int, int]:
        """Retorna el color reactivo según el gesto clasificado."""
        if gesture_type == 'ATTRACT':
            return self.COLOR_ATTRACT
        elif gesture_type == 'DISPERSE':
            return self.COLOR_DISPERSE
        elif gesture_type == 'CONDENSE':
            return self.COLOR_CONDENSE
        return self.COLOR_NEUTRAL

    def render_boids(
        self,
        surface: pygame.Surface,
        positions: np.ndarray,
        velocities: np.ndarray,
        gesture_type: str = 'NONE'
    ) -> None:
        """Dibuja las partículas del enjambre con cabezas de microrobot y orientación."""
        color = self.get_gesture_color(gesture_type)
        glow_color = (*color, 70)

        num_boids = len(positions)
        if num_boids == 0:
            return

        # Para eficiencia en Raspberry Pi, dibujamos boids como círculos con radio reactivo
        radius = 4 if gesture_type != 'CONDENSE' else 5

        for i in range(num_boids):
            px = int(positions[i, 0])
            py = int(positions[i, 1])

            # Punto central sólido
            pygame.draw.circle(surface, color, (px, py), radius)

            # Dirección de vector de velocidad
            vx = velocities[i, 0]
            vy = velocities[i, 1]
            speed = math.hypot(vx, vy)
            if speed > 10.0:
                end_x = int(px + (vx / speed) * 8.0)
                end_y = int(py + (vy / speed) * 8.0)
                pygame.draw.line(surface, color, (px, py), (end_x, end_y), 1)

    def render_hazards(
        self,
        surface: pygame.Surface,
        obstacles: List[Tuple[float, float, float]]
    ) -> None:
        """Dibuja campos de interferencia y calor electromagnético con pulso senoidal."""
        pulse = (math.sin(self._pulse_timer * 4.0) + 1.0) * 0.5
        for ox, oy, r in obstacles:
            center = (int(ox), int(oy))
            current_r = int(r + pulse * 6.0)

            # Círculo de advertencia externo
            pygame.draw.circle(surface, (180, 40, 30), center, current_r, 2)
            # Círculo interior
            pygame.draw.circle(surface, (80, 20, 15), center, int(r * 0.8))
            # Núcleo térmico
            pygame.draw.circle(surface, self.COLOR_HAZARD, center, int(r * 0.35))

    def render_phase1_portal(self, surface: pygame.Surface, checkpoint: Tuple[float, float, float]) -> None:
        """Dibuja el portal de paso de la Fase 1."""
        cx, cy, cr = checkpoint
        center = (int(cx), int(cy))
        pulse = (math.sin(self._pulse_timer * 3.0) + 1.0) * 0.5
        r_dyn = int(cr + pulse * 8.0)

        pygame.draw.circle(surface, (0, 120, 80), center, r_dyn, 3)
        pygame.draw.circle(surface, (0, 200, 130), center, int(cr * 0.6), 2)
        pygame.draw.circle(surface, self.COLOR_PORTAL, center, 12)

        label = self.font_small.render("PORTAL REACTOR", True, self.COLOR_PORTAL)
        surface.blit(label, (cx - label.get_width() // 2, cy - cr - 28))

    def render_phase2_nodes(self, surface: pygame.Surface, nodes: List[Dict[str, Any]]) -> None:
        """Dibuja los nodos de sincronización bimanual con barra de progreso circular."""
        for node in nodes:
            center = (int(node['x']), int(node['y']))
            r = int(node['radius'])
            charge_ratio = node['charge'] / node['required']

            # Anillo base
            pygame.draw.circle(surface, (30, 50, 80), center, r, 2)

            # Círculo de energía cargada
            if charge_ratio > 0.0:
                charged_r = max(5, int(r * charge_ratio))
                fill_color = (0, 230, 255) if node['id'] == 'alpha' else (190, 80, 255)
                pygame.draw.circle(surface, fill_color, center, charged_r, 0)

            node_name = f"NODO {node['id'].upper()} [{int(charge_ratio * 100)}%]"
            txt = self.font_small.render(node_name, True, self.COLOR_TEXT_PRIMARY)
            surface.blit(txt, (center[0] - txt.get_width() // 2, center[1] - r - 26))

    def render_phase3_core(
        self,
        surface: pygame.Surface,
        core_status: float,
        fuse_pos: Tuple[float, float]
    ) -> None:
        """Dibuja el reactor central y el fusible móvil arrastrable."""
        core_center = (1200, 540)
        pulse = (math.sin(self._pulse_timer * 5.0) + 1.0) * 0.5

        # Reactor Core
        core_r = int(80 + pulse * 10.0 * (1.0 + core_status))
        pygame.draw.circle(surface, (0, 100, 150), core_center, core_r, 3)
        pygame.draw.circle(surface, self.COLOR_CORE, core_center, int(40 + core_status * 25))

        # Fusible móvil
        fx, fy = int(fuse_pos[0]), int(fuse_pos[1])
        fuse_center = (fx, fy)

        # Línea de guía de conexión hacia el núcleo
        pygame.draw.line(surface, (40, 80, 120), fuse_center, core_center, 2)

        # Cápsula del fusible
        pygame.draw.circle(surface, (180, 150, 0), fuse_center, 45, 3)
        pygame.draw.circle(surface, self.COLOR_FUSE, fuse_center, 28)
        lbl = self.font_small.render("FUSIBLE", True, (10, 14, 23))
        surface.blit(lbl, (fx - lbl.get_width() // 2, fy - lbl.get_height() // 2))

    def render_hud(
        self,
        surface: pygame.Surface,
        mission_state: MissionState,
        gesture_data: HandGestureData,
        fps: float
    ) -> None:
        """Dibuja la interfaz de telemetría y mensajes superiores e inferiores."""
        # Barra superior traslúcida
        top_bar = pygame.Surface((self.width, 70), pygame.SRCALPHA)
        top_bar.fill((8, 12, 20, 210))
        surface.blit(top_bar, (0, 0))

        # 1. Título y Fase
        phase_lbl = self.font_medium.render(mission_state.phase_title, True, self.COLOR_ATTRACT)
        surface.blit(phase_lbl, (30, 20))

        # 2. Cronómetro de misión
        mins = int(mission_state.time_remaining) // 60
        secs = int(mission_state.time_remaining) % 60
        timer_str = f"TIEMPO: {mins:02d}:{secs:02d}"
        time_color = (255, 70, 70) if mission_state.time_remaining < 30.0 else self.COLOR_TEXT_PRIMARY
        timer_lbl = self.font_medium.render(timer_str, True, time_color)
        surface.blit(timer_lbl, (self.width // 2 - timer_lbl.get_width() // 2, 20))

        # 3. Integridad del Enjambre
        integrity_pct = int(mission_state.swarm_integrity * 100)
        integ_str = f"INTEGRIDAD: {integrity_pct}%"
        integ_color = (255, 80, 80) if integrity_pct < 40 else self.COLOR_PORTAL
        integ_lbl = self.font_medium.render(integ_str, True, integ_color)
        surface.blit(integ_lbl, (self.width - integ_lbl.get_width() - 30, 20))

        # Barra inferior para instrucciones y telemetría de manos
        bot_bar = pygame.Surface((self.width, 60), pygame.SRCALPHA)
        bot_bar.fill((8, 12, 20, 210))
        surface.blit(bot_bar, (0, self.height - 60))

        # Instrucción activa
        inst_lbl = self.font_medium.render(mission_state.phase_instruction, True, self.COLOR_TEXT_PRIMARY)
        surface.blit(inst_lbl, (30, self.height - 44))

        # Telemetría de gestos en la esquina inferior derecha
        if gesture_data.is_detected:
            g_name = gesture_data.gesture_type
            g_color = self.get_gesture_color(g_name)
            telemetry_str = f"MANOS: {gesture_data.hand_count} | GESTO: {g_name} | FPS: {int(fps)}"
        else:
            g_color = (255, 150, 40)
            telemetry_str = f"BUSCANDO MANOS... | FPS: {int(fps)}"

        tel_lbl = self.font_small.render(telemetry_str, True, g_color)
        surface.blit(tel_lbl, (self.width - tel_lbl.get_width() - 30, self.height - 40))

        # Overlays de Victoria / Derrota
        if mission_state.is_completed:
            self._render_overlay_banner(surface, "¡MISIÓN CUMPLIDA! REACTOR OPERATIVO", self.COLOR_PORTAL)
        elif mission_state.is_game_over:
            self._render_overlay_banner(surface, "SISTEMA COLAPSADO - REINTENTO", self.COLOR_HAZARD)

    def _render_overlay_banner(self, surface: pygame.Surface, text: str, color: Tuple[int, int, int]) -> None:
        """Banner central de fin de partida."""
        banner = pygame.Surface((self.width, 160), pygame.SRCALPHA)
        banner.fill((5, 8, 14, 230))
        surface.blit(banner, (0, self.height // 2 - 80))

        txt = self.font_large.render(text, True, color)
        surface.blit(txt, (self.width // 2 - txt.get_width() // 2, self.height // 2 - txt.get_height() // 2))

    def render_hand_reticles(self, surface: pygame.Surface, gesture_data: HandGestureData) -> None:
        """Dibuja cursores y retículas holográficas reactivas en pantalla para cada mano detectada."""
        if not gesture_data.is_detected or not gesture_data.hands:
            return

        pulse = (math.sin(self._pulse_timer * 6.0) + 1.0) * 0.5
        num_hands = len(gesture_data.hands)
        is_two_hands = num_hands > 1

        # Si hay dos manos, trazar enlace visual que muestra la división del enjambre
        if is_two_hands:
            p1 = (int(gesture_data.hands[0]['x'] * self.width), int(gesture_data.hands[0]['y'] * self.height))
            p2 = (int(gesture_data.hands[1]['x'] * self.width), int(gesture_data.hands[1]['y'] * self.height))
            pygame.draw.line(surface, (50, 90, 140), p1, p2, 1)

        for i, h in enumerate(gesture_data.hands):
            hx = int(h['x'] * self.width)
            hy = int(h['y'] * self.height)
            g_type = h['gesture']
            color = self.get_gesture_color(g_type)

            r_outer = int(32 + pulse * 6.0)
            r_inner = 8

            # Anillo exterior reactivo
            pygame.draw.circle(surface, color, (hx, hy), r_outer, 2)
            # Punto central
            pygame.draw.circle(surface, color, (hx, hy), r_inner, 0)

            # Cruz holográfica
            pygame.draw.line(surface, color, (hx - r_outer - 6, hy), (hx - r_inner - 2, hy), 2)
            pygame.draw.line(surface, color, (hx + r_inner + 2, hy), (hx + r_outer + 6, hy), 2)
            pygame.draw.line(surface, color, (hx, hy - r_outer - 6), (hx, hy - r_inner - 2), 2)
            pygame.draw.line(surface, color, (hx, hy + r_inner + 2), (hx, hy + r_outer + 6), 2)

            # Etiqueta clara del gesto
            if g_type == 'ATTRACT':
                tag = "1 DEDO [ATRAER]"
            elif g_type == 'DISPERSE':
                tag = "PALMA [DISPERSAR]"
            elif g_type == 'CONDENSE':
                tag = "PUNO [CONDENSAR]"
            else:
                tag = g_type

            if is_two_hands:
                tag = f"MANO {i+1} (50%): {tag}"

            lbl = self.font_small.render(tag, True, color)
            # Caja de fondo oscura para lectura fácil sobre partículas
            lbl_bg = pygame.Surface((lbl.get_width() + 10, lbl.get_height() + 4), pygame.SRCALPHA)
            lbl_bg.fill((8, 12, 22, 220))
            surface.blit(lbl_bg, (hx - lbl.get_width() // 2 - 5, hy + r_outer + 6))
            surface.blit(lbl, (hx - lbl.get_width() // 2, hy + r_outer + 8))

    def render_camera_pip(
        self,
        surface: pygame.Surface,
        preview_frame: Optional[np.ndarray],
        is_detected: bool,
        gesture_type: str,
        show_pip: bool = True
    ) -> None:
        """
        Dibuja el cuadro Picture-in-Picture (PiP) de la cámara en vivo con retroalimentación esquelética.
        Se posiciona de forma adaptativa según el ancho de la superficie dada.
        """
        if not show_pip:
            return

        pip_w, pip_h = 240, 180
        surf_w, surf_h = surface.get_size()
        pip_x = surf_w - pip_w - 30
        pip_y = 85

        border_color = self.get_gesture_color(gesture_type) if is_detected else (70, 90, 120)

        # 1. Marco exterior translúcido y fondo
        bg_surface = pygame.Surface((pip_w + 12, pip_h + 46), pygame.SRCALPHA)
        bg_surface.fill((8, 12, 22, 230))
        surface.blit(bg_surface, (pip_x - 6, pip_y - 20))

        # 2. Cabecera superior
        header_text = "SENSOR OPTICO // EN VIVO"
        hdr_lbl = self.font_small.render(header_text, True, (160, 190, 220))
        surface.blit(hdr_lbl, (pip_x, pip_y - 18))

        # 3. Contenido del video de la cámara
        if preview_frame is not None and preview_frame.ndim == 3:
            try:
                if preview_frame.shape[:2] == (pip_h, pip_w):
                    cam_surface = pygame.surfarray.make_surface(preview_frame.swapaxes(0, 1))
                else:
                    # Si tiene dimensiones distintas, convertir y redimensionar con Pygame
                    temp_surf = pygame.surfarray.make_surface(preview_frame.swapaxes(0, 1))
                    cam_surface = pygame.transform.scale(temp_surf, (pip_w, pip_h))
                surface.blit(cam_surface, (pip_x, pip_y + 4))
            except Exception:
                preview_frame = None

        if preview_frame is None or preview_frame.ndim != 3:
            grid_surf = pygame.Surface((pip_w, pip_h))
            grid_surf.fill((12, 16, 25))
            for gx in range(0, pip_w, 40):
                pygame.draw.line(grid_surf, (25, 35, 55), (gx, 0), (gx, pip_h), 1)
            for gy in range(0, pip_h, 30):
                pygame.draw.line(grid_surf, (25, 35, 55), (0, gy), (pip_w, gy), 1)
            surface.blit(grid_surf, (pip_x, pip_y + 4))

            no_cam_lbl = self.font_small.render("BUSCANDO CAMARA...", True, (255, 160, 40))
            surface.blit(no_cam_lbl, (pip_x + pip_w // 2 - no_cam_lbl.get_width() // 2, pip_y + 4 + pip_h // 2 - 10))

        # 4. Borde cibernético neón alrededor del video
        pygame.draw.rect(surface, border_color, (pip_x - 1, pip_y + 3, pip_w + 2, pip_h + 2), 2)

        # 5. Estado y clasificación del gesto debajo del video
        if is_detected:
            if gesture_type == 'ATTRACT':
                status_text = "● 1 DEDO: ATRAER"
            elif gesture_type == 'DISPERSE':
                status_text = "● PALMA: DISPERSAR"
            elif gesture_type == 'CONDENSE':
                status_text = "● PUNO: CONDENSAR"
            else:
                status_text = f"● MANO: {gesture_type}"
            status_lbl = self.font_small.render(status_text, True, border_color)
        else:
            status_text = "○ BUSCANDO MANOS..."
            status_lbl = self.font_small.render(status_text, True, (150, 150, 150))
        surface.blit(status_lbl, (pip_x, pip_y + pip_h + 8))

    def render_frame(
        self,
        surface: pygame.Surface,
        boid_positions: np.ndarray,
        boid_velocities: np.ndarray,
        mission_state: MissionState,
        gesture_data: HandGestureData,
        fps: float = 60.0,
        dt: float = 0.016
    ) -> None:
        """
        Renderiza el fotograma completo del juego.
        """
        self._pulse_timer += dt

        # 1. Aplicar estela bioluminiscente difuminando el cuadro anterior
        surface.blit(self.fade_surface, (0, 0))

        # 2. Renderizar elementos específicos de la fase
        if mission_state.phase == 1:
            self.render_phase1_portal(surface, (1680.0, 540.0, 150.0))
        elif mission_state.phase == 2:
            self.render_phase2_nodes(surface, mission_state.target_nodes)
        elif mission_state.phase == 3:
            self.render_phase3_core(surface, mission_state.core_status, mission_state.fuse_pos)

        # 3. Renderizar obstáculos activos
        if mission_state.active_obstacles:
            self.render_hazards(surface, mission_state.active_obstacles)

        # 4. Renderizar boids con color reactivo
        self.render_boids(surface, boid_positions, boid_velocities, gesture_data.gesture_type)

        # 5. Renderizar retículas holográficas de las manos (posiciones en pantalla y división 50/50)
        self.render_hand_reticles(surface, gesture_data)

        # 6. Renderizar HUD y telemetría
        self.render_hud(surface, mission_state, gesture_data, fps)

        # 7. Renderizar cuadro Picture-in-Picture de la cámara en vivo
        self.render_camera_pip(
            surface=surface,
            preview_frame=gesture_data.preview_frame,
            is_detected=gesture_data.is_detected,
            gesture_type=gesture_data.gesture_type,
            show_pip=self.show_camera_pip
        )

    def render_attract_mode(
        self,
        surface: pygame.Surface,
        boid_positions: np.ndarray,
        time_elapsed: float,
        preview_frame: Optional[np.ndarray] = None,
        is_detected: bool = False,
        gesture_type: str = 'NONE'
    ) -> None:
        """Modo reposo/demostración para atraer visitantes al tótem en AudacIA."""
        surface.blit(self.fade_surface, (0, 0))

        # Dibujar partículas con color cian suave
        for pos in boid_positions:
            pygame.draw.circle(surface, (0, 180, 220), (int(pos[0]), int(pos[1])), 4)

        # Título y llamada a la acción parpadeante
        alpha_pulse = (math.sin(time_elapsed * 3.0) + 1.0) * 0.5
        title = self.font_large.render("SWARMCORE // AUDACIA AI", True, self.COLOR_ATTRACT)
        prompt = self.font_medium.render(
            "ACÉRCATE Y LEVANTA TUS MANOS PARA COMENZAR",
            True,
            (int(255 * alpha_pulse), int(220 * alpha_pulse), int(100 * alpha_pulse))
        )

        surface.blit(title, (self.width // 2 - title.get_width() // 2, self.height // 2 - 60))
        surface.blit(prompt, (self.width // 2 - prompt.get_width() // 2, self.height // 2 + 10))

        # Mostrar también el cuadro de cámara en vivo durante el modo reposo
        self.render_camera_pip(
            surface=surface,
            preview_frame=preview_frame,
            is_detected=is_detected,
            gesture_type=gesture_type,
            show_pip=self.show_camera_pip
        )
