from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
from src.vision.gesture_tracker import HandGestureData

@dataclass
class MissionState:
    phase: int = 1
    time_remaining: float = 180.0
    score: int = 0
    is_completed: bool = False
    is_game_over: bool = False
    active_obstacles: List[Tuple[float, float, float]] = field(default_factory=list)
    target_nodes: List[Dict[str, Any]] = field(default_factory=list)
    core_status: float = 0.0
    swarm_integrity: float = 1.0
    fuse_pos: Tuple[float, float] = (500.0, 540.0)
    phase_title: str = "FASE 1: NAVEGACIÓN Y EVASIÓN"
    phase_instruction: str = "Apunta con 1 DEDO para guiar el enjambre hacia el portal | Abre la palma para esquivar calor"

class MissionDirector:
    """
    Director de misión y máquina de estados del juego 'SwarmCore'.
    Controla la progresión de las 3 fases, cronómetro, puntuación,
    daño de obstáculos y condiciones de victoria/derrota.
    """

    def __init__(
        self,
        total_time: float = 180.0,
        bounds: Tuple[float, float] = (1920.0, 1080.0)
    ):
        self.total_time = total_time
        self.time_remaining = total_time
        self.width, self.height = bounds
        self.phase = 1
        self.score = 0
        self.is_completed = False
        self.is_game_over = False
        self.swarm_integrity = 1.0

        # Parámetros Fase 1 (Portal Checkpoint y Obstáculos de Calor)
        self.phase1_checkpoint = (1680.0, 540.0, 150.0)  # (x, y, radio)
        self.phase1_obstacles: List[Tuple[float, float, float]] = [
            (700.0, 360.0, 75.0),
            (1050.0, 680.0, 90.0),
            (1250.0, 320.0, 80.0)
        ]

        # Parámetros Fase 2 (Nodos de Sincronización Bimanual)
        self.phase2_nodes: List[Dict[str, Any]] = [
            {'id': 'alpha', 'x': 1500.0, 'y': 280.0, 'radius': 110.0, 'charge': 0.0, 'required': 2.0},
            {'id': 'beta', 'x': 1500.0, 'y': 800.0, 'radius': 110.0, 'charge': 0.0, 'required': 2.0}
        ]
        self.phase2_obstacles: List[Tuple[float, float, float]] = [
            (1050.0, 540.0, 110.0)  # Barrera central divisoria
        ]

        # Parámetros Fase 3 (Inserción del Núcleo)
        self.core_position = np.array([1200.0, 540.0], dtype=np.float64)
        self.fuse_start_x = 500.0
        self.fuse_position = np.array([self.fuse_start_x, 540.0], dtype=np.float64)
        self.core_status = 0.0
        self.phase3_obstacles: List[Tuple[float, float, float]] = [
            (850.0, 300.0, 70.0),
            (850.0, 780.0, 70.0)
        ]

    def reset(self) -> None:
        """Reinicia el director de misión a su estado inicial."""
        self.time_remaining = self.total_time
        self.phase = 1
        self.score = 0
        self.is_completed = False
        self.is_game_over = False
        self.swarm_integrity = 1.0
        self.core_status = 0.0
        self.fuse_position = np.array([self.fuse_start_x, 540.0], dtype=np.float64)
        for node in self.phase2_nodes:
            node['charge'] = 0.0

    def get_obstacles(self) -> List[Tuple[float, float, float]]:
        """Retorna la lista de obstáculos activos según la fase actual."""
        if self.phase == 1:
            return self.phase1_obstacles
        elif self.phase == 2:
            return self.phase2_obstacles
        elif self.phase == 3:
            return self.phase3_obstacles
        return []

    def _update_phase1(self, dt: float, boid_positions: np.ndarray) -> None:
        # 1. Comprobar colisión con obstáculos de calor (daño a la integridad)
        num_boids = len(boid_positions)
        if num_boids > 0:
            for ox, oy, r in self.phase1_obstacles:
                diff = boid_positions - np.array([ox, oy])
                dists = np.linalg.norm(diff, axis=1)
                inside_hazard = np.sum(dists < r)
                if inside_hazard > 0:
                    damage = (inside_hazard / num_boids) * 0.4 * dt
                    self.swarm_integrity = max(0.0, self.swarm_integrity - damage)

        # 2. Comprobar avance al checkpoint
        cx, cy, cr = self.phase1_checkpoint
        diff = boid_positions - np.array([cx, cy])
        dists = np.linalg.norm(diff, axis=1)
        in_checkpoint = np.sum(dists < cr)

        # Si al menos el 60% del enjambre cruza el portal
        threshold = max(20, int(num_boids * 0.6))
        if in_checkpoint >= threshold:
            self.phase = 2
            self.score += 1000

    def _update_phase2(self, dt: float, boid_positions: np.ndarray, gesture_data: HandGestureData) -> None:
        node_a = self.phase2_nodes[0]
        node_b = self.phase2_nodes[1]

        diff_a = boid_positions - np.array([node_a['x'], node_a['y']])
        diff_b = boid_positions - np.array([node_b['x'], node_b['y']])

        dists_a = np.linalg.norm(diff_a, axis=1)
        dists_b = np.linalg.norm(diff_b, axis=1)

        count_a = np.sum(dists_a < node_a['radius'])
        count_b = np.sum(dists_b < node_b['radius'])

        # Carga activa si ambos nodos tienen presencia simultánea de microrobots
        min_required_boids = 20
        if count_a >= min_required_boids and count_b >= min_required_boids:
            node_a['charge'] = min(node_a['required'], node_a['charge'] + dt)
            node_b['charge'] = min(node_b['required'], node_b['charge'] + dt)
        else:
            # Pérdida lenta de carga si se abandona la posición
            node_a['charge'] = max(0.0, node_a['charge'] - dt * 0.4)
            node_b['charge'] = max(0.0, node_b['charge'] - dt * 0.4)

        if node_a['charge'] >= node_a['required'] and node_b['charge'] >= node_b['required']:
            self.phase = 3
            self.score += 2000

    def _update_phase3(self, dt: float, boid_positions: np.ndarray, gesture_data: HandGestureData) -> None:
        # Solo el gesto CONDENSE (puño cerrado) tiene densidad de masa suficiente para empujar el fusible
        if gesture_data.gesture_type == 'CONDENSE':
            diff_fuse = boid_positions - self.fuse_position
            dists = np.linalg.norm(diff_fuse, axis=1)
            boids_near_fuse = np.sum(dists < 90.0)

            if boids_near_fuse >= 20:
                # Vector hacia el núcleo central
                to_core = self.core_position - self.fuse_position
                dist_to_core = np.linalg.norm(to_core)

                if dist_to_core > 1e-3:
                    direction = to_core / dist_to_core
                    push_speed = 180.0  # px/s
                    self.fuse_position += direction * push_speed * dt

                    total_dist = np.linalg.norm(self.core_position - np.array([self.fuse_start_x, 540.0]))
                    self.core_status = float(np.clip(1.0 - (dist_to_core / total_dist), 0.0, 1.0))

                    # Inserción completada
                    if dist_to_core < 35.0:
                        self.core_status = 1.0
                        self.is_completed = True
                        self.score += 5000 + int(self.time_remaining * 25)

    def update(
        self,
        dt: float,
        boid_positions: np.ndarray,
        gesture_data: Optional[HandGestureData] = None
    ) -> MissionState:
        """
        Actualiza el estado de la misión y evalúa transiciones de juego.
        """
        if gesture_data is None:
            gesture_data = HandGestureData()

        if self.is_completed or self.is_game_over:
            return self.get_state()

        # Actualización de tiempo
        self.time_remaining -= dt
        if self.time_remaining <= 0.0:
            self.time_remaining = 0.0
            self.is_game_over = True
            return self.get_state()

        # Condición de derrota por integridad crítica
        if self.swarm_integrity <= 0.15:
            self.is_game_over = True
            return self.get_state()

        # Lógica de fases
        if self.phase == 1:
            self._update_phase1(dt, boid_positions)
        elif self.phase == 2:
            self._update_phase2(dt, boid_positions, gesture_data)
        elif self.phase == 3:
            self._update_phase3(dt, boid_positions, gesture_data)

        return self.get_state()

    def get_state(self) -> MissionState:
        """Genera instantánea inmutable del estado actual."""
        titles = {
            1: "FASE 1: NAVEGACIÓN Y EVASIÓN",
            2: "FASE 2: SINCRONIZACIÓN BIMANUAL",
            3: "FASE 3: INSERCIÓN DEL NÚCLEO",
            4: "MISIÓN CUMPLIDA",
            5: "MISIÓN FALLIDA"
        }
        instructions = {
            1: "Apunta con 1 DEDO para guiar el enjambre hacia el portal | Abre la palma para esquivar calor",
            2: "Usa 2 MANOS (1 dedo en cada una) para dividir el enjambre al 50% y alimentar ambos nodos",
            3: "Cierra el PUÑO (CONDENSAR) sobre el fusible para concentrar el enjambre y empujarlo al reactor",
            4: "¡Reactor reactivado exitosamente! Excelente trabajo en equipo",
            5: "El reactor colapsó. Vuelve a intentarlo para salvar AudacIA"
        }

        current_phase = self.phase
        if self.is_completed:
            current_phase = 4
        elif self.is_game_over:
            current_phase = 5

        return MissionState(
            phase=self.phase,
            time_remaining=self.time_remaining,
            score=self.score,
            is_completed=self.is_completed,
            is_game_over=self.is_game_over,
            active_obstacles=self.get_obstacles(),
            target_nodes=self.phase2_nodes if self.phase == 2 else [],
            core_status=self.core_status,
            swarm_integrity=self.swarm_integrity,
            fuse_pos=(float(self.fuse_position[0]), float(self.fuse_position[1])),
            phase_title=titles.get(current_phase, ""),
            phase_instruction=instructions.get(current_phase, "")
        )
