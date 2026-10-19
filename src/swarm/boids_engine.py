from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class BoidsEngine:
    """
    Motor físico y vectorial de simulación de enjambre (Swarm/Boids) en 2.5D.
    Implementa reglas de Reynolds (cohesión, separación, alineación),
    campos de potencial para evasión de obstáculos y fuerzas de control
    guiadas por visión artificial (atracción, dispersión y condensación).
    Optimizado en NumPy para ejecución en <8 ms sobre Raspberry Pi 4.
    """

    def __init__(
        self,
        num_boids: int = 300,
        bounds: Tuple[float, float] = (1920.0, 1080.0),
        max_speed: float = 400.0,
        min_speed: float = 60.0,
        max_force: float = 800.0,
        perception_radius: float = 120.0,
        separation_radius: float = 40.0
    ):
        self.num_boids = num_boids
        self.width, self.height = bounds
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_force = max_force
        self.perception_radius = perception_radius
        self.separation_radius = separation_radius

        # Pesos de comportamiento
        self.w_sep = 1.6
        self.w_ali = 0.9
        self.w_coh = 1.1
        self.w_attract = 2.2
        self.w_disperse = 4.5
        self.w_condense = 6.0
        self.w_obs = 6.0

        self.obstacles: List[Tuple[float, float, float]] = []

        # Inicialización de posiciones y velocidades en float32 para aceleración SIMD y L2 cache
        self.positions = np.zeros((self.num_boids, 2), dtype=np.float32)
        self.velocities = np.zeros((self.num_boids, 2), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Reinicia posiciones aleatorias dentro del área y asigna velocidades iniciales."""
        rng = np.random.default_rng()
        margin = 100.0
        self.positions[:, 0] = rng.uniform(margin, self.width - margin, size=self.num_boids).astype(np.float32)
        self.positions[:, 1] = rng.uniform(margin, self.height - margin, size=self.num_boids).astype(np.float32)

        angles = rng.uniform(0, 2 * np.pi, size=self.num_boids).astype(np.float32)
        speeds = rng.uniform(self.min_speed, self.max_speed * 0.5, size=self.num_boids).astype(np.float32)
        self.velocities[:, 0] = np.cos(angles) * speeds
        self.velocities[:, 1] = np.sin(angles) * speeds

    def get_positions(self) -> np.ndarray:
        return self.positions

    def get_velocities(self) -> np.ndarray:
        return self.velocities

    def set_obstacles(self, obstacles: List[Tuple[float, float, float]]) -> None:
        """Establece lista de obstáculos circulares: [(x, y, radio)]."""
        self.obstacles = list(obstacles)

    def _compute_reynolds_forces(self) -> np.ndarray:
        """Cálculo vectorizado de separación, alineación y cohesión optimizado en 2D float32."""
        forces = np.zeros_like(self.positions)

        # Matrices de diferencia 2D (N, N) en lugar de tensores 3D para residir en L2 cache
        px = self.positions[:, 0:1]
        py = self.positions[:, 1:2]
        dx = px.T - px  # shape (N, N)
        dy = py.T - py  # shape (N, N)
        dist_sq = dx * dx + dy * dy

        # 1. Separación (usando radio cuadrado)
        r_sep_sq = self.separation_radius * self.separation_radius
        sep_mask = (dist_sq < r_sep_sq) & (dist_sq > 1e-4)
        inv_dist_sq = np.where(sep_mask, 1.0 / (dist_sq + 1e-4), 0.0).astype(np.float32)

        sep_x = -np.sum(dx * inv_dist_sq, axis=1)
        sep_y = -np.sum(dy * inv_dist_sq, axis=1)
        forces[:, 0] += sep_x * (self.w_sep * 150.0)
        forces[:, 1] += sep_y * (self.w_sep * 150.0)

        # 2. Cohesión y Alineación (usando radio de percepción cuadrado)
        r_perc_sq = self.perception_radius * self.perception_radius
        neigh_mask = (dist_sq < r_perc_sq) & (dist_sq > 1e-4)
        neigh_count = np.sum(neigh_mask, axis=1, keepdims=True)

        has_neighbors = (neigh_count[:, 0] > 0)
        if np.any(has_neighbors):
            mask_f32 = neigh_mask.astype(np.float32)
            # Cohesión
            sum_pos_x = np.dot(mask_f32, self.positions[:, 0])
            sum_pos_y = np.dot(mask_f32, self.positions[:, 1])
            counts = np.maximum(neigh_count[:, 0], 1.0)
            mean_pos_x = sum_pos_x / counts
            mean_pos_y = sum_pos_y / counts
            forces[has_neighbors, 0] += (mean_pos_x[has_neighbors] - self.positions[has_neighbors, 0]) * (self.w_coh * 1.5)
            forces[has_neighbors, 1] += (mean_pos_y[has_neighbors] - self.positions[has_neighbors, 1]) * (self.w_coh * 1.5)

            # Alineación
            sum_vel_x = np.dot(mask_f32, self.velocities[:, 0])
            sum_vel_y = np.dot(mask_f32, self.velocities[:, 1])
            mean_vel_x = sum_vel_x / counts
            mean_vel_y = sum_vel_y / counts
            forces[has_neighbors, 0] += (mean_vel_x[has_neighbors] - self.velocities[has_neighbors, 0]) * (self.w_ali * 1.2)
            forces[has_neighbors, 1] += (mean_vel_y[has_neighbors] - self.velocities[has_neighbors, 1]) * (self.w_ali * 1.2)

        return forces

    def _compute_hand_forces(self, hand_targets: List[Tuple[float, float, str]]) -> np.ndarray:
        """Cálculo de fuerzas ejercidas por las manos del usuario."""
        forces = np.zeros_like(self.positions)
        if not hand_targets:
            return forces

        for target in hand_targets:
            hx, hy, gesture = target
            hand_pos = np.array([hx, hy], dtype=np.float64)

            diff = hand_pos - self.positions  # Vector hacia la mano
            dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-5
            direction = diff / dist

            if gesture == 'ATTRACT':
                # Atracción suave proporcional a distancia, con amortiguamiento cercano
                attract_mag = np.clip(dist * 1.5, 50.0, 500.0)
                forces += direction * attract_mag * self.w_attract

            elif gesture == 'DISPERSE':
                # Repulsión radial fuerte dentro de un radio de influencia
                influence_r = 450.0
                in_range = (dist < influence_r).astype(np.float64)
                repel_mag = (influence_r - dist) / influence_r * 800.0
                forces -= direction * repel_mag * in_range * self.w_disperse

            elif gesture == 'CONDENSE':
                # Máxima atracción concentrada hacia la mano para formar masa densa
                condense_mag = np.clip(dist * 3.0, 150.0, 1000.0)
                forces += direction * condense_mag * self.w_condense

        return forces

    def _compute_obstacle_forces(self) -> np.ndarray:
        """Fuerzas repulsivas de campos de potencial artificial alrededor de obstáculos."""
        forces = np.zeros_like(self.positions)
        if not self.obstacles:
            return forces

        for ox, oy, radius in self.obstacles:
            obs_pos = np.array([ox, oy], dtype=np.float64)
            diff = self.positions - obs_pos  # vector desde obstáculo hacia boids
            dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-5
            direction = diff / dist

            influence_r = radius + 80.0
            in_range = (dist < influence_r).astype(np.float64)
            repel_mag = (influence_r - dist) / influence_r * 900.0
            forces += direction * repel_mag * in_range * self.w_obs

        return forces

    def _compute_boundary_forces(self) -> np.ndarray:
        """Fuerzas de contención elásticas para mantener boids dentro de la pantalla."""
        forces = np.zeros_like(self.positions)
        margin = 80.0
        k_repel = 600.0

        # Borde izquierdo y derecho
        left_mask = self.positions[:, 0] < margin
        right_mask = self.positions[:, 0] > (self.width - margin)
        forces[left_mask, 0] += (margin - self.positions[left_mask, 0]) * k_repel / margin
        forces[right_mask, 0] -= (self.positions[right_mask, 0] - (self.width - margin)) * k_repel / margin

        # Borde superior e inferior
        top_mask = self.positions[:, 1] < margin
        bottom_mask = self.positions[:, 1] > (self.height - margin)
        forces[top_mask, 1] += (margin - self.positions[top_mask, 1]) * k_repel / margin
        forces[bottom_mask, 1] -= (self.positions[bottom_mask, 1] - (self.height - margin)) * k_repel / margin

        return forces

    def step(self, dt: float, hand_targets: Optional[List[Tuple[float, float, str]]] = None) -> np.ndarray:
        """
        Avanza la simulación física en un intervalo de tiempo dt (segundos).
        """
        if dt <= 0:
            return self.positions

        if hand_targets is None:
            hand_targets = []

        total_forces = np.zeros_like(self.positions)
        total_forces += self._compute_reynolds_forces()
        total_forces += self._compute_hand_forces(hand_targets)
        total_forces += self._compute_obstacle_forces()
        total_forces += self._compute_boundary_forces()

        # Limitar magnitud de aceleración máxima
        force_mags = np.linalg.norm(total_forces, axis=1, keepdims=True) + 1e-6
        clamped_forces = np.where(
            force_mags > self.max_force,
            total_forces / force_mags * self.max_force,
            total_forces
        )

        # Integración de Euler semi-implícita
        self.velocities += clamped_forces * dt

        # Frenado / amortiguamiento si está en modo CONDENSE para estabilizar la masa
        has_condense = any(t[2] == 'CONDENSE' for t in hand_targets)
        if has_condense:
            self.velocities *= 0.94
        else:
            self.velocities *= 0.985  # fricción aerodinámica ligera

        # Limitar velocidades
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True) + 1e-6
        too_fast = speeds > self.max_speed
        self.velocities = np.where(too_fast, self.velocities / speeds * self.max_speed, self.velocities)

        # Si no hay manos y caen por debajo de min_speed, restaurar velocidad suave
        if not hand_targets:
            too_slow = speeds < self.min_speed
            self.velocities = np.where(too_slow, self.velocities / speeds * self.min_speed, self.velocities)

        self.positions += self.velocities * dt

        # Clamping estricto a las dimensiones de la pantalla (evita que cualquier boid escape)
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0.0, self.width)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0.0, self.height)

        return self.positions

    def get_state(self) -> Dict[str, Any]:
        """Retorna telemetría del estado de la simulación."""
        speeds = np.linalg.norm(self.velocities, axis=1)
        center_of_mass = np.mean(self.positions, axis=0)
        return {
            'num_boids': self.num_boids,
            'avg_speed': float(np.mean(speeds)),
            'center_of_mass': (float(center_of_mass[0]), float(center_of_mass[1])),
            'active_obstacles': len(self.obstacles)
        }
