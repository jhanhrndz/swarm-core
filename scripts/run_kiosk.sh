#!/usr/bin/env bash
# ==============================================================================
# SwarmCore - Script de Auto-Arranque para Tótem Kiosk en Raspberry Pi OS
# Centro de Investigación AudacIA
# ==============================================================================

set -e

# Directorio raíz del proyecto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "Iniciando SwarmCore: Misión Núcleo // AudacIA AI & Robotics"
echo "=========================================================="

# 1. Configuración de pantalla X11 / Wayland para Kiosk
export DISPLAY=:0
export SDL_VIDEODRIVER=x11
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

# 2. Ocultar cursor del mouse tras 2 segundos de inactividad si unclutter está instalado
if command -v unclutter >/dev/null 2>&1; then
    unclutter -idle 2 -root &
fi

# 3. Activar entorno virtual
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -d "$SCRIPT_DIR/../.venv" ]; then
    source "$SCRIPT_DIR/../.venv/bin/activate"
else
    echo "Advertencia: No se encontró .venv, usando Python del sistema."
fi

# 4. Ejecutar aplicación en pantalla completa
exec python3 -m src.app.main --fullscreen --fps 60 --boids 350 "$@"
