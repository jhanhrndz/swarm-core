<#
==============================================================================
SwarmCore - Script de Ejecución Local para Laptop (Windows PowerShell)
Centro de Investigación AudacIA
==============================================================================
#>

$ScriptDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ScriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Iniciando SwarmCore // Modo Laptop Local" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Warning "No se encontró el entorno virtual en $VenvPython. Usando python del PATH."
    $VenvPython = "python"
}

# Ejecutar con paso de argumentos opcionales (ej: -mock-camera, -width, etc.)
& $VenvPython -m src.app.main @args
