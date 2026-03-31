# activate.ps1
.\venv\Scripts\Activate.ps1
# activate.ps1 — correr no terminal integrado do VS Code com: . .\activate.ps1
# O ponto antes é obrigatório (dot-sourcing) para as vars ficarem no terminal.
# NÃO correr via Code Runner ("Run Code") — usa processo filho e perde as vars.

$ROOT = $PSScriptRoot   # pasta onde este script está, independente de onde o terminal está

# 1. Activar venv
$venvActivate = Join-Path $ROOT "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "venv activado." -ForegroundColor Cyan
} else {
    Write-Host "AVISO: venv nao encontrado em $venvActivate" -ForegroundColor Yellow
    Write-Host "Cria com: python -m venv venv" -ForegroundColor Yellow
}

# 2. Injectar variaveis do .env
$envFile = Join-Path $ROOT ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.+)$') {
            $key   = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }
    Write-Host ".env carregado." -ForegroundColor Cyan
} else {
    Write-Host "AVISO: .env nao encontrado em $envFile" -ForegroundColor Yellow
    Write-Host "Copia .env.example para .env e preenche as keys." -ForegroundColor Yellow
}

Write-Host "Litterman MVP pronto." -ForegroundColor Green


