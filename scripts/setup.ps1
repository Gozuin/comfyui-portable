$ErrorActionPreference = "Stop"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction Stop
}

& $python.Source "$PSScriptRoot\comfyui_portable.py" setup @args
exit $LASTEXITCODE
