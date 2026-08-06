$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$Root\src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })
python -m l4stack.cli --config-dir "$Root\config" run @args
