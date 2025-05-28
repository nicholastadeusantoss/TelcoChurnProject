# Defina o diretório raiz do projeto
$root = "telco_churn_project"

# Lista de pastas a criar
$dirs = @(
    "$root/src/data",
    "$root/src/features",
    "$root/src/models",
    "$root/src/utils",
    "$root/tests",
    "$root/notebooks",
    "$root/models",
    "$root/configs",
    "$root/scripts"
)

# Cria as pastas
foreach ($dir in $dirs) {
    if (-Not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Criada pasta: $dir"
    } else {
        Write-Host "Pasta já existe: $dir"
    }
}

# Lista de arquivos __init__.py
$initFiles = @(
    "$root/src/__init__.py",
    "$root/src/data/__init__.py",
    "$root/src/models/__init__.py",
    "$root/src/utils/__init__.py"
)

# Lista de arquivos comuns
$files = @(
    "$root/src/data/load_and_clean_data.py",
    "$root/src/models/train_models.py",
    "$root/src/utils/helpers.py",
    "$root/scripts/main.py",
    "$root/requirements.txt",
    "$root/setup.py",
    "$root/README.md",
    "$root/Dockerfile"
)

# Cria os arquivos
foreach ($file in $initFiles + $files) {
    if (-Not (Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "Criado arquivo: $file"
    } else {
        Write-Host "Arquivo já existe: $file"
    }
}