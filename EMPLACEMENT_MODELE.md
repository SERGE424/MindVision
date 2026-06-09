# Emplacement du modèle SD-Turbo

## 📁 Où est stocké le modèle ?

### Windows :
```
C:\Users\<VotreNom>\.cache\huggingface\hub\models--stabilityai--sd-turbo
```

### Linux/Mac :
```
~/.cache/huggingface/hub/models--stabilityai--sd-turbo
```

## 📊 Taille du modèle

- **SD-Turbo complet** : ~2.1 GB
- Contient :
  - UNet : ~1.7 GB
  - VAE : ~300 MB
  - Text Encoder : ~100 MB

## 🔍 Vérifier l'emplacement exact

Ajoutez ce code dans votre application :

```python
from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()
for repo in cache_info.repos:
    if "sd-turbo" in repo.repo_id:
        print(f"Modèle : {repo.repo_id}")
        print(f"Emplacement : {repo.repo_path}")
        print(f"Taille : {repo.size_on_disk / (1024**3):.2f} GB")
```

## 🗑️ Supprimer le cache (libérer de l'espace)

### Méthode 1 : Supprimer manuellement
```bash
# Windows
rmdir /s "C:\Users\<VotreNom>\.cache\huggingface"

# Linux/Mac
rm -rf ~/.cache/huggingface
```

### Méthode 2 : Avec Python
```python
from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()
to_delete = cache_info.delete_revisions("stabilityai/sd-turbo")
print(f"Espace libéré : {to_delete.expected_freed_size / (1024**3):.2f} GB")
to_delete.execute()
```

## 📥 Télécharger le modèle manuellement

Si vous voulez pré-télécharger le modèle :

```python
from diffusers import AutoPipelineForText2Image

# Télécharge et met en cache
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    cache_dir="./mon_dossier_modeles"  # Emplacement personnalisé
)
```

## 🔄 Changer l'emplacement du cache

### Méthode 1 : Variable d'environnement
```bash
# Windows (PowerShell)
$env:HF_HOME = "D:\mes_modeles"

# Linux/Mac
export HF_HOME="/chemin/vers/mes_modeles"
```

### Méthode 2 : Dans le code
```python
import os
os.environ['HF_HOME'] = 'D:/mes_modeles'

# Puis charger le modèle
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo")
```

## 📦 Structure du cache

```
.cache/huggingface/hub/
└── models--stabilityai--sd-turbo/
    ├── blobs/              # Fichiers du modèle
    ├── refs/               # Références de version
    └── snapshots/          # Versions du modèle
        └── <hash>/
            ├── model_index.json
            ├── unet/
            ├── vae/
            ├── text_encoder/
            └── scheduler/
```

## 💡 Conseils

### 1. Vérifier si le modèle est déjà téléchargé
```python
import os
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_dir = cache_dir / "models--stabilityai--sd-turbo"

if model_dir.exists():
    print(f"✅ Modèle déjà téléchargé : {model_dir}")
    size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
    print(f"Taille : {size / (1024**3):.2f} GB")
else:
    print("❌ Modèle non téléchargé")
```

### 2. Utiliser un modèle local (hors ligne)
```python
# Une fois téléchargé, fonctionne hors ligne
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    local_files_only=True  # Ne télécharge pas, utilise le cache
)
```

### 3. Partager le modèle entre plusieurs projets
Le cache Hugging Face est partagé automatiquement entre tous vos projets Python !

### 4. Sauvegarder le modèle ailleurs
```python
# Sauvegarder dans un dossier spécifique
pipe.save_pretrained("./mon_modele_sd_turbo")

# Charger depuis ce dossier
pipe = AutoPipelineForText2Image.from_pretrained("./mon_modele_sd_turbo")
```

## 🚀 Optimiser l'espace disque

Si vous manquez d'espace :

1. **Supprimer les anciennes versions**
```python
from huggingface_hub import scan_cache_dir

cache = scan_cache_dir()
# Garde seulement la dernière version
for repo in cache.repos:
    if len(repo.revisions) > 1:
        old_revisions = list(repo.revisions)[:-1]
        cache.delete_revisions(*[r.commit_hash for r in old_revisions]).execute()
```

2. **Utiliser des modèles plus petits**
- `sd-turbo` : 2.1 GB
- `tiny-sd` : 500 MB (qualité inférieure)

3. **Utiliser float16 au lieu de float32**
```python
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16  # Divise la taille par 2
)
```

## 🔍 Commande rapide pour trouver le modèle

### Windows (PowerShell)
```powershell
Get-ChildItem -Path "$env:USERPROFILE\.cache\huggingface" -Recurse -Filter "*sd-turbo*"
```

### Linux/Mac
```bash
find ~/.cache/huggingface -name "*sd-turbo*"
```

## ✅ Résumé

- **Emplacement** : `~/.cache/huggingface/hub/models--stabilityai--sd-turbo`
- **Taille** : ~2.1 GB
- **Partagé** : Entre tous vos projets Python
- **Hors ligne** : Fonctionne une fois téléchargé
- **Personnalisable** : Via `HF_HOME` ou `cache_dir`
