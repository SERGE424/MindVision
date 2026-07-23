# Configuration de l'API Hugging Face (GRATUIT)

## ⚠️ Note importante sur l'API gratuite

L'API Inference gratuite de Hugging Face a des **limitations importantes** :
- Modèles souvent en veille (temps de chargement 20-60 secondes)
- Disponibilité non garantie
- Quotas limités

**Recommandation : Utilisez le mode Local pour une expérience stable.**

---

## Étapes pour obtenir votre token gratuit :

1. Allez sur : https://huggingface.co/join
2. Créez un compte gratuit
3. Allez dans : https://huggingface.co/settings/tokens
4. Cliquez sur "New token"
5. Donnez un nom (ex: "generateur_images")
6. Sélectionnez "Read" comme type
7. Copiez le token généré

## Configuration :

1. Ouvrez le fichier `config.py`
2. Remplacez `"votre_token_ici"` par votre token
3. Exemple : `HUGGING_FACE_TOKEN = "hf_abcdefghijklmnopqrstuvwxyz123456"`

## Utilisation :

- **Mode Local (Recommandé)** : Utilise votre CPU (stable et fiable)
- **Mode API (Expérimental)** : Utilise les serveurs Hugging Face (instable)

## Modèle API actuel :

Le code utilise : `prompthero/openjourney` (plus petit et plus fiable)

Autres modèles à essayer si celui-ci ne fonctionne pas :
- `stabilityai/stable-diffusion-2-1` (peut être indisponible)
- `runwayml/stable-diffusion-v1-5` (peut être indisponible)
- `CompVis/stable-diffusion-v1-4` (ancien mais stable)

Pour changer de modèle, modifiez la ligne 29 dans `Generateur.py` :
```python
self.api_url = "https://api-inference.huggingface.co/models/NOM_DU_MODELE"
```

## Installation de la dépendance :

```bash
pip install requests
```

## Erreurs courantes :

- **Erreur 410** : Le modèle n'est plus disponible → Utilisez le mode Local
- **Erreur 503** : Le modèle se charge → Attendez 30-60 secondes et réessayez
- **Erreur 401** : Token invalide → Vérifiez votre token dans config.py
- **Token manquant** : Configurez votre token dans config.py

## Alternative recommandée :

Si l'API ne fonctionne pas, le **mode Local reste la meilleure option** :
- Stable et fiable
- Pas de dépendance internet
- Pas de quotas
- Fonctionne même sur CPU (bien que lent)
