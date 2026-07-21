# Guide : Utiliser une image comme prompt

## 📌 Nouvelle fonctionnalité

Vous pouvez maintenant utiliser une **image comme prompt** dans l'onglet "Assistant Complet" ! Cette fonctionnalité permet de générer de nouvelles images basées sur une image de référence.

## 🎯 Comment l'utiliser

### Étape 1 : Charger une image prompt

Dans l'onglet **Assistant Complet**, dans le panneau "📝 Paramètres" :

1. Cliquez sur le bouton **"🖼️ Charger Image Prompt"**
2. Sélectionnez l'image que vous souhaitez utiliser comme référence
3. Une miniature de l'image s'affiche dans la zone prévue

### Étape 2 : Ajouter un prompt textuel (optionnel)

Vous pouvez combiner :
- **Image seule** : l'IA génère une image basée sur l'image de référence
- **Image + texte** : l'IA combine l'image de référence avec votre description textuelle

Par exemple :
```
Image : photo d'un chat
Texte : "in a cyberpunk style with neon lights"
```

### Étape 3 : Générer

1. Sélectionnez le mode **"Hugging Face / Flux"** (recommandé pour l'image-to-image)
2. Cliquez sur **"🚀 Créer"**
3. L'IA optimise votre prompt et génère l'image

### Étape 4 : Effacer l'image prompt

Pour revenir au mode texte uniquement :
- Cliquez sur le bouton **"❌"** à côté du bouton "Charger Image Prompt"

## 🔧 Modes compatibles

| Mode | Support Image Prompt |
|------|---------------------|
| **Hugging Face / Flux** | ✅ Oui (recommandé) |
| Pollinations.ai | ⚠️ Partiel |
| Replicate (FLUX-Schnell) | ⚠️ À tester |
| AWS Bedrock | ❌ Non (pour l'instant) |
| Local CPU (SD-Turbo) | ❌ Non (pour l'instant) |

## 💡 Cas d'usage

### 1. Variation de style
- **Image** : votre photo
- **Texte** : "in watercolor painting style"

### 2. Transformation artistique
- **Image** : photo de paysage
- **Texte** : "as a fantasy landscape with dragons"

### 3. Génération basée sur référence
- **Image** : croquis ou dessin
- **Texte** : "high quality photorealistic render"

### 4. Image seule
- **Image** : n'importe quelle image
- **Texte** : (laisser vide)
- Résultat : l'IA génère une variation de l'image

## ⚙️ Configuration avancée

### Modèle Hugging Face

Par défaut, le modèle utilisé est `black-forest-labs/FLUX.1-schnell`.

Pour utiliser un autre modèle compatible avec image-to-image, modifiez dans `config.py` :

```python
HUGGING_FACE_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
# ou tout autre modèle compatible
```

### Format de l'image

L'image est automatiquement :
- Convertie en PNG
- Encodée en base64
- Envoyée à l'API Hugging Face

L'API détermine la meilleure façon d'utiliser l'image selon le modèle.

## 🐛 Dépannage

### "Erreur Hugging Face API"
- Vérifiez que le modèle supporte l'image-to-image
- Certains modèles FLUX nécessitent des paramètres spécifiques
- Essayez avec une image plus petite (< 2 Mo)

### "L'image ne s'affiche pas"
- Formats supportés : PNG, JPG, JPEG, BMP, GIF
- Vérifiez que le fichier n'est pas corrompu

### "Le résultat ne ressemble pas à l'image"
- Ajoutez plus de détails dans le prompt textuel
- Essayez d'augmenter le "strength" (dans une future version)
- Certains modèles interprètent différemment l'image

## 📝 Notes techniques

### Architecture

1. **Interface** : Bouton + Label pour afficher la miniature
2. **Stockage** : `self.assistant_image_prompt` (PIL Image)
3. **Encodage** : Base64 pour l'envoi à l'API
4. **API** : Le payload JSON inclut le champ `"image"` avec l'image encodée

### Code ajouté

- Variables : `assistant_image_prompt`, `assistant_image_prompt_path`
- Méthodes : `charger_image_prompt()`, `effacer_image_prompt()`
- Modification : `_generer_huggingface()` accepte maintenant `input_image`
- Logique : `_assistant_thread()` passe l'image au générateur

## 🚀 Futures améliorations

- [ ] Support image-to-image pour les autres modes (Local, AWS, Replicate)
- [ ] Paramètre "strength" pour contrôler l'influence de l'image
- [ ] Prévisualisation de l'image avant génération
- [ ] Historique des images prompts utilisées
- [ ] Drag & drop pour charger l'image
- [ ] Support de plusieurs images en entrée

---

**Bon usage !** 🎨
