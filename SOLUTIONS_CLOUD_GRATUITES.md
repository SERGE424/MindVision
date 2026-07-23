# Solutions Cloud GRATUITES pour Génération d'Images IA

## 🎯 Meilleures alternatives gratuites (2024)

### 1. **Replicate** ⭐ RECOMMANDÉ
- **Quota gratuit** : $5 de crédit gratuit/mois (~50 images)
- **Fiabilité** : Excellente
- **Vitesse** : Rapide (GPU)
- **Site** : https://replicate.com

#### Installation :
```bash
pip install replicate
```

#### Configuration :
1. Créez un compte sur https://replicate.com
2. Obtenez votre token : https://replicate.com/account/api-tokens
3. Ajoutez dans `config.py` :
```python
REPLICATE_API_TOKEN = "r8_votre_token_ici"
```

#### Code d'intégration :
```python
import replicate

output = replicate.run(
    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
    input={"prompt": "votre description"}
)
```

---

### 2. **Stability AI (DreamStudio)**
- **Quota gratuit** : 25 crédits gratuits (25 images)
- **Fiabilité** : Excellente (officiel Stable Diffusion)
- **Site** : https://dreamstudio.ai

#### Installation :
```bash
pip install stability-sdk
```

---

### 3. **Segmind** 🆓 VRAIMENT GRATUIT
- **Quota gratuit** : 1000 requêtes/jour GRATUITES
- **Fiabilité** : Bonne
- **Site** : https://www.segmind.com

#### Configuration :
1. Créez un compte sur https://www.segmind.com
2. Obtenez votre API key : https://www.segmind.com/api-keys
3. Ajoutez dans `config.py` :
```python
SEGMIND_API_KEY = "SG_votre_key_ici"
```

#### Code d'intégration :
```python
import requests

url = "https://api.segmind.com/v1/sd1.5-txt2img"
headers = {"x-api-key": SEGMIND_API_KEY}
data = {"prompt": "votre description"}
response = requests.post(url, json=data, headers=headers)
```

---

### 4. **Pollinations.ai** 🆓 100% GRATUIT
- **Quota gratuit** : ILLIMITÉ
- **Fiabilité** : Moyenne (peut être lent)
- **Site** : https://pollinations.ai
- **Aucune inscription requise !**

#### Code d'intégration (le plus simple) :
```python
import requests
from PIL import Image
from io import BytesIO

prompt = "votre description"
url = f"https://image.pollinations.ai/prompt/{prompt}"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
```

---

## 🚀 Intégration recommandée : Pollinations.ai

C'est la solution la plus simple et vraiment gratuite. Voici comment l'intégrer :

### Modifiez `Generateur.py` :

Remplacez la fonction `generer_api` par :

```python
def generer_api(self, prompt):
    try:
        self.progress["maximum"] = 1
        self.progress["value"] = 0
        
        # Utiliser Pollinations.ai (gratuit et sans token)
        self.status.config(text="Connexion à Pollinations.ai...")
        
        # Encoder le prompt pour l'URL
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        
        self.progress["value"] = 1
        self.afficher_image(image)
        self.status.config(text="Terminé (Pollinations.ai) !")
        
    except Exception as e:
        self.status.config(text=f"Erreur API : {str(e)[:50]}")
    finally:
        self.btn.config(state='normal')
        self.progress["value"] = 0
```

---

## 📊 Comparaison

| Service | Gratuit | Quota | Inscription | Fiabilité |
|---------|---------|-------|-------------|-----------|
| **Pollinations.ai** | ✅ | Illimité | ❌ Non | ⭐⭐⭐ |
| **Segmind** | ✅ | 1000/jour | ✅ Oui | ⭐⭐⭐⭐ |
| **Replicate** | 💰 | $5/mois | ✅ Oui | ⭐⭐⭐⭐⭐ |
| **DreamStudio** | 💰 | 25 images | ✅ Oui | ⭐⭐⭐⭐⭐ |
| **Hugging Face** | ✅ | Limité | ✅ Oui | ⭐⭐ |

---

## 🎯 Ma recommandation finale :

**Utilisez Pollinations.ai** - C'est vraiment gratuit, illimité, et ne nécessite aucune inscription !

Voulez-vous que je modifie votre code pour utiliser Pollinations.ai ?
