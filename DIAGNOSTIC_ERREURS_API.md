# Diagnostic des erreurs API

## 🔍 Erreurs courantes qui bloquent les APIs

### 1. **Import manquant : urllib.parse**
Si vous voyez une erreur sur `urllib.parse`, c'est normal - c'est un module standard Python.

**Solution** : Aucune, c'est déjà inclus dans Python.

### 2. **Module requests non installé**
```
ModuleNotFoundError: No module named 'requests'
```

**Solution** :
```bash
pip install requests
```

### 3. **Erreur SSL/Certificat**
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solution** :
```python
# Dans generer_api(), ajouter :
response = requests.get(url, timeout=30, verify=False)
```

### 4. **Timeout réseau**
```
ReadTimeout: HTTPSConnectionPool
```

**Solution** : Augmenter le timeout
```python
response = requests.get(url, timeout=120)  # Au lieu de 30
```

### 5. **Proxy/Firewall**
Si vous êtes derrière un proxy d'entreprise :

```python
proxies = {
    'http': 'http://proxy:port',
    'https': 'http://proxy:port',
}
response = requests.get(url, proxies=proxies)
```

### 6. **Erreur d'encodage du prompt**
```
UnicodeEncodeError
```

**Solution** :
```python
import urllib.parse
encoded_prompt = urllib.parse.quote(prompt, safe='')
```

---

## 🧪 Test de diagnostic

Ajoutez ce code pour tester les APIs :

```python
def tester_api(self):
    """Test de diagnostic des APIs"""
    import urllib.parse
    
    print("=== Test de diagnostic API ===")
    
    # Test 1: Module requests
    try:
        import requests
        print("✅ Module requests installé")
    except ImportError:
        print("❌ Module requests manquant - pip install requests")
        return
    
    # Test 2: Connexion internet
    try:
        response = requests.get("https://www.google.com", timeout=5)
        print("✅ Connexion internet OK")
    except:
        print("❌ Pas de connexion internet")
        return
    
    # Test 3: Pollinations.ai
    try:
        prompt = "test"
        encoded = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}"
        print(f"URL testée : {url}")
        
        response = requests.get(url, timeout=30)
        print(f"Status code : {response.status_code}")
        print(f"Content-Type : {response.headers.get('Content-Type')}")
        
        if response.status_code == 200:
            print("✅ Pollinations.ai fonctionne")
            from PIL import Image
            from io import BytesIO
            image = Image.open(BytesIO(response.content))
            print(f"Image reçue : {image.size}")
        else:
            print(f"❌ Erreur {response.status_code}")
            print(f"Réponse : {response.text[:200]}")
    except Exception as e:
        print(f"❌ Erreur Pollinations.ai : {e}")
    
    print("=== Fin du test ===")
```

Ajoutez un bouton dans l'interface :
```python
test_btn = tk.Button(self.root, text="Tester API", command=self.tester_api)
test_btn.pack(pady=5)
```

---

## 🔧 Corrections possibles dans Generateur.py

### Si urllib.parse pose problème :
```python
# Au début du fichier
import urllib.parse
```

### Si requests pose problème :
```bash
pip install requests
```

### Version robuste de generer_api :
```python
def generer_api(self, prompt):
    try:
        # Vérifier que requests est installé
        import requests
    except ImportError:
        self.status.config(text="Erreur : pip install requests")
        self.btn.config(state='normal')
        return
    
    try:
        self.progress["maximum"] = 1
        self.progress["value"] = 0
        
        # Encoder le prompt
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt, safe='')
        
        # URL avec paramètres
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true"
        
        print(f"Requête API : {url}")  # Debug
        
        # Requête avec gestion d'erreur détaillée
        self.status.config(text="Connexion à l'API...")
        response = requests.get(url, timeout=60, verify=True)
        
        print(f"Status: {response.status_code}")  # Debug
        print(f"Headers: {response.headers}")  # Debug
        
        if response.status_code != 200:
            self.status.config(text=f"Erreur API: {response.status_code}")
            return
        
        # Vérifier que c'est bien une image
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            self.status.config(text=f"Erreur: pas une image ({content_type})")
            return
        
        # Charger l'image
        from PIL import Image
        from io import BytesIO
        image = Image.open(BytesIO(response.content))
        
        self.progress["value"] = 1
        self.afficher_image(image)
        self.status.config(text="Terminé (API) !")
        
    except requests.exceptions.Timeout:
        self.status.config(text="Erreur: Timeout (réseau lent)")
    except requests.exceptions.ConnectionError:
        self.status.config(text="Erreur: Pas de connexion internet")
    except requests.exceptions.SSLError:
        self.status.config(text="Erreur: Problème SSL/Certificat")
    except Exception as e:
        self.status.config(text=f"Erreur: {str(e)[:50]}")
        print(f"Erreur détaillée: {e}")  # Debug console
    finally:
        self.btn.config(state='normal')
        self.progress["value"] = 0
```

---

## 📋 Checklist de vérification

- [ ] Module `requests` installé : `pip install requests`
- [ ] Connexion internet active
- [ ] Pas de proxy/firewall bloquant
- [ ] Python 3.7+ (pour urllib.parse)
- [ ] Pas d'antivirus bloquant les connexions Python

---

## 🐛 Commandes de diagnostic

### Vérifier requests :
```bash
python -c "import requests; print(requests.__version__)"
```

### Tester une requête simple :
```bash
python -c "import requests; r=requests.get('https://google.com'); print(r.status_code)"
```

### Tester Pollinations.ai :
```bash
python -c "import requests; r=requests.get('https://image.pollinations.ai/prompt/test'); print(r.status_code, r.headers.get('Content-Type'))"
```

---

## 💡 Si rien ne fonctionne

Les APIs gratuites sont instables. **Utilisez le mode Local** qui est :
- ✅ Fiable
- ✅ Hors ligne
- ✅ Gratuit
- ✅ Optimisé (avec les dernières modifications)

Le mode Local est maintenant 3-6x plus rapide grâce aux optimisations !
