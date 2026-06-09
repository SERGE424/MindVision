# Configuration AWS Bedrock

## 🚀 Étapes de configuration

### 1. Créer un compte AWS
- Allez sur : https://aws.amazon.com
- Cliquez sur "Créer un compte AWS"
- Suivez les instructions (carte bancaire requise)

### 2. Configurer AWS CLI
```bash
pip install awscli
aws configure
```

Vous aurez besoin de :
- **AWS Access Key ID** : Votre clé d'accès
- **AWS Secret Access Key** : Votre clé secrète
- **Region** : `us-east-1` (recommandé)
- **Output format** : `json`

### 3. Obtenir les clés d'accès

1. Connectez-vous à la console AWS
2. Allez dans **IAM** (Identity and Access Management)
3. Cliquez sur **Users** → **Create user**
4. Nom : `bedrock-user`
5. Cochez **Programmatic access**
6. Permissions : **AmazonBedrockFullAccess**
7. Créez l'utilisateur
8. **Téléchargez les clés** (Access Key ID + Secret Access Key)

### 4. Configurer les credentials

**Option A : Via AWS CLI**
```bash
aws configure
```

**Option B : Fichier manuel**
Créez `~/.aws/credentials` (Linux/Mac) ou `C:\Users\<VotreNom>\.aws\credentials` (Windows) :
```
[default]
aws_access_key_id = VOTRE_ACCESS_KEY
aws_secret_access_key = VOTRE_SECRET_KEY
```

### 5. Activer Bedrock dans votre région

1. Allez dans la console AWS Bedrock
2. Région : **us-east-1** (Virginie du Nord)
3. Cliquez sur **Model access**
4. Activez **Stability AI SDXL**

---

## 💰 Tarification

### Stable Diffusion XL sur Bedrock :
- **$0.04 par image** (1024x1024)
- **$0.02 par image** (512x512)

### Exemple de coûts :
- 10 images : $0.40
- 100 images : $4.00
- 1000 images : $40.00

---

## 🔧 Test de configuration

Testez votre configuration avec ce script :

```python
import boto3

try:
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    print("✅ Connexion AWS réussie !")
    
    # Lister les modèles disponibles
    bedrock_client = boto3.client('bedrock', region_name='us-east-1')
    response = bedrock_client.list_foundation_models()
    print(f"✅ {len(response['modelSummaries'])} modèles disponibles")
    
except Exception as e:
    print(f"❌ Erreur : {e}")
```

---

## 🐛 Résolution de problèmes

### Erreur : "NoCredentialsError"
```bash
aws configure
# Entrez vos clés
```

### Erreur : "AccessDeniedException"
- Vérifiez que l'utilisateur IAM a les permissions **AmazonBedrockFullAccess**
- Vérifiez que Bedrock est activé dans votre région

### Erreur : "ModelNotFound"
- Allez dans AWS Bedrock Console
- Activez le modèle **Stability AI SDXL**

### Erreur : "Region not supported"
- Utilisez `us-east-1` (Virginie du Nord)
- C'est la région principale pour Bedrock

---

## 📊 Comparaison des modèles Bedrock

| Modèle | Coût | Résolution | Qualité |
|--------|------|------------|---------|
| **Stable Diffusion XL** | $0.04 | 1024x1024 | ⭐⭐⭐⭐⭐ |
| **Titan Image Generator** | $0.008 | 1024x1024 | ⭐⭐⭐⭐ |

---

## 🎯 Utilisation dans l'application

1. Configurez AWS (étapes ci-dessus)
2. Lancez l'application
3. Sélectionnez **"AWS Bedrock ($0.04)"**
4. Générez votre image !

---

## 💡 Conseils

- **Testez d'abord en mode Local** (gratuit)
- **Utilisez AWS pour la production** (qualité professionnelle)
- **Surveillez vos coûts** dans AWS Cost Explorer
- **Définissez des alertes budgétaires** dans AWS Budgets

---

## 🔒 Sécurité

⚠️ **NE PARTAGEZ JAMAIS vos clés AWS !**
- Ne les commitez pas dans Git
- Ne les partagez pas publiquement
- Utilisez IAM avec permissions minimales
- Activez MFA sur votre compte AWS

---

## 📞 Support

- Documentation AWS Bedrock : https://docs.aws.amazon.com/bedrock/
- Tarification : https://aws.amazon.com/bedrock/pricing/
- Support AWS : https://console.aws.amazon.com/support/
