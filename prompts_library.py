"""
Bibliothèque de prompts pour l'optimisation des descriptions d'images IA
Utilisée par Ollama pour générer de meilleurs prompts Stable Diffusion
"""

PROMPTS_LIBRARY = {
    "portrait": {
        "description": "Portraits et visages",
        "positive_examples": [
            "beautiful woman, professional headshot, cinematic lighting, soft focus, 8k, high quality",
            "handsome man, warm smile, studio photography, dramatic lighting, shallow depth of field",
            "elderly person with wise expression, detailed facial features, natural lighting, character portrait",
            "young girl with peaceful expression, golden hour light, soft bokeh background, ethereal",
            "close-up face portrait, sharp eyes, professional makeup, studio lighting, detailed skin texture"
        ],
        "negative_examples": [
            "blurry, low quality, distorted face, amateur photography",
            "ugly, asymmetrical features, deformed"
        ]
    },
    
    "landscape": {
        "description": "Paysages et environnements naturels",
        "positive_examples": [
            "majestic mountain landscape, golden hour sunlight, misty peaks, dramatic sky, professional photography, 4k",
            "serene forest clearing, ancient towering trees, sunbeams through canopy, peaceful, nature photography",
            "vast rolling hills with wildflowers, warm sunset glow, depth of field, landscape photography",
            "pristine lake reflection, snow-capped mountains, mirror-like water, fine art photography, high detail",
            "tropical paradise beach, crystal clear turquoise water, palm trees, sunset, cinematic"
        ],
        "negative_examples": [
            "blurry, low resolution, poorly composed",
            "artificial, plastic looking"
        ]
    },
    
    "still_life": {
        "description": "Nature morte et objets",
        "positive_examples": [
            "beautiful flowers in vintage vase, soft natural light, shallow depth of field, still life painting",
            "antique books stacked artfully, warm candlelight, moody atmosphere, detailed textures",
            "fresh fruit arrangement, studio lighting, precise shadows, professional food photography",
            "delicate porcelain cups with tea, elegant composition, soft shadows, classical still life",
            "luxury watch on dark background, professional product photography, sharp focus, dramatic lighting"
        ],
        "negative_examples": [
            "messy, cluttered composition",
            "harsh lighting, washed out colors"
        ]
    },
    
    "abstract": {
        "description": "Art abstrait et surréalisme",
        "positive_examples": [
            "abstract digital art, vibrant neon colors, flowing shapes, modern aesthetic, high resolution",
            "surreal dreamscape, impossible architecture, floating islands, ethereal lighting, digital painting",
            "abstract geometric composition, bold primary colors, balanced design, modern art",
            "cosmic abstract, galaxies and nebulae, deep space colors, mystical atmosphere, digital art",
            "liquid ink diffusion abstract, swirling colors, mesmerizing patterns, artistic, detailed"
        ],
        "negative_examples": [
            "realistic, too literal",
            "boring colors, random shapes"
        ]
    },
    
    "fantasy": {
        "description": "Fantaisie, magie et créatures",
        "positive_examples": [
            "fantasy elf warrior, intricate armor, magical aura, professional character art, fantasy illustration",
            "majestic dragon flying through clouds, detailed scales, dramatic lighting, fantasy art",
            "enchanted forest with magical creatures, glowing lights, mystical atmosphere, fantasy world",
            "powerful wizard casting spell, magical energy effects, dramatic pose, dark fantasy, digital art",
            "beautiful fairy with translucent wings, moonlight, magical glow, delicate features, fantasy illustration"
        ],
        "negative_examples": [
            "cartoonish, childish style",
            "poorly designed, ugly creatures"
        ]
    },
    
    "cyberpunk": {
        "description": "Cyberpunk et sci-fi",
        "positive_examples": [
            "cyberpunk city street, neon signs, flying cars, rain, night scene, moody atmospheric, high tech",
            "android woman with glowing cyber implants, sleek design, neon accents, cinematic, futuristic",
            "dense futuristic metropolis, holographic advertisements, crowded streets, blade runner aesthetic",
            "high-tech robot, metallic shiny surfaces, glowing circuits, professional rendering, sci-fi",
            "cyberpunk hacker at computer, neon keyboard, dark room, cold light, atmospheric, cinematic"
        ],
        "negative_examples": [
            "bright and cheerful, outdated",
            "low tech, primitive looking"
        ]
    },
    
    "oil_painting": {
        "description": "Style peinture à l'huile classique",
        "positive_examples": [
            "oil painting, impressionist style, soft brushstrokes, classic art, gallery quality",
            "baroque oil painting, rich colors, dramatic lighting, masterpiece, old masters",
            "landscape oil painting, van gogh style, swirling brushstrokes, vibrant colors, expressive",
            "portrait oil painting, renaissance style, detailed background, warm tones, classical",
            "still life oil painting, dutch golden age, rich textures, museum quality, highly detailed"
        ],
        "negative_examples": [
            "digital, photorealistic",
            "modern minimalist"
        ]
    },
    
    "anime": {
        "description": "Style anime et manga",
        "positive_examples": [
            "beautiful anime girl, large expressive eyes, detailed hair, vibrant colors, anime art style",
            "anime action hero, dynamic pose, detailed clothing, flying effects, manga illustration",
            "cozy anime scene, characters in warm room, soft colors, peaceful mood, slice of life anime",
            "anime landscape, dramatic sky, vibrant nature, anime color palette, studio ghibli style",
            "anime romance scene, two characters, soft lighting, emotional expression, beautiful illustration"
        ],
        "negative_examples": [
            "photorealistic, not anime style",
            "crude drawing, poor proportions"
        ]
    },
    
    "product": {
        "description": "Photographie de produits et marketing",
        "positive_examples": [
            "sleek modern smartphone on white background, professional product photography, sharp focus, studio lighting",
            "luxury watch on dark surface, product photography, detailed metalwork, professional lighting, high quality",
            "cosmetics beauty product, elegant packaging, soft studio lighting, professional advertising photography",
            "shoe product photography, clean white background, professional lighting, sharp focus, commercial ready",
            "beverage product shot, condensation on bottle, studio lighting, appetizing, professional advertisement"
        ],
        "negative_examples": [
            "blurry, unprofessional lighting",
            "cluttered background, poor composition"
        ]
    },
    
    "food": {
        "description": "Photographie culinaire",
        "positive_examples": [
            "delicious gourmet burger, appetizing composition, professional food photography, styled ingredients visible",
            "artfully plated gourmet dish, restaurant quality, warm studio lighting, fine dining, high resolution",
            "colorful sushi platter, fresh ingredients, artistic arrangement, professional food photography, appetizing",
            "freshly baked bread, warm golden crust, rustic styling, natural light, mouth-watering detail",
            "chocolate dessert, decadent toppings, studio lighting, artistic plating, magazine quality food photography"
        ],
        "negative_examples": [
            "unappetizing, poor lighting",
            "unappetizing, poor styling, amateurish"
        ]
    },
    
    "architecture": {
        "description": "Architecture et bâtiments",
        "positive_examples": [
            "modern architecture, sleek building design, geometric lines, urban, professional photography, 4k",
            "historic castle, grand stone structure, dramatic sky, landscape photography, romantic setting",
            "futuristic building design, innovative architecture, night lighting, architectural visualization",
            "classical architecture, ornate details, symmetrical composition, museum quality photography",
            "minimalist modern house, clean lines, large windows, natural light, architectural photography"
        ],
        "negative_examples": [
            "ugly building, depressing atmosphere",
            "photon-stretched, distorted perspective"
        ]
    },
    
    "nature_wildlife": {
        "description": "Faune sauvage et animaux",
        "positive_examples": [
            "majestic lion with full mane, golden hour lighting, detailed fur, professional wildlife photography",
            "graceful antelope leaping, motion blur, savanna landscape, dynamic pose, nature photography",
            "colorful bird with detailed plumage, macro photography, sharp detail, natural background",
            "powerful grizzly bear in water, capturing fish, wild nature, cinematic wildlife photography",
            "herd of elephants in sunset landscape, warm golden light, family group, emotional wildlife shot"
        ],
        "negative_examples": [
            "artificial looking, zoo photo",
            "poor composition, blurry animals"
        ]
    },
    
    "underwater": {
        "description": "Scènes sous-marines",
        "positive_examples": [
            "colorful coral reef, diverse fish species, crystal clear water, underwater photography, vibrant",
            "majestic whale swimming, ocean depths, bioluminescent creatures, mysterious underwater world",
            "sunlight rays penetrating ocean, underwater garden of anemones, peaceful aquatic scene, detailed",
            "deep sea creatures, glowing organisms, dark depths, mysterious and alien underwater ecosystem",
            "underwater wreck exploration, vintage ship, fish swimming around, atmospheric, underwater photography"
        ],
        "negative_examples": [
            "murky, dirty water, low visibility",
            "unrealistic sea creatures"
        ]
    }
}


def get_examples_for_category(category, negative=False):
    """
    Récupère les exemples pour une catégorie
    
    Args:
        category (str): Nom de la catégorie
        negative (bool): Si True, retourne les exemples négatifs
    
    Returns:
        list: Liste des exemples
    """
    if category not in PROMPTS_LIBRARY:
        return []
    
    key = "negative_examples" if negative else "positive_examples"
    return PROMPTS_LIBRARY[category].get(key, [])


def get_all_categories():
    """Retourne la liste de toutes les catégories"""
    return list(PROMPTS_LIBRARY.keys())


def get_category_description(category):
    """Récupère la description d'une catégorie"""
    if category not in PROMPTS_LIBRARY:
        return ""
    return PROMPTS_LIBRARY[category].get("description", "")


def format_prompt_examples(category, negative=False, max_examples=3):
    """
    Formate les exemples pour inclusion dans un prompt Ollama
    
    Args:
        category (str): Catégorie
        negative (bool): Exemples positifs ou négatifs
        max_examples (int): Nombre max d'exemples à inclure
    
    Returns:
        str: Texte formaté pour le prompt
    """
    examples = get_examples_for_category(category, negative)
    if not examples:
        return ""
    
    examples = examples[:max_examples]
    
    if negative:
        header = "Voici des exemples de choses à ÉVITER:\n"
    else:
        header = "Voici des exemples de bons prompts:\n"
    
    formatted = header
    for i, example in enumerate(examples, 1):
        formatted += f"{i}. {example}\n"
    
    return formatted


def create_system_prompt_with_examples(demande, category="general", negative_prompt=""):
    """
    Crée un système prompt enrichi avec des exemples
    
    Args:
        demande (str): Demande de l'utilisateur
        category (str): Catégorie pour les exemples
        negative_prompt (str): Prompt négatif optionnel
    
    Returns:
        tuple: (prompt_positif, prompt_négatif)
    """
    # Exemples positifs
    examples = format_prompt_examples(category, negative=False, max_examples=2)
    
    positive_prompt = (
        f"Tu es un expert en génération d'images IA pour Stable Diffusion.\n"
        f"{examples}\n"
        f"L'utilisateur veut : '{demande}'.\n"
        f"Crée un prompt détaillé en anglais, style bien structuré (max 50 mots), "
        f"en suivant le format des exemples.\n"
        f"Réponds UNIQUEMENT avec le prompt, sans explication."
    )
    
    # Exemples négatifs
    negative_examples = format_prompt_examples(category, negative=True, max_examples=2)
    
    if negative_prompt and "Exemple:" not in negative_prompt:
        negative_system_prompt = (
            f"Tu es un expert en génération d'images IA pour Stable Diffusion.\n"
            f"{negative_examples}\n"
            f"L'utilisateur veut éviter : '{negative_prompt}'.\n"
            f"Crée un prompt négatif détaillé en anglais (max 30 mots), "
            f"en suivant le format des exemples.\n"
            f"Réponds UNIQUEMENT avec le prompt, sans explication."
        )
    else:
        negative_system_prompt = ""
    
    return positive_prompt, negative_system_prompt


if __name__ == "__main__":
    # Test pour vérifier que tout fonctionne
    print("Catégories disponibles :")
    for cat in get_all_categories():
        print(f"  - {cat}: {get_category_description(cat)}")
    
    print("\n--- Démo : Catégorie 'portrait' ---")
    positive, negative = create_system_prompt_with_examples(
        "Un chat mignon",
        category="portrait"
    )
    print("POSITIF:")
    print(positive)
    print("\nNÉGATIF:")
    print(negative)
