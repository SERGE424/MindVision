#!/usr/bin/env python3
"""
Démonstration de la bibliothèque de prompts
Exécutez ce script pour voir la bibliothèque en action
"""

from prompts_library import (
    get_all_categories,
    get_category_description,
    get_examples_for_category,
    create_system_prompt_with_examples,
    format_prompt_examples
)


def demo_categories():
    """Affiche les catégories et descriptions"""
    print("\n" + "="*60)
    print("📚 CATÉGORIES DISPONIBLES")
    print("="*60 + "\n")
    
    for i, category in enumerate(get_all_categories(), 1):
        description = get_category_description(category)
        positive = get_examples_for_category(category, negative=False)
        negative = get_examples_for_category(category, negative=True)
        
        print(f"{i:2}. {category.upper():20} - {description}")
        print(f"    └─ {len(positive)} exemples positifs, {len(negative)} négatifs")


def demo_category(category):
    """Affiche les détails d'une catégorie"""
    print("\n" + "="*60)
    print(f"📖 CATÉGORIE: {category.upper()}")
    print("="*60)
    
    description = get_category_description(category)
    print(f"\nDescription: {description}\n")
    
    # Exemples positifs
    print("✅ EXEMPLES POSITIFS:")
    positive = get_examples_for_category(category, negative=False)
    for i, example in enumerate(positive, 1):
        print(f"   {i}. {example}")
    
    # Exemples négatifs
    print("\n❌ EXEMPLES NÉGATIFS (À ÉVITER):")
    negative = get_examples_for_category(category, negative=True)
    for i, example in enumerate(negative, 1):
        print(f"   {i}. {example}")


def demo_prompt_generation():
    """Démontre la génération de prompts avec exemples"""
    print("\n" + "="*60)
    print("🤖 DÉMONSTRATION: GÉNÉRATION DE PROMPTS AVEC EXEMPLES")
    print("="*60)
    
    # Exemple 1: Portrait
    print("\n--- EXEMPLE 1: Portrait ---")
    print("Demande utilisateur: 'Je veux un portrait d'une belle femme'")
    
    positive, negative = create_system_prompt_with_examples(
        "Je veux un portrait d'une belle femme",
        category="portrait"
    )
    
    print("\n✅ PROMPT POSITIF GÉNÉRÉ POUR OLLAMA:")
    print("-" * 60)
    print(positive)
    print("-" * 60)
    
    if negative:
        print("\n❌ PROMPT NÉGATIF GÉNÉRÉ POUR OLLAMA:")
        print("-" * 60)
        print(negative)
        print("-" * 60)
    
    # Exemple 2: Fantasy
    print("\n\n--- EXEMPLE 2: Fantasy ---")
    print("Demande utilisateur: 'Un dragon volant dans le ciel'")
    
    positive, negative = create_system_prompt_with_examples(
        "Un dragon volant dans le ciel",
        category="fantasy"
    )
    
    print("\n✅ PROMPT POSITIF GÉNÉRÉ POUR OLLAMA:")
    print("-" * 60)
    print(positive)
    print("-" * 60)


def demo_format():
    """Démontre le formatage des exemples"""
    print("\n" + "="*60)
    print("📝 EXEMPLE DE FORMATAGE")
    print("="*60 + "\n")
    
    formatted = format_prompt_examples("landscape", negative=False, max_examples=3)
    print(formatted)


def main():
    """Exécute les démonstrations"""
    
    # 1. Afficher les catégories
    demo_categories()
    
    # 2. Détails d'une catégorie
    demo_category("portrait")
    
    # 3. Génération de prompts
    demo_prompt_generation()
    
    # 4. Exemple de formatage
    demo_format()
    
    print("\n" + "="*60)
    print("✅ FIN DE LA DÉMONSTRATION")
    print("="*60)
    print("\nPour utiliser dans votre application:")
    print("  from prompts_library import create_system_prompt_with_examples")
    print("  pos, neg = create_system_prompt_with_examples('votre demande', category='portrait')")


if __name__ == "__main__":
    main()
