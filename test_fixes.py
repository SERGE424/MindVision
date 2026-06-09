#!/usr/bin/env python
# Test de vérification des imports et de la syntaxe
import sys
print(f"Python {sys.version}")

try:
    print("✓ Test 1: Import de AssistantIA_Complet...")
    from AssistantIA_Complet import AssistantIA
    print("✅ Import réussi!")
    
    print("\n✓ Test 2: Import de Generateur...")
    from Generateur import GenerateurIA
    print("✅ Import réussi!")
    
    print("\n✓ Test 3: Vérification tkinter...")
    import tkinter as tk
    from tkinter import ttk
    
    # Test création de widgets
    root = tk.Tk()
    root.withdraw()
    
    # Test Frame ttk
    frame = ttk.Frame(root)
    print("✅ ttk.Frame créé")
    
    # Test Canvas avec couleur blanche
    canvas = tk.Canvas(frame, bg="white", highlightthickness=0)
    print("✅ tk.Canvas avec bg='white' créé")
    
    # Test Label normal
    label = tk.Label(frame, relief="sunken", background="gray20")
    print("✅ tk.Label avec relief et background créé")
    
    root.destroy()
    
    print("\n🎉 Tous les tests passent! L'application devrait démarrer correctement.")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
