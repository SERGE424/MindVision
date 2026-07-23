#!/usr/bin/env python
# Test de vérification des imports et de la syntaxe
import sys
print(f"Python {sys.version}")

try:
    print("Test 1: Import de AssistantIA_Complet...")
    from AssistantIA_Complet import AssistantIA
    print("[OK] Import reussi!")

    print("\nTest 2: Import de Generateur...")
    from Generateur import GenerateurIA
    print("[OK] Import reussi!")

    print("\nTest 3: Verification tkinter...")
    import tkinter as tk
    from tkinter import ttk

    # Test création de widgets
    root = tk.Tk()
    root.withdraw()

    # Test Frame ttk
    frame = ttk.Frame(root)
    print("[OK] ttk.Frame cree")

    # Test Canvas avec couleur blanche
    canvas = tk.Canvas(frame, bg="white", highlightthickness=0)
    print("[OK] tk.Canvas avec bg='white' cree")

    # Test Label normal
    label = tk.Label(frame, relief="sunken", background="gray20")
    print("[OK] tk.Label avec relief et background cree")

    root.destroy()

    print("\n[OK] Tous les tests passent! L'application devrait demarrer correctement.")

except Exception as e:
    print(f"[ERREUR]: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
