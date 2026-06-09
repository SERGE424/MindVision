#!/usr/bin/env python
import sys
import tkinter as tk
from tkinter import ttk

print("Test de démarrage de MindVision...")
print(f"Version de Python: {sys.version}")
print(f"Version de tkinter: {tk.TkVersion}")

try:
    # Test basique de tkinter
    root = tk.Tk()
    root.withdraw()  # Masquer la fenêtre
    
    # Test configuration avec bg
    root.configure(bg="black")
    print("✅ Configuration bg sur root: OK")
    
    # Test tk.Label avec relief et bg
    label = tk.Label(root, relief="sunken", background="gray20")
    print("✅ tk.Label avec relief et bg: OK")
    
    # Test ttk.Label
    ttk_label = ttk.Label(root, text="Test")
    print("✅ ttk.Label: OK")
    
    # Test tk.Label avec foreground
    fg_label = tk.Label(root, foreground="green")
    print("✅ tk.Label avec foreground: OK")
    
    root.destroy()
    
    print("\n🎉 Tests basiques tkinter réussis!")
    print("Tentative de démarrage de Generateur.py...")
    
    # Maintenant, essayer d'importer et de démarrer le générateur
    from Generateur import GenerateurIA
    root2 = tk.Tk()
    app = GenerateurIA(root2)
    print("✅ Generateur.py chargé avec succès!")
    root2.quit()
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Tous les tests sont passés!")
