"""
Fenêtre indépendante pour consulter et gérer la bibliothèque de prompts
Peut rester ouverte sans bloquer l'application principale
"""

import tkinter as tk
from tkinter import ttk, messagebox
from prompts_library import (
    get_all_categories,
    get_category_description,
    get_examples_for_category,
    PROMPTS_LIBRARY
)


class PromptsViewerWindow:
    """Fenêtre indépendante pour visualiser la bibliothèque de prompts"""
    
    def __init__(self, parent=None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("📚 Bibliothèque de Prompts IA")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e1e")
        
        # Variables
        self.current_category = tk.StringVar(value="portrait")
        
        # Créer l'interface
        self._create_interface()
        
        # Charger la catégorie par défaut
        self._load_category("portrait")
    
    def _create_interface(self):
        """Crée l'interface utilisateur"""
        
        # === BARRE SUPÉRIEURE : Sélecteur de catégorie ===
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(header_frame, text="📖 Catégories:", font=("Arial", 11, "bold")).pack(side="left", padx=5)
        
        # Dropdown des catégories
        categories = get_all_categories()
        category_combo = ttk.Combobox(
            header_frame,
            values=categories,
            textvariable=self.current_category,
            state="readonly",
            width=30,
            font=("Arial", 10)
        )
        category_combo.pack(side="left", padx=5)
        category_combo.bind("<<ComboboxSelected>>", lambda e: self._load_category(self.current_category.get()))
        
        ttk.Button(header_frame, text="🔄 Rafraîchir", command=lambda: self._load_category(self.current_category.get())).pack(side="left", padx=5)
        ttk.Button(header_frame, text="💾 Copier Catégories", command=self._copy_all_categories).pack(side="left", padx=5)
        
        # === CONTENU PRINCIPAL ===
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Panneau gauche - Description et infos
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=2)
        
        ttk.Label(left_panel, text="Description:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.desc_label = ttk.Label(left_panel, text="", foreground="#90EE90", wraplength=250, justify="left")
        self.desc_label.pack(anchor="w", pady=(5, 15))
        
        # Stats
        ttk.Label(left_panel, text="📊 Statistiques:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self.stats_label = ttk.Label(left_panel, text="", foreground="#87CEEB", justify="left")
        self.stats_label.pack(anchor="w", pady=5)
        
        # Boutons d'action
        ttk.Label(left_panel, text="🎯 Actions:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 0))
        ttk.Button(left_panel, text="📋 Copier Positifs", command=self._copy_positive_examples).pack(fill="x", pady=2)
        ttk.Button(left_panel, text="❌ Copier Négatifs", command=self._copy_negative_examples).pack(fill="x", pady=2)
        ttk.Button(left_panel, text="📝 Copier Tout", command=self._copy_all_examples).pack(fill="x", pady=2)
        
        # Panneau droit - Exemples
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Onglets pour positifs/négatifs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill="both", expand=True)
        
        # Onglet positif
        self.positive_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.positive_frame, text="✅ Prompts Positifs")
        
        self.positive_text = tk.Text(
            self.positive_frame,
            height=20,
            width=60,
            bg="#2d2d2d",
            fg="#00FF00",
            font=("Courier New", 9),
            wrap="word"
        )
        self.positive_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Scrollbar pour positiv
        positive_scroll = ttk.Scrollbar(self.positive_frame, command=self.positive_text.yview)
        positive_scroll.pack(side="right", fill="y")
        self.positive_text.config(yscrollcommand=positive_scroll.set)
        
        # Onglet négatif
        self.negative_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.negative_frame, text="❌ Prompts Négatifs")
        
        self.negative_text = tk.Text(
            self.negative_frame,
            height=20,
            width=60,
            bg="#2d2d2d",
            fg="#FF6B6B",
            font=("Courier New", 9),
            wrap="word"
        )
        self.negative_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Scrollbar pour négatif
        negative_scroll = ttk.Scrollbar(self.negative_frame, command=self.negative_text.yview)
        negative_scroll.pack(side="right", fill="y")
        self.negative_text.config(yscrollcommand=negative_scroll.set)
        
        # === BARRE INFÉRIEURE ===
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill="x", padx=10, pady=10)
        
        self.info_label = ttk.Label(footer_frame, text="", foreground="#FFD700")
        self.info_label.pack(anchor="w")
    
    def _load_category(self, category):
        """Charge une catégorie et affiche ses exemples"""
        
        # Description
        description = get_category_description(category)
        self.desc_label.config(text=description)
        
        # Exemples positifs
        positive_examples = get_examples_for_category(category, negative=False)
        positive_text = "\n\n".join([f"#{i+1}:\n{ex}" for i, ex in enumerate(positive_examples)])
        self.positive_text.config(state="normal")
        self.positive_text.delete("1.0", "end")
        self.positive_text.insert("1.0", positive_text)
        self.positive_text.config(state="disabled")
        
        # Exemples négatifs
        negative_examples = get_examples_for_category(category, negative=True)
        negative_text = "\n\n".join([f"#{i+1}:\n{ex}" for i, ex in enumerate(negative_examples)])
        self.negative_text.config(state="normal")
        self.negative_text.delete("1.0", "end")
        self.negative_text.insert("1.0", negative_text)
        self.negative_text.config(state="disabled")
        
        # Statistiques
        stats = f"Positifs: {len(positive_examples)}\nNégatifs: {len(negative_examples)}"
        self.stats_label.config(text=stats)
        
        self.info_label.config(text=f"✅ Catégorie '{category}' chargée")
    
    def _copy_positive_examples(self):
        """Copie tous les exemples positifs dans le presse-papiers"""
        content = self.positive_text.get("1.0", "end")
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.info_label.config(text="✅ Exemples positifs copiés !")
    
    def _copy_negative_examples(self):
        """Copie tous les exemples négatifs dans le presse-papiers"""
        content = self.negative_text.get("1.0", "end")
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.info_label.config(text="✅ Exemples négatifs copiés !")
    
    def _copy_all_examples(self):
        """Copie tous les exemples de la catégorie"""
        category = self.current_category.get()
        positive = self.positive_text.get("1.0", "end")
        negative = self.negative_text.get("1.0", "end")
        content = f"=== {category.upper()} ===\n\n✅ POSITIFS:\n{positive}\n\n❌ NÉGATIFS:\n{negative}"
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.info_label.config(text="✅ Tous les exemples copiés !")
    
    def _copy_all_categories(self):
        """Copie un résumé de toutes les catégories"""
        content = "📚 RÉSUMÉ DE LA BIBLIOTHÈQUE\n" + "="*50 + "\n\n"
        
        for category in get_all_categories():
            positive = get_examples_for_category(category, negative=False)
            negative = get_examples_for_category(category, negative=True)
            description = get_category_description(category)
            
            content += f"🏷️ {category.upper()}\n"
            content += f"   Description: {description}\n"
            content += f"   Exemples positifs: {len(positive)}\n"
            content += f"   Exemples négatifs: {len(negative)}\n\n"
        
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.info_label.config(text="✅ Résumé de toutes les catégories copié !")


def open_prompts_viewer(parent=None):
    """Ouvre la fenêtre de visualisation des prompts en mode non-bloquant"""
    window = PromptsViewerWindow(parent)
    if parent:
        return window.root
    else:
        window.root.mainloop()


if __name__ == "__main__":
    # Demo si exécuté indépendamment
    root = tk.Tk()
    root.withdraw()  # Masquer la fenêtre root
    open_prompts_viewer(root)
    root.deiconify()
