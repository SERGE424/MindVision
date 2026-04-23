import warnings
warnings.filterwarnings('ignore')
import os

# Evite le crash OpenMP (libiomp5md.dll chargee plusieurs fois via dependances)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import random
from io import BytesIO
import requests
from urllib.parse import quote
import json
import re
from datetime import datetime
from pathlib import Path

# Import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama non installé : pip install ollama")

# Import AWS Bedrock
try:
    import boto3
    import json
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# Import Replicate
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False

# Charger les tokens depuis config.py
try:
    from config import REPLICATE_API_TOKEN
except ImportError:
    REPLICATE_API_TOKEN = None

# Import service TTS du ChatBot
try:
    from chatbot_tts import ChatbotTTSService
    CHAT_TTS_AVAILABLE = True
except ImportError:
    CHAT_TTS_AVAILABLE = False
    print("⚠️ TTS ChatBot non disponible : verifier le dossier chatbot_tts")

# Import de la bibliothèque de prompts
try:
    from prompts_library import create_system_prompt_with_examples, get_all_categories
    from prompts_viewer import open_prompts_viewer
    PROMPTS_LIBRARY_AVAILABLE = True
except ImportError:
    PROMPTS_LIBRARY_AVAILABLE = False
    print("⚠️ Bibliothèque de prompts non disponible")


class AssistantIA:
    def __init__(self, root):
        self.root = root
        self.root.title("Votre Assistant IA Complet - ChatBot + Générateur d'Images")
        self.root.geometry("1200x900")
        
        # Configuration
        self.ollama_model = "mistral-large-3:675b-cloud"  # Modèle rapide
        self.sd_pipe = None
        self.torch = None
        self.current_image = None
        self.generation_mode = "local"  # Mode par défaut
        self.generator = None  # Pour le seed
        self.last_optimized_prompt = ""  # Stocker le dernier prompt optimisé
        self.last_optimized_negative_prompt = ""  # Stocker le dernier prompt négatif optimisé
        self.last_generator_prompt = ""  # Dernier prompt utilisé dans l'onglet Générateur
        self.last_generator_negative_prompt = ""  # Dernier prompt négatif du Générateur
        self.gallery_images = []  # Galerie d'images partagée
        self.auto_recreate = False  # Mode recréation automatique
        self.auto_recreate_generator = False  # Recréation continue côté Générateur
        self.auto_recreate_delay_sec = 0.4  # Délai entre deux recréations automatiques
        self.tooltip_window = None  # Pour gérer les tooltips
        self.hf_endpoint_strategy = "Root (router)"
        
        # Images originales pour redimensionnement adaptatif
        self.original_assistant_image = None
        self.original_image = None
        
        # Charger le modèle SD en arrière-plan
        threading.Thread(target=self.charger_sd_turbo, daemon=True).start()
        
        # Configuration TTS
        self.lang = "fr"  # Langue française
        self.tts_service = ChatbotTTSService(lang=self.lang) if CHAT_TTS_AVAILABLE else None
        self.images_dir = Path(os.path.dirname(__file__)) / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        # Image en attente d'envoi dans le chatbot
        self.chat_pending_image = None
        self._chat_image_refs = []  # Références PhotoImage pour éviter le GC
        
        # Gestion des conversations
        self.conversations_dir = Path(os.path.dirname(__file__)) / "conversations"
        self.conversations_dir.mkdir(exist_ok=True)
        self.conversations = {}  # {id: {nom, date, messages}}
        self.current_conversation_id = None
        self.charger_conversations()
        self.creer_nouvelle_conversation()  # Créer une première conversation
        
        self.creer_interface()
    
    def charger_sd_turbo(self):
        """Charge SD-Turbo en arrière-plan"""
        try:
            print("Chargement de SD-Turbo...")
            import torch
            from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image

            self.torch = torch
            self.sd_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo")
            self.sd_pipe.enable_attention_slicing()
            self.sd_pipe.to("cpu")
            print("✅ SD-Turbo chargé !")
        except Exception as e:
            print(f"❌ Erreur SD-Turbo : {e}")

    def speak(self, text):
        """Prononce le texte via le service TTS du ChatBot."""
        if not self.tts_service:
            return
        self.tts_service.speak(text)
    
    # === GESTION CONVERSATIONS ===
    
    def charger_conversations(self):
        """Charge toutes les conversations depuis les fichiers JSON"""
        self.conversations = {}
        if not self.conversations_dir.exists():
            return
        
        for file_path in sorted(self.conversations_dir.glob("*.json"), reverse=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conv_id = file_path.stem
                    self.conversations[conv_id] = data
            except Exception as e:
                print(f"Erreur chargement conversation {file_path}: {e}")
    
    def creer_nouvelle_conversation(self):
        """Crée une nouvelle conversation"""
        conv_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.conversations[conv_id] = {
            "nom": f"Conversation {datetime.now().strftime('%d/%m %H:%M')}",
            "date_creation": datetime.now().isoformat(),
            "messages": []
        }
        self.current_conversation_id = conv_id
        self.sauvegarder_conversation(conv_id)
    
    def sauvegarder_conversation(self, conv_id):
        """Sauvegarde une conversation dans un fichier JSON"""
        if conv_id not in self.conversations:
            return
        
        file_path = self.conversations_dir / f"{conv_id}.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversations[conv_id], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur sauvegarde conversation: {e}")
    
    def ajouter_message(self, sender, texte):
        """Ajoute un message à la conversation actuelle"""
        if not self.current_conversation_id or self.current_conversation_id not in self.conversations:
            return
        
        message = {
            "sender": sender,
            "texte": texte,
            "timestamp": datetime.now().isoformat()
        }
        self.conversations[self.current_conversation_id]["messages"].append(message)
        
        # Générer un nom si c'est le premier message (autre que système)
        if len(self.conversations[self.current_conversation_id]["messages"]) <= 2 and sender == "Vous":
            # Prendre les 40 premiers caractères du message
            nom_base = texte[:40].replace('\n', ' ').strip()
            self.conversations[self.current_conversation_id]["nom"] = nom_base if nom_base else "Conversation"
        
        self.sauvegarder_conversation(self.current_conversation_id)
        self.rafraichir_liste_conversations()
    
    def charger_conversation(self, conv_id):
        """Charge une conversation et l'affiche"""
        if conv_id not in self.conversations:
            return
        
        self.current_conversation_id = conv_id
        self.afficher_conversation_actuelle()
    
    def afficher_conversation_actuelle(self):
        """Affiche tous les messages de la conversation actuelle"""
        self.chat_display.config(state='normal')
        self.chat_display.delete("1.0", tk.END)
        
        if not self.current_conversation_id or self.current_conversation_id not in self.conversations:
            self.chat_display.config(state='disabled')
            return
        
        messages = self.conversations[self.current_conversation_id]["messages"]
        for msg in messages:
            sender = msg["sender"]
            texte = msg["texte"]
            self.chat_display.insert(tk.END, f"\n{sender} : {texte}\n", sender)
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')
    
    def supprimer_conversation(self, conv_id):
        """Supprime une conversation"""
        if conv_id not in self.conversations:
            return
        
        # Supprimer le fichier
        file_path = self.conversations_dir / f"{conv_id}.json"
        try:
            file_path.unlink()
        except:
            pass
        
        # Supprimer de la mémoire
        del self.conversations[conv_id]
        
        # Si c'est la conversation actuelle, charger une autre ou en créer une
        if self.current_conversation_id == conv_id:
            if self.conversations:
                first_id = next(iter(self.conversations))
                self.charger_conversation(first_id)
            else:
                self.creer_nouvelle_conversation()
        
        self.rafraichir_liste_conversations()
    
    def rafraichir_liste_conversations(self):
        """Rafraîchit le dropdown des conversations"""
        if not hasattr(self, 'chat_conversations_combo'):
            return
        
        values = [f"{self.conversations[cid]['nom']}" for cid in sorted(self.conversations.keys(), reverse=True)]
        self.chat_conversations_combo['values'] = values
        
        if self.current_conversation_id and self.current_conversation_id in self.conversations:
            idx = list(sorted(self.conversations.keys(), reverse=True)).index(self.current_conversation_id)
            self.chat_conversations_combo.current(idx)
    
    def creer_interface(self):
        # Notebook (onglets)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Onglet 1 : ChatBot
        self.tab_chat = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_chat, text="💬 ChatBot")
        self.creer_onglet_chatbot()
        
        # Onglet 2 : Générateur d'images
        self.tab_images = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_images, text="🎨 Générateur d'Images")
        self.creer_onglet_generateur()
        
        # Onglet 3 : Assistant combiné
        self.tab_assistant = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_assistant, text="🤖 Assistant Complet")
        self.creer_onglet_assistant()
    
    def creer_onglet_chatbot(self):
        """Onglet ChatBot Ollama"""
        # Titre
        ttk.Label(self.tab_chat, text="Conversation avec Ollama", font=("Arial", 28, "bold")).pack(pady=5)
        
        # Barre de gestion des conversations (fixe en haut)
        conv_frame = ttk.Frame(self.tab_chat)
        conv_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(conv_frame, text="Conversations :", font=("Arial", 18)).pack(side="left", padx=5)
        self.chat_conversations_combo = ttk.Combobox(conv_frame, state="readonly", width=40)
        self.chat_conversations_combo.pack(side="left", padx=5, fill="x", expand=True)
        self.chat_conversations_combo.bind("<<ComboboxSelected>>", self._on_conversation_selected)
        
        tk.Button(conv_frame, text="➕ Nouvelle", command=self._on_nouvelle_conversation, bg="#2E7D32", fg="white", width=12).pack(side="left", padx=2)
        tk.Button(conv_frame, text="🗑️ Supprimer", command=self._on_supprimer_conversation, bg="#C62828", fg="white", width=12).pack(side="left", padx=2)
        
        # Rafraîchir la liste
        self.rafraichir_liste_conversations()
        
        # PanedWindow vertical pour diviser l'espace entre conversation et paramètres
        main_paned = ttk.PanedWindow(self.tab_chat, orient=tk.VERTICAL)
        main_paned.pack(fill="both", expand=True, padx=10, pady=5)
        
        # === PANEAU 1 : Zone de conversation (adaptative) ===
        chat_frame = tk.LabelFrame(main_paned, text="📝 Conversation", relief=tk.GROOVE)
        main_paned.add(chat_frame, weight=2)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, font=("Arial", 20), bg="black", fg="white")
        self.chat_display.pack(fill="both", expand=True, padx=5, pady=5)
        self.chat_display.config(state='disabled')
        
        # Configuration des styles pour les messages
        self.chat_display.tag_config("Vous", foreground="#4DA6FF", font=("Arial", 22, "bold"))  # Bleu ciel clair + gras + agrandi
        self.chat_display.tag_config("IA", foreground="#7CFC00", font=("Arial", 20))  # Vert clair pour visibilité
        self.chat_display.tag_config("Système", foreground="#FF6B6B", font=("Arial", 20))  # Rouge clair pour les erreurs
        
        # Afficher la conversation initiale
        self.afficher_conversation_actuelle()
        
        # === PANEAU 2 : Paramètres et saisie (adaptative) ===
        params_frame = tk.LabelFrame(main_paned, text="⚙️ Paramètres & Saisie", relief=tk.GROOVE)
        main_paned.add(params_frame, weight=1)
        
        # Cadre scrollable pour les instructions et saisie
        scroll_frame = ttk.Frame(params_frame)
        scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        canvas = tk.Canvas(scroll_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Molette de souris
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Sélecteur de source IA
        source_frame = ttk.Frame(scrollable_frame)
        source_frame.pack(pady=(8, 1), padx=5, fill="x")
        ttk.Label(source_frame, text="Source IA :", font=("Arial", 16)).pack(side="left", padx=5)
        self.chat_source_combo = ttk.Combobox(source_frame, values=[
            "Ollama (local)",
            "Hugging Face API",
        ], state="readonly", width=22)
        self.chat_source_combo.current(0)
        self.chat_source_combo.pack(side="left", padx=5)

        self.chat_voice_enabled = tk.BooleanVar(value=True)
        self.chat_voice_toggle = ttk.Checkbutton(
            source_frame,
            text="Lecture vocale ON/OFF",
            variable=self.chat_voice_enabled
        )
        self.chat_voice_toggle.pack(side="left", padx=12)

        # Sélecteur de modèle HuggingFace texte
        hf_text_frame = ttk.Frame(scrollable_frame)
        hf_text_frame.pack(pady=(2, 1), padx=5, fill="x")
        ttk.Label(hf_text_frame, text="Modèle HF texte :", font=("Arial", 14)).pack(side="left", padx=5)
        self.chat_hf_model_combo = ttk.Combobox(hf_text_frame, values=[
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "HuggingFaceH4/zephyr-7b-beta",
        ], state="readonly", width=38)
        self.chat_hf_model_combo.current(0)
        self.chat_hf_model_combo.pack(side="left", padx=5)

        # Instructions négatives
        ttk.Label(scrollable_frame, text="Instructions négatives (comportements à éviter) :", font=("Arial", 16)).pack(pady=(8, 1), padx=5)
        self.chat_negative = tk.Text(scrollable_frame, height=1, font=("Arial", 14))
        self.chat_negative.pack(fill="x", padx=5, pady=1)
        self.chat_negative.insert(tk.END, "")
        
        # Zone de saisie
        ttk.Label(scrollable_frame, text="Votre message :", font=("Arial", 16)).pack(pady=(8, 1), padx=5)
        input_frame = ttk.Frame(scrollable_frame)
        input_frame.pack(fill="x", padx=5, pady=1)
        
        self.chat_input = tk.Text(input_frame, height=2, font=("Arial", 16))
        self.chat_input.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.chat_input.bind("<Return>", lambda e: self.envoyer_message_chat() if not (isinstance(e.state, int) and e.state & 1) else None)
        
        # Boutons droite (Envoyer + Import Image)
        btns_right = ttk.Frame(input_frame)
        btns_right.pack(side="right")
        btn_send = tk.Button(btns_right, text="Envoyer", command=self.envoyer_message_chat, bg="blue", fg="white", width=10, font=("Arial", 14, "bold"))
        btn_send.pack(side="top", pady=(0, 3))
        btn_img_import = tk.Button(btns_right, text="📎 Image", command=self._importer_image_chat, bg="#5C6BC0", fg="white", width=10, font=("Arial", 12))
        btn_img_import.pack(side="top")
        
        # Zone de prévisualisation de l'image sélectionnée (masquée par défaut)
        self.chat_img_preview_frame = tk.Frame(scrollable_frame, bg="#1a1a2e", relief=tk.GROOVE, bd=1)
        self.chat_img_thumbnail_label = tk.Label(self.chat_img_preview_frame, bg="#1a1a2e")
        self.chat_img_thumbnail_label.pack(side="left", padx=5, pady=3)
        self.chat_img_name_label = tk.Label(self.chat_img_preview_frame, bg="#1a1a2e", fg="white", font=("Arial", 11))
        self.chat_img_name_label.pack(side="left", padx=5)
        tk.Button(self.chat_img_preview_frame, text="✖", command=self._supprimer_image_chat,
                  bg="#C62828", fg="white", font=("Arial", 12, "bold"), width=3).pack(side="right", padx=5)
        
        # Status
        self.chat_status = tk.Label(scrollable_frame, text="Prêt", foreground="green")
        self.chat_status.pack(pady=5)
    
    def _importer_image_chat(self):
        """Ouvre un sélecteur de fichier pour joindre une image au prochain message"""
        path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.gif *.webp *.bmp"),
                ("Tous les fichiers", "*.*")
            ]
        )
        if not path:
            return
        self.chat_pending_image = path
        # Créer et afficher la miniature
        try:
            img = Image.open(path)
            img.thumbnail((80, 80))
            photo = ImageTk.PhotoImage(img)
            self.chat_img_thumbnail_label.config(image=photo)
            self.chat_img_thumbnail_label.image = photo  # référence
        except Exception:
            self.chat_img_thumbnail_label.config(image="")
        self.chat_img_name_label.config(text=os.path.basename(path))
        self.chat_img_preview_frame.pack(fill="x", padx=5, pady=2, before=self.chat_status)

    def _supprimer_image_chat(self):
        """Retire l'image en attente"""
        self.chat_pending_image = None
        self.chat_img_thumbnail_label.config(image="")
        self.chat_img_thumbnail_label.image = None
        self.chat_img_name_label.config(text="")
        self.chat_img_preview_frame.pack_forget()

    def _afficher_image_dans_chat(self, image_path):
        """Insère une miniature de l'image dans la zone de conversation"""
        try:
            img = Image.open(image_path)
            img.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(img)
            self.chat_display.config(state='normal')
            self.chat_display.image_create(tk.END, image=photo)
            self.chat_display.insert(tk.END, "\n")
            self.chat_display.see(tk.END)
            self.chat_display.config(state='disabled')
            self._chat_image_refs.append(photo)  # éviter le garbage collection
        except Exception as e:
            print(f"Erreur affichage image dans chat : {e}")

    def _on_conversation_selected(self, event):
        """Callback quand on sélectionne une conversation"""
        idx = self.chat_conversations_combo.current()
        if idx >= 0:
            conv_ids = list(sorted(self.conversations.keys(), reverse=True))
            if idx < len(conv_ids):
                self.charger_conversation(conv_ids[idx])
    
    def _on_nouvelle_conversation(self):
        """Callback pour créer une nouvelle conversation"""
        self.creer_nouvelle_conversation()
        self.afficher_conversation_actuelle()
        self.rafraichir_liste_conversations()
        self.chat_input.delete("1.0", tk.END)
    
    def _on_supprimer_conversation(self):
        """Callback pour supprimer la conversation actuelle"""
        if not self.current_conversation_id:
            return
        
        self.supprimer_conversation(self.current_conversation_id)
    
    def creer_onglet_generateur(self):
        """Onglet Générateur d'images"""
        # Titre
        ttk.Label(self.tab_images, text="Générateur d'Images IA", font=("Arial", 28, "bold")).pack(pady=5)
        
        # Cadre scrollable pour les paramètres
        params_frame = ttk.Frame(self.tab_images)
        params_frame.pack(fill="x", padx=10, pady=5)
        
        canvas = tk.Canvas(params_frame, bg="white", highlightthickness=0, height=200)
        scrollbar = ttk.Scrollbar(params_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Sélection du mode
        mode_frame = ttk.Frame(scrollable_frame)
        mode_frame.pack(pady=5, padx=5, fill="x")
        ttk.Label(mode_frame, text="Mode de génération :").pack(side="left", padx=5)
        self.mode_combo = ttk.Combobox(mode_frame, values=[
            "Local CPU (SD-Turbo)",
            "AWS Bedrock (Titan)",
            "Replicate (FLUX-Schnell)",
            "Pollinations.ai (Gratuit)",
            "Hugging Face API"
        ], state="readonly", width=25)
        self.mode_combo.current(0)
        self.mode_combo.pack(side="left", padx=5)

        # Choix de l'endpoint Hugging Face (sans modifier le code)
        hf_frame = ttk.Frame(scrollable_frame)
        hf_frame.pack(pady=2, padx=5, fill="x")
        ttk.Label(hf_frame, text="Endpoint Hugging Face :").pack(side="left", padx=5)
        self.hf_endpoint_combo = ttk.Combobox(
            hf_frame,
            values=[
                "Root (router)",
            ],
            state="readonly",
            width=25,
        )
        self.hf_endpoint_combo.set(self.hf_endpoint_strategy)
        self.hf_endpoint_combo.pack(side="left", padx=5)
        
        # Prompt
        ttk.Label(scrollable_frame, text="Description de l'image :").pack(pady=2, padx=5)
        self.image_prompt = tk.Text(scrollable_frame, height=2, width=60, font=("Arial", 16))
        self.image_prompt.pack(pady=2, padx=5)
        
        # Prompt négatif
        ttk.Label(scrollable_frame, text="Prompt négatif (à éviter) :").pack(pady=2, padx=5)
        self.image_negative_prompt = tk.Text(scrollable_frame, height=2, width=60, font=("Arial", 14))
        self.image_negative_prompt.pack(pady=2, padx=5)
        self.image_negative_prompt.insert(tk.END, "blurry, low quality, distorted")
        
        # Seed
        seed_frame = ttk.Frame(scrollable_frame)
        seed_frame.pack(pady=5, padx=5, fill="x")
        ttk.Label(seed_frame, text="Seed (optionnel) :").pack(side="left", padx=5)
        self.seed_entry = ttk.Entry(seed_frame, width=15)
        self.seed_entry.pack(side="left", padx=5)
        ttk.Label(seed_frame, text="(vide = aléatoire)").pack(side="left")
        
        # Boutons
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(pady=5, padx=5, fill="x")
        
        tk.Button(btn_frame, text="Générer", command=self.generer_image, bg="green", fg="white", width=11, font=("Arial", 12, "bold")).pack(side="left", padx=2)
        self.btn_recreer_gen = tk.Button(btn_frame, text="🔄 Recréer", command=self.recreer_image, bg="orange", fg="white", width=11, font=("Arial", 12, "bold"))
        self.btn_recreer_gen.pack(side="left", padx=2)
        tk.Button(btn_frame, text="Enregistrer", command=self.enregistrer_image, width=11, font=("Arial", 12, "bold")).pack(side="left", padx=2)
        tk.Button(btn_frame, text="📋 Galerie", command=self.ajouter_galerie, 
                 bg="green", fg="white", width=11, font=("Arial", 12, "bold")).pack(side="left", padx=2)
        
        # Status et progression
        self.image_status = ttk.Label(scrollable_frame, text="Prêt")
        self.image_status.pack(pady=1, padx=5)
        
        self.image_progress = ttk.Progressbar(scrollable_frame, length=400, mode='indeterminate')
        self.image_progress.pack(pady=1, padx=5)
        
        # PanedWindow pour séparer image et galerie
        paned = ttk.PanedWindow(self.tab_images, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Panneau 1 : Zone d'affichage de l'image
        image_frame = ttk.LabelFrame(paned, text="Résultat")
        paned.add(image_frame, weight=3)
        
        self.image_label = tk.Label(image_frame, relief="sunken", background="gray20")
        self.image_label.pack(fill="both", expand=True)
        
        # Panneau 2 : Galerie
        gallery_frame_gen = ttk.LabelFrame(paned, text="🖼️ Galerie (cliquez pour afficher)")
        paned.add(gallery_frame_gen, weight=1)
        
        gallery_canvas_frame_gen = ttk.Frame(gallery_frame_gen)
        gallery_canvas_frame_gen.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.gallery_canvas_gen = tk.Canvas(gallery_canvas_frame_gen, height=120, bg="#f0f0f0")
        gallery_scrollbar_gen = ttk.Scrollbar(gallery_canvas_frame_gen, orient="horizontal", command=self.gallery_canvas_gen.xview)
        self.gallery_canvas_gen.configure(xscrollcommand=gallery_scrollbar_gen.set)
        
        self.gallery_canvas_gen.pack(side="top", fill="both", expand=True)
        gallery_scrollbar_gen.pack(side="bottom", fill="x")
        
        self.gallery_frame_inner_gen = ttk.Frame(self.gallery_canvas_gen)
        self.gallery_canvas_gen.create_window((0, 0), window=self.gallery_frame_inner_gen, anchor="nw")
        
        # Défilement avec la molette
        self.gallery_canvas_gen.bind("<MouseWheel>", lambda e: self.gallery_canvas_gen.xview_scroll(-1 * (e.delta // 120), "units"))
        self.gallery_canvas_gen.bind("<Button-4>", lambda e: self.gallery_canvas_gen.xview_scroll(-1, "units"))
        self.gallery_canvas_gen.bind("<Button-5>", lambda e: self.gallery_canvas_gen.xview_scroll(1, "units"))
    
    def creer_onglet_assistant(self):
        """Onglet Assistant combiné"""
        ttk.Label(self.tab_assistant, text="Votre Assistant IA Complet", font=("Arial", 28, "bold")).pack(pady=5)
        
        # Instructions
        instructions = tk.Text(self.tab_assistant, height=3, wrap=tk.WORD, font=("Arial", 16))
        instructions.pack(fill="x", padx=10, pady=2)
        instructions.insert("1.0", 
            "💡 Décrivez ce que vous voulez en langage naturel. L'IA optimise votre demande et génère l'image.\n"
            "Exemple : 'Je veux une image relaxante pour méditer'")
        instructions.config(state='disabled', bg="#f0f0f0")

        # === SÉLECTEUR DE MODE (dédié à l'assistant) ===
        mode_frame = ttk.Frame(self.tab_assistant)
        mode_frame.pack(pady=5, padx=10, fill="x")
        
        ttk.Label(mode_frame, text="Mode de génération :", font=("Arial", 14, "bold")).pack(side="left", padx=5)
        self.assistant_mode_combo = ttk.Combobox(mode_frame, values=[
            "Local CPU (SD-Turbo)",
            "AWS Bedrock (Titan)",
            "Replicate (FLUX-Schnell)",
            "Pollinations.ai (Gratuit)",
            "Hugging Face API"
        ], state="readonly", width=28, font=("Arial", 12))
        self.assistant_mode_combo.current(0)
        self.assistant_mode_combo.pack(side="left", padx=5, fill="x", expand=True)

        # Sélecteur d'endpoint HuggingFace pour l'assistant
        hf_frame = ttk.Frame(self.tab_assistant)
        hf_frame.pack(pady=2, padx=10, fill="x")
        ttk.Label(hf_frame, text="Endpoint HF :", font=("Arial", 12)).pack(side="left", padx=5)
        self.assistant_hf_endpoint_combo = ttk.Combobox(
            hf_frame,
            values=[
                "Root (router)",
            ],
            state="readonly",
            width=25,
            font=("Arial", 12)
        )
        self.assistant_hf_endpoint_combo.set("Root (router)")
        self.assistant_hf_endpoint_combo.pack(side="left", padx=5)
        
        # PanedWindow PRINCIPAL - Vertical
        main_paned = ttk.PanedWindow(self.tab_assistant, orient=tk.VERTICAL)
        main_paned.pack(fill="both", expand=True, padx=10, pady=5)
        
        # PANNEAU 1 : PanedWindow HORIZONTAL (Inputs + Résultats)
        horizontal_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
        main_paned.add(horizontal_paned, weight=4)
        
        # === GAUCHE : PANNEAU D'ENTRÉE ===
        input_frame = tk.LabelFrame(horizontal_paned, text="📝 Paramètres", relief=tk.GROOVE)
        horizontal_paned.add(input_frame, weight=1)
        
        # Zone de saisie
        ttk.Label(input_frame, text="Votre demande :", font=("Arial", 14, "bold")).pack(pady=2)
        self.assistant_input = tk.Text(input_frame, height=2, font=("Arial", 14))
        self.assistant_input.pack(fill="both", expand=True, padx=8, pady=1)
        
        # Prompt négatif pour Assistant
        ttk.Label(input_frame, text="Prompt négatif :", font=("Arial", 14, "bold")).pack(pady=2)
        self.assistant_negative_prompt = tk.Text(input_frame, height=1, font=("Arial", 12))
        self.assistant_negative_prompt.pack(fill="both", expand=True, padx=8, pady=1)
        self.assistant_negative_prompt.insert(tk.END, "blurry, low quality, distorted")
        
        # Seed pour Assistant
        seed_frame_assistant = ttk.Frame(input_frame)
        seed_frame_assistant.pack(pady=2, padx=8, fill="x")
        ttk.Label(seed_frame_assistant, text="Seed :", font=("Arial", 12)).pack(side="left", padx=2)
        self.assistant_seed_entry = ttk.Entry(seed_frame_assistant, width=10, font=("Arial", 12))
        self.assistant_seed_entry.pack(side="left", padx=2)
        ttk.Label(seed_frame_assistant, text="(aléa)", font=("Arial", 10)).pack(side="left")
        
        # Boutons en colonne
        ttk.Separator(input_frame, orient="horizontal").pack(fill="x", pady=3)
        
        tk.Button(input_frame, text="🚀 Créer", command=self.assistant_creer, 
                 bg="purple", fg="white", font=("Arial", 14, "bold"), width=18).pack(padx=3, pady=1, fill="x")
        
        tk.Button(input_frame, text="📚 Bibliotheque", command=self.ouvrir_bibliotheque_prompts, 
                 bg="blue", fg="white", font=("Arial", 12), width=18).pack(padx=3, pady=1, fill="x")
        
        self.btn_recreer = tk.Button(input_frame, text="🔄 Recréer", command=self.assistant_recreer, 
                 bg="orange", fg="white", font=("Arial", 12), width=18)
        self.btn_recreer.pack(padx=3, pady=1, fill="x")
        
        self.btn_stop = tk.Button(input_frame, text="⏹️ Arrêter", command=self.stop_auto_recreate, 
                 bg="red", fg="white", font=("Arial", 12), width=18, state="disabled")
        self.btn_stop.pack(padx=3, pady=1, fill="x")
        
        tk.Button(input_frame, text="💾 Enregistrer", command=self.enregistrer_image, 
                 font=("Arial", 12), width=18).pack(padx=3, pady=1, fill="x")
        tk.Button(input_frame, text="📋 Ajouter Gal", command=self.ajouter_galerie, 
                 bg="green", fg="white", font=("Arial", 12), width=18).pack(padx=3, pady=1, fill="x")
        
        # === DROITE : PANNEAU DE RÉSULTATS ===
        results_frame = tk.LabelFrame(horizontal_paned, text="✨ Résultats", relief=tk.GROOVE)
        horizontal_paned.add(results_frame, weight=3)
        
        # PanedWindow HORIZONTAL pour les prompts et l'image
        vertical_results = ttk.PanedWindow(results_frame, orient=tk.VERTICAL)
        vertical_results.pack(fill="both", expand=True, padx=3, pady=3)
        
        # En haut : Image générée
        right_frame = tk.LabelFrame(vertical_results, text="Image générée (cliquez pour ajouter)", relief=tk.GROOVE)
        vertical_results.add(right_frame, weight=3)
        
        self.assistant_image = tk.Label(right_frame, relief="sunken", background="#e0e0e0", cursor="hand2")
        self.assistant_image.pack(fill="both", expand=True, padx=3, pady=3)
        self.assistant_image.bind("<Button-1>", lambda e: self.ajouter_galerie())
        # Redessiner l'image quand le label est redimensionné
        self.assistant_image.bind("<Configure>", self._redessiner_assistant_image)
        
        # En bas : Prompts optimisés
        prompts_frame = tk.LabelFrame(vertical_results, text="📝 Prompts optimisés par l'IA", relief=tk.GROOVE)
        vertical_results.add(prompts_frame, weight=2)
        
        # PanedWindow horizontal pour les deux prompts
        prompts_horizontal = ttk.PanedWindow(prompts_frame, orient=tk.HORIZONTAL)
        prompts_horizontal.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Prompt positif optimisé
        left_prompt_frame = tk.LabelFrame(prompts_horizontal, text="Positif", relief=tk.GROOVE)
        prompts_horizontal.add(left_prompt_frame, weight=1)
        self.assistant_prompt = scrolledtext.ScrolledText(left_prompt_frame, wrap=tk.WORD, height=4, font=("Arial", 12))
        self.assistant_prompt.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Prompt négatif optimisé
        right_prompt_frame = tk.LabelFrame(prompts_horizontal, text="Négatif", relief=tk.GROOVE)
        prompts_horizontal.add(right_prompt_frame, weight=1)
        self.assistant_negative_optimized = scrolledtext.ScrolledText(right_prompt_frame, wrap=tk.WORD, height=4, font=("Arial", 12))
        self.assistant_negative_optimized.pack(fill="both", expand=True, padx=2, pady=2)
        
        # PANNEAU 2 : Galerie (en bas, prise entière largeur)
        gallery_frame = tk.LabelFrame(main_paned, text="🖼️ Galerie (cliquez pour afficher)", relief=tk.GROOVE)
        main_paned.add(gallery_frame, weight=1)
        
        # Canvas avec scrollbar pour la galerie
        gallery_canvas_frame = ttk.Frame(gallery_frame)
        gallery_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.gallery_canvas = tk.Canvas(gallery_canvas_frame, height=100, bg="#f0f0f0")
        gallery_scrollbar = ttk.Scrollbar(gallery_canvas_frame, orient="horizontal", command=self.gallery_canvas.xview)
        self.gallery_canvas.configure(xscrollcommand=gallery_scrollbar.set)
        
        self.gallery_canvas.pack(side="top", fill="both", expand=True)
        gallery_scrollbar.pack(side="bottom", fill="x")
        
        self.gallery_frame_inner = ttk.Frame(self.gallery_canvas)
        self.gallery_canvas.create_window((0, 0), window=self.gallery_frame_inner, anchor="nw")
        
        # Défilement avec la molette
        self.gallery_canvas.bind("<MouseWheel>", lambda e: self.gallery_canvas.xview_scroll(-1 * (e.delta // 120), "units"))
        self.gallery_canvas.bind("<Button-4>", lambda e: self.gallery_canvas.xview_scroll(-1, "units"))
        self.gallery_canvas.bind("<Button-5>", lambda e: self.gallery_canvas.xview_scroll(1, "units"))
        
        # PANNEAU 3 : Conseils & Idées
        tips_frame = tk.LabelFrame(main_paned, text="💡 Conseils, Suggestions & Idées", relief=tk.GROOVE)
        main_paned.add(tips_frame, weight=1)
        
        # Notebook pour les conseils
        self.tips_notebook = ttk.Notebook(tips_frame)
        self.tips_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # === ONGLET 1 : CONSEILS ===
        tips_tab = ttk.Frame(self.tips_notebook)
        self.tips_notebook.add(tips_tab, text="📖 Conseils")
        
        tips_text = scrolledtext.ScrolledText(tips_tab, wrap=tk.WORD, font=("Arial", 12), height=5)
        tips_text.pack(fill="both", expand=True, padx=5, pady=5)
        tips_text.insert("1.0", 
            "✅ CONSEILS POUR DE MEILLEURS PROMPTS :\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "1️⃣ Soyez spécifique : Décrivez les couleurs, styles, ambiance\n"
            "2️⃣ Utilisez des références : 'dans le style de Van Gogh', 'photographie 8K'\n"
            "3️⃣ Ajoutez des détails d'éclairage : 'lumière dorée', 'cinématique', 'contre-jour'\n"
            "4️⃣ Évitez les négations : Utiliser le prompt négatif plutôt que 'pas de...'\n"
            "5️⃣ Soyez court mais dense : 50-150 mots c'est parfait\n"
            "6️⃣ Testez le même prompt plusieurs fois avec des seeds différents\n"
            "7️⃣ Combinez styles : 'oil painting mixed with watercolor, illustration'\n"
            "8️⃣ Précisez la composition : 'portrait, full body, macro, wide angle'\n"
            "\n🎨 STYLES POPULAIRES :\n"
            "• 3D rendering, Unreal Engine, Blender\n"
            "• Oil painting, Watercolor, Digital art\n"
            "• Photography, Cinematic, Movie poster\n"
            "• Anime, Cartoon, Comic book style")
        tips_text.config(state='disabled', bg="#f9f9f9")
        
        # === ONGLET 2 : SUGGESTIONS ===
        suggestions_tab = ttk.Frame(self.tips_notebook)
        self.tips_notebook.add(suggestions_tab, text="💬 Suggestions")
        
        suggestions_inner = ttk.Frame(suggestions_tab)
        suggestions_inner.pack(fill="both", expand=True, padx=5, pady=5)
        
        suggestions_text = scrolledtext.ScrolledText(suggestions_inner, wrap=tk.WORD, font=("Arial", 12), height=5)
        suggestions_text.pack(fill="both", expand=True, side="left")
        
        scrollbar_suggestions = ttk.Scrollbar(suggestions_inner, command=suggestions_text.yview)
        scrollbar_suggestions.pack(side="right", fill="y")
        suggestions_text.config(yscrollcommand=scrollbar_suggestions.set)
        
        suggestions_text.insert("1.0",
            "🎯 SUGGESTIONS D'AMÉLIORATIONS :\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📌 Si votre prompt est trop court :\n"
            "→ Ajouter : style, qualité, éclairage, composition\n\n"
            "📌 Si l'image n'a pas les détails désirés :\n"
            "→ Soyez plus précis sur les couleurs et matériaux\n\n"
            "📌 Si la qualité est faible :\n"
            "→ Ajoutez : '4K', 'cinematic', 'professional', 'detailed'\n\n"
            "📌 Si les proportions sont mauvaises :\n"
            "→ Spécifiez : 'anatomically correct', 'proper proportions'\n\n"
            "📌 Combiner styles pour des résultats uniques :\n"
            "→ 'oil painting + digital art', 'vintage + modern'\n\n"
            "📌 Pour plus de contrôle, utilisez des seeds :\n"
            "→ Même seed = même composition, différent prompt = variation\n\n"
            "⚡ PROMPT NÉGATIF EFFICACE :\n"
            "'blurry, low quality, distorted, deformed, ugly, bad anatomy,\n"
            "watermark, text, out of frame, oversaturated'")
        suggestions_text.config(state='disabled', bg="#fafafa")
        
        # === ONGLET 3 : IDÉES D'IMAGES ===
        ideas_tab = ttk.Frame(self.tips_notebook)
        self.tips_notebook.add(ideas_tab, text="🎨 Idées d'Images")
        
        # Frame scrollable pour les idées cliquables
        ideas_canvas_frame = ttk.Frame(ideas_tab)
        ideas_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ideas_canvas = tk.Canvas(ideas_canvas_frame, bg="#ffffff", height=150)
        ideas_scrollbar = ttk.Scrollbar(ideas_canvas_frame, orient="vertical", command=ideas_canvas.yview)
        ideas_canvas.configure(yscrollcommand=ideas_scrollbar.set)
        
        ideas_canvas.pack(side="left", fill="both", expand=True)
        ideas_scrollbar.pack(side="right", fill="y")
        
        ideas_frame_inner = ttk.Frame(ideas_canvas)
        ideas_canvas.create_window((0, 0), window=ideas_frame_inner, anchor="nw")
        
        # Idées prédéfinies cliquables
        ideas_list = [
            ("🌅 Coucher de soleil tropical", "A stunning tropical sunset over crystal clear ocean, golden hour lighting, cinematic, 8K, detailed sky"),
            ("🏰 Château fantasy", "A magical enchanted castle with floating islands, fantasy art style, bioluminescent lights, magical atmosphere, intricate details"),
            ("🤖 Cyborg futuriste", "A sleek retro-futuristic cyborg character, cyberpunk style, neon lights, detailed metallic parts, cinematic lighting, 8K"),
            ("🌌 Espace galactique", "A beautiful galaxy with nebulas and stars, cosmic art, vibrant colors, deep space, 3D rendering, cinematic"),
            ("🦁 Animal majestueux", "A majestic lion portrait in natural light, wildlife photography style, sharp focus, professional details, 4K"),
            ("🎭 Portrait artistique", "An artistic portrait with surreal elements, oil painting style, dramatic lighting, detailed features, masterpiece"),
            ("🌿 Nature relaxante", "A serene forest scene with a misty waterfall, peaceful atmosphere, natural lighting, 8K resolution, botanical details"),
            ("👽 Créature alien", "A fascinating alien creature design, sci-fi concept art, bioluminescent features, extraterrestrial, detailed anatomy, cinematic"),
            ("🏙️ Ville cyberpunk", "A dark futuristic mega-city with neon lights, cyberpunk aesthetic, flying vehicles, detailed architecture, moody atmosphere"),
            ("✨ Monde magique", "A magical fantasy world with floating objects, glowing runes, mystical atmosphere, detailed environment, cinematic, 8K")
        ]
        
        for emoji_title, prompt in ideas_list:
            btn = tk.Button(
                ideas_frame_inner,
                text=emoji_title,
                font=("Arial", 12),
                bg="#e3f2fd",
                fg="#1976d2",
                relief=tk.RAISED,
                padx=10,
                pady=6,
                wraplength=250,
                justify="center",
                cursor="hand2",
                command=lambda p=prompt: self.utiliser_idee(p)
            )
            btn.pack(fill="x", padx=3, pady=2)
            btn.bind("<Enter>", lambda e, p=prompt: self.afficher_tooltip_idee(e, p))
            btn.bind("<Leave>", lambda e: self.cacher_tooltip_idee())
        
        # Recalculer la zone de scroll
        ideas_frame_inner.update_idletasks()
        ideas_canvas.configure(scrollregion=ideas_canvas.bbox("all"))
        
        # Défilement à la molette
        ideas_canvas.bind("<MouseWheel>", lambda e: ideas_canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        ideas_canvas.bind("<Button-4>", lambda e: ideas_canvas.yview_scroll(-1, "units"))
        ideas_canvas.bind("<Button-5>", lambda e: ideas_canvas.yview_scroll(1, "units"))
        
        # Status
        self.assistant_status = ttk.Label(self.tab_assistant, text="Prêt")
        self.assistant_status.pack(pady=5)
    
    # === FONCTIONS CHATBOT ===
    
    def envoyer_message_chat(self):
        """Envoie un message au ChatBot"""
        source = self.chat_source_combo.get() if hasattr(self, "chat_source_combo") else "Ollama (local)"
        if "Ollama" in source and not OLLAMA_AVAILABLE:
            self.afficher_chat("Système", "❌ Ollama non installé : pip install ollama")
            return

        message = self.chat_input.get("1.0", tk.END).strip()
        image_path = self.chat_pending_image

        if not message and not image_path:
            return

        self.chat_input.delete("1.0", tk.END)

        # Afficher le message utilisateur (avec indication de l'image si jointe)
        if image_path:
            img_name = os.path.basename(image_path)
            display_msg = f"{message}\n[🖼️ Image jointe : {img_name}]" if message else f"[🖼️ Image jointe : {img_name}]"
        else:
            display_msg = message
        self.afficher_chat("Vous", display_msg)

        # Afficher la miniature dans le chat puis réinitialiser la sélection
        if image_path:
            self._afficher_image_dans_chat(image_path)
            self._supprimer_image_chat()

        self.chat_status.config(text="Réflexion en cours...", foreground="orange")
        threading.Thread(target=self._chat_thread, args=(message, image_path), daemon=True).start()
    
    def _chat_thread(self, message, image_path=None):
        """Thread pour le ChatBot"""
        try:
            # Récupérer les instructions négatives
            negative_instructions = self.chat_negative.get("1.0", tk.END).strip()

            # Construire le prompt avec instructions négatives
            if negative_instructions:
                full_prompt = f"{message}\n\n[Directives à respecter] {negative_instructions}"
            else:
                full_prompt = message if message else "Décris cette image."

            # Système prompt pour forcer la réponse en français
            system_prompt = "Tu es un assistant utile, honnête et inoffensif. Tu DOIS toujours répondre UNIQUEMENT en français, sans aucune exception. Tous tes messages doivent être en français."

            source = self.chat_source_combo.get() if hasattr(self, "chat_source_combo") else "Ollama (local)"

            if "Hugging Face" in source:
                response = self._chat_huggingface(full_prompt, system_prompt, image_path=image_path)
            else:
                response = ""
                if image_path:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    for chunk in ollama.generate(
                        model=self.ollama_model, prompt=full_prompt,
                        system=system_prompt, stream=True, images=[image_bytes]
                    ):
                        response += chunk.get('response', '')
                else:
                    for chunk in ollama.generate(
                        model=self.ollama_model, prompt=full_prompt,
                        system=system_prompt, stream=True
                    ):
                        response += chunk.get('response', '')

            self.afficher_chat("IA", response)

            # Lire la réponse à voix haute
            voice_enabled = self.chat_voice_enabled.get() if hasattr(self, "chat_voice_enabled") else True
            if self.tts_service and voice_enabled:
                self.speak(response)

            self.chat_status.config(text="Prêt", foreground="green")
        except Exception as e:
            self.afficher_chat("Système", f"❌ Erreur : {e}")
            self.chat_status.config(text="Erreur", foreground="red")

    def _chat_huggingface(self, full_prompt, system_prompt, image_path=None):
        """Génère une réponse texte via HuggingFace Inference API (OpenAI-compatible)"""
        try:
            from config import HUGGING_FACE_TOKEN
        except ImportError:
            raise Exception("Token Hugging Face manquant dans config.py")

        try:
            from config import HUGGING_FACE_API_ROOT
        except ImportError:
            HUGGING_FACE_API_ROOT = "https://router.huggingface.co/hf-inference"

        model = "mistralai/Mistral-7B-Instruct-v0.3"
        if hasattr(self, "chat_hf_model_combo"):
            model = self.chat_hf_model_combo.get().strip() or model

        url = f"{HUGGING_FACE_API_ROOT.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
            "Content-Type": "application/json",
        }

        # Construire le contenu utilisateur (texte seul ou texte + image)
        if image_path:
            import base64
            ext = os.path.splitext(image_path)[1].lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
            mime = mime_map.get(ext, "image/jpeg")
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": full_prompt},
            ]
        else:
            user_content = full_prompt

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 1024,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if not response.ok:
            details = response.text[:400] if response.text else "Aucun détail"
            raise Exception(f"Hugging Face API ({response.status_code}): {details}")

        data = response.json()
        return data["choices"][0]["message"]["content"]


    
    def afficher_chat(self, sender, message):
        """Affiche un message dans le chat et le sauvegarde"""
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"\n{sender} : {message}\n", sender)
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')
        
        # Sauvegarder le message dans la conversation
        self.ajouter_message(sender, message)
    
    # === FONCTIONS GÉNÉRATEUR ===
    
    def generer_image(self):
        """Génère une image"""
        prompt = self.image_prompt.get("1.0", tk.END).strip()
        if not prompt:
            return

        # Stocker les derniers paramètres pour permettre la recréation.
        self.last_generator_prompt = prompt
        self.last_generator_negative_prompt = self.image_negative_prompt.get("1.0", tk.END).strip()

        mode = self.mode_combo.get()
        
        if "Local" in mode and not self.sd_pipe:
            self.image_status.config(text="❌ Modèle SD-Turbo en chargement...")
            return
        
        self.image_status.config(text="Génération en cours...")
        self.image_progress.start()
        
        threading.Thread(
            target=self._generer_image_thread,
            args=(prompt, self.last_generator_negative_prompt),
            daemon=True,
        ).start()

    def recreer_image(self):
        """Active/désactive la recréation continue avec le dernier prompt du Générateur."""
        if self.auto_recreate_generator:
            self.stop_generator_auto_recreate()
            return

        prompt = self.last_generator_prompt.strip()
        if not prompt:
            prompt = self.image_prompt.get("1.0", tk.END).strip()

        if not prompt:
            self.image_status.config(text="❌ Générez d'abord une image")
            return

        mode = self.mode_combo.get()
        if "Local" in mode and not self.sd_pipe:
            self.image_status.config(text="❌ Modèle SD-Turbo en chargement...")
            return

        last_negative_prompt = self.last_generator_negative_prompt.strip()
        if not last_negative_prompt:
            last_negative_prompt = self.image_negative_prompt.get("1.0", tk.END).strip()

        self.last_generator_prompt = prompt
        self.last_generator_negative_prompt = last_negative_prompt
        self.auto_recreate_generator = True
        self.btn_recreer_gen.config(text="⏹️ Stop", bg="#c62828")
        self.image_status.config(text="🔄 Recréation continue activée...")
        self.image_progress.start()
        threading.Thread(
            target=self._generator_recreate_loop_thread,
            args=(prompt, last_negative_prompt),
            daemon=True,
        ).start()

    def stop_generator_auto_recreate(self):
        """Arrête la recréation continue côté Générateur."""
        self.auto_recreate_generator = False
        self.btn_recreer_gen.config(text="🔄 Recréer", bg="orange")
        self.image_progress.stop()
        self.image_status.config(text="⏹️ Recréation continue arrêtée")

    def _generator_recreate_loop_thread(self, prompt, negative_prompt):
        """Boucle de recréation continue pour le Générateur."""
        iteration = 0
        while self.auto_recreate_generator:
            try:
                self.image_progress.start()
                success = self._generer_image_thread(prompt, negative_prompt)
                if not success:
                    self.auto_recreate_generator = False
                    break

                if not self.auto_recreate_generator:
                    break

                iteration += 1
                self.image_status.config(text=f"🔄 Recréation continue #{iteration} terminée")

                if self.auto_recreate_delay_sec > 0:
                    time.sleep(self.auto_recreate_delay_sec)
            except Exception as e:
                self.auto_recreate_generator = False
                self.image_status.config(text=f"❌ Erreur recréation continue : {e}")
                break

        self.btn_recreer_gen.config(text="🔄 Recréer", bg="orange")
        self.image_progress.stop()
    
    def _generer_image_thread(self, prompt, negative_prompt=None):
        """Thread pour générer l'image"""
        try:
            mode = self.mode_combo.get()
            start = time.time()
            
            # Gérer le seed
            seed_text = self.seed_entry.get().strip()
            generation_seed = None
            if seed_text:
                generation_seed = int(seed_text)
                if self.torch is None:
                    raise Exception("Torch n'est pas encore chargé (attendre SD-Turbo)")
                self.generator = self.torch.Generator().manual_seed(generation_seed)
            else:
                generation_seed = random.randint(0, 2**31 - 1)
                if "Local" in mode and self.torch is not None:
                    self.generator = self.torch.Generator().manual_seed(generation_seed)
                else:
                    self.generator = None

            # Sécurité: si mode local et seed explicite mais torch non chargé.
            if "Local" in mode and self.generator is None and self.torch is not None:
                self.generator = self.torch.Generator().manual_seed(generation_seed)

            if generation_seed is None:
                generation_seed = random.randint(0, 2**31 - 1)

            if not seed_text and "Local" not in mode:
                self.generator = None
            
            # Récupérer le prompt négatif
            if negative_prompt is None:
                negative_prompt = self.image_negative_prompt.get("1.0", tk.END).strip()
            
            if "Local" in mode:
                if not self.sd_pipe:
                    raise Exception("SD-Turbo en chargement...")
                image = self.sd_pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=2, guidance_scale=0.0, generator=self.generator).images[0]
            
            elif "AWS" in mode:
                image = self._generer_aws(prompt, seed=generation_seed)
            
            elif "Replicate" in mode:
                image = self._generer_replicate(prompt, seed=generation_seed)
            
            elif "Pollinations" in mode:
                image = self._generer_pollinations(prompt, seed=generation_seed)
            
            elif "Hugging Face" in mode:
                image = self._generer_huggingface(prompt, seed=generation_seed)
            
            else:
                raise Exception("Mode inconnu")
            
            elapsed = time.time() - start
            self.current_image = image
            self.afficher_image(image, self.image_label)
            chemin, save_error = self.sauvegarder_image_auto_safe(image, prompt, mode)
            if chemin:
                self.image_status.config(text=f"✅ Terminé en {elapsed:.1f}s - Sauvé: {chemin.name}")
            else:
                self.image_status.config(text=f"✅ Terminé en {elapsed:.1f}s - ⚠️ Sauvegarde auto impossible ({save_error})")
            return True
        except Exception as e:
            self.image_status.config(text=f"❌ Erreur : {e}")
            return False
        finally:
            self.image_progress.stop()
    
    def _generer_aws(self, prompt, seed=None):
        """Génère avec AWS Bedrock"""
        if not AWS_AVAILABLE:
            raise Exception("boto3 non installé")
        
        image_generation_config = {
            "numberOfImages": 1,
            "quality": "standard",
            "cfgScale": 8.0,
            "width": 512,
            "height": 512
        }
        if seed is not None:
            image_generation_config["seed"] = int(seed)

        bedrock = boto3.client('bedrock-runtime', region_name='eu-central-1')
        body = json.dumps({
            "textToImageParams": {"text": prompt},
            "taskType": "TEXT_IMAGE",
            "imageGenerationConfig": image_generation_config
        })
        response = bedrock.invoke_model(body=body, modelId="amazon.titan-image-generator-v1")
        response_body = json.loads(response['body'].read())
        import base64
        image_data = base64.b64decode(response_body['images'][0])
        return Image.open(BytesIO(image_data))
    
    def _generer_replicate(self, prompt, seed=None):
        """Génère avec Replicate"""
        if not REPLICATE_AVAILABLE or not REPLICATE_API_TOKEN:
            raise Exception("Replicate non configuré")
        
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        replicate_input = {
            "prompt": prompt,
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "png"
        }
        if seed is not None:
            replicate_input["seed"] = int(seed)

        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input=replicate_input
        )
        # Convertir l'itérateur en liste si nécessaire
        if hasattr(output, '__iter__') and not isinstance(output, (str, bytes)):
            output = list(output)
        image_url = output[0] if isinstance(output, list) else output
        response = requests.get(str(image_url))
        return Image.open(BytesIO(response.content))
    
    def _generer_pollinations(self, prompt, seed=None):
        """Génère avec Pollinations.ai"""
        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
        if seed is not None:
            url = f"{url}?seed={int(seed)}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    
    def _generer_huggingface(self, prompt, seed=None):
        """Génère avec Hugging Face API"""
        try:
            from config import HUGGING_FACE_TOKEN
        except ImportError:
            raise Exception("Token Hugging Face manquant dans config.py")

        # Nouvelle API root Hugging Face (configurable), avec fallback legacy.
        try:
            from config import HUGGING_FACE_API_ROOT
        except ImportError:
            HUGGING_FACE_API_ROOT = "https://router.huggingface.co/hf-inference"

        try:
            from config import HUGGING_FACE_IMAGE_MODEL
        except ImportError:
            HUGGING_FACE_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

        api_url = f"{HUGGING_FACE_API_ROOT.rstrip('/')}/models/{HUGGING_FACE_IMAGE_MODEL}"
        headers = {
            "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
            "Accept": "image/png",
            "Content-Type": "application/json",
        }

        payload = {"inputs": prompt}
        if seed is not None:
            payload["parameters"] = {"seed": int(seed)}

        response = requests.post(api_url, headers=headers, json=payload, timeout=60)

        if not response.ok:
            details = response.text[:300] if response.text else "Aucun détail"
            if response.status_code == 410:
                raise Exception(
                    "Erreur Hugging Face API (410): endpoint obsolète. "
                    "Utilisez HUGGING_FACE_API_ROOT='https://router.huggingface.co/hf-inference'."
                )
            raise Exception(f"Erreur Hugging Face API ({response.status_code}): {details}")

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            details = response.text[:300] if response.text else "Réponse JSON sans image"
            raise Exception(f"Réponse Hugging Face inattendue: {details}")

        return Image.open(BytesIO(response.content))
    
    def afficher_image(self, image, label):
        """Affiche une image dans un label de manière adaptative"""
        # Stocker l'image originale pour redimensionnement adaptatif
        if label == self.assistant_image:
            self.original_assistant_image = image.copy()
        else:
            self.original_image = image.copy()
        
        # Redessiner avec les dimensions actuelles du label
        self._redessiner_image(image, label)
    
    def _redessiner_image(self, image, label):
        """Redessine l'image en fonction de la taille actuelle du label"""
        # Obtenir les dimensions du label
        label.update_idletasks()
        width = label.winfo_width()
        height = label.winfo_height()
        
        # Si le label n'a pas encore de taille, utiliser des dimensions par défaut
        if width <= 1 or height <= 1:
            width = 512
            height = 512
        
        # Redimensionner l'image en gardant les proportions
        img_copy = image.copy()
        img_copy.thumbnail((width, height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img_copy)
        label.config(image=photo)
        label.image = photo
    
    def _redessiner_assistant_image(self, event):
        """Redessine l'image de l'assistant quand la fenêtre est redimensionnée"""
        if self.original_assistant_image is not None:
            self._redessiner_image(self.original_assistant_image, self.assistant_image)

    def _slugifier_nom_image(self, texte, longueur_max=60):
        """Construit un nom de fichier lisible a partir du prompt."""
        texte = (texte or "image_generee").strip().lower()
        texte = re.sub(r"[^a-z0-9]+", "_", texte)
        texte = texte.strip("_") or "image_generee"
        return texte[:longueur_max].rstrip("_") or "image_generee"

    def sauvegarder_image_auto(self, image, prompt, origine):
        """Enregistre automatiquement une image generee dans le dossier images."""
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_nom = self._slugifier_nom_image(prompt)
        origine = self._slugifier_nom_image(origine, longueur_max=20)
        chemin = self.images_dir / f"{horodatage}_{origine}_{base_nom}.png"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        image.save(chemin, format="PNG")
        return chemin

    def sauvegarder_image_auto_safe(self, image, prompt, origine):
        """Version non bloquante: renvoie (chemin, erreur)."""
        try:
            chemin = self.sauvegarder_image_auto(image, prompt, origine)
            return chemin, None
        except Exception as e:
            return None, str(e)
    
    def enregistrer_image(self):
        """Enregistre l'image actuelle"""
        if not self.current_image:
            return
        
        filepath = filedialog.asksaveasfilename(
            initialdir=str(self.images_dir),
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if filepath:
            self.current_image.save(filepath)
            self.image_status.config(text=f"✅ Enregistré : {os.path.basename(filepath)}")
    
    # === FONCTIONS ASSISTANT COMBINÉ ===
    
    def ouvrir_bibliotheque_prompts(self):
        """Ouvre la fenêtre de visualisation de la bibliothèque de prompts"""
        if not PROMPTS_LIBRARY_AVAILABLE:
            messagebox.showwarning("⚠️ Non disponible", "Bibliothèque de prompts non chargée")
            return
        
        # Ouvre la fenêtre en mode non-bloquant (threading)
        threading.Thread(target=lambda: open_prompts_viewer(self.root), daemon=True).start()
    
    def assistant_creer(self):
        """Mode assistant : optimise le prompt puis génère l'image"""
        demande = self.assistant_input.get("1.0", tk.END).strip()
        if not demande:
            return
        
        mode = self.assistant_mode_combo.get()

        if not OLLAMA_AVAILABLE:
            self.assistant_status.config(text="❌ Services non disponibles")
            return

        if "Local" in mode and not self.sd_pipe:
            self.assistant_status.config(text="❌ SD-Turbo en chargement (mode local)")
            return
        
        self.assistant_status.config(text="🤖 L'IA optimise votre demande...")
        threading.Thread(target=self._assistant_thread, args=(demande,), daemon=True).start()
    
    def _assistant_thread(self, demande):
        """Thread pour l'assistant combiné"""
        try:
            # Déterminer la catégorie (vous pouvez améliorer cette logique)
            categories = get_all_categories() if PROMPTS_LIBRARY_AVAILABLE else ["portrait"]
            category = "portrait"  # Catégorie par défaut
            
            # Essayer de détecter la catégorie à partir de la demande
            demande_lower = demande.lower()
            keywords = {
                "portrait": ["visage", "personne", "homme", "femme", "headshot", "portrait"],
                "landscape": ["paysage", "montagne", "forêt", "rivière", "nature", "landscape"],
                "fantasy": ["dragon", "magie", "fantaisie", "créature", "wizard", "fantasy"],
                "abstract": ["abstrait", "art", "moderne", "géométrique", "abstract"],
                "cyberpunk": ["cyberpunk", "neon", "robot", "technologie", "futur", "cyber"]
            }
            
            for cat, keywords_list in keywords.items():
                if any(kw in demande_lower for kw in keywords_list):
                    category = cat
                    break
            
            # Étape 1 : Optimiser le prompt avec la bibliothèque si disponible
            if PROMPTS_LIBRARY_AVAILABLE:
                user_negative_prompt = self.assistant_negative_prompt.get("1.0", tk.END).strip()
                prompt_optimization, negative_optimization = create_system_prompt_with_examples(
                    demande, 
                    category=category, 
                    negative_prompt=user_negative_prompt
                )
            else:
                # Fallback: sans bibliothèque
                prompt_optimization = (
                    f"Tu es un expert en génération d'images IA. "
                    f"L'utilisateur veut : '{demande}'. "
                    f"Crée un prompt détaillé en anglais pour Stable Diffusion (max 50 mots). "
                    f"Réponds UNIQUEMENT avec le prompt, sans explication."
                )
                negative_optimization = ""
            
            self.assistant_status.config(text=f"🤖 Optimisation du prompt (catégorie: {category})...")
            
            # Système prompt pour optimisation (en français)
            system_prompt = "Tu es un expert en génération d'images IA. Tu DOIS répondre UNIQUEMENT en français et fournir des prompts optimisés en anglais pour les modèles IA."
            
            optimized_prompt = ""
            for chunk in ollama.generate(model=self.ollama_model, prompt=prompt_optimization, system=system_prompt, stream=True):
                optimized_prompt += chunk.get('response', '')
            
            # Étape 1b : Optimiser le prompt négatif avec Ollama
            user_negative_prompt = self.assistant_negative_prompt.get("1.0", tk.END).strip()
            if user_negative_prompt and "Exemple:" not in user_negative_prompt:
                if not PROMPTS_LIBRARY_AVAILABLE:
                    negative_optimization = (
                        f"Tu es un expert en génération d'images IA. "
                        f"L'utilisateur veut éviter : '{user_negative_prompt}'. "
                        f"Crée un prompt négatif détaillé en anglais pour Stable Diffusion (max 30 mots). "
                        f"Réponds UNIQUEMENT avec le prompt, sans explication."
                    )
                
                optimized_negative = ""
                for chunk in ollama.generate(model=self.ollama_model, prompt=negative_optimization, system=system_prompt, stream=True):
                    optimized_negative += chunk.get('response', '')
            else:
                optimized_negative = user_negative_prompt
            
            # Afficher les prompts optimisés
            self.assistant_prompt.delete("1.0", tk.END)
            self.assistant_prompt.insert("1.0", optimized_prompt.strip())
            
            self.assistant_negative_optimized.delete("1.0", tk.END)
            self.assistant_negative_optimized.insert("1.0", optimized_negative.strip())
            
            self.assistant_status.config(text="🎨 Génération de l'image...")
            
            # Étape 2 : Générer l'image (utilise le mode sélectionné)
            start = time.time()
            mode = self.assistant_mode_combo.get()
            
            # Gérer le seed pour Assistant
            seed_text = self.assistant_seed_entry.get().strip()
            generation_seed = None
            if seed_text:
                generation_seed = int(seed_text)
                if self.torch is None:
                    raise Exception("Torch n'est pas encore chargé (attendre SD-Turbo)")
                self.generator = self.torch.Generator().manual_seed(generation_seed)
            else:
                generation_seed = random.randint(0, 2**31 - 1)
                if "Local" in mode and self.torch is not None:
                    self.generator = self.torch.Generator().manual_seed(generation_seed)
                else:
                    self.generator = None

            if generation_seed is None:
                generation_seed = random.randint(0, 2**31 - 1)
            
            if "Local" in mode:
                if not self.sd_pipe:
                    raise Exception("SD-Turbo en chargement...")
                image = self.sd_pipe(prompt=optimized_prompt.strip(), negative_prompt=optimized_negative.strip(), num_inference_steps=2, guidance_scale=0.0, generator=self.generator).images[0]
            elif "AWS" in mode:
                image = self._generer_aws(optimized_prompt.strip(), seed=generation_seed)
            elif "Replicate" in mode:
                image = self._generer_replicate(optimized_prompt.strip(), seed=generation_seed)
            elif "Pollinations" in mode:
                image = self._generer_pollinations(optimized_prompt.strip(), seed=generation_seed)
            elif "Hugging Face" in mode:
                image = self._generer_huggingface(optimized_prompt.strip(), seed=generation_seed)
            else:
                raise Exception("Mode inconnu")
            
            elapsed = time.time() - start
            
            # Stocker les prompts optimisés pour la fonction Recréer
            self.last_optimized_prompt = optimized_prompt.strip()
            self.last_optimized_negative_prompt = optimized_negative.strip()
            
            self.current_image = image
            self.afficher_image(image, self.assistant_image)
            chemin, save_error = self.sauvegarder_image_auto_safe(image, optimized_prompt.strip(), mode)
            if chemin:
                self.assistant_status.config(text=f"✅ Terminé en {elapsed:.1f}s - Catégorie: {category} - Sauvé: {chemin.name}")
            else:
                self.assistant_status.config(
                    text=f"✅ Terminé en {elapsed:.1f}s - Catégorie: {category} - ⚠️ Sauvegarde auto impossible ({save_error})"
                )
            
        except Exception as e:
            self.assistant_status.config(text=f"❌ Erreur : {e}")
    
    def assistant_recreer(self):
        """Recrée l'image avec le même prompt mais un nouveau seed aléatoire"""
        if not self.last_optimized_prompt:
            self.assistant_status.config(text="❌ Créez d'abord une image avec '🚀 Créer avec l'IA'")
            return

        mode = self.assistant_mode_combo.get()
        if "Local" in mode and not self.sd_pipe:
            self.assistant_status.config(text="❌ SD-Turbo en chargement (mode local)")
            return
        
        # Activer le mode auto-recréation
        self.auto_recreate = True
        self.btn_recreer.config(state="disabled")
        self.btn_stop.config(state="normal")
        
        self.assistant_status.config(text="🔄 Recréation automatique activée...")
        threading.Thread(target=self._assistant_recreer_thread, daemon=True).start()
    
    def stop_auto_recreate(self):
        """Arrête la recréation automatique"""
        self.auto_recreate = False
        self.btn_recreer.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.assistant_status.config(text="⏹️ Recréation automatique arrêtée")

    def _assistant_recreate_success_ui(self, image, elapsed, chemin, save_error):
        """Applique le résultat de recréation dans le thread UI."""
        try:
            self.current_image = image
            self.afficher_image(image, self.assistant_image)

            # Ajouter automatiquement à la galerie (opération Tkinter: thread UI uniquement)
            self.ajouter_galerie()

            if chemin:
                self.assistant_status.config(
                    text=f"✅ Recréé en {elapsed:.1f}s - Galerie: {len(self.gallery_images)} - Sauvé: {chemin.name}"
                )
            else:
                self.assistant_status.config(
                    text=f"✅ Recréé en {elapsed:.1f}s - Galerie: {len(self.gallery_images)} - ⚠️ Sauvegarde auto impossible ({save_error})"
                )
        except Exception as e:
            self.auto_recreate = False
            self.btn_recreer.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.assistant_status.config(text=f"❌ Erreur UI recréation : {e}")

    def _assistant_recreate_error_ui(self, error_message):
        """Affiche une erreur de recréation dans le thread UI."""
        self.btn_recreer.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.assistant_status.config(text=f"❌ Erreur : {error_message}")
    
    def _assistant_recreer_thread(self):
        """Thread pour recréer l'image avec nouveau seed en boucle"""
        while self.auto_recreate:
            try:
                start = time.time()
                mode = self.assistant_mode_combo.get()
                
                # Toujours utiliser un seed aléatoire explicite pour garantir la variation.
                recreate_seed = random.randint(0, 2**31 - 1)
                if "Local" in mode and self.torch is not None:
                    self.generator = self.torch.Generator().manual_seed(recreate_seed)
                else:
                    self.generator = None
                
                if "Local" in mode:
                    if not self.sd_pipe:
                        raise Exception("SD-Turbo en chargement...")
                    image = self.sd_pipe(prompt=self.last_optimized_prompt, negative_prompt=self.last_optimized_negative_prompt, num_inference_steps=2, guidance_scale=0.0, generator=self.generator).images[0]
                elif "AWS" in mode:
                    image = self._generer_aws(self.last_optimized_prompt, seed=recreate_seed)
                elif "Replicate" in mode:
                    image = self._generer_replicate(self.last_optimized_prompt, seed=recreate_seed)
                elif "Pollinations" in mode:
                    image = self._generer_pollinations(self.last_optimized_prompt, seed=recreate_seed)
                elif "Hugging Face" in mode:
                    image = self._generer_huggingface(self.last_optimized_prompt, seed=recreate_seed)
                else:
                    raise Exception("Mode inconnu")
                
                elapsed = time.time() - start
                chemin, save_error = self.sauvegarder_image_auto_safe(
                    image,
                    self.last_optimized_prompt,
                    f"{mode}_recreation"
                )

                # Toute opération Tkinter doit être exécutée sur le thread principal.
                self.root.after(0, self._assistant_recreate_success_ui, image, elapsed, chemin, save_error)

                # Limiter la cadence pour eviter de saturer CPU/API en mode auto.
                if self.auto_recreate and self.auto_recreate_delay_sec > 0:
                    time.sleep(self.auto_recreate_delay_sec)
                
            except Exception as e:
                self.auto_recreate = False
                self.root.after(0, self._assistant_recreate_error_ui, str(e))
                break
    
    # === FONCTIONS GALERIE ===
    
    def ajouter_galerie(self):
        """Ajoute l'image actuelle à la galerie"""
        if not self.current_image:
            print("Aucune image à ajouter")
            return
        
        try:
            self.gallery_images.append(self.current_image.copy())
            
            thumb = self.current_image.copy()
            thumb.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(thumb)
            
            idx = len(self.gallery_images) - 1
            
            # Ajouter dans l'onglet Générateur (si accessible)
            try:
                if hasattr(self, 'gallery_frame_inner_gen') and hasattr(self, 'gallery_canvas_gen'):
                    btn_gen = tk.Button(self.gallery_frame_inner_gen, image=photo, relief="raised", bd=2,
                                       command=lambda i=idx: self.afficher_depuis_galerie(i))
                    btn_gen.image = photo
                    btn_gen.pack(side="left", padx=5, pady=5)
                    self.gallery_frame_inner_gen.update_idletasks()
                    self.gallery_canvas_gen.configure(scrollregion=self.gallery_canvas_gen.bbox("all"))
            except Exception as e:
                print(f"Erreur ajout galerie Générateur: {e}")
            
            # Ajouter dans l'onglet Assistant (si accessible)
            try:
                if hasattr(self, 'gallery_frame_inner') and hasattr(self, 'gallery_canvas'):
                    photo2 = ImageTk.PhotoImage(self.current_image.copy().resize((100, 100)))
                    btn_asst = tk.Button(self.gallery_frame_inner, image=photo2, relief="raised", bd=2,
                                        command=lambda i=idx: self.afficher_depuis_galerie(i))
                    btn_asst.image = photo2
                    btn_asst.pack(side="left", padx=5, pady=5)
                    self.gallery_frame_inner.update_idletasks()
                    self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
            except Exception as e:
                print(f"Erreur ajout galerie Assistant: {e}")
            
            status_msg = f"✅ Ajouté à la galerie ({len(self.gallery_images)} images)"
            if hasattr(self, 'image_status'):
                self.image_status.config(text=status_msg)
            if hasattr(self, 'assistant_status'):
                self.assistant_status.config(text=status_msg)
        except Exception as e:
            print(f"Erreur générale ajouter_galerie: {e}")
    
    def afficher_depuis_galerie(self, index):
        """Affiche une image depuis la galerie"""
        if 0 <= index < len(self.gallery_images):
            self.current_image = self.gallery_images[index]
            self.afficher_image(self.current_image, self.assistant_image)
            self.afficher_image(self.current_image, self.image_label)
            status_msg = f"🖼️ Image {index+1}/{len(self.gallery_images)} affichée"
            self.assistant_status.config(text=status_msg)
            self.image_status.config(text=status_msg)
    
    # === FONCTIONS CONSEILS & IDÉES ===
    
    def utiliser_idee(self, prompt):
        """Insère une idée prédéfinie dans le champ de demande de l'Assistant"""
        self.assistant_input.delete("1.0", tk.END)
        self.assistant_input.insert("1.0", prompt)
        self.assistant_status.config(text="💡 Idée insérée ! Cliquez sur '🚀 Créer' pour générer", foreground="blue")
    
    def afficher_tooltip_idee(self, event, prompt):
        """Affiche un tooltip avec le prompt complet lors du survol"""
        window = tk.Toplevel(self.root)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        
        label = tk.Label(
            window,
            text=prompt,
            background="#fffacd",
            fg="#333333",
            font=("Arial", 11),
            wraplength=300,
            justify="left",
            padx=8,
            pady=6,
            relief=tk.SOLID,
            bd=1
        )
        label.pack()
        
        # Stocker la référence pour pouvoir la détruire
        self.tooltip_window = window
        
        # Détruire automatiquement après 5 secondes
        self.root.after(5000, lambda: self.cacher_tooltip_idee())
    
    def cacher_tooltip_idee(self):
        """Cache le tooltip des idées"""
        if hasattr(self, 'tooltip_window'):
            try:
                self.tooltip_window.destroy()
            except:
                pass
            self.tooltip_window = None


if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantIA(root)
    root.mainloop()
