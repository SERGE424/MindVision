import warnings
warnings.filterwarnings('ignore')
import os
import sys

# Evite le crash OpenMP (libiomp5md.dll chargee plusieurs fois via dependances)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import random
import tempfile
import subprocess
from io import BytesIO
import requests
import json
import re
from datetime import datetime
from pathlib import Path
import base64


COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR.parent) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR.parent))

try:
    from common.coherence import append_event as append_shared_event
    from common.coherence import build_character_system_prompt as build_shared_system_prompt
    from common.coherence import build_context_for_chat as build_shared_context_for_chat
    from common.coherence import get_default_character_id as get_shared_default_character_id
    from common.coherence import get_recent_events as get_shared_recent_events
    from common.coherence import resolve_character_id as resolve_shared_character_id
except Exception:
    append_shared_event = None
    build_shared_context_for_chat = None
    build_shared_system_prompt = None
    get_shared_default_character_id = None
    get_shared_recent_events = None
    resolve_shared_character_id = None

# Import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[ATTENTION] Ollama non installe : pip install ollama")

# Import service TTS du ChatBot
try:
    from chatbot_tts import ChatbotTTSService  # type: ignore
    CHAT_TTS_AVAILABLE = True
except ImportError:
    from gtts import gTTS
    try:
        import pyttsx3
    except ImportError:
        pyttsx3 = None

    try:
        import pygame
    except ImportError:
        pygame = None

    class ChatbotTTSService:
        def __init__(self, lang="fr", speech_rate=235, volume=1.0, max_chars_per_chunk=2500):
            self.lang = lang
            self.speech_rate = speech_rate
            self.volume = max(0.0, min(1.0, float(volume)))
            self._mixer_ready = False
            self.max_chars_per_chunk = max_chars_per_chunk
            self._pyttsx3_engine = None

        @staticmethod
        def _strip_markdown(text):
            """Supprime le formatage Markdown pour une lecture TTS naturelle."""
            text = re.sub(r'```[\s\S]*?```', '', text)
            text = re.sub(r'`[^`]*`', '', text)
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)
            text = re.sub(r'_{1,3}([^_\n]+)_{1,3}', r'\1', text)
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
            text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'  +', ' ', text)
            return text.strip()

        def _ensure_mixer(self):
            if pygame is None or self._mixer_ready:
                return pygame is not None and self._mixer_ready
            pygame.mixer.init()
            self._mixer_ready = True
            return True

        def _set_french_voice(self):
            if self._pyttsx3_engine is None:
                return
            try:
                voices = self._pyttsx3_engine.getProperty("voices")
                if not isinstance(voices, (list, tuple)):
                    return
                for voice in voices:
                    voice_data = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
                    langs = getattr(voice, "languages", [])
                    langs_data = " ".join(str(lang).lower() for lang in langs)
                    if "fr" in voice_data or "french" in voice_data or "fr" in langs_data:
                        self._pyttsx3_engine.setProperty("voice", voice.id)
                        break
            except Exception:
                pass

        def _ensure_pyttsx3_engine(self):
            if pyttsx3 is None:
                return None

            if self._pyttsx3_engine is not None:
                return self._pyttsx3_engine

            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.speech_rate)
                engine.setProperty("volume", self.volume)
                self._pyttsx3_engine = engine
                self._set_french_voice()
                return self._pyttsx3_engine
            except Exception:
                self._pyttsx3_engine = None
                return None

        def set_speech_rate(self, speech_rate):
            try:
                self.speech_rate = int(float(speech_rate))
            except Exception:
                return

            if self._pyttsx3_engine is not None:
                try:
                    self._pyttsx3_engine.setProperty("rate", self.speech_rate)
                except Exception:
                    pass

        def set_volume(self, volume):
            try:
                self.volume = max(0.0, min(1.0, float(volume)))
            except Exception:
                return

            if self._pyttsx3_engine is not None:
                try:
                    self._pyttsx3_engine.setProperty("volume", self.volume)
                except Exception:
                    pass

        def _split_for_tts(self, text):
            if len(text) <= self.max_chars_per_chunk:
                return [text]

            sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
            chunks = []
            current = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(sentence) > self.max_chars_per_chunk:
                    if current:
                        chunks.append(current)
                        current = ""
                    for i in range(0, len(sentence), self.max_chars_per_chunk):
                        part = sentence[i:i + self.max_chars_per_chunk].strip()
                        if part:
                            chunks.append(part)
                    continue

                candidate = sentence if not current else f"{current} {sentence}"
                if len(candidate) <= self.max_chars_per_chunk:
                    current = candidate
                else:
                    chunks.append(current)
                    current = sentence

            if current:
                chunks.append(current)

            return chunks if chunks else [text]

        def _speak_with_pyttsx3(self, clean_text):
            engine = self._ensure_pyttsx3_engine()
            if engine is None:
                return False
            try:
                engine.say(clean_text)
                engine.runAndWait()
                return True
            except Exception:
                self._pyttsx3_engine = None
                return False

        def _speak_with_gtts(self, clean_text, pygame_module):
            if pygame_module is None:
                return False

            if not self._ensure_mixer():
                return False

            for text_chunk in self._split_for_tts(clean_text):
                tts = gTTS(text=text_chunk, lang=self.lang, slow=False)
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                        tmp_path = tmp_file.name
                        tts.write_to_fp(tmp_file)

                    pygame_module.mixer.music.load(tmp_path)
                    pygame_module.mixer.music.play()

                    while pygame_module.mixer.music.get_busy():
                        time.sleep(0.05)
                finally:
                    if tmp_path:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

            return True

        def _speak_with_windows_sapi(self, clean_text):
            """Fallback natif Windows via System.Speech quand pyttsx3/gTTS échouent."""
            if os.name != "nt":
                return False

            text_escaped = clean_text.replace("'", "''")
            sapi_rate = max(-10, min(10, int(round((self.speech_rate - 200) / 12))))
            sapi_volume = max(0, min(100, int(round(self.volume * 100))))

            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$synth.Rate = {sapi_rate}; "
                f"$synth.Volume = {sapi_volume}; "
                "try { "
                "$synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet, "
                "[System.Speech.Synthesis.VoiceAge]::NotSet, 0, "
                "(New-Object System.Globalization.CultureInfo('fr-FR'))) "
                "} catch { }; "
                f"$synth.Speak('{text_escaped}')"
            )

            try:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        script,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return result.returncode == 0
            except Exception:
                return False

        def speak(self, text, status_callback=None):
            pygame_module = pygame
            if not text:
                return

            def _speak():
                try:
                    if status_callback:
                        status_callback("Audio : préparation...", "gray")
                    clean = self._strip_markdown(text)
                    if not clean:
                        if status_callback:
                            status_callback("Audio : aucun texte à lire", "gray")
                        return

                    if self._speak_with_pyttsx3(clean):
                        if status_callback:
                            status_callback("Audio : lu via pyttsx3", "green")
                        return

                    if status_callback:
                        status_callback("Audio : fallback gTTS", "orange")
                    if self._speak_with_gtts(clean, pygame_module):
                        if status_callback:
                            status_callback("Audio : lu via gTTS", "green")
                        return

                    if status_callback:
                        status_callback("Audio : fallback Windows SAPI", "orange")
                    if self._speak_with_windows_sapi(clean):
                        if status_callback:
                            status_callback("Audio : lu via Windows SAPI", "green")
                        return

                    raise RuntimeError("Aucun moteur audio disponible (pyttsx3, gTTS+pygame, SAPI Windows)")
                except Exception as e:
                    print(f"Erreur lors de la synthèse vocale: {e}")
                    if status_callback:
                        status_callback(f"Audio : erreur - {e}", "red")

            thread = threading.Thread(target=_speak, daemon=True)
            thread.start()

    CHAT_TTS_AVAILABLE = True
    print("TTS ChatBot externe non disponible : fallback interne (pyttsx3/gTTS/Windows SAPI) active")

# Import de la bibliothèque de prompts
try:
    from prompts_library import create_system_prompt_with_examples, get_all_categories
    from prompts_viewer import open_prompts_viewer
    PROMPTS_LIBRARY_AVAILABLE = True
except ImportError:
    PROMPTS_LIBRARY_AVAILABLE = False
    print("[ATTENTION] Bibliotheque de prompts non disponible")


class AssistantIA:
    def __init__(self, root):
        self.root = root
        self.root.title("Votre Assistant IA Complet - ChatBot + Générateur d'Images")
        self.root.geometry("1200x900")
        
        # Configuration
        self.ollama_model = "gemma4:31b-cloud"  # Modèle cloud rapide
        self.ollama_models_disponibles = []  # Liste des modèles Ollama disponibles
        self.charger_modeles_ollama()  # Charger la liste des modèles
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
        self.hf_min_request_interval_sec = 1.8  # Cadence mini entre 2 appels HF
        self.hf_retry_backoff_sec = 3.0  # Pause additionnelle en cas de surcharge HF
        self.hf_request_jitter_sec = 0.15  # Jitter léger pour lisser les rafales
        self._hf_last_request_ts = 0.0
        self._hf_rate_lock = threading.Lock()
        self.tooltip_window = None  # Pour gérer les tooltips
        self.hf_endpoint_strategy = "API Inference (images)"
        
        # Images originales pour redimensionnement adaptatif
        self.original_assistant_image = None
        self.original_image = None
        self.assistant_generated_seed_var = tk.StringVar(value="")
        
        # Image prompt pour l'Assistant (prompt par image)
        self.assistant_image_prompt = None
        self.assistant_image_prompt_path = None
        self.assistant_image_prompt_preview_window = None
        
        # Charger le modèle SD en arrière-plan
        threading.Thread(target=self.charger_sd_turbo, daemon=True).start()
        
        # Configuration TTS
        self.lang = "fr"  # Langue française
        self.tts_service = ChatbotTTSService(lang=self.lang) if CHAT_TTS_AVAILABLE else None
        self._loaded_chat_tts_rate = getattr(self, "_loaded_chat_tts_rate", 235)
        self._loaded_chat_tts_volume = getattr(self, "_loaded_chat_tts_volume", 1.0)
        self._loaded_chat_voice_enabled = getattr(self, "_loaded_chat_voice_enabled", True)
        self.max_chat_history_messages = 12
        self.max_assistant_context_events = 20
        if get_shared_default_character_id is not None:
            try:
                self.shared_character_id = get_shared_default_character_id()
            except Exception:
                self.shared_character_id = "severine"
        else:
            self.shared_character_id = "severine"
        self.images_dir = Path(os.path.dirname(__file__)) / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.ui_settings_path = Path(os.path.dirname(__file__)) / "ui_settings_32_cartes.json"
        self.profile_memory_path = Path(os.path.dirname(__file__)) / "serge424_profile_memory.json"
        self.last_image_dir_chat = str(self.images_dir)
        self._charger_ui_settings()
        self._charger_hf_rate_limits_config()
        self.serge_profile_memory = self._charger_profil_serge424()
        
        # Image en attente d'envoi dans le chatbot
        self.chat_pending_image = None
        self._chat_image_refs = []  # Références PhotoImage pour éviter le GC
        self.last_chat_image_for_story = None
        self.stories_dir = Path(os.path.dirname(__file__)) / "histoires"
        self.stories_dir.mkdir(exist_ok=True)
        
        # Gestion des conversations
        self.conversations_dir = Path(os.path.dirname(__file__)) / "conversations"
        self.conversations_dir.mkdir(exist_ok=True)
        self.conversations = {}  # {id: {nom, date, messages}}
        self.current_conversation_id = None
        self.charger_conversations()
        if self.conversations:
            self.current_conversation_id = sorted(self.conversations.keys(), reverse=True)[0]
        else:
            self.creer_nouvelle_conversation()  # Créer une première conversation
        
        self.creer_interface()
        self.root.after(150, self._initialiser_paned_chatbot)
    
    def charger_modeles_ollama(self):
        """Charge la liste des modèles Ollama disponibles."""
        try:
            if OLLAMA_AVAILABLE:
                models = ollama.list()
                if 'models' in models and models['models']:
                    self.ollama_models_disponibles = [
                        model.get('name') or model.get('model', 'Inconnu')
                        for model in models['models']
                    ]
                else:
                    self.ollama_models_disponibles = ["gemma4:31b-cloud"]
            else:
                self.ollama_models_disponibles = ["gemma4:31b-cloud"]
        except Exception as e:
            print(f"Erreur lors du chargement des modèles Ollama: {e}")
            self.ollama_models_disponibles = ["gemma4:31b-cloud"]
    
    def charger_sd_turbo(self):
        """Charge SD-Turbo en arrière-plan"""
        try:
            print("Chargement de SD-Turbo...")
            import torch
            from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image

            self.torch = torch
            
            # Charger le pipeline avec safety checker actif
            self.sd_pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sd-turbo"
            )
            
            self.sd_pipe.enable_attention_slicing()
            self.sd_pipe.to("cpu")
            print("[OK] SD-Turbo charge (safety checker actif) !")
        except Exception as e:
            print(f"[ERREUR] SD-Turbo : {e}")

    def speak(self, text):
        """Prononce le texte via le service TTS du ChatBot."""
        if not self.tts_service:
            self._update_chat_tts_status("Audio : indisponible", "red")
            return
        self._update_chat_tts_status("Audio : lancement...", "orange")
        self.tts_service.speak(text, status_callback=self._update_chat_tts_status)

    def _update_chat_tts_status(self, text, color="gray"):
        """Met à jour l'indicateur audio du ChatBot depuis n'importe quel thread."""
        if not hasattr(self, "chat_tts_status"):
            return

        def _apply():
            try:
                self.chat_tts_status.config(text=text, foreground=color)
            except Exception:
                pass

        if hasattr(self, "root") and self.root is not None:
            try:
                self.root.after(0, _apply)
                return
            except Exception:
                pass

        _apply()

    def _charger_hf_rate_limits_config(self):
        """Charge la cadence Hugging Face depuis config.py (si définie)."""
        try:
            from config import HUGGING_FACE_MIN_INTERVAL_SEC
            self.hf_min_request_interval_sec = max(0.0, float(HUGGING_FACE_MIN_INTERVAL_SEC))
        except Exception:
            pass

        try:
            from config import HUGGING_FACE_RETRY_BACKOFF_SEC
            self.hf_retry_backoff_sec = max(0.0, float(HUGGING_FACE_RETRY_BACKOFF_SEC))
        except Exception:
            pass

    def _attendre_si_hf_trop_rapide(self, source_label="HF"):
        """Espace les appels Hugging Face pour réduire la surcharge en recréation continue."""
        min_interval = max(0.0, float(getattr(self, "hf_min_request_interval_sec", 0.0)))
        if min_interval <= 0:
            return

        with self._hf_rate_lock:
            now = time.monotonic()
            elapsed = now - self._hf_last_request_ts
            wait_time = min_interval - elapsed
            if wait_time > 0:
                jitter = random.uniform(0.0, max(0.0, float(self.hf_request_jitter_sec)))
                total_wait = wait_time + jitter
                print(f"[DELAI] Espacement {source_label}: pause {total_wait:.2f}s")
                time.sleep(total_wait)
            self._hf_last_request_ts = time.monotonic()
    
    def _relire_message(self, texte):
        """Lit un message spécifique avec TTS."""
        texte = "" if texte is None else str(texte)
        if not texte.strip():
            self._update_chat_tts_status("Audio : aucun texte à relire", "gray")
            return

        if self.tts_service:
            self.speak(texte)
        else:
            messagebox.showinfo("TTS non disponible", "Le service de lecture vocale n'est pas disponible.")

    def _tester_audio_chatbot(self):
        """Lance une courte phrase pour vérifier la voix en un clic."""
        if not self.tts_service:
            self._update_chat_tts_status("Audio : indisponible", "red")
            messagebox.showinfo("TTS non disponible", "Le service de lecture vocale n'est pas disponible.")
            return

        # Applique les réglages en cours avant de lire la phrase de test.
        self._appliquer_parametres_tts(save=True)
        self.speak("Test audio du ChatBot. Si vous entendez cette phrase, la voix fonctionne correctement.")
    
    # === GESTION CONVERSATIONS ===

    def _charger_ui_settings(self):
        """Charge les réglages UI persistés."""
        if not self.ui_settings_path.exists():
            return

        try:
            with open(self.ui_settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        except Exception as e:
            print(f"Erreur chargement réglages UI: {e}")
            return

        last_dir = settings.get("last_image_dir_chat")
        if isinstance(last_dir, str) and os.path.isdir(last_dir):
            self.last_image_dir_chat = last_dir

        try:
            self._loaded_chat_tts_rate = int(float(settings.get("chat_tts_rate", 235)))
        except Exception:
            self._loaded_chat_tts_rate = 235

        try:
            self._loaded_chat_tts_volume = max(0.0, min(1.0, float(settings.get("chat_tts_volume", 1.0))))
        except Exception:
            self._loaded_chat_tts_volume = 1.0

        self._loaded_chat_voice_enabled = bool(settings.get("chat_voice_enabled", True))

    def _sauvegarder_ui_settings(self):
        """Sauvegarde les réglages UI persistés."""
        settings = {}

        if self.ui_settings_path.exists():
            try:
                with open(self.ui_settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            except Exception:
                settings = {}

        settings["last_image_dir_chat"] = self.last_image_dir_chat
        settings["chat_tts_rate"] = int(getattr(getattr(self, "chat_tts_rate_var", None), "get", lambda: getattr(self, "_loaded_chat_tts_rate", 235))())
        settings["chat_tts_volume"] = float(getattr(getattr(self, "chat_tts_volume_var", None), "get", lambda: getattr(self, "_loaded_chat_tts_volume", 1.0))())
        settings["chat_voice_enabled"] = bool(getattr(getattr(self, "chat_voice_enabled", None), "get", lambda: getattr(self, "_loaded_chat_voice_enabled", True))())

        try:
            with open(self.ui_settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur sauvegarde réglages UI: {e}")

    def _appliquer_parametres_tts(self, save=True):
        """Synchronise les contrôles TTS avec le moteur vocal et la persistance."""
        rate_var = getattr(self, "chat_tts_rate_var", None)
        volume_var = getattr(self, "chat_tts_volume_var", None)

        try:
            rate = int(float(rate_var.get())) if rate_var is not None else 235
        except Exception:
            rate = 235

        try:
            volume = max(0.0, min(1.0, float(volume_var.get()))) if volume_var is not None else 1.0
        except Exception:
            volume = 1.0

        if hasattr(self, "chat_tts_rate_value_label"):
            try:
                self.chat_tts_rate_value_label.config(text=f"{rate}")
            except Exception:
                pass

        if hasattr(self, "chat_tts_volume_value_label"):
            try:
                self.chat_tts_volume_value_label.config(text=f"{int(volume * 100)}%")
            except Exception:
                pass

        if self.tts_service:
            self.tts_service.set_speech_rate(rate)
            self.tts_service.set_volume(volume)

        if save:
            self._sauvegarder_ui_settings()

    def _profil_serge_template(self):
        """Retourne la structure standard de mémoire pour Profils de Karine, Gabriel et Iris."""
        return {
            "physique": "",
            "caractere": "",
            "activites": "",
            "biens": "",
            "autres": "",
            "updated_at": ""
        }

    def _normaliser_profil_serge(self, data):
        """Normalise les donnees de Profils de Karine, Gabriel et Iris pour éviter les formats invalides."""
        profil = self._profil_serge_template()
        if isinstance(data, dict):
            for key in ("physique", "caractere", "activites", "biens", "autres"):
                value = data.get(key, "")
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value if v is not None)
                elif value is None:
                    value = ""
                else:
                    value = str(value)
                profil[key] = value.strip()

            updated_at = data.get("updated_at", "")
            if isinstance(updated_at, str):
                profil["updated_at"] = updated_at.strip()

        if not profil["updated_at"]:
            profil["updated_at"] = datetime.now().isoformat()

        return profil

    def _charger_profil_serge424(self):
        """Charge la mémoire persistante de Profils de Karine, Gabriel et Iris."""
        if not self.profile_memory_path.exists():
            return self._profil_serge_template()

        try:
            with open(self.profile_memory_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._normaliser_profil_serge(data)
        except Exception as e:
            print(f"Erreur chargement Profils de Karine, Gabriel et Iris: {e}")
            return self._profil_serge_template()

    def _sauvegarder_profil_serge424(self):
        """Sauvegarde la mémoire persistante de Profils de Karine, Gabriel et Iris."""
        try:
            self.serge_profile_memory = self._normaliser_profil_serge(self.serge_profile_memory)
            self.serge_profile_memory["updated_at"] = datetime.now().isoformat()
            with open(self.profile_memory_path, 'w', encoding='utf-8') as f:
                json.dump(self.serge_profile_memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur sauvegarde Profils de Karine, Gabriel et Iris: {e}")

    def _construire_contexte_profil_serge424(self):
        """Construit un résumé des Profils de Karine, Gabriel et Iris pour le contexte IA."""
        profil = getattr(self, "serge_profile_memory", None)
        if not isinstance(profil, dict):
            return ""

        lignes = []
        champs = [
            ("Physique", "physique"),
            ("Caractère", "caractere"),
            ("Activités", "activites"),
            ("Biens", "biens"),
            ("Autres détails", "autres"),
        ]

        for label, key in champs:
            valeur = self._resumer_texte(profil.get(key, ""), max_len=260)
            if valeur:
                lignes.append(f"- {label}: {valeur}")

        if not lignes:
            return ""

        return "\n".join(lignes)

    def ouvrir_editeur_profil_serge424(self):
        """Ouvre une fenêtre pour mémoriser les composants de Profils de Karine, Gabriel et Iris."""
        win = tk.Toplevel(self.root)
        win.title("Mémorisation - Profils de Karine, Gabriel et Iris")
        win.geometry("760x620")

        container = ttk.Frame(win, padding=10)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Renseignez les composants mémorisés pour Profils de Karine, Gabriel et Iris (physique, caractère, activités, biens, etc.)",
            font=("Arial", 12, "bold"),
            wraplength=700,
            justify="left"
        ).pack(anchor="w", pady=(0, 8))

        form = ttk.Frame(container)
        form.pack(fill="both", expand=True)

        fields = {}
        specs = [
            ("Physique", "physique", 4),
            ("Caractère", "caractere", 4),
            ("Activités", "activites", 4),
            ("Biens", "biens", 4),
            ("Autres", "autres", 5),
        ]

        profil = self._normaliser_profil_serge(self.serge_profile_memory)

        for idx, (label, key, height) in enumerate(specs):
            ttk.Label(form, text=f"{label} :", font=("Arial", 11, "bold")).grid(
                row=idx * 2,
                column=0,
                sticky="w",
                pady=(0 if idx == 0 else 6, 2)
            )
            txt = tk.Text(form, height=height, font=("Arial", 11), wrap=tk.WORD)
            txt.grid(row=idx * 2 + 1, column=0, sticky="nsew", pady=(0, 2))
            txt.insert("1.0", profil.get(key, ""))
            fields[key] = txt

        form.columnconfigure(0, weight=1)

        btns = ttk.Frame(container)
        btns.pack(fill="x", pady=(10, 0))

        def _save_profile():
            new_profile = self._profil_serge_template()
            for key, widget in fields.items():
                new_profile[key] = widget.get("1.0", tk.END).strip()

            self.serge_profile_memory = self._normaliser_profil_serge(new_profile)
            self._sauvegarder_profil_serge424()
            self.assistant_status.config(text="✅ Profils de Karine, Gabriel et Iris mémorisés")
            win.destroy()

        tk.Button(
            btns,
            text="💾 Mémoriser le profil",
            command=_save_profile,
            bg="#2E7D32",
            fg="white",
            font=("Arial", 12, "bold"),
            width=22
        ).pack(side="left", padx=(0, 6))

        tk.Button(
            btns,
            text="Annuler",
            command=win.destroy,
            bg="#616161",
            fg="white",
            font=("Arial", 11),
            width=12
        ).pack(side="left")

    def _extraire_json_depuis_texte(self, texte):
        """Extrait un objet JSON depuis une réponse texte potentiellement bruitée."""
        if not texte:
            return None

        texte = texte.strip()
        if texte.startswith("```"):
            texte = re.sub(r"^```(?:json)?\s*", "", texte)
            texte = re.sub(r"\s*```$", "", texte)

        try:
            data = json.loads(texte)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", texte)
        if not match:
            return None

        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _extraire_profil_serge_fallback(self, texte):
        """Extraction simple locale de Profils de Karine, Gabriel et Iris sans LLM."""
        profil = self._profil_serge_template()
        brut = (texte or "").strip()
        if not brut:
            return profil

        lignes = [l.strip(" -\t") for l in re.split(r"[\n;]+", brut) if l.strip()]
        if not lignes:
            lignes = [brut]

        key_map = {
            "physique": ["physique", "apparence", "taille", "cheveux", "yeux", "visage", "corpulence"],
            "caractere": ["caractere", "personnalite", "temperament", "qualites", "defauts", "humeur"],
            "activites": ["activite", "hobby", "loisir", "travail", "metier", "passion", "sport"],
            "biens": ["bien", "possede", "possession", "maison", "voiture", "instrument", "objet"],
        }

        autres = []
        for ligne in lignes:
            lower = ligne.lower()
            found = False
            for key, words in key_map.items():
                if any(word in lower for word in words):
                    profil[key] = (profil[key] + " ; " + ligne).strip(" ;") if profil[key] else ligne
                    found = True
                    break
            if not found:
                autres.append(ligne)

        if autres:
            profil["autres"] = " ; ".join(autres)

        return profil

    def _extraire_profil_serge_depuis_demande(self, demande):
        """Extrait les composants du profil depuis une demande texte."""
        texte = (demande or "").strip()
        if not texte:
            return self._profil_serge_template()

        if OLLAMA_AVAILABLE:
            prompt = (
                "Analyse ce texte et extrais uniquement les informations factuelles de Profils de Karine, Gabriel et Iris. "
                "Retourne strictement un JSON valide avec les cles: physique, caractere, activites, biens, autres. "
                "Valeurs: courtes chaines en francais, sans inventer si absent.\n\n"
                f"Texte:\n{texte}"
            )
            try:
                response = ollama.generate(
                    model=self.ollama_model,
                    prompt=prompt,
                    system="Tu extrais des donnees de profil de maniere fiable. Reponds uniquement en JSON.",
                    stream=False,
                    options={"temperature": 0.1, "num_predict": 300},
                )
                contenu = response.get("response", "") if isinstance(response, dict) else ""
                data = self._extraire_json_depuis_texte(contenu)
                if isinstance(data, dict):
                    return self._normaliser_profil_serge(data)
            except Exception:
                pass

        return self._normaliser_profil_serge(self._extraire_profil_serge_fallback(texte))

    def _fusionner_profil_serge(self, profil_courant, profil_nouveau):
        """Fusionne les nouvelles infos de profil sans perdre l'existant."""
        courant = self._normaliser_profil_serge(profil_courant)
        nouveau = self._normaliser_profil_serge(profil_nouveau)
        fusion = self._profil_serge_template()

        for key in ("physique", "caractere", "activites", "biens", "autres"):
            old_val = (courant.get(key, "") or "").strip()
            new_val = (nouveau.get(key, "") or "").strip()

            if not old_val:
                fusion[key] = new_val
                continue
            if not new_val:
                fusion[key] = old_val
                continue

            old_low = old_val.lower()
            new_low = new_val.lower()
            if new_low in old_low:
                fusion[key] = old_val
            elif old_low in new_low:
                fusion[key] = new_val
            else:
                fusion[key] = f"{old_val} ; {new_val}"

        fusion["updated_at"] = datetime.now().isoformat()
        return fusion

    def memoriser_profil_depuis_demande(self):
        """Extrait et mémorise automatiquement Profils de Karine, Gabriel et Iris depuis la demande actuelle."""
        demande = self.assistant_input.get("1.0", tk.END).strip()
        if not demande:
            self.assistant_status.config(text="⚠️ Saisissez d'abord une demande pour extraire le profil")
            return

        self.assistant_status.config(text="🧠 Extraction pour Profils de Karine, Gabriel et Iris depuis la demande...")
        threading.Thread(target=self._memoriser_profil_depuis_demande_thread, args=(demande,), daemon=True).start()

    def _memoriser_profil_depuis_demande_thread(self, demande):
        """Thread d'extraction de profil à partir de la demande utilisateur."""
        try:
            profil_extrait = self._extraire_profil_serge_depuis_demande(demande)
            profil_fusionne = self._fusionner_profil_serge(self.serge_profile_memory, profil_extrait)
            self.serge_profile_memory = profil_fusionne
            self._sauvegarder_profil_serge424()

            champs_majs = [
                key for key in ("physique", "caractere", "activites", "biens", "autres")
                if (profil_extrait.get(key, "") or "").strip()
            ]
            resume_champs = ", ".join(champs_majs) if champs_majs else "aucun champ detecte"

            def _ui_success():
                self.assistant_status.config(text=f"✅ Profils de Karine, Gabriel et Iris mémorisés depuis la demande ({resume_champs})")

            self.root.after(0, _ui_success)
        except Exception as e:
            self.root.after(0, lambda: self.assistant_status.config(text=f"❌ Erreur mémorisation profil : {e}"))

    def _normaliser_message_conversation(self, message):
        """Convertit différents formats de messages en format chatbot."""
        if not isinstance(message, dict):
            return None

        sender = message.get("sender") or message.get("role") or "IA"
        texte = message.get("texte") or message.get("message") or ""
        timestamp = message.get("timestamp") or datetime.now().isoformat()

        if sender == "Assistant":
            sender = "IA"
        elif sender == "User":
            sender = "Vous"

        return {
            "sender": sender,
            "texte": texte,
            "timestamp": timestamp
        }

    def _normaliser_conversation_chargee(self, conv_id, data):
        """Normalise les anciens formats de conversation chargés depuis JSON."""
        if not isinstance(data, dict):
            return None

        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list):
            raw_messages = []

        messages = []
        for raw_message in raw_messages:
            message = self._normaliser_message_conversation(raw_message)
            if message is not None:
                messages.append(message)

        nom = data.get("nom")
        if not isinstance(nom, str) or not nom.strip():
            premier_texte = next((msg["texte"].strip() for msg in messages if msg["texte"].strip()), "")
            nom = premier_texte[:40] if premier_texte else conv_id.replace("_", " ")

        date_creation = data.get("date_creation") or data.get("session_started") or datetime.now().isoformat()

        raw_assistant_context = data.get("assistant_context")
        if not isinstance(raw_assistant_context, list):
            raw_assistant_context = []

        assistant_context = []
        for item in raw_assistant_context:
            if not isinstance(item, dict):
                continue
            assistant_context.append({
                "timestamp": item.get("timestamp") or datetime.now().isoformat(),
                "demande": str(item.get("demande") or ""),
                "prompt": str(item.get("prompt") or ""),
                "negative_prompt": str(item.get("negative_prompt") or ""),
                "mode": str(item.get("mode") or ""),
                "image": str(item.get("image") or ""),
            })

        return {
            "nom": nom,
            "date_creation": date_creation,
            "messages": messages,
            "assistant_context": assistant_context[-self.max_assistant_context_events:]
        }

    def _est_fichier_conversation_chatbot(self, file_path, data):
        """Détermine si un JSON appartient à l'historique chatbot."""
        if not isinstance(data, dict):
            return False

        stem = file_path.stem
        if stem.startswith("chat_cartes_"):
            return False

        if "nom" in data or "date_creation" in data:
            return True

        # Certains anciens exports chatbot n'ont pas de méta, mais gardent sender/texte.
        messages = data.get("messages")
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict) and ("sender" in first or "texte" in first):
                return True

        return False
    
    def charger_conversations(self):
        """Charge toutes les conversations depuis les fichiers JSON"""
        self.conversations = {}
        if not self.conversations_dir.exists():
            return
        
        for file_path in sorted(self.conversations_dir.glob("*.json"), reverse=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if not self._est_fichier_conversation_chatbot(file_path, data):
                        continue

                    conv_id = file_path.stem
                    conversation = self._normaliser_conversation_chargee(conv_id, data)
                    if conversation is None:
                        continue
                    self.conversations[conv_id] = conversation
            except Exception as e:
                print(f"Erreur chargement conversation {file_path}: {e}")
    
    def creer_nouvelle_conversation(self):
        """Crée une nouvelle conversation"""
        conv_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.conversations[conv_id] = {
            "nom": f"Conversation {datetime.now().strftime('%d/%m %H:%M')}",
            "date_creation": datetime.now().isoformat(),
            "messages": [],
            "assistant_context": []
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
        self._append_shared_chat_event(sender=sender, texte=texte)
        
        # Générer un titre intelligent après 2-3 messages utilisateur
        self._auto_generer_titre()
        
        self.sauvegarder_conversation(self.current_conversation_id)
        self.rafraichir_liste_conversations()

    def _append_shared_chat_event(self, sender, texte):
        if append_shared_event is None:
            return

        sender_norm = (sender or "").strip().lower()
        character_id = ""
        if sender_norm in {"ia", "assistant", "severine", "séverine"}:
            if resolve_shared_character_id is not None:
                character_id = resolve_shared_character_id(str(self.shared_character_id or "severine"), fallback="severine")
            else:
                character_id = str(self.shared_character_id or "severine")

        event = {
            "type": "chat_message",
            "source_app": "mindvision",
            "session_id": str(self.current_conversation_id or ""),
            "character_id": character_id,
            "sender": sender,
            "text": str(texte or ""),
        }
        try:
            append_shared_event(event)
        except Exception:
            return
    
    def _auto_generer_titre(self):
        """Génère automatiquement un titre intelligent pour la conversation."""
        if not self.current_conversation_id or self.current_conversation_id not in self.conversations:
            return
        
        conv = self.conversations[self.current_conversation_id]
        messages = conv.get("messages", [])
        
        # Compter les messages utilisateur
        user_messages = [m for m in messages if m.get("sender") == "Vous"]
        
        # Générer un titre après 2-3 messages utilisateur
        if len(user_messages) < 2 or len(user_messages) > 3:
            return
        
        current_nom = conv.get("nom", "")
        # Ne pas régénérer si le titre a été modifié manuellement
        if not current_nom.startswith("Conversation"):
            return
        
        # Générer un titre intelligent avec Ollama
        titre = self._generer_titre_intelligent(messages)
        if titre:
            conv["nom"] = titre
    
    def _generer_titre_intelligent(self, messages):
        """Utilise Ollama pour générer un titre contextuel intelligent."""
        if not OLLAMA_AVAILABLE or not messages:
            # Fallback : utiliser les premiers mots du premier message
            for msg in messages:
                if msg.get("sender") == "Vous":
                    texte = msg.get("texte", "").strip()
                    if texte:
                        return texte[:40].replace('\n', ' ').strip() + ("..." if len(texte) > 40 else "")
            return "Conversation"
        
        # Construire un résumé des premiers échanges
        context_parts = []
        for msg in messages[:6]:  # Premiers 3 échanges max
            sender = msg.get("sender", "")
            content = msg.get("texte", "").strip()
            if content:
                context_parts.append(f"{sender}: {content[:150]}")
        
        if not context_parts:
            return "Conversation"
        
        conversation_snippet = "\n".join(context_parts)
        
        # Prompt pour générer un titre court
        prompt = f"""Analyse cette conversation et génère un titre court (maximum 5 mots) qui résume le sujet principal.
Ne mets PAS de guillemets autour du titre. Réponds UNIQUEMENT avec le titre, sans ponctuation finale.

Conversation:
{conversation_snippet}

Titre:"""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.7, "num_predict": 50}
            )
            titre = response.get("message", {}).get("content", "").strip()
            # Nettoyer le titre
            titre = titre.strip('"\'.,!?;:')
            titre = re.sub(r'\s+', ' ', titre)
            # Limiter à 50 caractères
            if len(titre) > 50:
                titre = titre[:47] + "..."
            return titre if titre else "Conversation"
        except Exception as e:
            print(f"Erreur génération titre intelligent: {e}")
            # Fallback
            for msg in messages:
                if msg.get("sender") == "Vous":
                    texte = msg.get("texte", "").strip()
                    if texte:
                        return texte[:40].replace('\n', ' ').strip() + ("..." if len(texte) > 40 else "")
            return "Conversation"


    def _resumer_texte(self, texte, max_len=220):
        """Compacte un texte pour stockage de contexte."""
        texte = (texte or "").replace("\n", " ").strip()
        if len(texte) <= max_len:
            return texte
        return texte[:max_len - 1].rstrip() + "…"

    def _enregistrer_contexte_assistant(self, demande, prompt, negative_prompt, mode, image_name=""):
        """Persist le contexte Assistant pour l'injecter ensuite dans le ChatBot."""
        conv_id = self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return

        conversation = self.conversations[conv_id]
        events = conversation.setdefault("assistant_context", [])
        events.append({
            "timestamp": datetime.now().isoformat(),
            "demande": self._resumer_texte(demande),
            "prompt": self._resumer_texte(prompt),
            "negative_prompt": self._resumer_texte(negative_prompt),
            "mode": mode or "",
            "image": image_name or "",
        })
        conversation["assistant_context"] = events[-self.max_assistant_context_events:]
        self.sauvegarder_conversation(conv_id)
        if append_shared_event is not None:
            try:
                append_shared_event(
                    {
                        "type": "assistant_context",
                        "source_app": "mindvision",
                        "session_id": str(conv_id),
                        "character_id": "severine",
                        "demande": self._resumer_texte(demande),
                        "prompt": self._resumer_texte(prompt),
                        "negative_prompt": self._resumer_texte(negative_prompt),
                        "mode": mode or "",
                        "image_path": image_name or "",
                    }
                )
            except Exception:
                pass

    def _construire_historique_chat(self):
        """Construit un historique compact pour donner de la mémoire au modèle."""
        conv_id = self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return ""

        messages = self.conversations[conv_id].get("messages", [])
        if not messages:
            return ""

        # Exclure le dernier message utilisateur: il est déjà fourni dans le prompt courant.
        messages_for_history = messages[:-1] if messages and messages[-1].get("sender") == "Vous" else messages
        messages_for_history = messages_for_history[-self.max_chat_history_messages:]

        lignes = []
        for msg in messages_for_history:
            sender = msg.get("sender", "")
            texte = self._resumer_texte(msg.get("texte", ""), max_len=280)
            if not texte:
                continue
            lignes.append(f"{sender}: {texte}")

        return "\n".join(lignes)

    def _construire_contexte_assistant(self):
        """Construit un résumé des dernières actions de l'Assistant Complet."""
        conv_id = self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return ""

        events = self.conversations[conv_id].get("assistant_context", [])
        if not events:
            return ""

        lignes = []
        for event in events[-6:]:
            demande = event.get("demande", "")
            prompt = event.get("prompt", "")
            negative = event.get("negative_prompt", "")
            mode = event.get("mode", "")
            image_name = event.get("image", "")

            ligne = f"- Demande: {demande} | Prompt: {prompt}"
            if negative:
                ligne += f" | Negatif: {negative}"
            if mode:
                ligne += f" | Mode: {mode}"
            if image_name:
                ligne += f" | Image: {image_name}"
            lignes.append(ligne)

        return "\n".join(lignes)
    
    def charger_conversation(self, conv_id):
        """Charge une conversation et l'affiche"""
        if conv_id not in self.conversations:
            return
        
        self.current_conversation_id = conv_id
        self.afficher_conversation_actuelle()
        self._mettre_a_jour_champ_titre()
    
    def _mettre_a_jour_champ_titre(self):
        """Met à jour le champ de titre avec le nom de la conversation actuelle"""
        if not hasattr(self, 'chat_titre_entry'):
            return
        
        if self.current_conversation_id and self.current_conversation_id in self.conversations:
            nom = self.conversations[self.current_conversation_id]["nom"]
            self.chat_titre_entry.delete(0, tk.END)
            self.chat_titre_entry.insert(0, nom)
        else:
            self.chat_titre_entry.delete(0, tk.END)
    
    def afficher_conversation_actuelle(self):
        """Affiche tous les messages de la conversation actuelle"""
        self.chat_display.config(state='normal')
        self.chat_display.delete("1.0", tk.END)
        
        if not self.current_conversation_id or self.current_conversation_id not in self.conversations:
            return
        
        messages = self.conversations[self.current_conversation_id]["messages"]
        for msg in messages:
            sender = msg["sender"]
            texte = msg["texte"]
            self.chat_display.insert(tk.END, f"\n{sender} : {texte}\n", sender)
            
            # Ajouter un bouton "Relire" sous chaque message
            btn_relire = tk.Button(
                self.chat_display,
                text="🔊 Relire",
                command=lambda t=texte: self._relire_message(t),
                bg="#424242",
                fg="white",
                font=("Arial", 10),
                relief=tk.FLAT,
                cursor="hand2",
                padx=8,
                pady=2
            )
            self.chat_display.window_create(tk.END, window=btn_relire)
            self.chat_display.insert(tk.END, "\n")
        
        self.chat_display.see(tk.END)
    
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
        
        # Mettre à jour le champ de titre
        self._mettre_a_jour_champ_titre()
    
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
              
        # Barre de gestion des conversations (fixe en haut)
        conv_frame = ttk.Frame(self.tab_chat)
        conv_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(conv_frame, text="Conversations :", font=("Arial", 18)).pack(side="left", padx=5)
        self.chat_conversations_combo = ttk.Combobox(conv_frame, state="readonly", width=40)
        self.chat_conversations_combo.pack(side="left", padx=5, fill="x", expand=True)
        self.chat_conversations_combo.bind("<<ComboboxSelected>>", self._on_conversation_selected)
        
        tk.Button(conv_frame, text="➕ Nouvelle", command=self._on_nouvelle_conversation, bg="#2E7D32", fg="white", width=12).pack(side="left", padx=2)
        tk.Button(conv_frame, text="🗑️ Supprimer", command=self._on_supprimer_conversation, bg="#C62828", fg="white", width=12).pack(side="left", padx=2)
        tk.Button(conv_frame, text="🧭 Contexte partagé", command=self.ouvrir_contexte_partage, bg="#455A64", fg="white", width=16).pack(side="left", padx=2)
        
        # Barre de renommage de conversation
        rename_frame = ttk.Frame(self.tab_chat)
        rename_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(rename_frame, text="Titre/Contexte :", font=("Arial", 14)).pack(side="left", padx=5)
        self.chat_titre_entry = ttk.Entry(rename_frame, font=("Arial", 14))
        self.chat_titre_entry.pack(side="left", padx=5, fill="x", expand=True)
        self.chat_titre_entry.bind("<Return>", lambda e: self._on_renommer_conversation())
        
        tk.Button(rename_frame, text="✏️ Renommer", command=self._on_renommer_conversation, bg="#1976D2", fg="white", width=12).pack(side="left", padx=2)
        
        # Rafraîchir la liste
        self.rafraichir_liste_conversations()
        
        # PanedWindow horizontal pour afficher conversation et paramètres côte à côte
        self.chatbot_main_paned = ttk.PanedWindow(self.tab_chat, orient=tk.HORIZONTAL)
        self.chatbot_main_paned.pack(fill="both", expand=True, padx=10, pady=5)
        
        # === PANEAU 1 : Zone de conversation (adaptative) ===
        chat_frame = tk.LabelFrame(self.chatbot_main_paned, text="📝 Conversation", relief=tk.GROOVE)
        self.chatbot_main_paned.add(chat_frame, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, font=("Arial", 20), bg="black", fg="white")
        self.chat_display.pack(fill="both", expand=True, padx=5, pady=5)
        # Lecture seule via bindings (au lieu de state='disabled') pour garder les boutons Relire cliquables.
        self.chat_display.bind("<Key>", lambda _e: "break")
        self.chat_display.bind("<<Paste>>", lambda _e: "break")
        self.chat_display.bind("<<Cut>>", lambda _e: "break")
        
        # Configuration des styles pour les messages
        self.chat_display.tag_config("Vous", foreground="#4DA6FF", font=("Arial", 22, "bold"))  # Bleu ciel clair + gras + agrandi
        self.chat_display.tag_config("IA", foreground="#7CFC00", font=("Arial", 20))  # Vert clair pour visibilité
        self.chat_display.tag_config("Système", foreground="#FF6B6B", font=("Arial", 20))  # Rouge clair pour les erreurs
        
        # Afficher la conversation initiale
        self.afficher_conversation_actuelle()
        
        # === PANEAU 2 : Paramètres et saisie (adaptative) ===
        params_frame = tk.LabelFrame(self.chatbot_main_paned, text="⚙️ Paramètres & Saisie", relief=tk.GROOVE)
        self.chatbot_main_paned.add(params_frame, weight=1)
        
        # Cadre scrollable pour les instructions et saisie
        scroll_frame = ttk.Frame(params_frame)
        scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        canvas = tk.Canvas(scroll_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        def _ajuster_largeur_canvas(event):
            canvas.itemconfigure(scrollable_window, width=event.width)

        canvas.bind("<Configure>", _ajuster_largeur_canvas)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Molette de souris
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        content_frame = ttk.Frame(scrollable_frame)
        content_frame.pack(fill="x", padx=5, pady=5)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)

        left_params_frame = ttk.Frame(content_frame)
        left_params_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")

        right_params_frame = ttk.Frame(content_frame)
        right_params_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(6, 0))

        # Sélecteur de source IA
        source_frame = ttk.Frame(left_params_frame)
        source_frame.pack(pady=(8, 1), fill="x")
        ttk.Label(source_frame, text="Source IA :", font=("Arial", 16)).pack(side="left", padx=5)
        self.chat_source_combo = ttk.Combobox(source_frame, values=[
            "Ollama (local)",
            "Hugging Face API",
        ], state="readonly", width=22)
        self.chat_source_combo.current(0)
        self.chat_source_combo.pack(side="left", padx=5)

        self.chat_voice_enabled = tk.BooleanVar(value=getattr(self, "_loaded_chat_voice_enabled", True))
        self.chat_voice_toggle = ttk.Checkbutton(
            source_frame,
            text="Lecture vocale ON/OFF",
            variable=self.chat_voice_enabled,
            command=self._sauvegarder_ui_settings
        )
        self.chat_voice_toggle.pack(side="left", padx=12)

        tts_controls_frame = ttk.Frame(left_params_frame)
        tts_controls_frame.pack(pady=(2, 2), fill="x")

        ttk.Label(tts_controls_frame, text="Rythme TTS :", font=("Arial", 12)).grid(row=0, column=0, sticky="w", padx=5)
        self.chat_tts_rate_var = tk.DoubleVar(value=float(getattr(self, "_loaded_chat_tts_rate", 235)))
        self.chat_tts_rate_scale = ttk.Scale(
            tts_controls_frame,
            from_=120,
            to=320,
            orient="horizontal",
            variable=self.chat_tts_rate_var,
            command=lambda _value: self._appliquer_parametres_tts(save=True)
        )
        self.chat_tts_rate_scale.grid(row=0, column=1, sticky="ew", padx=5)
        self.chat_tts_rate_value_label = ttk.Label(tts_controls_frame, text="235", width=5)
        self.chat_tts_rate_value_label.grid(row=0, column=2, sticky="e", padx=5)

        ttk.Label(tts_controls_frame, text="Volume TTS :", font=("Arial", 12)).grid(row=1, column=0, sticky="w", padx=5, pady=(6, 0))
        self.chat_tts_volume_var = tk.DoubleVar(value=float(getattr(self, "_loaded_chat_tts_volume", 1.0)))
        self.chat_tts_volume_scale = ttk.Scale(
            tts_controls_frame,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.chat_tts_volume_var,
            command=lambda _value: self._appliquer_parametres_tts(save=True)
        )
        self.chat_tts_volume_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=(6, 0))
        self.chat_tts_volume_value_label = ttk.Label(tts_controls_frame, text="100%", width=5)
        self.chat_tts_volume_value_label.grid(row=1, column=2, sticky="e", padx=5, pady=(6, 0))

        tts_controls_frame.columnconfigure(1, weight=1)
        ttk.Label(
            tts_controls_frame,
            text="Le rythme et le volume pilotent pyttsx3; gTTS garde ses limites de tonalité.",
            font=("Arial", 10),
            foreground="#666666"
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 0))

        self.chat_test_audio_btn = tk.Button(
            tts_controls_frame,
            text="🔊 Test audio",
            command=self._tester_audio_chatbot,
            bg="#00796B",
            fg="white",
            font=("Arial", 11, "bold"),
            width=14
        )
        self.chat_test_audio_btn.grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(6, 0))

        self.chat_auto_profile_from_image = tk.BooleanVar(value=True)
        self.chat_auto_profile_toggle = ttk.Checkbutton(
            source_frame,
            text="Auto-profil photo (Karine, Gabriel, Iris)",
            variable=self.chat_auto_profile_from_image
        )
        self.chat_auto_profile_toggle.pack(side="left", padx=8)

        # Sélecteur de modèle Ollama
        ollama_model_frame = ttk.Frame(left_params_frame)
        ollama_model_frame.pack(pady=(2, 1), fill="x")
        ttk.Label(ollama_model_frame, text="Modèle Ollama :", font=("Arial", 14)).pack(side="left", padx=5)
        self.chat_ollama_model_combo = ttk.Combobox(
            ollama_model_frame,
            values=self.ollama_models_disponibles,
            state="readonly",
            width=38
        )
        if self.ollama_models_disponibles:
            self.chat_ollama_model_combo.current(0)
        self.chat_ollama_model_combo.pack(side="left", padx=5)

        # Sélecteur de modèle HuggingFace texte
        hf_text_frame = ttk.Frame(left_params_frame)
        hf_text_frame.pack(pady=(2, 1), fill="x")
        ttk.Label(hf_text_frame, text="Modèle HF texte :", font=("Arial", 14)).pack(side="left", padx=5)
        self.chat_hf_model_combo = ttk.Combobox(hf_text_frame, values=[
            "gemma4:31b-cloud",
        ], state="readonly", width=38)
        self.chat_hf_model_combo.current(0)
        self.chat_hf_model_combo.pack(side="left", padx=5)

        # Instructions négatives
        ttk.Label(right_params_frame, text="Instructions négatives (comportements à éviter) :", font=("Arial", 16)).pack(pady=(8, 1), padx=5)
        self.chat_negative = tk.Text(right_params_frame, height=1, width=34, font=("Arial", 14))
        self.chat_negative.pack(fill="x", padx=5, pady=1)
        self.chat_negative.insert(tk.END, "")
        
        # Zone de saisie
        ttk.Label(right_params_frame, text="Votre message :", font=("Arial", 16)).pack(pady=(8, 1), padx=5)
        input_frame = ttk.Frame(right_params_frame)
        input_frame.pack(fill="x", padx=5, pady=1)
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=0)
        
        self.chat_input = tk.Text(input_frame, height=3, width=34, font=("Arial", 16))
        self.chat_input.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.chat_input.bind("<Return>", lambda e: self.envoyer_message_chat() if not (isinstance(e.state, int) and e.state & 1) else None)
        
        # Boutons droite (Envoyer + Import Image + Histoire)
        btns_right = ttk.Frame(input_frame)
        btns_right.grid(row=0, column=1, sticky="ne")
        btn_send = tk.Button(btns_right, text="Envoyer", command=self.envoyer_message_chat, bg="blue", fg="white", width=10, font=("Arial", 14, "bold"))
        btn_send.pack(side="top", pady=(0, 3))
        btn_img_import = tk.Button(btns_right, text="📎 Image", command=self._importer_image_chat, bg="#5C6BC0", fg="white", width=10, font=("Arial", 12))
        btn_img_import.pack(side="top", pady=(0, 3))
        btn_story = tk.Button(btns_right, text="Ecris une histoire", command=self.ecrire_histoire_depuis_image_chat, bg="#8E24AA", fg="white", width=16, font=("Arial", 11, "bold"))
        btn_story.pack(side="top")

        # Zone de prévisualisation de l'image sélectionnée (masquée par défaut)
        self.chat_img_preview_frame = tk.Frame(right_params_frame, bg="#1a1a2e", relief=tk.GROOVE, bd=1)
        self.chat_img_thumbnail_label = tk.Label(self.chat_img_preview_frame, bg="#1a1a2e")
        self.chat_img_thumbnail_label.pack(side="left", padx=5, pady=3)
        self.chat_img_name_label = tk.Label(self.chat_img_preview_frame, bg="#1a1a2e", fg="white", font=("Arial", 11))
        self.chat_img_name_label.pack(side="left", padx=5)
        tk.Button(self.chat_img_preview_frame, text="✖", command=self._supprimer_image_chat,
                  bg="#C62828", fg="white", font=("Arial", 12, "bold"), width=3).pack(side="right", padx=5)
        
        # Status
        self.chat_status = tk.Label(right_params_frame, text="Prêt", foreground="green")
        self.chat_status.pack(pady=5)
        self.chat_tts_status = tk.Label(right_params_frame, text="Audio : prêt", foreground="gray")
        self.chat_tts_status.pack(pady=(0, 6))

        self._appliquer_parametres_tts(save=False)

    def _initialiser_paned_chatbot(self):
        """Place la séparation principale du ChatBot au milieu au premier affichage."""
        paned = getattr(self, "chatbot_main_paned", None)
        if paned is None:
            return

        if not paned.winfo_ismapped():
            self.root.after(150, self._initialiser_paned_chatbot)
            return

        paned.update_idletasks()
        width = paned.winfo_width()
        if width <= 100:
            self.root.after(150, self._initialiser_paned_chatbot)
            return

        paned.sashpos(0, width // 2)
    
    def _importer_image_chat(self):
        """Ouvre un sélecteur de fichier pour joindre une image au prochain message"""
        path = filedialog.askopenfilename(
            title="Sélectionner une image",
            initialdir=self.last_image_dir_chat,
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.gif *.webp *.bmp"),
                ("Tous les fichiers", "*.*")
            ]
        )
        if not path:
            return

        try:
            self.last_image_dir_chat = os.path.dirname(path)
            self._sauvegarder_ui_settings()
        except Exception:
            pass

        self.chat_pending_image = path
        self.last_chat_image_for_story = path
        # Créer et afficher la miniature
        try:
            img = Image.open(path)
            img.thumbnail((80, 80))
            photo = ImageTk.PhotoImage(img)
            self.chat_img_thumbnail_label.config(image=photo)
            setattr(self.chat_img_thumbnail_label, "image", photo)  # référence
        except Exception:
            self.chat_img_thumbnail_label.config(image="")
        self.chat_img_name_label.config(text=os.path.basename(path))
        self.chat_img_preview_frame.pack(fill="x", padx=5, pady=2, before=self.chat_status)

    def _supprimer_image_chat(self):
        """Retire l'image en attente"""
        self.chat_pending_image = None
        self.chat_img_thumbnail_label.config(image="")
        setattr(self.chat_img_thumbnail_label, "image", None)
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
    
    def _on_renommer_conversation(self, nouveau_titre=None):
        """Callback pour renommer la conversation actuelle avec le texte du champ titre"""
        if not self.current_conversation_id or self.current_conversation_id not in self.conversations:
            return
        
        if nouveau_titre is None:
            nouveau_titre = self.chat_titre_entry.get().strip()
        else:
            nouveau_titre = (nouveau_titre or "").strip()

        if not nouveau_titre:
            messagebox.showwarning("Titre vide", "Veuillez entrer un titre pour la conversation.")
            return
        
        # Mettre à jour le nom de la conversation
        self.conversations[self.current_conversation_id]["nom"] = nouveau_titre
        self.sauvegarder_conversation(self.current_conversation_id)
        self.rafraichir_liste_conversations()
        
        # Afficher un message de confirmation
        self.chat_status.config(text=f"✓ Conversation renommée : {nouveau_titre}", foreground="green")
        self.root.after(3000, lambda: self.chat_status.config(text="Prêt", foreground="green"))
    
    def creer_onglet_generateur(self):
        """Onglet Générateur d'images"""
        # Titre
        ttk.Label(self.tab_images, text="Générateur d'Images IA", font=("Arial", 28, "bold")).pack(pady=5)

        # Fenêtre principale en 2 panneaux: zone haute + galerie
        main_paned = ttk.PanedWindow(self.tab_images, orient=tk.VERTICAL)
        main_paned.pack(fill="both", expand=True, padx=10, pady=5)

        top_panel_frame = ttk.Frame(main_paned)
        main_paned.add(top_panel_frame, weight=4)
        top_panel_frame.columnconfigure(0, weight=3)
        top_panel_frame.columnconfigure(1, weight=2)
        top_panel_frame.rowconfigure(0, weight=1)
        
        # Cadre scrollable pour les paramètres
        params_frame = ttk.Frame(top_panel_frame)
        params_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # Zone de résultat placée à droite pour occuper l'espace vide
        result_frame = ttk.LabelFrame(top_panel_frame, text="Résultat")
        result_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        
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
            "Hugging Face API"
        ], state="readonly", width=25)
        self.mode_combo.current(0)
        self.mode_combo.pack(side="left", padx=5, fill="x", expand=True)

        # Choix de l'endpoint Hugging Face (sans modifier le code)
        hf_frame = ttk.Frame(scrollable_frame)
        hf_frame.pack(pady=2, padx=5, fill="x")
        ttk.Label(hf_frame, text="Endpoint Hugging Face :").pack(side="left", padx=5)
        self.hf_endpoint_combo = ttk.Combobox(
            hf_frame,
            values=[
                "API Inference (images)",
                "Router HF Inference (fallback TLS)",
                "api-inference.huggingface.co",
                "router.huggingface.co/hf-inference",
            ],
            state="readonly",
            width=25,
        )
        self.hf_endpoint_combo.set(self.hf_endpoint_strategy)
        self.hf_endpoint_combo.pack(side="left", padx=5)
        
        prompt_inputs_frame = ttk.Frame(scrollable_frame)
        prompt_inputs_frame.pack(pady=2, padx=5, fill="x", expand=True)

        # Prompt
        ttk.Label(prompt_inputs_frame, text="Description de l'image :").pack(pady=2, padx=5, anchor="w")
        self.image_prompt = tk.Text(prompt_inputs_frame, height=2, width=60, font=("Arial", 16))
        self.image_prompt.pack(pady=2, padx=5, fill="x")

        # Prompt négatif
        ttk.Label(prompt_inputs_frame, text="Prompt négatif (à éviter) :").pack(pady=2, padx=5, anchor="w")
        self.image_negative_prompt = tk.Text(prompt_inputs_frame, height=2, width=60, font=("Arial", 14))
        self.image_negative_prompt.pack(pady=2, padx=5, fill="x")
        self.image_negative_prompt.insert(tk.END, "blurry, low quality, distorted")

        self.image_label = tk.Label(result_frame, relief="sunken", background="gray20", width=44, height=12)
        self.image_label.pack(fill="both", expand=True, padx=5, pady=5)
        
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
        
        # Galerie (panneau du bas)
        gallery_frame_gen = ttk.LabelFrame(main_paned, text="🖼️ Galerie (cliquez pour afficher)")
        main_paned.add(gallery_frame_gen, weight=1)

        # Coulisse supérieure: agit sur la séparation entre la 1re fenêtre et la galerie
        gallery_handle = tk.Frame(gallery_frame_gen, height=8, bg="#cfcfcf", cursor="sb_v_double_arrow")
        gallery_handle.pack(fill="x", padx=5, pady=(2, 0))
        self.gallery_collapsed = False
        self._gallery_drag_origin_y = None
        self._gallery_drag_start_sash = None
        self._gallery_saved_sash = None
        
        gallery_canvas_frame_gen = ttk.Frame(gallery_frame_gen)
        gallery_canvas_frame_gen.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.gallery_canvas_gen = tk.Canvas(gallery_canvas_frame_gen, height=120, bg="#f0f0f0")
        gallery_scrollbar_gen = ttk.Scrollbar(gallery_canvas_frame_gen, orient="horizontal", command=self.gallery_canvas_gen.xview)
        self.gallery_canvas_gen.configure(xscrollcommand=gallery_scrollbar_gen.set)
        
        self.gallery_canvas_gen.pack(side="top", fill="both", expand=True)
        gallery_scrollbar_gen.pack(side="bottom", fill="x")
        
        self.gallery_frame_inner_gen = ttk.Frame(self.gallery_canvas_gen)
        self.gallery_canvas_gen.create_window((0, 0), window=self.gallery_frame_inner_gen, anchor="nw")

        def _gallery_start_drag(event):
            self._gallery_drag_origin_y = event.y_root
            self._gallery_drag_start_sash = int(main_paned.sashpos(0))

        def _gallery_on_drag(event):
            if self._gallery_drag_origin_y is None or self._gallery_drag_start_sash is None:
                return
            delta = event.y_root - self._gallery_drag_origin_y
            total_h = max(1, main_paned.winfo_height())
            min_top = 220
            min_gallery = 28
            new_sash = self._gallery_drag_start_sash + delta
            new_sash = max(min_top, min(total_h - min_gallery, new_sash))
            main_paned.sashpos(0, int(new_sash))
            self.gallery_collapsed = (new_sash >= total_h - min_gallery)

        def _gallery_toggle(_event=None):
            if self.gallery_collapsed:
                total_h = max(1, main_paned.winfo_height())
                min_top = 220
                restore = self._gallery_saved_sash if self._gallery_saved_sash is not None else (total_h - 180)
                restore = max(min_top, min(total_h - 60, restore))
                main_paned.sashpos(0, int(restore))
                self.gallery_collapsed = False
            else:
                total_h = max(1, main_paned.winfo_height())
                min_top = 220
                min_gallery = 28
                self._gallery_saved_sash = int(main_paned.sashpos(0))
                collapsed_sash = max(min_top, total_h - min_gallery)
                main_paned.sashpos(0, int(collapsed_sash))
                self.gallery_collapsed = True

        def _init_gallery_split():
            total_h = max(1, main_paned.winfo_height())
            min_top = 220
            start_sash = max(min_top, total_h - 180)
            main_paned.sashpos(0, int(start_sash))

        self.root.after(80, _init_gallery_split)

        gallery_handle.bind("<ButtonPress-1>", _gallery_start_drag)
        gallery_handle.bind("<B1-Motion>", _gallery_on_drag)
        gallery_handle.bind("<Double-Button-1>", _gallery_toggle)
        
        # Défilement avec la molette
        self.gallery_canvas_gen.bind("<MouseWheel>", lambda e: self.gallery_canvas_gen.xview_scroll(-1 * (e.delta // 120), "units"))
        self.gallery_canvas_gen.bind("<Button-4>", lambda e: self.gallery_canvas_gen.xview_scroll(-1, "units"))
        self.gallery_canvas_gen.bind("<Button-5>", lambda e: self.gallery_canvas_gen.xview_scroll(1, "units"))
    
    def creer_onglet_assistant(self):
        """Onglet Assistant combiné"""
        #ttk.Label(self.tab_assistant, text="Votre Assistant IA Complet", font=("Arial", 28, "bold")).pack(pady=5)
        
        # Instructions
        #instructions = tk.Text(self.tab_assistant, height=3, wrap=tk.WORD, font=("Arial", 16))
        #instructions.pack(fill="x", padx=10, pady=2)
        #instructions.insert("1.0", 
        #    "💡 Décrivez ce que vous voulez en langage naturel. L'IA optimise votre demande et génère l'image.\n"
        #    "Exemple : 'Je veux une image relaxante pour méditer'")
        #instructions.config(state='disabled', bg="#f0f0f0")

        # === SÉLECTEUR DE MODE (dédié à l'assistant) ===
        mode_frame = ttk.Frame(self.tab_assistant)
        mode_frame.pack(pady=5, padx=10, fill="x")
        
        ttk.Label(mode_frame, text="Mode de génération :", font=("Arial", 14, "bold")).pack(side="left", padx=5)
        self.assistant_mode_combo = ttk.Combobox(mode_frame, values=[
            "Local CPU (SD-Turbo)",
            "Hugging Face / Flux"
        ], state="readonly", width=28, font=("Arial", 12))
        self.assistant_mode_combo.set("Hugging Face / Flux")
        self.assistant_mode_combo.pack(side="left", padx=5, fill="x", expand=True)

        # Sélecteur d'endpoint HuggingFace pour l'assistant
        hf_frame = ttk.Frame(self.tab_assistant)
        hf_frame.pack(pady=2, padx=10, fill="x")
        ttk.Label(hf_frame, text="Hugging Face / Flux :", font=("Arial", 12)).pack(side="left", padx=5)
        self.assistant_hf_endpoint_combo = ttk.Combobox(
            hf_frame,
            values=[
                "API Inference (images)",
                "Router HF Inference (fallback TLS)",
                "api-inference.huggingface.co",
                "router.huggingface.co/hf-inference",
            ],
            state="readonly",
            width=25,
            font=("Arial", 12)
        )
        self.assistant_hf_endpoint_combo.set(self.hf_endpoint_strategy)
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
        ttk.Label(input_frame, text="Votre demande (texte ou image) :", font=("Arial", 14, "bold")).pack(pady=2)
        self.assistant_input = tk.Text(input_frame, height=2, font=("Arial", 14))
        self.assistant_input.pack(fill="both", expand=True, padx=8, pady=1)
        
        # Bouton pour charger une image prompt
        image_prompt_frame = ttk.Frame(input_frame)
        image_prompt_frame.pack(pady=2, padx=8, fill="x")
        tk.Button(image_prompt_frame, text="🖼️ Charger Image Prompt", 
                 command=self.charger_image_prompt, bg="#4a90e2", fg="white", 
                 font=("Arial", 11)).pack(side="left", padx=2)
        self.btn_effacer_image_prompt = tk.Button(image_prompt_frame, text="❌", 
                 command=self.effacer_image_prompt, bg="#e74c3c", fg="white", 
                 font=("Arial", 10), state="disabled")
        self.btn_effacer_image_prompt.pack(side="left", padx=2)
        
        # Label pour afficher l'image prompt (miniature)
        self.assistant_image_prompt_label = tk.Label(input_frame, text="Aucune image chargée", 
                                                      relief="sunken", bg="#f5f5f5", 
                                                      height=3, cursor="hand2")
        self.assistant_image_prompt_label.pack(fill="x", padx=8, pady=2)
        self.assistant_image_prompt_label.bind(
            "<Button-1>",
            lambda e: self.ouvrir_apercu_image_prompt() if self.assistant_image_prompt is not None else self.charger_image_prompt()
        )
        
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
        
        # Boutons sur 3 colonnes (gain de place en hauteur)
        ttk.Separator(input_frame, orient="horizontal").pack(fill="x", pady=3)

        buttons_grid_frame = ttk.Frame(input_frame)
        buttons_grid_frame.pack(fill="x", padx=3, pady=2)

        # Ligne 1 : Créer | Recréer | Enregistrer
        tk.Button(buttons_grid_frame, text="🚀 Créer", command=self.assistant_creer,
                  bg="purple", fg="white", font=("Arial", 11, "bold"), width=10).grid(
            row=0, column=0, padx=2, pady=1, sticky="ew"
        )

        self.btn_recreer = tk.Button(buttons_grid_frame, text="🔄 Recréer", command=self.assistant_recreer,
                                     bg="orange", fg="white", font=("Arial", 10), width=10)
        self.btn_recreer.grid(row=0, column=1, padx=2, pady=1, sticky="ew")

        tk.Button(buttons_grid_frame, text="💾 Enregistrer", command=self.enregistrer_image,
                  font=("Arial", 10), width=10).grid(row=0, column=2, padx=2, pady=1, sticky="ew")

        # Ligne 2 : Bibliothèque | Arrêter | Vider
        tk.Button(buttons_grid_frame, text="📚 Bibliotheque", command=self.ouvrir_menu_bibliotheque,
                  bg="blue", fg="white", font=("Arial", 10), width=10).grid(
            row=1, column=0, padx=2, pady=1, sticky="ew"
        )

        self.btn_stop = tk.Button(buttons_grid_frame, text="⏹️ Arrêter", command=self.stop_auto_recreate,
                                  bg="red", fg="white", font=("Arial", 10), width=10, state="disabled")
        self.btn_stop.grid(row=1, column=1, padx=2, pady=1, sticky="ew")

        tk.Button(buttons_grid_frame, text="🗑️ Vider", command=self.vider_resultats_assistant,
                  bg="#FF5722", fg="white", font=("Arial", 10), width=10).grid(
            row=1, column=2, padx=2, pady=1, sticky="ew"
        )

        # Ligne 3 : Profils de Karine, Gabriel et Iris | Mémoriser depuis demande
        tk.Button(buttons_grid_frame, text="🧠 Profils de Karine, Gabriel et Iris", command=self.ouvrir_editeur_profil_serge424,
                  bg="#455A64", fg="white", font=("Arial", 10, "bold"), width=10).grid(
            row=2, column=0, padx=2, pady=3, sticky="ew"
        )

        tk.Button(buttons_grid_frame, text="📝 Mémoriser depuis demande", command=self.memoriser_profil_depuis_demande,
                  bg="#00838F", fg="white", font=("Arial", 10, "bold"), width=10).grid(
            row=2, column=1, padx=2, pady=3, sticky="ew"
        )

        # Configurer les colonnes pour qu'elles s'étendent également
        buttons_grid_frame.columnconfigure(0, weight=1)
        buttons_grid_frame.columnconfigure(1, weight=1)
        buttons_grid_frame.columnconfigure(2, weight=1)
        
        # === DROITE : PANNEAU DE RÉSULTATS ===
        results_frame = tk.LabelFrame(horizontal_paned, text="✨ Résultats", relief=tk.GROOVE)
        horizontal_paned.add(results_frame, weight=3)
        
        # PanedWindow HORIZONTAL pour les prompts et l'image
        vertical_results = ttk.PanedWindow(results_frame, orient=tk.VERTICAL)
        vertical_results.pack(fill="both", expand=True, padx=3, pady=3)
        
        # En haut : Image générée
        right_frame = tk.LabelFrame(vertical_results, text="Image générée (cliquez pour ajouter)", relief=tk.GROOVE)
        vertical_results.add(right_frame, weight=3)

        assistant_seed_result_frame = ttk.Frame(right_frame)
        assistant_seed_result_frame.pack(pady=(3, 0), padx=3, fill="x")
        ttk.Label(assistant_seed_result_frame, text="Seed photo :", font=("Arial", 11)).pack(side="left", padx=(2, 4))
        self.assistant_generated_seed_entry = ttk.Entry(
            assistant_seed_result_frame,
            width=14,
            font=("Arial", 11),
            textvariable=self.assistant_generated_seed_var,
            state="readonly"
        )
        self.assistant_generated_seed_entry.pack(side="left", padx=2)
        ttk.Label(assistant_seed_result_frame, text="(dernière génération)", font=("Arial", 10)).pack(side="left", padx=2)
        
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
        
        # PANNEAU 3 : Conseils & Idées - DÉPLACÉ DANS LE MENU BIBLIOTHÈQUE
        # Le panneau "Conseils, Suggestions & Idées" est maintenant accessible via le bouton Bibliothèque
        # tips_frame = tk.LabelFrame(main_paned, text="💡 Conseils, Suggestions & Idées", relief=tk.GROOVE)
        # main_paned.add(tips_frame, weight=1)
        
        # Notebook pour les conseils (maintenant dans une fenêtre séparée)
        # self.tips_notebook = ttk.Notebook(tips_frame)
        # # self.tips_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # === ONGLET 1 : CONSEILS ===
        # tips_tab = ttk.Frame(self.tips_notebook)
        # self.tips_notebook.add(tips_tab, text="📖 Conseils")
        
        # tips_text = scrolledtext.ScrolledText(tips_tab, wrap=tk.WORD, font=("Arial", 12), height=5)
        # tips_text.pack(fill="both", expand=True, padx=5, pady=5)
        # tips_text.insert("1.0", 
        #     "✅ CONSEILS POUR DE MEILLEURS PROMPTS :\n"
        #     "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        #     "1️⃣ Soyez spécifique : Décrivez les couleurs, styles, ambiance\n"
        #     "2️⃣ Utilisez des références : 'dans le style de Van Gogh', 'photographie 8K'\n"
        #     "3️⃣ Ajoutez des détails d'éclairage : 'lumière dorée', 'cinématique', 'contre-jour'\n"
        #     "4️⃣ Évitez les négations : Utiliser le prompt négatif plutôt que 'pas de...'\n"
        #     "5️⃣ Soyez court mais dense : 50-150 mots c'est parfait\n"
        #     "6️⃣ Testez le même prompt plusieurs fois avec des seeds différents\n"
        #     "7️⃣ Combinez styles : 'oil painting mixed with watercolor, illustration'\n"
        #     "8️⃣ Précisez la composition : 'portrait, full body, macro, wide angle'\n"
        #     "\n🎨 STYLES POPULAIRES :\n"
        #     "• 3D rendering, Unreal Engine, Blender\n"
        #     "• Oil painting, Watercolor, Digital art\n"
        #     "• Photography, Cinematic, Movie poster\n"
        #     "• Anime, Cartoon, Comic book style")
        # tips_text.config(state='disabled', bg="#f9f9f9")
        
        # === ONGLET 2 : SUGGESTIONS ===
        # suggestions_tab = ttk.Frame(self.tips_notebook)
        # self.tips_notebook.add(suggestions_tab, text="💬 Suggestions")
        
        # suggestions_inner = ttk.Frame(suggestions_tab)
        # suggestions_inner.pack(fill="both", expand=True, padx=5, pady=5)
        
        # suggestions_text = scrolledtext.ScrolledText(suggestions_inner, wrap=tk.WORD, font=("Arial", 12), height=5)
        # suggestions_text.pack(fill="both", expand=True, side="left")
        
        # scrollbar_suggestions = ttk.Scrollbar(suggestions_inner, command=suggestions_text.yview)
        # scrollbar_suggestions.pack(side="right", fill="y")
        # suggestions_text.config(yscrollcommand=scrollbar_suggestions.set)
        
        # suggestions_text.insert("1.0",
        #     "🎯 SUGGESTIONS D'AMÉLIORATIONS :\n"
        #     "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        #     "📌 Si votre prompt est trop court :\n"
        #     "→ Ajouter : style, qualité, éclairage, composition\n\n"
        #     "📌 Si l'image n'a pas les détails désirés :\n"
        #     "→ Soyez plus précis sur les couleurs et matériaux\n\n"
        #     "📌 Si la qualité est faible :\n"
        #     "→ Ajoutez : '4K', 'cinematic', 'professional', 'detailed'\n\n"
        #     "📌 Si les proportions sont mauvaises :\n"
        #     "→ Spécifiez : 'anatomically correct', 'proper proportions'\n\n"
        #     "📌 Combiner styles pour des résultats uniques :\n"
        #     "→ 'oil painting + digital art', 'vintage + modern'\n\n"
        #     "📌 Pour plus de contrôle, utilisez des seeds :\n"
        #     "→ Même seed = même composition, différent prompt = variation\n\n"
        #     "⚡ PROMPT NÉGATIF EFFICACE :\n"
        #     "'blurry, low quality, distorted, deformed, ugly, bad anatomy,\n"
        #     "watermark, text, out of frame, oversaturated'")
        # suggestions_text.config(state='disabled', bg="#fafafa")
        
        # === ONGLET 3 : IDÉES D'IMAGES ===
        # ideas_tab = ttk.Frame(self.tips_notebook)
        # self.tips_notebook.add(ideas_tab, text="🎨 Idées d'Images")
        
        # Frame scrollable pour les idées cliquables
        # ideas_canvas_frame = ttk.Frame(ideas_tab)
        # ideas_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # ideas_canvas = tk.Canvas(ideas_canvas_frame, bg="#ffffff", height=150)
        # ideas_scrollbar = ttk.Scrollbar(ideas_canvas_frame, orient="vertical", command=ideas_canvas.yview)
        # ideas_canvas.configure(yscrollcommand=ideas_scrollbar.set)
        
        # ideas_canvas.pack(side="left", fill="both", expand=True)
        # ideas_scrollbar.pack(side="right", fill="y")
        
        # ideas_frame_inner = ttk.Frame(ideas_canvas)
        # ideas_canvas.create_window((0, 0), window=ideas_frame_inner, anchor="nw")
        
        # Idées prédéfinies cliquables
        # ideas_list = [
        #     ("🌅 Coucher de soleil tropical", "A stunning tropical sunset over crystal clear ocean, golden hour lighting, cinematic, 8K, detailed sky"),
        #     ("🏰 Château fantasy", "A magical enchanted castle with floating islands, fantasy art style, bioluminescent lights, magical atmosphere, intricate details"),
        #     ("🤖 Cyborg futuriste", "A sleek retro-futuristic cyborg character, cyberpunk style, neon lights, detailed metallic parts, cinematic lighting, 8K"),
        #     ("🌌 Espace galactique", "A beautiful galaxy with nebulas and stars, cosmic art, vibrant colors, deep space, 3D rendering, cinematic"),
        #     ("🦁 Animal majestueux", "A majestic lion portrait in natural light, wildlife photography style, sharp focus, professional details, 4K"),
        #     ("🎭 Portrait artistique", "An artistic portrait with surreal elements, oil painting style, dramatic lighting, detailed features, masterpiece"),
        #     ("🌿 Nature relaxante", "A serene forest scene with a misty waterfall, peaceful atmosphere, natural lighting, 8K resolution, botanical details"),
        #     ("👽 Créature alien", "A fascinating alien creature design, sci-fi concept art, bioluminescent features, extraterrestrial, detailed anatomy, cinematic"),
        #     ("🏙️ Ville cyberpunk", "A dark futuristic mega-city with neon lights, cyberpunk aesthetic, flying vehicles, detailed architecture, moody atmosphere"),
        #     ("✨ Monde magique", "A magical fantasy world with floating objects, glowing runes, mystical atmosphere, detailed environment, cinematic, 8K")
        # ]
        
        # for emoji_title, prompt in ideas_list:
        #     btn = tk.Button(
        #         ideas_frame_inner,
        #         text=emoji_title,
        #         font=("Arial", 12),
        #         bg="#e3f2fd",
        #         fg="#1976d2",
        #         relief=tk.RAISED,
        #         padx=10,
        #         pady=6,
        #         wraplength=250,
        #         justify="center",
        #         cursor="hand2",
        #         command=lambda p=prompt: self.utiliser_idee(p)
        #     )
        #     btn.pack(fill="x", padx=3, pady=2)
        #     btn.bind("<Enter>", lambda e, p=prompt: self.afficher_tooltip_idee(e, p))
        #     btn.bind("<Leave>", lambda e: self.cacher_tooltip_idee())
        
        # Recalculer la zone de scroll
        # ideas_frame_inner.update_idletasks()
        # ideas_canvas.configure(scrollregion=ideas_canvas.bbox("all"))
        
        # Défilement à la molette
        # ideas_canvas.bind("<MouseWheel>", lambda e: ideas_canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        # ideas_canvas.bind("<Button-4>", lambda e: ideas_canvas.yview_scroll(-1, "units"))
        # ideas_canvas.bind("<Button-5>", lambda e: ideas_canvas.yview_scroll(1, "units"))
        
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

        # Commandes locales du chatbot (ne pas envoyer au modèle).
        if self._traiter_commande_chat(message, image_path=image_path):
            self.chat_input.delete("1.0", tk.END)
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
            self.last_chat_image_for_story = image_path
            self._afficher_image_dans_chat(image_path)
            self._supprimer_image_chat()

        self.chat_status.config(text="Réflexion en cours...", foreground="orange")
        threading.Thread(target=self._chat_thread, args=(message, image_path), daemon=True).start()

    def _traiter_commande_chat(self, message, image_path=None):
        """Traite les commandes locales (ex: renommer) et retourne True si gérée."""
        if image_path:
            return False

        texte = (message or "").strip()
        if not texte:
            return False

        # Accepte la faute fréquente "ronommer" en plus de "renommer".
        match_rename = re.match(r"^(?:renommer|ronommer)\b\s*(.*)$", texte, flags=re.IGNORECASE)
        if not match_rename:
            return False

        nouveau_titre = match_rename.group(1).strip()
        self._on_renommer_conversation(nouveau_titre=nouveau_titre)
        return True
    
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

            system_prompt = self._build_chat_system_prompt()

            history_text = self._construire_historique_chat()
            assistant_context_text = self._construire_contexte_assistant()
            serge_profile_context = self._construire_contexte_profil_serge424()
            prompt_sections = []

            if history_text:
                prompt_sections.append(
                    "Historique récent de la conversation (à utiliser comme mémoire de contexte):\n"
                    f"{history_text}"
                )

            if assistant_context_text:
                prompt_sections.append(
                    "Contexte Assistant Complet (prompts et générations déjà faits):\n"
                    f"{assistant_context_text}"
                )

            if serge_profile_context:
                prompt_sections.append(
                    "Profils de Karine, Gabriel et Iris (à respecter dans la réponse):\n"
                    f"{serge_profile_context}"
                )

            if build_shared_context_for_chat is not None:
                try:
                    shared_context_text = build_shared_context_for_chat(
                        session_id=str(self.current_conversation_id or ""),
                        character_id=str(self.shared_character_id or "severine"),
                        max_events=6,
                    )
                except Exception:
                    shared_context_text = ""
                if shared_context_text:
                    prompt_sections.append(
                        "Contexte partage recent (inter-apps):\n"
                        f"{shared_context_text}"
                    )

            prompt_sections.append(f"Message utilisateur actuel:\n{full_prompt}")
            full_prompt = "\n\n".join(prompt_sections)

            source = self.chat_source_combo.get() if hasattr(self, "chat_source_combo") else "Ollama (local)"
            hf_model = self.chat_hf_model_combo.get().strip() if hasattr(self, "chat_hf_model_combo") else ""
            ollama_model = self.chat_ollama_model_combo.get().strip() if hasattr(self, "chat_ollama_model_combo") else self.ollama_model

            if "Hugging Face" in source:
                response = self._chat_huggingface(full_prompt, system_prompt, image_path=image_path)
            else:
                response = ""
                # Utiliser le modèle Ollama sélectionné
                model_to_use = ollama_model if ollama_model else self.ollama_model
                if image_path:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    for chunk in ollama.generate(
                        model=model_to_use,
                        prompt=full_prompt,
                        system=system_prompt, stream=True, images=[image_bytes]
                    ):
                        response += chunk.get('response', '')
                else:
                    for chunk in ollama.generate(
                        model=model_to_use,
                        prompt=full_prompt,
                        system=system_prompt, stream=True
                    ):
                        response += chunk.get('response', '')

            self.afficher_chat("IA", response)

            if image_path:
                self._lancer_memorisation_auto_profil_image(message, response, image_path)

            # Lire la réponse à voix haute
            voice_enabled = self.chat_voice_enabled.get() if hasattr(self, "chat_voice_enabled") else True
            if self.tts_service and voice_enabled:
                self.speak(response)

            self.chat_status.config(text="Prêt", foreground="green")
        except Exception as e:
            self.afficher_chat("Système", f"❌ Erreur : {e}")
            self.chat_status.config(text="Erreur", foreground="red")

    def _build_chat_system_prompt(self):
        if build_shared_system_prompt is None:
            return (
                "Tu es un assistant utile, honnete et inoffensif. "
                "Tu dois toujours repondre uniquement en francais."
            )
        try:
            return build_shared_system_prompt(str(self.shared_character_id or "severine"))
        except Exception:
            return (
                "Tu es un assistant utile, honnete et inoffensif. "
                "Tu dois toujours repondre uniquement en francais."
            )

    def ouvrir_contexte_partage(self):
        lines = []
        if get_shared_recent_events is not None:
            try:
                events = get_shared_recent_events(max_events=50, session_id=str(self.current_conversation_id or ""))
                if not events:
                    events = get_shared_recent_events(max_events=50)
                for item in events:
                    lines.append(json.dumps(item, ensure_ascii=False))
            except Exception as e:
                lines.append(f"Erreur lecture contexte partage: {e}")

        content = "\n".join(lines).strip() or "Aucun evenement partage disponible."

        win = tk.Toplevel(self.root)
        win.title("Contexte partage")
        win.geometry("980x560")

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill="both", expand=True)

        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 10))
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", content)
        text_widget.config(state="disabled")

    def _texte_reference_serge424(self, texte):
        """Retourne True si le texte semble parler de Serge424."""
        lower = (texte or "").lower()
        marqueurs = ("serge424", "serge 424", "serge")
        return any(m in lower for m in marqueurs)

    def _lancer_memorisation_auto_profil_image(self, message, response, image_path):
        """Lance l'enrichissement auto du profil depuis une photo et sa description."""
        try:
            enabled = self.chat_auto_profile_from_image.get() if hasattr(self, "chat_auto_profile_from_image") else False
        except Exception:
            enabled = False

        if not enabled:
            return

        contexte = "\n".join([
            f"Message utilisateur: {message or ''}",
            f"Réponse IA: {response or ''}",
            f"Nom du fichier image: {os.path.basename(image_path) if image_path else ''}",
        ])

        # Garde-fou: ne pas toucher a Profils de Karine, Gabriel et Iris si Serge n'est pas mentionne.
        if not self._texte_reference_serge424(contexte):
            return

        threading.Thread(
            target=self._memoriser_profil_depuis_image_thread,
            args=(contexte,),
            daemon=True
        ).start()

    def _memoriser_profil_depuis_image_thread(self, contexte):
        """Extrait un profil depuis la description d'image puis fusionne en mémoire."""
        try:
            profil_extrait = self._extraire_profil_serge_depuis_demande(contexte)
            champs_majs = [
                key for key in ("physique", "caractere", "activites", "biens", "autres")
                if (profil_extrait.get(key, "") or "").strip()
            ]
            if not champs_majs:
                return

            profil_fusionne = self._fusionner_profil_serge(self.serge_profile_memory, profil_extrait)
            self.serge_profile_memory = profil_fusionne
            self._sauvegarder_profil_serge424()

            resume_champs = ", ".join(champs_majs)

            def _ui_success():
                self.chat_status.config(text=f"✅ Profils de Karine, Gabriel et Iris enrichis via photo ({resume_champs})", foreground="green")

            self.root.after(0, _ui_success)
        except Exception as e:
            print(f"Erreur mémorisation profil depuis image: {e}")

    def _chat_huggingface(self, full_prompt, system_prompt, image_path=None):
        """Génère une réponse texte via HuggingFace Inference API (OpenAI-compatible)"""
        return self._chat_huggingface_with_model(
            full_prompt,
            system_prompt,
            image_path=image_path,
            model_override=None,
            max_tokens=1024,
        )

    def _chat_huggingface_with_model(self, full_prompt, system_prompt, image_path=None, model_override=None, max_tokens=1024, request_timeout=60):
        """Génère une réponse texte via HF avec modèle imposé optionnel."""
        try:
            from config import HUGGING_FACE_TOKEN
        except ImportError:
            raise Exception("Token Hugging Face manquant dans config.py")

        try:
            from config import HUGGING_FACE_API_ROOT
        except ImportError:
            HUGGING_FACE_API_ROOT = "https://router.huggingface.co/hf-inference"
            
        HUGGING_FACE_API_ROOT = globals().get("HUGGING_FACE_API_ROOT") or "https://router.huggingface.co/hf-inference"

        def _normalize_chat_api_root(value):
            root = (value or "").strip()
            if not root:
                return ""
            if "://" not in root:
                root = f"https://{root}"
            root = root.replace("api-inference.co", "api-inference.huggingface.co")
            return root.rstrip("/")

        model = (model_override or "").strip()
        if not model:
            model = "gemma4:31b-cloud"
            if hasattr(self, "chat_hf_model_combo"):
                model = self.chat_hf_model_combo.get().strip() or model

        # Normaliser les alias locaux vers des IDs de modèles Hugging Face.
        model_alias_map = {
            # Compatibilite: redirige aussi les anciens alias retires.
            "gemma3:27b-cloud": "gemma4:31b-cloud",
            "ministral-3:8b-cloud": "gemma4:31b-cloud",
        }
        hf_model = model_alias_map.get(model, model)

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
            "model": hf_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": int(max(128, max_tokens)),
        }

        # Essayer plusieurs endpoints compatibles OpenAI pour éviter les erreurs provider.
        api_roots = []
        normalized_primary_root = _normalize_chat_api_root(HUGGING_FACE_API_ROOT)
        if normalized_primary_root:
            api_roots.append(normalized_primary_root)
        router_root = "https://router.huggingface.co"
        if router_root not in api_roots:
            api_roots.append(router_root)

        last_error = None
        for root in api_roots:
            url = f"{root}/v1/chat/completions"
            try:
                self._attendre_si_hf_trop_rapide("HF chat")
                response = requests.post(url, headers=headers, json=payload, timeout=max(30, int(request_timeout)), verify=True)
                if not response.ok:
                    details = response.text[:400] if response.text else "Aucun détail"
                    if response.status_code in (429, 503):
                        retry_after = response.headers.get("Retry-After")
                        try:
                            retry_after_sec = float(retry_after) if retry_after else 0.0
                        except (TypeError, ValueError):
                            retry_after_sec = 0.0
                        extra_wait = max(self.hf_retry_backoff_sec, retry_after_sec)
                        if extra_wait > 0:
                            time.sleep(extra_wait)
                    last_error = Exception(f"Hugging Face API ({response.status_code}): {details}")
                    continue

                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e

        raise last_error or Exception("Hugging Face API: échec de tous les endpoints")

    def ecrire_histoire_depuis_image_chat(self):
        """Crée un titre puis une histoire de 300 lignes depuis l'image importée du ChatBot."""
        image_path = self.chat_pending_image or self.last_chat_image_for_story
        if not image_path:
            self.afficher_chat("Système", "⚠️ Importez d'abord une image dans le ChatBot (bouton 📎 Image).")
            return

        if not os.path.exists(image_path):
            self.afficher_chat("Système", "⚠️ L'image sélectionnée est introuvable. Réimportez-la.")
            return

        self.chat_status.config(text="Création du titre et de l'histoire en cours...", foreground="orange")
        threading.Thread(target=self._histoire_image_thread, args=(image_path,), daemon=True).start()

    def _normaliser_titre_histoire(self, titre_brut):
        """Nettoie un titre généré par l'IA pour l'utiliser en affichage et nom de fichier."""
        titre = (titre_brut or "").strip()
        if not titre:
            return "Image sans titre"

        titre = titre.replace("\r", "\n")
        titre = titre.split("\n")[0].strip()
        titre = re.sub(r'^"+|"+$', "", titre).strip()
        titre = re.sub(r"^titre\s*[:\-]\s*", "", titre, flags=re.IGNORECASE).strip()
        if len(titre) > 120:
            titre = titre[:120].rstrip()
        return titre or "Image sans titre"

    def _extraire_lignes_non_vides(self, texte):
        """Retourne les lignes non vides d'un texte en conservant l'ordre."""
        lignes = []
        for raw_line in (texte or "").splitlines():
            line = raw_line.strip()
            if line:
                lignes.append(line)
        return lignes

    def _retirer_numerotation_lignes(self, lignes):
        """Supprime les préfixes de type 'Ligne N:' pour un rendu narratif naturel."""
        nettoyees = []
        for ligne in lignes or []:
            texte = (ligne or "").strip()
            texte = re.sub(r"^\s*(?:ligne|line)\s*\d+\s*[:\-\.)]\s*", "", texte, flags=re.IGNORECASE)
            texte = texte.strip()
            if texte:
                nettoyees.append(texte)
        return nettoyees

    def _completer_histoire_jusqua_300(self, titre, lignes_existantes):
        """Demande au modèle de compléter une histoire jusqu'à 300 lignes."""
        start_index = len(lignes_existantes) + 1
        if start_index > 300:
            return []

        extrait = "\n".join(lignes_existantes[-40:])
        prompt = (
            f"Tu continues une histoire dont le titre est: {titre}\n"
            f"Le texte contient déjà {len(lignes_existantes)} lignes.\n"
            f"Rédige UNIQUEMENT les lignes manquantes de Ligne {start_index}: à Ligne 300:.\n"
            "Respecte exactement le format 'Ligne N: ...' et n'ajoute aucun commentaire.\n\n"
            f"Dernières lignes déjà écrites:\n{extrait}"
        )
        system_prompt = "Tu es un auteur francophone. Tu respectes strictement le format demandé."
        try:
            suite = self._story_generate_with_ministral(
                prompt,
                system_prompt,
                image_path=None,
                max_tokens=3500,
            )
        except Exception:
            return []

        return self._extraire_lignes_non_vides(suite)

    def _story_generate_with_ministral(self, prompt, system_prompt, image_path=None, max_tokens=1024, route_tracker=None):
        """Génère via gemma4:31b-cloud (Ollama en priorité), fallback HF si nécessaire."""
        last_error = None

        def _mark_route(label):
            if route_tracker is not None:
                try:
                    route_tracker.add(label)
                except Exception:
                    pass

        if OLLAMA_AVAILABLE:
            try:
                kwargs = {
                    "model": "gemma4:31b-cloud",
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": int(max(128, max_tokens)),
                    },
                }

                if image_path:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    kwargs["images"] = [image_bytes]

                response = ollama.generate(**kwargs)
                text = response.get("response", "") if isinstance(response, dict) else ""
                if text and text.strip():
                    _mark_route("Ollama: gemma4:31b-cloud")
                    return text
                last_error = Exception("Réponse vide de gemma4:31b-cloud")
            except Exception as e:
                last_error = e

        # Fallback 2: autre modele Ollama selectionne dans l'UI.
        alt_ollama_model = ""
        if hasattr(self, "chat_ollama_model_combo"):
            alt_ollama_model = (self.chat_ollama_model_combo.get() or "").strip()
        if not alt_ollama_model:
            alt_ollama_model = (self.ollama_model or "").strip()

        if OLLAMA_AVAILABLE and alt_ollama_model and alt_ollama_model != "gemma4:31b-cloud":
            try:
                kwargs = {
                    "model": alt_ollama_model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": int(max(128, max_tokens)),
                    },
                }
                if image_path:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    kwargs["images"] = [image_bytes]

                response = ollama.generate(**kwargs)
                text = response.get("response", "") if isinstance(response, dict) else ""
                if text and text.strip():
                    _mark_route(f"Ollama: {alt_ollama_model}")
                    return text
            except Exception as e:
                last_error = e

        # Fallback robuste si ministral indisponible (token/model endpoint).
        try:
            fallback_model = "gemma4:31b-cloud"
            for timeout_s in (90, 140):
                try:
                    hf_text = self._chat_huggingface_with_model(
                        prompt,
                        system_prompt,
                        image_path=image_path,
                        model_override=fallback_model,
                        max_tokens=max_tokens,
                        request_timeout=timeout_s,
                    )
                    _mark_route(f"HF: {fallback_model}")
                    return hf_text
                except Exception as hf_err:
                    last_error = hf_err
                    continue
        except Exception as e:
            if last_error:
                raise Exception(f"ministral indisponible ({last_error}) ; fallback HF échoué ({e})")
            raise

        raise Exception(f"ministral indisponible ({last_error}) ; fallback HF échoué")

    def _titre_secours_depuis_fichier(self, image_path):
        """Construit un titre de secours lisible à partir du nom de fichier."""
        base_name = os.path.splitext(os.path.basename(image_path or ""))[0].strip()
        if not base_name:
            return "Souvenir d'image"
        human = re.sub(r"[_\-]+", " ", base_name)
        human = re.sub(r"\s+", " ", human).strip()
        return self._normaliser_titre_histoire(human.title())

    def _generer_histoire_locale_secours(self, titre, start_line=1, total_lines=300, route_tracker=None):
        """Produit une histoire locale de secours quand les modèles externes sont indisponibles."""
        if route_tracker is not None:
            try:
                route_tracker.add("Secours local")
            except Exception:
                pass

        sujets = [
            "la brume du matin", "une rue silencieuse", "un souffle de vent", "une lumière dorée",
            "un vieux carnet", "un sourire discret", "le murmure des arbres", "un pas hésitant"
        ]
        actions = [
            "ouvre un chemin inattendu", "réveille un souvenir ancien", "invite au courage",
            "change la trajectoire du héros", "donne naissance à une promesse", "éclaire une décision",
            "relie deux destins", "annonce un tournant décisif"
        ]
        tons = [
            "avec douceur", "dans un calme étrange", "avec une énergie nouvelle",
            "comme une évidence", "dans une émotion sincère", "avec une patience tranquille"
        ]

        lignes = []
        for n in range(start_line, total_lines + 1):
            sujet = random.choice(sujets)
            action = random.choice(actions)
            ton = random.choice(tons)
            lignes.append(f"Ligne {n}: Dans {titre}, {sujet} {action} {ton}.")
        return lignes

    def _generer_histoire_par_blocs(self, titre, total_lines=300, block_size=60, route_tracker=None):
        """Génère l'histoire par blocs pour limiter les timeouts sur les appels cloud."""
        lignes = []
        current = 1

        while current <= total_lines:
            block_end = min(total_lines, current + block_size - 1)
            contexte = "\n".join(lignes[-25:])
            prompt = (
                f"Titre: {titre}\n"
                f"Écris UNIQUEMENT les lignes de Ligne {current}: à Ligne {block_end}:.\n"
                "Chaque ligne doit suivre strictement le format 'Ligne N: ...'.\n"
                "N'ajoute aucun commentaire, aucun titre, aucun texte hors lignes.\n\n"
                f"Contexte narratif déjà écrit:\n{contexte}"
            )
            system_prompt = "Tu es un écrivain francophone rigoureux. Tu suis exactement le format demandé."

            try:
                block_text = self._story_generate_with_ministral(
                    prompt,
                    system_prompt,
                    image_path=None,
                    max_tokens=1800,
                    route_tracker=route_tracker,
                )
                block_lines = self._extraire_lignes_non_vides(block_text)
            except Exception:
                block_lines = []

            # Si le bloc échoue, on génère localement juste la portion manquante.
            if not block_lines:
                block_lines = self._generer_histoire_locale_secours(
                    titre,
                    start_line=current,
                    total_lines=block_end,
                    route_tracker=route_tracker,
                )

            # Limiter au nombre attendu de lignes pour le bloc courant.
            expected = block_end - current + 1
            if len(block_lines) < expected:
                secours = self._generer_histoire_locale_secours(
                    titre,
                    start_line=current + len(block_lines),
                    total_lines=block_end,
                    route_tracker=route_tracker,
                )
                block_lines.extend(secours)

            lignes.extend(block_lines[:expected])
            current = block_end + 1

        return lignes[:total_lines]

    def _sauvegarder_histoire_image(self, titre, histoire):
        """Sauvegarde l'histoire générée dans un fichier texte horodaté."""
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", titre).strip("_")
        if not slug:
            slug = "histoire"
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.stories_dir / f"histoire_{horodatage}_{slug[:50]}.txt"
        contenu = f"Titre: {titre}\n\n{histoire}\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(contenu)
        return file_path

    def _ouvrir_fenetre_histoire(self, titre, histoire, file_path):
        """Affiche l'histoire complète dans une fenêtre dédiée du ChatBot."""
        win = tk.Toplevel(self.root)
        win.title("Histoire générée depuis l'image")
        win.geometry("920x760")

        header = ttk.Frame(win, padding=10)
        header.pack(fill="x")
        ttk.Label(header, text=f"Titre : {titre}", font=("Arial", 14, "bold")).pack(anchor="w")
        ttk.Label(header, text=f"Fichier : {file_path}", font=("Arial", 10)).pack(anchor="w", pady=(2, 0))

        txt = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Arial", 12))
        txt.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        txt.insert("1.0", histoire)
        txt.config(state="disabled")

    def _histoire_image_thread(self, image_path):
        """Thread de création: titre image puis histoire de 300 lignes."""
        try:
            routes_utilisees = set()

            title_prompt = (
                "Observe cette image et propose un titre court, évocateur et poétique en français. "
                "Réponds uniquement par le titre sur une seule ligne."
            )
            title_system = "Tu es un expert en titrage d'images. Réponds en français, une ligne, sans commentaire."
            try:
                titre_brut = self._story_generate_with_ministral(
                    title_prompt,
                    title_system,
                    image_path=image_path,
                    max_tokens=120,
                    route_tracker=routes_utilisees,
                )
                titre = self._normaliser_titre_histoire(titre_brut)
            except Exception:
                titre = self._titre_secours_depuis_fichier(image_path)
                routes_utilisees.add("Titre secours (nom de fichier)")

            lignes = self._generer_histoire_par_blocs(
                titre,
                total_lines=300,
                block_size=60,
                route_tracker=routes_utilisees,
            )
            if len(lignes) < 300:
                lignes.extend(
                    self._generer_histoire_locale_secours(
                        titre,
                        start_line=len(lignes) + 1,
                        total_lines=300,
                        route_tracker=routes_utilisees,
                    )
                )

            lignes = self._retirer_numerotation_lignes(lignes[:300])
            if len(lignes) < 300:
                secours = self._generer_histoire_locale_secours(
                    titre,
                    start_line=len(lignes) + 1,
                    total_lines=300,
                    route_tracker=routes_utilisees,
                )
                secours = self._retirer_numerotation_lignes(secours)
                lignes.extend(secours[: max(0, 300 - len(lignes))])

            histoire_finale = "\n".join(lignes[:300])
            file_path = self._sauvegarder_histoire_image(titre, histoire_finale)

            def _ui_done():
                routes_text = " | ".join(sorted(routes_utilisees)) if routes_utilisees else "Non déterminé"
                self.afficher_chat(
                    "IA",
                    (
                        f"🖼️ Titre proposé : {titre}\n"
                        f"🧭 Voie utilisée : {routes_text}\n"
                        f"💾 Fichier : {file_path}\n\n"
                        f"📖 Histoire :\n\n{histoire_finale}"
                    )
                )
                self.chat_status.config(text=f"✅ Histoire terminée ({routes_text})", foreground="green")

            self.root.after(0, _ui_done)
        except Exception as e:
            self.root.after(0, lambda: self.afficher_chat("Système", f"❌ Erreur génération histoire : {e}"))
            self.root.after(0, lambda: self.chat_status.config(text="Erreur", foreground="red"))


    
    def afficher_chat(self, sender, message):
        """Affiche un message dans le chat et le sauvegarde"""
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, lambda s=sender, m=message: self.afficher_chat(s, m))
            return

        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"\n{sender} : {message}\n", sender)
        
        # Ajouter un bouton "Relire" sous le message
        btn_relire = tk.Button(
            self.chat_display,
            text="🔊 Relire",
            command=lambda t=message: self._relire_message(t),
            bg="#424242",
            fg="white",
            font=("Arial", 10),
            relief=tk.FLAT,
            cursor="hand2",
            padx=8,
            pady=2
        )
        self.chat_display.window_create(tk.END, window=btn_relire)
        self.chat_display.insert(tk.END, "\n")
        
        self.chat_display.see(tk.END)
        
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

            if generation_seed is None:
                generation_seed = random.randint(0, 2**31 - 1)

            if not seed_text and "Local" not in mode:
                self.generator = None
            
            # Récupérer le prompt négatif
            if negative_prompt is None:
                negative_prompt = self.image_negative_prompt.get("1.0", tk.END).strip()
            
            image = self._generer_image_par_mode(
                mode,
                prompt,
                negative_prompt=negative_prompt,
                seed=generation_seed,
                endpoint_override=self.hf_endpoint_combo.get().strip() if hasattr(self, "hf_endpoint_combo") else None,
            )
            
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
    
    def _generer_image_par_mode(self, mode, prompt, negative_prompt="", seed=None, input_image=None, endpoint_override=None):
        """Centralise la génération d'image pour limiter la duplication entre onglets."""
        if "Local" in mode:
            if not self.sd_pipe:
                raise Exception("SD-Turbo en chargement...")
            return self.sd_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=2,
                guidance_scale=0.0,
                generator=self.generator,
            ).images[0]

        if "Hugging Face" in mode:
            try:
                return self._generer_huggingface(
                    prompt,
                    seed=seed,
                    input_image=input_image,
                    endpoint_override=endpoint_override,
                    negative_prompt=negative_prompt,
                )
            except Exception as hf_error:
                # Si HF est indisponible, basculer automatiquement sur le local quand il est prêt.
                if self.sd_pipe is not None:
                    print(f"[ATTENTION] HF indisponible ({hf_error}) -> fallback local SD-Turbo")
                    return self.sd_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=2,
                        guidance_scale=0.0,
                        generator=self.generator,
                    ).images[0]
                raise

        raise Exception("Mode inconnu")

    def _is_non_retryable_hf_error(self, error):
        """Détecte les erreurs réseau locales qui ne méritent pas de retries HF."""
        message = str(error)
        markers = (
            "NameResolutionError",
            "getaddrinfo failed",
            "Failed to resolve",
            "No address associated with hostname",
            "SSLV3_ALERT_HANDSHAKE_FAILURE",
            "SSLError",
            "SSL: ",
            "handshake failure",
            "certificate verify failed",
        )
        return any(marker in message for marker in markers)

    def _describe_hf_endpoint_error(self, message):
        """Retourne un résumé plus précis et actionnable pour une erreur HF."""
        message = str(message or "")
        lower_message = message.lower()

        endpoint_label = "HF"
        if "api-inference.huggingface.co" in message:
            endpoint_label = "api-inference.huggingface.co"
        elif "router.huggingface.co/hf-inference" in message:
            endpoint_label = "router.huggingface.co/hf-inference"
        elif "router.huggingface.co" in message:
            endpoint_label = "router.huggingface.co"

        if "handshake" in lower_message or "ssl" in lower_message or "certificate" in lower_message:
            return (
                f"{endpoint_label}: échec TLS/SSL local "
                "(proxy/VPN/antivirus/inspection HTTPS ou chaîne certificat)"
            )

        if (
            "name resolution" in lower_message
            or "getaddrinfo" in lower_message
            or "failed to resolve" in lower_message
            or "no address associated with hostname" in lower_message
        ):
            return f"{endpoint_label}: résolution DNS impossible"

        if self._is_non_retryable_hf_error(message):
            return (
                f"{endpoint_label}: erreur réseau locale non récupérable"
            )

        if "Read timed out" in message or "Timeout" in message:
            return f"{endpoint_label}: délai d'attente dépassé"

        if any(code in message for code in ("(500)", "(502)", "(503)", "(504)")):
            return f"{endpoint_label}: erreur serveur ou passerelle côté Hugging Face"

        if "(429)" in message:
            return f"{endpoint_label}: limite de requêtes atteinte"

        compact = re.sub(r"https?://\S+", "[url]", message)
        compact = re.sub(r"\s+", " ", compact).strip()
        return compact[:180]

    def _format_hf_error_summary(self, endpoint_errors):
        """Condense les erreurs HF pour un affichage utilisateur lisible."""
        summaries = []
        for raw_error in endpoint_errors[-3:]:
            message = str(raw_error)

            summaries.append(self._describe_hf_endpoint_error(message))

        return " | ".join(summaries)
    
    def _generer_huggingface(self, prompt, seed=None, input_image=None, endpoint_override=None, negative_prompt=""):
        """Génère avec Hugging Face API
        
        Args:
            prompt: Le prompt textuel
            seed: La graine pour la génération
            input_image: Une image PIL à utiliser comme input (pour image-to-image)
            endpoint_override: Endpoint Hugging Face à privilégier pour cet appel
            negative_prompt: Le prompt négatif (à éviter dans l'image)
        """
        try:
            from config import HUGGING_FACE_TOKEN
        except ImportError:
            raise Exception("Token Hugging Face manquant dans config.py")

        try:
            from config import HUGGING_FACE_IMAGE_API_ROOT
        except ImportError:
            HUGGING_FACE_IMAGE_API_ROOT = "https://api-inference.huggingface.co"

        try:
            from config import HUGGING_FACE_IMAGE_MODEL
        except ImportError:
            HUGGING_FACE_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

        def _normalize_api_root(value):
            root = (value or "").strip()
            if not root:
                return ""
            alias_map = {
                # Compatibilite: conserver les anciens labels d'UI.
                "Root (router)": "https://router.huggingface.co/hf-inference",
                "API Inference (images)": "https://api-inference.huggingface.co",
                "Router HF Inference (fallback TLS)": "https://router.huggingface.co/hf-inference",
                "api-inference.co": "https://api-inference.huggingface.co",
                "https://api-inference.co": "https://api-inference.huggingface.co",
            }
            root = alias_map.get(root, root)
            if "://" not in root:
                root = f"https://{root}"
            root = root.replace("api-inference.co", "api-inference.huggingface.co")
            if root.rstrip("/") == "https://router.huggingface.co":
                root = "https://router.huggingface.co/hf-inference"
            return root.rstrip("/")

        configured_root = _normalize_api_root(HUGGING_FACE_IMAGE_API_ROOT)
        selected_root = _normalize_api_root(endpoint_override)

        api_roots = []
        for candidate in (
            selected_root,
            configured_root,
            "https://router.huggingface.co/hf-inference",
            "https://api-inference.huggingface.co",
        ):
            if candidate and candidate not in api_roots:
                api_roots.append(candidate)

        # Si on a une image en entrée, utiliser un modèle qui supporte img2img
        if input_image is not None:
            # Utiliser un modèle qui supporte l'image-to-image
            img2img_model = "timbrooks/instruct-pix2pix"
            
            # Préparer l'image comme fichier binaire
            buffered = BytesIO()
            input_image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            
            data = {
                "inputs": prompt
            }
            if seed is not None:
                data["seed"] = str(int(seed))
            
            headers = {
                "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
            }

            img2img_errors = []
            for api_root in api_roots:
                api_url = f"{api_root}/models/{img2img_model}"
                try:
                    self._attendre_si_hf_trop_rapide("HF img2img")
                    files = {
                        "image": ("input.png", BytesIO(image_bytes), "image/png")
                    }
                    response = requests.post(api_url, headers=headers, data=data, files=files, timeout=90, verify=True)

                    if not response.ok:
                        details = response.text[:300] if response.text else "Aucun détail"
                        img2img_errors.append(
                            f"Erreur Hugging Face API ({response.status_code}) sur {api_root}: {details}"
                        )
                        continue

                    content_type = response.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        details = response.text[:300] if response.text else "Réponse JSON sans image"
                        img2img_errors.append(
                            f"Réponse Hugging Face inattendue sur {api_root}: {details}"
                        )
                        continue

                    return Image.open(BytesIO(response.content))
                except Exception as e:
                    img2img_errors.append(f"{api_root}: {e}")

            print(f"[ATTENTION] Img2img indisponible avec {img2img_model}: {self._format_hf_error_summary(img2img_errors)}")
            print(f"   Fallback sans image d'entrée avec {HUGGING_FACE_IMAGE_MODEL}")
            input_image = None  # Désactiver l'image pour fallback

        # Mode classique : texte seulement (ou fallback)
        headers = {
            "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
            "Accept": "image/png",
            "Content-Type": "application/json",
        }
        
        payload = {"inputs": prompt}
        payload["parameters"] = {}
        if seed is not None:
            payload["parameters"]["seed"] = int(seed)
        
        # Ajouter le prompt négatif s'il est fourni
        if negative_prompt and negative_prompt.strip():
            payload["parameters"]["negative_prompt"] = negative_prompt.strip()

        last_error = None
        endpoint_errors = []
        transient_statuses = {429, 500, 502, 503, 504}
        max_attempts_per_root = 3
        for api_root in api_roots:
            api_url = f"{api_root}/models/{HUGGING_FACE_IMAGE_MODEL}"

            for attempt in range(1, max_attempts_per_root + 1):
                try:
                    self._attendre_si_hf_trop_rapide("HF image")
                    response = requests.post(api_url, headers=headers, json=payload, timeout=60, verify=True)

                    if not response.ok:
                        details = response.text[:300] if response.text else "Aucun détail"
                        if response.status_code == 410:
                            last_error = Exception(
                                f"Erreur Hugging Face API (410) sur {api_root}: endpoint obsolète"
                            )
                            endpoint_errors.append(str(last_error))
                            break

                        last_error = Exception(
                            f"Erreur Hugging Face API ({response.status_code}) sur {api_root}: {details}"
                        )

                        if response.status_code in transient_statuses and attempt < max_attempts_per_root:
                            retry_after = response.headers.get("Retry-After")
                            try:
                                retry_after_sec = float(retry_after) if retry_after else 0.0
                            except (TypeError, ValueError):
                                retry_after_sec = 0.0
                            extra_wait = max(self.hf_retry_backoff_sec, retry_after_sec, float(attempt))
                            print(
                                f"⚠️ HF image transitoire ({response.status_code}) sur {api_root}, "
                                f"nouvelle tentative {attempt + 1}/{max_attempts_per_root} dans {extra_wait:.1f}s"
                            )
                            time.sleep(extra_wait)
                            continue

                        endpoint_errors.append(str(last_error))
                        break

                    content_type = response.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        details = response.text[:300] if response.text else "Réponse JSON sans image"
                        last_error = Exception(f"Réponse Hugging Face inattendue sur {api_root}: {details}")
                        endpoint_errors.append(str(last_error))
                        break

                    return Image.open(BytesIO(response.content))

                except requests.exceptions.RequestException as e:
                    last_error = e
                    if self._is_non_retryable_hf_error(e):
                        endpoint_errors.append(f"{api_root}: {e}")
                        break
                    if attempt < max_attempts_per_root:
                        extra_wait = max(self.hf_retry_backoff_sec, float(attempt))
                        print(
                            f"⚠️ HF image indisponible sur {api_root} ({e}), "
                            f"nouvelle tentative {attempt + 1}/{max_attempts_per_root} dans {extra_wait:.1f}s"
                        )
                        time.sleep(extra_wait)
                        continue

                    endpoint_errors.append(f"{api_root}: {e}")
                    break

        if endpoint_errors:
            raise Exception(
                "Erreur Hugging Face: génération indisponible. "
                + self._format_hf_error_summary(endpoint_errors)
            )

        raise last_error or Exception("Erreur Hugging Face: aucun endpoint n'a répondu correctement")
    
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
        setattr(label, "image", photo)
    
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
    
    def charger_image_prompt(self):
        """Charge une image pour l'utiliser comme prompt"""
        filepath = filedialog.askopenfilename(
            title="Sélectionner une image prompt",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Tous", "*.*")]
        )
        if not filepath:
            return
        
        try:
            # Charger l'image
            image = Image.open(filepath)
            self.assistant_image_prompt = image.copy()
            self.assistant_image_prompt_path = filepath
            
            # Afficher une miniature dans le label
            thumb = image.copy()
            thumb.thumbnail((150, 80), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(thumb)
            self.assistant_image_prompt_label.config(image=photo, text="")
            setattr(self.assistant_image_prompt_label, "image", photo)  # Garder une référence
            
            # Activer le bouton effacer
            self.btn_effacer_image_prompt.config(state="normal")
            self.ouvrir_apercu_image_prompt()
            
            # Optionnel : vider le texte pour indiquer que l'image est prioritaire
            # self.assistant_input.delete("1.0", tk.END)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image: {e}")

    def ouvrir_apercu_image_prompt(self):
        """Affiche l'image prompt dans une fenêtre dédiée, plus lisible que la miniature intégrée."""
        if self.assistant_image_prompt is None:
            return

        try:
            if self.assistant_image_prompt_preview_window is not None:
                try:
                    if self.assistant_image_prompt_preview_window.winfo_exists():
                        self.assistant_image_prompt_preview_window.lift()
                        self.assistant_image_prompt_preview_window.focus_force()
                        return
                except Exception:
                    pass

            window = tk.Toplevel(self.root)
            window.title("Aperçu de l'image prompt")
            window.geometry("900x700")
            window.transient(self.root)
            window.lift()
            window.attributes("-topmost", True)
            window.after(250, lambda: window.attributes("-topmost", False))
            self.assistant_image_prompt_preview_window = window
            window.protocol("WM_DELETE_WINDOW", self._fermer_apercu_image_prompt)

            container = ttk.Frame(window, padding=10)
            container.pack(fill="both", expand=True)

            info = ttk.Label(
                container,
                text=os.path.basename(self.assistant_image_prompt_path or "Image prompt"),
                font=("Arial", 12, "bold")
            )
            info.pack(pady=(0, 8))

            preview_label = tk.Label(container, bg="#111111", bd=1, relief=tk.SUNKEN)
            preview_label.pack(fill="both", expand=True)

            image = self.assistant_image_prompt.copy()
            max_width = max(300, min(860, self.root.winfo_screenwidth() - 120))
            max_height = max(240, min(600, self.root.winfo_screenheight() - 180))
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            preview_label.config(image=photo)
            setattr(preview_label, "image", photo)

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher l'aperçu de l'image: {e}")

    def _fermer_apercu_image_prompt(self):
        """Ferme la fenêtre dédiée à l'aperçu de l'image prompt."""
        if self.assistant_image_prompt_preview_window is not None:
            try:
                self.assistant_image_prompt_preview_window.destroy()
            except Exception:
                pass
            self.assistant_image_prompt_preview_window = None
    
    def effacer_image_prompt(self):
        """Efface l'image prompt chargée"""
        self.assistant_image_prompt = None
        self.assistant_image_prompt_path = None
        self.assistant_image_prompt_label.config(image="", text="Aucune image chargée", bg="#f5f5f5")
        setattr(self.assistant_image_prompt_label, "image", None)
        self.btn_effacer_image_prompt.config(state="disabled")
        self._fermer_apercu_image_prompt()
    
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
    
    def vider_resultats_assistant(self):
        """Vide la fenêtre de résultats de l'Assistant Complet"""
        # Effacer l'image affichée
        self.assistant_image.config(image="")
        self.current_image = None
        self.original_assistant_image = None
        self.assistant_generated_seed_var.set("")
        
        # Vider les prompts optimisés
        self.assistant_prompt.delete("1.0", tk.END)
        self.assistant_negative_optimized.delete("1.0", tk.END)
        
        # Réinitialiser les prompts stockés
        self.last_optimized_prompt = ""
        self.last_optimized_negative_prompt = ""
        
        # Mettre à jour le status
        self.assistant_status.config(text="Résultats vidés")
    
    # === FONCTIONS ASSISTANT COMBINÉ ===
    
    def ouvrir_menu_bibliotheque(self):
        """Ouvre un menu avec Bibliothèque de prompts et Conseils & Idées"""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="📚 Bibliothèque de prompts", command=self.ouvrir_bibliotheque_prompts)
        menu.add_command(label="💡 Conseils & Idées", command=self.ouvrir_conseils_idees)
        
        # Afficher le menu à la position du curseur
        try:
            x, y = self.root.winfo_pointerx(), self.root.winfo_pointery()
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()
    
    def ouvrir_conseils_idees(self):
        """Ouvre une fenêtre séparée avec les conseils et idées"""
        conseils_win = tk.Toplevel(self.root)
        conseils_win.title("💡 Conseils, Suggestions & Idées")
        conseils_win.geometry("800x600")
        
        # Notebook pour les conseils
        tips_notebook = ttk.Notebook(conseils_win)
        tips_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # === ONGLET 1 : CONSEILS ===
        tips_tab = ttk.Frame(tips_notebook)
        tips_notebook.add(tips_tab, text="📖 Conseils")
        
        tips_text = scrolledtext.ScrolledText(tips_tab, wrap=tk.WORD, font=("Arial", 12), height=20)
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
        suggestions_tab = ttk.Frame(tips_notebook)
        tips_notebook.add(suggestions_tab, text="💬 Suggestions")
        
        suggestions_inner = ttk.Frame(suggestions_tab)
        suggestions_inner.pack(fill="both", expand=True, padx=5, pady=5)
        
        suggestions_text = scrolledtext.ScrolledText(suggestions_inner, wrap=tk.WORD, font=("Arial", 12), height=20)
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
        ideas_tab = ttk.Frame(tips_notebook)
        tips_notebook.add(ideas_tab, text="🎨 Idées d'Images")
        
        # Frame scrollable pour les idées cliquables
        ideas_canvas_frame = ttk.Frame(ideas_tab)
        ideas_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ideas_canvas = tk.Canvas(ideas_canvas_frame, bg="#ffffff")
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
                command=lambda p=prompt: self.utiliser_idee_depuis_fenetre(p, conseils_win)
            )
            btn.pack(fill="x", padx=3, pady=2)
        
        # Recalculer la zone de scroll
        ideas_frame_inner.update_idletasks()
        ideas_canvas.configure(scrollregion=ideas_canvas.bbox("all"))
        
        # Défilement à la molette
        ideas_canvas.bind("<MouseWheel>", lambda e: ideas_canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        ideas_canvas.bind("<Button-4>", lambda e: ideas_canvas.yview_scroll(-1, "units"))
        ideas_canvas.bind("<Button-5>", lambda e: ideas_canvas.yview_scroll(1, "units"))
    
    def utiliser_idee_depuis_fenetre(self, prompt, fenetre):
        """Utilise une idée cliquée depuis la fenêtre des conseils"""
        self.utiliser_idee(prompt)
        fenetre.destroy()
    
    def ouvrir_bibliotheque_prompts(self):
        """Ouvre la fenêtre de visualisation de la bibliothèque de prompts"""
        if not PROMPTS_LIBRARY_AVAILABLE:
            messagebox.showwarning("⚠️ Non disponible", "Bibliothèque de prompts non chargée")
            return
        
        # Ouvre la fenêtre en mode non-bloquant (threading)
        threading.Thread(target=lambda: open_prompts_viewer(self.root), daemon=True).start()
    
    def assistant_creer(self):
        """Mode assistant : optimise le prompt puis génère l'image"""
        print("DEBUG: assistant_creer() appelée")
        demande = self.assistant_input.get("1.0", tk.END).strip()
        has_image_prompt = self.assistant_image_prompt is not None
        print(f"DEBUG: demande='{demande}', has_image_prompt={has_image_prompt}")
        
        # Vérifier qu'il y a au moins un prompt (texte ou image)
        if not demande and not has_image_prompt:
            messagebox.showwarning("Attention", "Veuillez saisir une demande textuelle ou charger une image prompt.")
            return
        
        mode = self.assistant_mode_combo.get()
        print(f"DEBUG: mode='{mode}', OLLAMA_AVAILABLE={OLLAMA_AVAILABLE}, sd_pipe={self.sd_pipe is not None}")

        if not OLLAMA_AVAILABLE:
            self.assistant_status.config(text="❌ Services non disponibles")
            messagebox.showerror("Erreur", "Ollama n'est pas disponible.\nVeuillez installer Ollama : pip install ollama")
            return

        if "Local" in mode and not self.sd_pipe:
            self.assistant_status.config(text="❌ SD-Turbo en chargement (mode local)")
            messagebox.showwarning("Attention", "SD-Turbo est en cours de chargement.\nVeuillez patienter ou choisir un autre mode de génération.")
            return
        
        status_msg = "🤖 L'IA optimise votre demande..."
        if has_image_prompt and not demande:
            status_msg = "🖼️ Traitement de l'image prompt..."
        elif has_image_prompt:
            status_msg = "🤖 L'IA optimise votre demande avec l'image..."
        
        print(f"DEBUG: Lancement du thread avec status_msg='{status_msg}'")    
        self.assistant_status.config(text=status_msg)
        threading.Thread(target=self._assistant_thread, args=(demande,), daemon=True).start()
    
    def _assistant_thread(self, demande):
        """Thread pour l'assistant combiné"""
        print(f"DEBUG: _assistant_thread démarré avec demande='{demande}'")
        try:
            # Récupérer l'image prompt si elle existe
            input_image = self.assistant_image_prompt
            
            # Si on a une image mais pas de texte, créer une demande par défaut
            if input_image and not demande:
                demande = "Generate an image based on the provided reference image, maintaining its style and characteristics"
            
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

            # Injecter la mémoire du ChatBot et du contexte Assistant dans l'optimisation.
            history_text = self._construire_historique_chat()
            assistant_context_text = self._construire_contexte_assistant()
            serge_profile_context = self._construire_contexte_profil_serge424()
            context_parts = []
            if history_text:
                context_parts.append(
                    "Historique recent de la conversation (memoire contextuelle):\n"
                    f"{history_text}"
                )
            if assistant_context_text:
                context_parts.append(
                    "Contexte Assistant Complet (prompts/generations precedents):\n"
                    f"{assistant_context_text}"
                )
            if serge_profile_context:
                context_parts.append(
                    "Profils de Karine, Gabriel et Iris (physique, caractere, activites, biens, autres):\n"
                    f"{serge_profile_context}"
                )

            if context_parts:
                contexte_optimisation = "\n\n".join(context_parts)
                prompt_optimization = (
                    f"{contexte_optimisation}\n\n"
                    f"Demande actuelle de l'utilisateur:\n{demande}\n\n"
                    f"Instruction d'optimisation:\n{prompt_optimization}"
                )

                if negative_optimization:
                    negative_optimization = (
                        f"{contexte_optimisation}\n\n"
                        f"Demande actuelle de l'utilisateur:\n{demande}\n\n"
                        f"Instruction d'optimisation negative:\n{negative_optimization}"
                    )
            
            self.assistant_status.config(text=f"🤖 Optimisation du prompt (catégorie: {category})...")
            
            # Système prompts distincts pour positif et négatif
            system_prompt_positif = (
                "Tu es un expert en génération d'images IA. "
                "Réponds UNIQUEMENT avec le prompt positif en anglais, sans titre, sans explication, sans tiret."
            )
            system_prompt_negatif = (
                "Tu es un expert en génération d'images IA. "
                "Réponds UNIQUEMENT avec le prompt NÉGATIF en anglais (éléments à éviter), "
                "sans titre, sans explication, sans prompt positif."
            )
            
            optimized_prompt = ""
            try:
                for chunk in ollama.generate(model=self.ollama_model, prompt=prompt_optimization, system=system_prompt_positif, stream=True):
                    optimized_prompt += chunk.get('response', '')
            except Exception as _ollama_err:
                print(f"[ATTENTION] Ollama indisponible ({_ollama_err}) -> prompt original conserve")
                optimized_prompt = demande
            
            # Étape 1b : Optimiser le prompt négatif avec Ollama
            user_negative_prompt = self.assistant_negative_prompt.get("1.0", tk.END).strip()
            if user_negative_prompt and "Exemple:" not in user_negative_prompt:
                if not PROMPTS_LIBRARY_AVAILABLE:
                    negative_optimization = (
                        f"Tu es un expert en génération d'images IA. "
                        f"L'utilisateur veut éviter : '{user_negative_prompt}'. "
                        f"Crée un prompt négatif détaillé en anglais pour Stable Diffusion (max 30 mots). "
                        f"Réponds UNIQUEMENT avec le prompt négatif, sans explication."
                    )
                
                optimized_negative = ""
                try:
                    for chunk in ollama.generate(model=self.ollama_model, prompt=negative_optimization, system=system_prompt_negatif, stream=True):
                        optimized_negative += chunk.get('response', '')
                except Exception as _ollama_neg_err:
                    print(f"[ATTENTION] Ollama negatif indisponible ({_ollama_neg_err}) -> prompt negatif original conserve")
                    optimized_negative = user_negative_prompt
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
            
            image = self._generer_image_par_mode(
                mode,
                optimized_prompt.strip(),
                negative_prompt=optimized_negative.strip(),
                seed=generation_seed,
                input_image=input_image,
                endpoint_override=self.assistant_hf_endpoint_combo.get().strip() if hasattr(self, "assistant_hf_endpoint_combo") else None,
            )
            
            elapsed = time.time() - start
            
            # Stocker les prompts optimisés pour la fonction Recréer
            self.last_optimized_prompt = optimized_prompt.strip()
            self.last_optimized_negative_prompt = optimized_negative.strip()
            
            self.current_image = image
            self.assistant_generated_seed_var.set(str(generation_seed))
            self.afficher_image(image, self.assistant_image)
            chemin, save_error = self.sauvegarder_image_auto_safe(image, optimized_prompt.strip(), mode)

            image_name = chemin.name if chemin else ""
            self._enregistrer_contexte_assistant(
                demande=demande,
                prompt=optimized_prompt.strip(),
                negative_prompt=optimized_negative.strip(),
                mode=mode,
                image_name=image_name,
            )

            if chemin:
                self.assistant_status.config(text=f"✅ Terminé en {elapsed:.1f}s - Catégorie: {category} - Sauvé: {chemin.name}")
            else:
                self.assistant_status.config(
                    text=f"✅ Terminé en {elapsed:.1f}s - Catégorie: {category} - ⚠️ Sauvegarde auto impossible ({save_error})"
                )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERREUR] _assistant_thread: {error_details}")
            self.assistant_status.config(text=f"❌ Erreur : {e}")
            messagebox.showerror("Erreur de génération", f"Une erreur s'est produite :\n\n{str(e)[:200]}")
    
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

    def _assistant_recreate_success_ui(self, image, recreate_seed, elapsed, chemin, save_error):
        """Applique le résultat de recréation dans le thread UI."""
        try:
            self.current_image = image
            self.assistant_generated_seed_var.set(str(recreate_seed))
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
                
                image = self._generer_image_par_mode(
                    mode,
                    self.last_optimized_prompt,
                    negative_prompt=self.last_optimized_negative_prompt,
                    seed=recreate_seed,
                    endpoint_override=self.assistant_hf_endpoint_combo.get().strip() if hasattr(self, "assistant_hf_endpoint_combo") else None,
                )
                
                elapsed = time.time() - start
                chemin, save_error = self.sauvegarder_image_auto_safe(
                    image,
                    self.last_optimized_prompt,
                    f"{mode}_recreation"
                )

                # Toute opération Tkinter doit être exécutée sur le thread principal.
                self.root.after(0, self._assistant_recreate_success_ui, image, recreate_seed, elapsed, chemin, save_error)

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
                    setattr(btn_gen, "image", photo)
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
                    setattr(btn_asst, "image", photo2)
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
        if hasattr(self, 'tooltip_window') and self.tooltip_window is not None:
            try:
                self.tooltip_window.destroy()
            except Exception:
                pass
            self.tooltip_window = None


if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantIA(root)
    root.mainloop()
