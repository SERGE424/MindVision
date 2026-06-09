import warnings
warnings.filterwarnings('ignore')
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
import threading
from tqdm import tqdm
import os
import requests
from io import BytesIO
try:
    from config import HUGGING_FACE_TOKEN, REPLICATE_API_TOKEN
except ImportError:
    HUGGING_FACE_TOKEN = ""
    REPLICATE_API_TOKEN = ""
import time
import json
import base64


class GenerateurIA:
    def __init__(self, root):
        self.root = root
        self.root.title("Générateur IA Léger")
        self.root.geometry("900x1200")
        self.root.configure(bg="black")

        # Configuration API Cloud
        self.hf_token = HUGGING_FACE_TOKEN
        self.replicate_token = REPLICATE_API_TOKEN
        self.use_api = False  # Mode par défaut : local
        
        # Charger le modèle local
        self.pipe = None
        self.charger_modele_local()
        
        self.photo_ref = None
        self.current_image = None
        self.creer_interface()
    
    def charger_modele_local(self):
        try:
            print("Chargement du modèle local...")
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sd-turbo",
            )
            self.pipe.enable_attention_slicing()
            self.pipe.to("cpu")
            self.device = "cpu"
            print("Modèle local chargé !")
        except Exception as e:
            print(f"Erreur chargement modèle local : {e}")
            self.pipe = None
            self.device = "cpu"
    
    def creer_interface(self):
        # Sélecteur de mode
        mode_frame = ttk.Frame(self.root)
        mode_frame.pack(pady=5)
        ttk.Label(mode_frame, text="Mode :").pack(side="left", padx=5)
        self.mode_var = tk.StringVar(value="local")
        ttk.Radiobutton(mode_frame, text="Local (CPU)", variable=self.mode_var, value="local", command=self.changer_mode).pack(side="left")
        ttk.Radiobutton(mode_frame, text="AWS Bedrock ($0.04)", variable=self.mode_var, value="aws", command=self.changer_mode).pack(side="left")
        ttk.Radiobutton(mode_frame, text="Replicate", variable=self.mode_var, value="replicate", command=self.changer_mode).pack(side="left")
        
        # Prompt
        ttk.Label(self.root, text="Description :").pack(pady=5)
        self.prompt_entry = tk.Text(self.root, height=3, width=60)
        self.prompt_entry.pack(pady=5)
        
        # Bouton
        self.btn = tk.Button(self.root, text="Générer", command=self.lancer_generation, bg="green", fg="white")
        self.btn.pack(pady=10)
        
        
        # Bouton pour enregistrer l'image
        self.save_button = tk.Button(self.root, text="Enregistrer l'image", command=self.enregistrer_image)
        self.save_button.pack(pady=5)
        
        # Bouton pour upscaler l'image
        self.upscale_button = tk.Button(self.root, text="Upscaler (2x)", command=self.upscaler_image, bg="blue", fg="white")
        self.upscale_button.pack(pady=5)
        
        # Bouton pour génération automatique infinie
        self.auto_gen_button = ttk.Button(self.root, text="Génération Automatique", command=self.lancer_generation_infinie)
        self.auto_gen_button.pack(pady=5)
        
        # Bouton pour arrêter la génération automatique
        self.stop_button = ttk.Button(self.root, text="Arrêter", command=self.arreter_generation)
        self.stop_button.pack(pady=5)
        self.stop_event = threading.Event()
        
        # Status
        self.status = ttk.Label(self.root, text="Prêt")
        self.status.pack(pady=5)
        
        # Barre de progression
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=5)
        
        # Conteneur principal pour image et galerie côte à côte
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Cadre gauche - Image
        self.left_frame = ttk.Frame(self.main_container)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        ttk.Label(self.left_frame, text="Image actuelle :").pack()
        self.image_label = tk.Label(self.left_frame, relief="sunken", background="gray20")
        self.image_label.pack(fill="both", expand=True)
        
        # Cadre droit - Galerie
        self.right_frame = ttk.Frame(self.main_container)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        ttk.Label(self.right_frame, text="Galerie d'images :").pack()
        self.gallery_frame = ttk.Frame(self.right_frame)
        self.gallery_frame.pack(fill="both", expand=True)
        
        # Canvas avec scrollbars
        self.canvas = tk.Canvas(self.gallery_frame, bg="white")
        self.scrollbar_vertical = ttk.Scrollbar(self.gallery_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar_horizontal = ttk.Scrollbar(self.gallery_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scrollbar_vertical.set, xscrollcommand=self.scrollbar_horizontal.set)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Positionnement des éléments avec grid
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar_vertical.grid(row=0, column=1, sticky="ns")
        self.scrollbar_horizontal.grid(row=1, column=0, sticky="ew")
        
        self.gallery_frame.grid_rowconfigure(0, weight=1)
        self.gallery_frame.grid_columnconfigure(0, weight=1)
        
        # Configuration des colonnes du conteneur principal
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=1)

        self.image_thumbnails = []
        self.thumbnail_photos = []  # Pour stocker les références PhotoImage
        self.gallery_images = []  # Pour stocker les images complètes de la galerie

    def lancer_generation(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        if not prompt:
            return
        
        self.btn.config(state='disabled')
        self.status.config(text="Génération en cours...")
        
        # Thread pour ne pas bloquer l'interface
        thread = threading.Thread(target=self.generer, args=(prompt,))
        thread.start()
    
    def changer_mode(self):
        mode = self.mode_var.get()
        self.use_api = mode
        if mode == "replicate" and not self.replicate_token:
            messagebox.showwarning("Token manquant",
                "Configurez REPLICATE_API_TOKEN dans config.py")
            self.mode_var.set("local")
            self.use_api = "local"
    
    def generer(self, prompt):
        if self.use_api == "aws":
            self.generer_aws(prompt)
        elif self.use_api == "replicate":
            self.generer_replicate(prompt)
        elif self.use_api == "pollinations":
            self.generer_pollinations(prompt)
        else:
            self.generer_local(prompt)
    
    def generer_aws(self, prompt):
        try:
            import boto3
        except ImportError:
            self.status.config(text="Erreur : pip install boto3")
            self.btn.config(state='normal')
            return
        
        try:
            self.progress["maximum"] = 1
            self.progress["value"] = 0
            
            self.status.config(text="Connexion à AWS Bedrock...")
            
            start = time.time()
            
            # Client Bedrock (région Europe - Francfort)
            bedrock = boto3.client('bedrock-runtime', region_name='eu-central-1')
            
            # Paramètres pour Stable Diffusion XL
            body = json.dumps({
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "steps": 30,
                "seed": 0,
                "width": 1024,
                "height": 1024
            })
            
            # Appel à Bedrock
            response = bedrock.invoke_model(
                modelId='stability.stable-diffusion-xl-v1',
                body=body
            )
            
            # Décoder la réponse
            response_body = json.loads(response['body'].read())
            image_data = base64.b64decode(response_body['artifacts'][0]['base64'])
            image = Image.open(BytesIO(image_data))
            
            elapsed = time.time() - start
            
            self.progress["value"] = 1
            self.afficher_image(image)
            self.status.config(text=f"Terminé (AWS Bedrock) en {elapsed:.1f}s !")
            print(f"Génération AWS : {elapsed:.2f}s - Coût : ~$0.04")
            
        except Exception as e:
            self.status.config(text=f"Erreur AWS : {str(e)[:50]}")
            print(f"Erreur détaillée : {e}")
        finally:
            self.btn.config(state='normal')
            self.progress["value"] = 0
    
    def generer_replicate(self, prompt):
        try:
            import replicate
        except ImportError:
            self.status.config(text="Erreur : pip install replicate")
            self.btn.config(state='normal')
            return
        
        try:
            import time
            self.progress["maximum"] = 1
            self.progress["value"] = 0
            
            self.status.config(text="Connexion à Replicate (FLUX)...")
            
            # Configurer le token
            os.environ["REPLICATE_API_TOKEN"] = self.replicate_token
            
            start = time.time()
            
            # Appel à FLUX-Schnell (rapide et gratuit avec crédits)
            output = replicate.run(
                "black-forest-labs/flux-schnell",
                input={"prompt": prompt}
            )
            
            # Récupérer l'image
            image_data = list(output)[0].read()
            image = Image.open(BytesIO(image_data))
            
            elapsed = time.time() - start
            
            self.progress["value"] = 1
            self.afficher_image(image)
            self.status.config(text=f"Terminé (Replicate FLUX) en {elapsed:.1f}s !")
            print(f"Génération Replicate : {elapsed:.2f}s")
            
        except Exception as e:
            self.status.config(text=f"Erreur Replicate : {str(e)[:50]}")
            print(f"Erreur détaillée : {e}")
        finally:
            self.btn.config(state='normal')
            self.progress["value"] = 0
    
    def generer_pollinations(self, prompt):
        try:
            self.progress["maximum"] = 1
            self.progress["value"] = 0
            
            import urllib.parse
            encoded_prompt = urllib.parse.quote(prompt)
            
            # Essayer plusieurs APIs gratuites
            apis = [
                (f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true", "Pollinations.ai"),
                (f"https://pollinations.ai/p/{encoded_prompt}", "Pollinations v2"),
            ]
            
            for url, nom in apis:
                try:
                    self.status.config(text=f"Connexion à {nom}...")
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        self.progress["value"] = 1
                        self.afficher_image(image)
                        self.status.config(text=f"Terminé ({nom}) !")
                        return
                except:
                    continue
            
            self.status.config(text="APIs indisponibles, utilisez le mode Local")
            
        except Exception as e:
            self.status.config(text=f"Erreur : {str(e)[:50]}")
        finally:
            self.btn.config(state='normal')
            self.progress["value"] = 0
    
    def generer_local(self, prompt):
        import time
        try:
            if not self.pipe:
                self.status.config(text="Modèle local non disponible")
                return
            
            num_inference_steps = 1
            self.progress["maximum"] = num_inference_steps
            self.progress["value"] = 0
            
            # Mesurer le temps
            start = time.time()
            
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0
            ).images[0]
            
            elapsed = time.time() - start
            
            self.progress["value"] = num_inference_steps
            self.root.update_idletasks()
            self.afficher_image(image)
            self.status.config(text=f"Terminé en {elapsed:.1f}s !")
            print(f"Temps de génération : {elapsed:.2f}s")

        except Exception as e:
            self.status.config(text=f"Erreur : {e}")
        finally:
            self.btn.config(state='normal')
            self.progress["value"] = 0
    
    def afficher_image(self, image):
        image.thumbnail((512, 512))
        self.current_image = image
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.photo_ref = photo

        # Ajouter l'image à la galerie
        self.ajouter_galerie(image)

    def ajouter_galerie(self, image):
        # Stocker l'image complète
        self.gallery_images.append(image.copy())
        
        # Créer la miniature
        thumbnail = image.copy()
        thumbnail.thumbnail((128, 128))
        photo = ImageTk.PhotoImage(thumbnail)
        label = tk.Label(self.scrollable_frame, image=photo, relief="raised", borderwidth=2, bg="white")
        
        # Disposition en grille avec retour à la ligne
        # 3 colonnes par défaut
        num_cols = 3
        image_index = len(self.gallery_images) - 1
        row = image_index // num_cols
        col = image_index % num_cols
        
        label.grid(row=row, column=col, padx=5, pady=5)

        # Ajouter un binding pour cliquer sur la miniature
        label.bind("<Button-1>", lambda e, idx=image_index: self.selectionner_image_galerie(idx))

        # Stocker la référence de l'image pour éviter le garbage collection
        self.thumbnail_photos.append(photo)
        self.image_thumbnails.append(label)

        # Message de débogage
        print("Image ajoutée à la galerie.")

    def selectionner_image_galerie(self, index):
        """Sélectionner une image depuis la galerie et l'afficher"""
        if 0 <= index < len(self.gallery_images):
            self.current_image = self.gallery_images[index]
            photo = ImageTk.PhotoImage(self.current_image)
            self.image_label.config(image=photo)
            self.photo_ref = photo
            self.status.config(text=f"Image {index + 1} sélectionnée depuis la galerie")

    def enregistrer_image(self):
        if self.current_image:
            # Ouvrir une boîte de dialogue pour choisir l'emplacement
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialdir=os.path.join(os.getcwd(), "image"),
                title="Enregistrer l'image"
            )
            if filepath:
                self.current_image.save(filepath)
                self.status.config(text=f"Image enregistrée : {filepath}")
        else:
            self.status.config(text="Aucune image à enregistrer.")
    
    def upscaler_image(self):
        """Upscaler l'image actuelle avec LANCZOS (2x)"""
        if not self.current_image:
            self.status.config(text="Aucune image à upscaler")
            return
        
        try:
            self.status.config(text="Upscaling en cours...")
            self.upscale_button.config(state='disabled')
            
            # Thread pour ne pas bloquer l'interface
            thread = threading.Thread(target=self._upscaler_thread)
            thread.start()
        except Exception as e:
            self.status.config(text=f"Erreur upscaling : {e}")
            self.upscale_button.config(state='normal')
    
    def _upscaler_thread(self):
        """Thread pour upscaler l'image"""
        try:
            import time
            start = time.time()
            
            # Récupérer la taille originale
            original_size = self.current_image.size
            new_size = (original_size[0] * 2, original_size[1] * 2)
            
            # Upscaler avec LANCZOS (meilleure qualité)
            upscaled = self.current_image.resize(new_size, Image.Resampling.LANCZOS)
            
            elapsed = time.time() - start
            
            # Mettre à jour l'image actuelle
            self.current_image = upscaled
            
            # Afficher (avec thumbnail pour l'interface)
            display_image = upscaled.copy()
            display_image.thumbnail((512, 512))
            photo = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=photo)
            self.photo_ref = photo
            
            self.status.config(text=f"Upscalé en {elapsed:.1f}s ({new_size[0]}x{new_size[1]}) !")
            print(f"Upscaling : {original_size} -> {new_size} en {elapsed:.2f}s")
            
        except Exception as e:
            self.status.config(text=f"Erreur : {e}")
        finally:
            self.upscale_button.config(state='normal')

    def lancer_generation_infinie(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        if not prompt:
            return

        self.auto_gen_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status.config(text="Génération automatique en cours...")
        self.stop_event.clear()

        # Thread pour ne pas bloquer l'interface
        thread = threading.Thread(target=self.generer_en_boucle, args=(prompt,))
        thread.daemon = True  # Permet d'arrêter le thread avec l'application
        thread.start()

    def generer_en_boucle(self, prompt):
        try:
            while not self.stop_event.is_set():
                if self.use_api == "aws":
                    self.generer_aws(prompt)
                elif self.use_api == "replicate":
                    self.generer_replicate(prompt)
                elif self.use_api == "pollinations":
                    self.generer_pollinations(prompt)
                else:
                    if not self.pipe:
                        self.status.config(text="Modèle local non disponible")
                        break
                    
                    num_inference_steps = 1
                    self.progress["maximum"] = num_inference_steps
                    self.progress["value"] = 0

                    for step in range(num_inference_steps):
                        image = self.pipe(
                            prompt=prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=0.0
                        ).images[0]
                        self.progress["value"] += 1
                        self.root.update_idletasks()

                    self.afficher_image(image)
        except Exception as e:
            self.status.config(text=f"Erreur : {e}")
        finally:
            self.auto_gen_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.progress["value"] = 0

    def arreter_generation(self):
        self.stop_event.set()
        self.status.config(text="Génération arrêtée.")
        self.stop_button.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = GenerateurIA(root)
    root.mainloop()
