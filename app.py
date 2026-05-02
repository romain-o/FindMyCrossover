import os
import subprocess
import threading
import customtkinter as ctk

# --- CONFIGURATION VISUELLE ---
ctk.set_appearance_mode("Dark")  # Mode sombre pro
ctk.set_default_color_theme("blue")  # Thème d'accentuation

class FindMyCrossoverApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuration de la fenêtre
        self.title("FindMyCrossover - Studio")
        self.geometry("700x550")
        self.resizable(False, False)

        # Scan des haut-parleurs disponibles (On garde une liste "Master" intacte)
        self.all_woofers = self.scan_drivers("Woofers")
        self.all_tweeters = self.scan_drivers("Tweeters")

        self.build_ui()

    def scan_drivers(self, category):
        """Cherche les fichiers .zma pour lister les haut-parleurs existants."""
        path = os.path.join("data", category, "ZMA")
        if not os.path.exists(path):
            return ["Dossier introuvable"]
        
        drivers = [f.replace(".zma", "") for f in os.listdir(path) if f.endswith(".zma")]
        return drivers if drivers else ["Aucun trouvé"]

    def build_ui(self):
        # --- TITRE ---
        title_label = ctk.CTkLabel(self, text="FindMyCrossover", font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(20, 10))

        # --- CONTENEUR PRINCIPAL ---
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="x", padx=40)

        # == Colonne Gauche : Paramètres du projet ==
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ctk.CTkLabel(left_frame, text="Haut-Parleurs", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 5))

        # Variables de texte
        self.woofer_var = ctk.StringVar(value=self.all_woofers[0])
        self.tweeter_var = ctk.StringVar(value=self.all_tweeters[0])

        # NOUVEAU : CTkComboBox (Menu avec saisie) au lieu de CTkOptionMenu
        self.w_menu = ctk.CTkComboBox(left_frame, variable=self.woofer_var, values=self.all_woofers, command=self.update_project_name)
        self.w_menu.pack(pady=5, padx=20, fill="x")

        self.t_menu = ctk.CTkComboBox(left_frame, variable=self.tweeter_var, values=self.all_tweeters, command=self.update_project_name)
        self.t_menu.pack(pady=5, padx=20, fill="x")

        # NOUVEAU : "Triggers" de frappe. À chaque lettre tapée, on filtre la liste !
        self.woofer_var.trace_add("write", self.filter_woofers)
        self.tweeter_var.trace_add("write", self.filter_tweeters)

        ctk.CTkLabel(left_frame, text="Nom du Projet", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        self.name_entry = ctk.CTkEntry(left_frame)
        self.name_entry.pack(pady=(0, 15), padx=20, fill="x")
        self.update_project_name(None) # Initialisation auto

        # == Colonne Droite : Moteur Génétique ==
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        ctk.CTkLabel(right_frame, text="Moteur IA", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 5))

        # --- NOUVEAU WIDGET UI : FREQUENCE ---
        ctk.CTkLabel(right_frame, text="Fréq. Croisement (Hz, 0=Auto) :").pack(anchor="w", padx=20)
        self.fc_entry = ctk.CTkEntry(right_frame)
        self.fc_entry.insert(0, "0") # Valeur classique par défaut
        self.fc_entry.pack(pady=(0, 10), padx=20, fill="x")

        ctk.CTkLabel(right_frame, text="Générations :").pack(anchor="w", padx=20)
        self.gen_entry = ctk.CTkEntry(right_frame)
        self.gen_entry.insert(0, "100")
        self.gen_entry.pack(pady=(0, 10), padx=20, fill="x")

        ctk.CTkLabel(right_frame, text="Taille Population :").pack(anchor="w", padx=20)
        self.pop_entry = ctk.CTkEntry(right_frame)
        self.pop_entry.insert(0, "120")
        self.pop_entry.pack(pady=(0, 15), padx=20, fill="x")

        # --- BOUTON RUN ---
        self.run_btn = ctk.CTkButton(self, text="🚀 LANCER L'OPTIMISATION", height=40, font=ctk.CTkFont(weight="bold"), command=self.start_optimization)
        self.run_btn.pack(pady=20, padx=40, fill="x")

        # --- CONSOLE ---
        self.console = ctk.CTkTextbox(self, height=150, font=ctk.CTkFont(family="Consolas", size=12))
        self.console.pack(padx=40, fill="x", pady=(0, 20))
        self.console.insert("0.0", "Prêt. Sélectionnez ou tapez le nom de vos haut-parleurs.\n")
        self.console.configure(state="disabled")

    # --- MÉTHODES DE FILTRAGE DYNAMIQUE ---
    def filter_woofers(self, *args):
        """Met à jour les choix du menu déroulant du Woofer selon la saisie."""
        typed_text = self.woofer_var.get().lower()
        filtered_list = [w for w in self.all_woofers if typed_text in w.lower()]
        self.w_menu.configure(values=filtered_list if filtered_list else ["Aucun résultat"])
        self.update_project_name()

    def filter_tweeters(self, *args):
        """Met à jour les choix du menu déroulant du Tweeter selon la saisie."""
        typed_text = self.tweeter_var.get().lower()
        filtered_list = [t for t in self.all_tweeters if typed_text in t.lower()]
        self.t_menu.configure(values=filtered_list if filtered_list else ["Aucun résultat"])
        self.update_project_name()

    def update_project_name(self, *args):
        """Génère automatiquement le nom du projet en fonction des choix."""
        w = self.woofer_var.get()
        t = self.tweeter_var.get()
        suggested_name = f"{w}_X_{t}"
        self.name_entry.delete(0, "end")
        self.name_entry.insert(0, suggested_name)

    def write_console(self, text):
        """Écrit dans la console graphique de manière sécurisée (Thread-safe)."""
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end") # Autoscroll vers le bas
        self.console.configure(state="disabled")

    def start_optimization(self):
        """Désactive le bouton et lance le thread de calcul."""
        w = self.woofer_var.get()
        t = self.tweeter_var.get()
        name = self.name_entry.get()
        gen = self.gen_entry.get()
        pop = self.pop_entry.get()
        fc = self.fc_entry.get()
        
        # Vérification si l'utilisateur a tapé n'importe quoi ou validé un champ vide
        if w not in self.all_woofers or t not in self.all_tweeters:
            self.write_console("[-] Erreur : Nom de haut-parleur inconnu dans la base de données.")
            return

        self.run_btn.configure(state="disabled", text="⏳ OPTIMISATION EN COURS...")
        self.console.configure(state="normal")
        self.console.delete("0.0", "end")
        self.console.configure(state="disabled")
        
        # Lancement dans un Thread séparé pour ne pas figer l'interface
        threading.Thread(target=self._run_process, args=(w, t, name, gen, pop, fc), daemon=True).start()

    def _run_process(self, w, t, name, gen, pop, fc):
        """Exécute run.py en interceptant ce qu'il affiche."""
        out_dir = os.path.join("crossovers", name)
        cmd = [
            "python", "run.py",
            "--woofer", w,
            "--tweeter", t,
            "--name", name,
            "--out_dir", out_dir,
            "--gen", gen,
            "--pop", pop,
            "--fc", fc
        ]

        # --- L'ASTUCE MAGIQUE ---
        # On force le script 'run.py' à parler en UTF-8 pur, ignorant le vieux standard Windows
        custom_env = os.environ.copy()
        custom_env["PYTHONIOENCODING"] = "utf-8"
        # ------------------------

        try:
            # On passe custom_env au processus, et on garde encoding='utf-8'
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                encoding='utf-8', 
                bufsize=1,
                env=custom_env  # <-- Ajouté ici
            )

            for line in process.stdout:
                self.after(0, self.write_console, line.strip())

            process.wait()

            if process.returncode == 0:
                self.after(0, self.write_console, f"\n[+] MISSION ACCOMPLIE ! Fichiers dans : {out_dir}")
            else:
                self.after(0, self.write_console, f"\n[-] Erreur fatale de l'optimiseur (Code {process.returncode})")

        except Exception as e:
            self.after(0, self.write_console, f"[-] Erreur de lancement: {e}")

        # Réactivation du bouton
        self.after(0, lambda: self.run_btn.configure(state="normal", text="🚀 LANCER L'OPTIMISATION"))

if __name__ == "__main__":
    app = FindMyCrossoverApp()
    app.mainloop()