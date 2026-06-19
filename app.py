import os
import subprocess
import threading
import json
import customtkinter as ctk

# --- PARAMÈTRES PAR DÉFAUT DE L'IA ---
DEFAULT_CONFIG = {
    "weights": {
        "mse_sum": 1.0,
        "crossover": 3.25,
        "overshoot": 10.0,
        "impedance": 142.9,
        "high_low_leak": 48.0,
        "low_high_leak": 10.0,
        "components": 0.15,
        "n_comps_free": 6,
        "resistors": 0.3,
        "thermal": 20.0
    },
    "spl_settings": {
        "target_offset_db": 0.0
    },
    "complexity": {
        "max_comp_per_branch": 6
    }
}

# --- CONFIGURATION VISUELLE ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class FindMyCrossoverIA(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FindMyCrossover - IA Designer")
        self.geometry("1000x550")
        self.resizable(False, False)

        self.all_woofers = self.scan_drivers("Woofers")
        self.all_tweeters = self.scan_drivers("Tweeters")

        self.build_ui()

    def scan_drivers(self, category):
        path = os.path.join("data", category)
        if not os.path.exists(path):
            return []
        drivers = set()
        for root_dir, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(('.frd', '.txt')):
                    base = f.rsplit('.', 1)[0]
                    for suf in ["_0deg", "-0deg", "_0", " 0deg"]:
                        if base.lower().endswith(suf.lower()):
                            base = base[:-len(suf)]
                            break
                    drivers.add(base.strip())
        return sorted(list(drivers))
    
    def write_console(self, text):
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def build_ui(self):
        # --- TITRE ET ENGRENAGE ---
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(10, 5))

        ctk.CTkLabel(
            header, text="Générateur IA (Inférence)",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(side="left")

        # Bouton Paramètres ⚙️
        ctk.CTkButton(
            header, text="⚙️", width=36, height=36,
            font=ctk.CTkFont(size=16),
            fg_color="transparent", hover_color="#333333",
            command=self.open_settings
        ).pack(side="right")

        # --- CONTENEUR PRINCIPAL ---
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20)

        # == Colonne Gauche : Sélection des HP ==
        left_frame = ctk.CTkFrame(main_frame, width=300)
        left_frame.pack(side="left", fill="both", padx=(0, 10))

        ctk.CTkLabel(left_frame, text="1. Composants Acoustiques", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 10))

        self.woofer_var = ctk.StringVar(value=self.all_woofers[0] if self.all_woofers else "Aucun woofer trouvé")
        self.tweeter_var = ctk.StringVar(value=self.all_tweeters[0] if self.all_tweeters else "Aucun tweeter trouvé")

        ctk.CTkLabel(left_frame, text="Woofer (Voie Basse)", text_color="gray").pack(anchor="w", padx=20)
        self.w_menu = ctk.CTkComboBox(left_frame, variable=self.woofer_var, values=self.all_woofers or ["Vide"])
        self.w_menu.pack(pady=(0, 15), padx=20, fill="x")

        ctk.CTkLabel(left_frame, text="Tweeter (Voie Haute)", text_color="gray").pack(anchor="w", padx=20)
        self.t_menu = ctk.CTkComboBox(left_frame, variable=self.tweeter_var, values=self.all_tweeters or ["Vide"])
        self.t_menu.pack(pady=(0, 20), padx=20, fill="x")

        self.woofer_var.trace_add("write", lambda *args: self.filter_menu(self.woofer_var, self.w_menu, self.all_woofers))
        self.tweeter_var.trace_add("write", lambda *args: self.filter_menu(self.tweeter_var, self.t_menu, self.all_tweeters))

        # == Colonne Centrale : Puissance du GPU ==
        center_frame = ctk.CTkFrame(main_frame, width=250)
        center_frame.pack(side="left", fill="both", padx=(0, 10))

        ctk.CTkLabel(center_frame, text="2. Puissance d'Optimisation", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 10))

        ctk.CTkLabel(center_frame, text="Restarts Simultanés (GPU) :", text_color="gray").pack(anchor="w", padx=20)
        self.restarts_entry = ctk.CTkEntry(center_frame)
        self.restarts_entry.insert(0, "1024")
        self.restarts_entry.pack(pady=(0, 15), padx=20, fill="x")

        ctk.CTkLabel(center_frame, text="Itérations (Steps) :", text_color="gray").pack(anchor="w", padx=20)
        self.steps_entry = ctk.CTkEntry(center_frame)
        self.steps_entry.insert(0, "400")
        self.steps_entry.pack(pady=(0, 20), padx=20, fill="x")

        presets_frame = ctk.CTkFrame(center_frame, fg_color="transparent")
        presets_frame.pack(fill="x", padx=20)
        
        ctk.CTkButton(presets_frame, text="Rapide", fg_color="#555", width=80,
                      command=lambda: self.set_power("128", "150")).pack(side="left", expand=True, padx=(0,2))
        ctk.CTkButton(presets_frame, text="Précis", fg_color="#005500", width=80,
                      command=lambda: self.set_power("1024", "400")).pack(side="right", expand=True, padx=(2,0))

        # == Colonne Droite : Console ==
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="left", fill="both", expand=True)
        
        ctk.CTkLabel(right_frame, text="3. Console de Sortie", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        self.console = ctk.CTkTextbox(right_frame, font=ctk.CTkFont(family="Consolas", size=12))
        self.console.pack(padx=15, pady=(0, 15), fill="both", expand=True)
        self.console.configure(state="disabled")
        
        # --- BOUTON RUN ---
        self.run_btn = ctk.CTkButton(self, text="⚡ CONCEVOIR LE FILTRE", height=45, 
                                     font=ctk.CTkFont(size=14, weight="bold"), 
                                     fg_color="#0055AA", hover_color="#004488",
                                     command=self.start_design)
        self.run_btn.pack(pady=(10, 15), padx=40, fill="x")

    # ==========================================
    # GESTION DES PARAMÈTRES (LOSS)
    # ==========================================
    def open_settings(self):
        win = ctk.CTkToplevel(self)
        win.title("⚙️ Configuration Globale (config.json)")
        win.geometry("450x650")
        win.resizable(False, True)
        win.grab_set()

        ctk.CTkLabel(
            win, text="Paramètres de Loss (IA)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 5))

        scroll = ctk.CTkScrollableFrame(win)
        scroll.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        # Lecture de config.json
        config_path = "config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            self.write_console("[!] config.json introuvable. Création avec les paramètres par défaut.")
            config_data = DEFAULT_CONFIG.copy()

        local_entries = {}
        row = 0

        # Génération dynamique des champs
        for section, params in config_data.items():
            ctk.CTkLabel(
                scroll, text=section.replace('_', ' ').upper(),
                font=ctk.CTkFont(size=13, weight="bold"), text_color="#00A86B"
            ).grid(row=row, column=0, columnspan=2, pady=(15, 5), sticky="w")
            row += 1
            
            if isinstance(params, dict):
                local_entries[section] = {}
                for key, value in params.items():
                    ctk.CTkLabel(scroll, text=key, font=ctk.CTkFont(size=12)).grid(row=row, column=0, sticky="w", padx=(10, 20), pady=2)
                    entry = ctk.CTkEntry(scroll, width=120)
                    entry.insert(0, str(value))
                    entry.grid(row=row, column=1, sticky="e", pady=2)
                    local_entries[section][key] = entry
                    row += 1

        def save_and_close():
            for section, params in local_entries.items():
                for key, entry in params.items():
                    val_str = entry.get()
                    if val_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                        config_data[section][key] = float(val_str) if "." in val_str else int(val_str)
                    else:
                        config_data[section][key] = val_str

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
                
            self.write_console("[+] Configuration IA mise à jour (config.json).")
            win.destroy()

        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=(0, 15))
        
        ctk.CTkButton(btn_frame, text="💾 Sauvegarder & Fermer", font=ctk.CTkFont(weight="bold"), command=save_and_close).pack(fill="x")

    def filter_menu(self, string_var, menu_widget, full_list):
        typed_text = string_var.get().lower()
        filtered = [item for item in full_list if typed_text in item.lower()]
        menu_widget.configure(values=filtered if filtered else ["Aucun résultat"])

    def set_power(self, restarts, steps):
        self.restarts_entry.delete(0, "end")
        self.restarts_entry.insert(0, restarts)
        self.steps_entry.delete(0, "end")
        self.steps_entry.insert(0, steps)

    def start_design(self):
        w = self.woofer_var.get()
        t = self.tweeter_var.get()
        restarts = self.restarts_entry.get().strip()
        steps = self.steps_entry.get().strip()

        if w not in self.all_woofers or t not in self.all_tweeters:
            self.write_console("[-] Erreur : Veuillez sélectionner des haut-parleurs valides de la liste.")
            return

        self.run_btn.configure(state="disabled", text="⏳ RÉFLEXION DE L'IA EN COURS...")
        self.console.configure(state="normal")
        self.console.delete("0.0", "end")
        self.console.configure(state="disabled")
        
        threading.Thread(target=self._run_process, args=(w, t, restarts, steps), daemon=True).start()

    def _run_process(self, w, t, restarts, steps):
        cmd = ["python", "-m", "design", "--low", w, "--high", t, "--restarts", restarts, "--steps", steps]
        custom_env = os.environ.copy()
        custom_env["PYTHONIOENCODING"] = "utf-8"

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1, env=custom_env)
            for line in process.stdout:
                self.after(0, self.write_console, line.strip())
            process.wait()

            if process.returncode == 0:
                self.after(0, self.write_console, f"\n[+] TERMINE ! Consultez 'design_response.png'.")
            else:
                self.after(0, self.write_console, f"\n[-] Erreur fatale (Code {process.returncode})")

        except Exception as e:
            self.after(0, self.write_console, f"[-] Erreur de lancement: {e}")

        self.after(0, lambda: self.run_btn.configure(state="normal", text="⚡ CONCEVOIR LE FILTRE"))

if __name__ == "__main__":
    app = FindMyCrossoverIA()
    app.mainloop()