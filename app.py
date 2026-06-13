import os
import subprocess
import threading
import json
import customtkinter as ctk
from src.optimizer import WEIGHTS as DEFAULT_WEIGHTS

DEFAULT_CONFIG = {
    "weights": {
        "mse_sum": 1.0,
        "n_comps": 12,
        "crossover": 3.2549,
        "fc_err": 20.0,
        "impedance": 142.9232,
        "woofer_attenuation": 218.2858,
        "midrange_attenuation": 200.0,
        "thermal": 20.0,
        "tweeter_low": 48.4022,
        "woofer_high": 10.0,
        "midrange_low": 25.0,
        "midrange_high": 25.0,
        "midrange_participation": 60.0,
        "components": 0.5,
        "resistors": 0.4
    },
    "spl_settings": {
        "target_offset_db": 0.0
    },
    "optimization_range": {
        "mode": "auto", 
        "manual_min_hz": 300,
        "manual_max_hz": 15000
    }
}
WEIGHT_DEFS = {
    # Groupe : Objectif Principal
    "mse_sum":                (DEFAULT_WEIGHTS['mse_sum'],    "Poids de l'erreur SPL globale",         "float"),
    "n_comps":                (DEFAULT_WEIGHTS['n_comps'],     "Nb de composants idéal (seuil)",         "int"),
    # Groupe : Zone de Croisement
    "crossover":              (DEFAULT_WEIGHTS['crossover'],   "Multiplicateur zone de raccord",          "float"),
    "fc_err":                 (DEFAULT_WEIGHTS['fc_err'],   "Pénalité écart fréq. coupure (octaves²)", "float"),
    # Groupe : Sécurité Électrique
    "impedance":              (DEFAULT_WEIGHTS['impedance'], "Pénalité impédance < 3.2 Ω",             "float"),
    "woofer_attenuation":     (DEFAULT_WEIGHTS['woofer_attenuation'], "Pénalité atténuation woofer (Vtension)",  "float"),
    'midrange_attenuation':   (DEFAULT_WEIGHTS['midrange_attenuation'],  "Pénalité atténuation médium (Vtension)",  "float"),
    "thermal":                (DEFAULT_WEIGHTS['thermal'],   "Pénalité thermique résistances > 20 W",   "float"),
    # Groupe : Sécurité Drivers
    "tweeter_low":            (DEFAULT_WEIGHTS['tweeter_low'],  "Pénalité tweeter qui joue en grave",      "float"),
    'woofer_high':            (DEFAULT_WEIGHTS['woofer_high'],  "Pénalité woofer qui joue en aigu",        "float"),
    "midrange_low":           (DEFAULT_WEIGHTS['midrange_low'],   "Pénalité médium qui joue trop bas",       "float"),
    "midrange_high":          (DEFAULT_WEIGHTS['midrange_high'],   "Pénalité médium qui joue trop haut",      "float"),
    "midrange_participation": (DEFAULT_WEIGHTS['midrange_participation'],   "Pénalité médium inactif (bypassé)",       "float"),
    # Groupe : Complexité
    "components":             (DEFAULT_WEIGHTS['components'],   "Pénalité par composant au-delà du seuil", "float"),
    "resistors":              (DEFAULT_WEIGHTS['resistors'],   "Pénalité linéaire par résistance",        "float"),
}

WEIGHT_GROUPS = [
    ("🎯  Objectif Principal",   ["mse_sum", "n_comps"]),
    ("✂️  Zone de Croisement",   ["crossover", "fc_err"]),
    ("⚡  Sécurité Électrique",  ["impedance", "woofer_attenuation", "midrange_attenuation",  "thermal"]),
    ("🔊  Sécurité Drivers",     ["tweeter_low", "woofer_high", "midrange_low", "midrange_high", "midrange_participation"]),
    ("🧩  Complexité du Filtre", ["components", "resistors"]),
]

# --- CONFIGURATION VISUELLE ---
ctk.set_appearance_mode("Dark")  # Mode sombre pro
ctk.set_default_color_theme("blue")  # Thème d'accentuation

class FindMyCrossoverApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuration de la fenêtre
        self.title("FindMyCrossover - Studio")
        self.geometry("1200x600")
        self.resizable(False, False)

        # Scan des haut-parleurs disponibles (On garde une liste "Master" intacte)
        self.all_woofers = self.scan_drivers("Woofers")
        self.all_midranges = self.scan_drivers("Midranges")
        self.all_tweeters = self.scan_drivers("Tweeters")

        self.build_ui()

    def scan_drivers(self, category):
        """Cherche les fichiers .zma pour lister les haut-parleurs existants."""
        path = os.path.join("data", category, "ZMA")
        if not os.path.exists(path):
            return []
        drivers = sorted(f.replace(".zma", "") for f in os.listdir(path) if f.endswith(".zma"))
        return drivers
    
    def write_console(self, text):
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")


    def build_ui(self):
        # --- TITRE ---
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(10, 5)) # Espacement réduit

        ctk.CTkLabel(
            header, text="FindMyCrossover",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(side="left")

        ctk.CTkButton(
            header, text="⚙️", width=36, height=36,
            font=ctk.CTkFont(size=16),
            fg_color="transparent", hover_color="#333333",
            command=self.open_settings
        ).pack(side="right")

        # --- CONTENEUR PRINCIPAL ---
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20)

        # == Colonne Gauche : Paramètres du projet ==
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ctk.CTkLabel(left_frame, text="Haut-Parleurs", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 2))

        # Variables de texte
        self.woofer_var = ctk.StringVar(value=self.all_woofers[0] if self.all_woofers else "")
        self.tweeter_var = ctk.StringVar(value=self.all_tweeters[0] if self.all_tweeters else "")
        self.w_qty_var = ctk.StringVar(value="1")

        # --- WOOFER ---
        ctk.CTkLabel(left_frame, text="Woofer", text_color="gray").pack(anchor="w", padx=20)
        w_row = ctk.CTkFrame(left_frame, fg_color="transparent")
        w_row.pack(pady=(0, 2), padx=20, fill="x")
        
        self.w_menu = ctk.CTkComboBox(w_row, variable=self.woofer_var, values=self.all_woofers, command=self.update_project_name)
        self.w_menu.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        # Dans build_ui, remplacez la ligne du w_qty_menu par :
        self.w_qty_menu = ctk.CTkComboBox(w_row, variable=self.w_qty_var, values=["1", "2"], width=60, command=self._on_qty_change)
        self.w_qty_menu.pack(side="right")

        # --- MEDIUM ---
        ctk.CTkLabel(left_frame, text="Médium  (optionnel — 3 voies)", text_color="gray").pack(anchor="w", padx=14)
        self.midrange_var = ctk.StringVar(value="(Aucun)")
        self.m_menu = ctk.CTkComboBox(
            left_frame, variable=self.midrange_var,
            values=["(Aucun)"] + (self.all_midranges or []),
            command=self._on_midrange_change
        )
        self.m_menu.pack(pady=(0, 2), padx=14, fill="x")
        self.midrange_var.trace_add("write", self.filter_midranges)
        
        # --- TWEETER ---
        ctk.CTkLabel(left_frame, text="Tweeter", text_color="gray").pack(anchor="w", padx=20)
        self.t_menu = ctk.CTkComboBox(left_frame, variable=self.tweeter_var, values=self.all_tweeters, command=self.update_project_name)
        self.t_menu.pack(pady=(0, 2), padx=20, fill="x")

        self.woofer_var.trace_add("write", self.filter_woofers)
        self.tweeter_var.trace_add("write", self.filter_tweeters)

        # ==========================================
        # SECTION POSITIONS PHYSIQUES (ONGLETS)
        # ==========================================
        ctk.CTkLabel(left_frame, text="Positions des Haut-Parleurs (m)", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 0))
        
        self.pos_tabs = ctk.CTkTabview(left_frame, height=140)
        self.pos_tabs.pack(padx=20, pady=(0, 5), fill="x")

        # Configuration des onglets permanents
        self.pos_tabs.add("W1")
        self.pos_tabs.add("Tweeter")
        
        # On crée les champs pour chaque onglet
        self.w1_entries = self._create_pos_tab_content("W1", "0.0", "-0.150", "0.0")
        self.t_entries  = self._create_pos_tab_content("Tweeter", "0.0", "0.0", "0.0")
        
        # Onglets conditionnels (seront gérés par les fonctions de changement)
        self.w2_entries = None
        self.mid_entries = None

        # --- NOM DU PROJET ---
        # CORRECTION : On assigne le label à self.name_label pour s'en servir de point de repère
        self.name_label = ctk.CTkLabel(left_frame, text="Nom du Projet", font=ctk.CTkFont(weight="bold"))
        self.name_label.pack(pady=(10, 2))
        self.name_entry = ctk.CTkEntry(left_frame)
        self.name_entry.pack(pady=(0, 10), padx=20, fill="x")
        self.update_project_name(None) 

        # == Colonne Centrale : Moteur Génétique ==
        center_frame = ctk.CTkFrame(main_frame)
        center_frame.pack(side="left", fill="both", padx=(0, 10))

        ctk.CTkLabel(center_frame, text="Moteur IA", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 2))

        self.auto_fc_var = ctk.BooleanVar(value=True) 
        self.auto_fc_checkbox = ctk.CTkCheckBox(
            center_frame, text="Fréq. coupure automatiques",
            variable=self.auto_fc_var,
            command=self._toggle_fc_state,
            text_color="#00A86B" 
        )
        self.auto_fc_checkbox.pack(anchor="w", padx=14, pady=(0, 5))

        self.fc1_label = ctk.CTkLabel(center_frame, text="Fréq. coupure (Hz) :", text_color="gray")
        self.fc1_label.pack(anchor="w", padx=14)
        self.fc_entry = ctk.CTkEntry(center_frame)
        self.fc_entry.insert(0, "2000") 
        self.fc_entry.pack(pady=(0, 4), padx=14, fill="x")
        self.fc_entry.configure(state="disabled") 

        self.fc2_frame = ctk.CTkFrame(center_frame, fg_color="transparent")
        ctk.CTkLabel(self.fc2_frame, text="Fréq. coupure Médium-Tweeter (Hz) :", text_color="gray").pack(anchor="w")
        self.fc2_entry = ctk.CTkEntry(self.fc2_frame)
        self.fc2_entry.insert(0, "4500") 
        self.fc2_entry.pack(fill="x")
        self.fc2_entry.configure(state="disabled") 

        self.gen_label = ctk.CTkLabel(center_frame, text="Générations :", text_color="gray")
        self.gen_label.pack(anchor="w", padx=14)
        self.gen_entry = ctk.CTkEntry(center_frame)
        self.gen_entry.insert(0, "50")
        self.gen_entry.pack(pady=(0, 4), padx=14, fill="x")

        ctk.CTkLabel(center_frame, text="Taille Population :", text_color="gray").pack(anchor="w", padx=14)
        self.pop_entry = ctk.CTkEntry(center_frame)
        self.pop_entry.insert(0, "500")
        self.pop_entry.pack(pady=(0, 10), padx=14, fill="x")

        # == Colonne Droite : Console ==
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="left", fill="both", expand=True)
        
        ctk.CTkLabel(right_frame, text="Console de Sortie", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 2))
        self.console = ctk.CTkTextbox(right_frame, font=ctk.CTkFont(family="Consolas", size=12))
        self.console.pack(padx=10, pady=(0, 10), fill="both", expand=True)
        self.console.configure(state="disabled")
        
        # --- BOUTON RUN ---
        self.run_btn = ctk.CTkButton(self, text="🚀 LANCER L'OPTIMISATION", height=40, font=ctk.CTkFont(weight="bold"), command=self.start_optimization)
        self.run_btn.pack(pady=(5, 15), padx=40, fill="x")

    def _create_pos_tab_content(self, tab_name, dx, dy, dz):
        """Crée la grille X, Y, Z à l'intérieur d'un onglet spécifique."""
        parent = self.pos_tabs.tab(tab_name)
        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(expand=True, fill="both", padx=5, pady=5)
        
        entries = {}
        for i, (axis, default) in enumerate([("X", dx), ("Y", dy), ("Z", dz)]):
            col = ctk.CTkFrame(grid, fg_color="transparent")
            col.pack(side="left", expand=True, fill="x", padx=2)
            ctk.CTkLabel(col, text=axis, text_color="gray", font=ctk.CTkFont(size=10)).pack()
            entry = ctk.CTkEntry(col, height=25)
            entry.insert(0, default)
            entry.pack(fill="x")
            entries[axis.lower()] = entry
            
        return entries
    
    def _on_qty_change(self, *_):
        qty = self.w_qty_var.get()
        if qty == "2":
            if "W2" not in self.pos_tabs._tab_dict:
                self.pos_tabs.add("W2")
                self.w2_entries = self._create_pos_tab_content("W2", "0.0", "0.150", "0.0")
        else:
            if "W2" in self.pos_tabs._tab_dict:
                self.pos_tabs.delete("W2")
                self.w2_entries = None
        self.update_project_name()

    # --- MÉTHODES DE FILTRAGE DYNAMIQUE ---
    def filter_woofers(self, *args):
        """Met à jour les choix du menu déroulant du Woofer selon la saisie."""
        typed_text = self.woofer_var.get().lower()
        filtered_list = [w for w in self.all_woofers if typed_text in w.lower()]
        self.w_menu.configure(values=filtered_list if filtered_list else ["Aucun résultat"])
        self.update_project_name()
        
    def filter_midranges(self, *_):
        typed = self.midrange_var.get().lower()
        if typed == "" or typed == "(aucun)":
            filtered = ["(Aucun)"] + self.all_midranges
        else:
            filtered = [m for m in self.all_midranges if typed in m.lower()]
        self.m_menu.configure(values=filtered or ["(aucun résultat)"])
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
        m = self.midrange_var.get().strip()
        t = self.tweeter_var.get()
        qty = self.w_qty_var.get()
        
        if m == "(Aucun)":
            m = ""
        
        # Ajout du préfixe "2x_" si l'utilisateur a choisi 2 woofers
        prefix = f"2x_" if str(qty) == "2" else ""
        parts = [p for p in [w, m, t] if p]
        name = "_X_".join(parts)
        name = prefix + name
        self.name_entry.delete(0, "end")
        self.name_entry.insert(0, name)

    def write_console(self, text):
        """Écrit dans la console graphique de manière sécurisée (Thread-safe)."""
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end") # Autoscroll vers le bas
        self.console.configure(state="disabled")
        
    def _create_geo_field(self, parent, label, default):
        """Utilitaire pour créer les petits champs X, Y, Z"""
        col = ctk.CTkFrame(parent, fg_color="transparent")
        col.pack(side="left", expand=True, fill="x", padx=2)
        ctk.CTkLabel(col, text=label, text_color="gray", font=ctk.CTkFont(size=10)).pack()
        entry = ctk.CTkEntry(col, height=25)
        entry.insert(0, default)
        entry.pack(fill="x")
        return entry
        
    def _on_midrange_change(self, *_):
        val = self.midrange_var.get().strip()
        if val and val != "(Aucun)":
            # Gestion des fréquences (code existant)
            self.fc1_label.configure(text="Fréq. coupure Woofer-Médium (Hz) :")
            self.fc2_frame.pack(padx=14, fill="x", pady=(0, 4), before=self.gen_label)
            
            # Gestion de l'onglet
            if "Mid" not in self.pos_tabs._tab_dict:
                self.pos_tabs.add("Mid")
                self.mid_entries = self._create_pos_tab_content("Mid", "0.0", "-0.050", "0.0")
        else:
            self.fc1_label.configure(text="Fréq. coupure (Hz) :")
            self.fc2_frame.pack_forget()
            if "Mid" in self.pos_tabs._tab_dict:
                self.pos_tabs.delete("Mid")
                self.mid_entries = None
        self.update_project_name()

    def _toggle_fc_state(self):
        """Active ou grise les champs de fréquence selon l'état de la Checkbox."""
        if self.auto_fc_var.get():
            self.fc_entry.configure(state="disabled")
            self.fc2_entry.configure(state="disabled")
        else:
            self.fc_entry.configure(state="normal")
            self.fc2_entry.configure(state="normal")
        
    def open_settings(self):
        win = ctk.CTkToplevel(self)
        win.title("⚙️ Configuration Globale (config.json)")
        win.geometry("450x650")
        win.resizable(False, True)
        win.grab_set()

        ctk.CTkLabel(
            win, text="Paramètres de l'Optimiseur",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 5))

        scroll = ctk.CTkScrollableFrame(win)
        scroll.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        # 1. Chargement ou Création du fichier JSON
        config_path = "config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            self.write_console("[!] Fichier config.json introuvable. Création avec les paramètres par défaut.")
            config_data = DEFAULT_CONFIG.copy()

        # 2. Génération dynamique de l'interface
        local_entries = {}
        row = 0

        for section, params in config_data.items():
            # Titre de section stylisé
            ctk.CTkLabel(
                scroll, text=section.replace('_', ' ').upper(),
                font=ctk.CTkFont(size=13, weight="bold"), text_color="#00A86B"
            ).grid(row=row, column=0, columnspan=2, pady=(15, 5), sticky="w")
            row += 1
            
            if isinstance(params, dict):
                local_entries[section] = {}
                for key, value in params.items():
                    # Nom du paramètre
                    ctk.CTkLabel(
                        scroll, text=key, font=ctk.CTkFont(size=12)
                    ).grid(row=row, column=0, sticky="w", padx=(10, 20), pady=2)
                    
                    # Champ de saisie
                    entry = ctk.CTkEntry(scroll, width=120)
                    entry.insert(0, str(value))
                    entry.grid(row=row, column=1, sticky="e", pady=2)
                    
                    local_entries[section][key] = entry
                    row += 1

        # 3. Fonction de Sauvegarde
        def save_and_close():
            for section, params in local_entries.items():
                for key, entry in params.items():
                    val_str = entry.get()
                    # Typage intelligent
                    if val_str.lower() in ['true', 'false']:
                        config_data[section][key] = val_str.lower() == 'true'
                    elif val_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                        config_data[section][key] = float(val_str) if "." in val_str else int(val_str)
                    else:
                        config_data[section][key] = val_str

            # Écriture physique sur le disque
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
                
            self.write_console("[+] Configuration mise à jour et sauvegardée dans config.json.")
            win.destroy()

        # Bouton d'action
        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=(0, 15))
        
        ctk.CTkButton(
            btn_frame, text="💾 Sauvegarder & Fermer",
            font=ctk.CTkFont(weight="bold"),
            command=save_and_close
        ).pack(fill="x")

        def reset_defaults():
            """Remet les valeurs par défaut dans les champs affichés."""
            for k, e in local_entries.items():
                e.delete(0, "end")
                e.insert(0, str(WEIGHT_DEFS[k][0]))

        def apply_and_close():
            """Sauvegarde les valeurs saisies dans les variables principales puis ferme."""
            for k, e in local_entries.items():
                self.weight_vars[k].set(e.get())
            win.destroy()

        ctk.CTkButton(
            btn_frame, text="↩ Réinitialiser",
            fg_color="#555555", hover_color="#666666",
            command=reset_defaults
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_frame, text="✔ Appliquer & Fermer",
            command=apply_and_close
        ).pack(side="right")
        
        # Si l'utilisateur ferme violemment avec la croix rouge, on sauvegarde quand même !
        win.protocol("WM_DELETE_WINDOW", apply_and_close)

    def start_optimization(self):
        """Désactive le bouton et lance le thread de calcul."""
        w = self.woofer_var.get()
        w_qty = self.w_qty_var.get() # NOUVEAU : Récupération de la quantité
        m  = self.midrange_var.get().strip()
        if m == "(Aucun)":
            m = ""
        t = self.tweeter_var.get()
        name = self.name_entry.get()
        gen = self.gen_entry.get()
        pop = self.pop_entry.get()
            
        # --- MODIFICATION ICI ---
        if self.auto_fc_var.get():
            fc1 = "None"
            fc2 = "None"
        else:
            fc1 = self.fc_entry.get().strip()
            fc2 = self.fc2_entry.get().strip() if m else ""
        # ------------------------
            
        # Vérification si l'utilisateur a tapé n'importe quoi ou validé un champ vide
        if w not in self.all_woofers or t not in self.all_tweeters:
            self.write_console("[-] Erreur : Nom de haut-parleur inconnu dans la base de données.")
            return

        self.run_btn.configure(state="disabled", text="⏳ OPTIMISATION EN COURS...")
        self.console.configure(state="normal")
        self.console.delete("0.0", "end")
        self.console.configure(state="disabled")
        
        # Lancement dans un Thread séparé pour ne pas figer l'interface
        threading.Thread(target=self._run_process, args=(w, w_qty,m , t, 
                                                         name, gen, pop, 
                                                         fc1, fc2, 
                                                         ), 
                         daemon=True).start()

    def _run_process(self, w, w_qty, m, t, name, gen, pop, fc1, fc2):
        """Exécute run.py en interceptant ce qu'il affiche."""
        out_dir = os.path.join("crossovers", name)
        
        # --- MODIFICATION ICI : Omission intelligente de l'argument ---
        fc_args = []
        if fc1 != "None":
            fc_args = ["--fc", fc1]
            if m and fc2 and fc2 != "None":
                fc_args += [fc2]
        # --------------------------------------------------------------
        
        pos_args = [
            "--wx", self.w1_entries['x'].get(), "--wy", self.w1_entries['y'].get(), "--wz", self.w1_entries['z'].get(),
            "--tx", self.t_entries['x'].get(), "--ty", self.t_entries['y'].get(), "--tz", self.t_entries['z'].get()
        ]
        if self.w2_entries:
            pos_args += [
                "--wx2", self.w2_entries['x'].get(), 
                "--wy2", self.w2_entries['y'].get(), 
                "--wz2", self.w2_entries['z'].get()
            ]
            
        # Ajout conditionnel du Médium
        if self.mid_entries:
            pos_args += [
                "--mx", self.mid_entries['x'].get(), 
                "--my", self.mid_entries['y'].get(), 
                "--mz", self.mid_entries['z'].get()
            ]
        
        cmd = [
            "python", "run.py",
            "--woofer", w,
            "--woofer_count", str(w_qty), # NOUVEAU : Envoi de la quantité au script run.py
            "--tweeter", t,
            "--name", name,
            "--out_dir", out_dir,
            "--gen", gen,
            "--pop", pop,
            *pos_args,
            *fc_args, 
        ]
        if m:
            cmd += ["--midrange", m]

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