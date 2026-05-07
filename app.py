import os
import json
import subprocess
import threading
import customtkinter as ctk

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ── Définition des poids avec valeurs par défaut et descriptions ──────────────
# Structure : clé → (valeur_défaut, description, type)
WEIGHT_DEFS = {
    # Groupe : Objectif Principal
    "mse_sum":                (1.0,    "Poids de l'erreur SPL globale",         "float"),
    "n_comps":                (10,     "Nb de composants idéal (seuil)",         "int"),
    # Groupe : Zone de Croisement
    "crossover":              (3.25,   "Multiplicateur zone de raccord",          "float"),
    "fc_err":                 (20.0,   "Pénalité écart fréq. coupure (octaves²)", "float"),
    # Groupe : Sécurité Électrique
    "impedance":              (142.92, "Pénalité impédance < 3.2 Ω",             "float"),
    "woofer_attenuation":     (218.29, "Pénalité atténuation woofer (Vtension)",  "float"),
    "thermal":                (0.12,   "Pénalité thermique résistances > 20 W",   "float"),
    # Groupe : Sécurité Drivers
    "tweeter_low":            (48.40,  "Pénalité tweeter qui joue en grave",      "float"),
    "midrange_low":           (25.0,   "Pénalité médium qui joue trop bas",       "float"),
    "midrange_high":          (25.0,   "Pénalité médium qui joue trop haut",      "float"),
    "midrange_participation": (80.0,   "Pénalité médium inactif (bypassé)",       "float"),
    # Groupe : Complexité
    "components":             (0.68,   "Pénalité par composant au-delà du seuil", "float"),
    "resistors":              (0.65,   "Pénalité linéaire par résistance",        "float"),
}

WEIGHT_GROUPS = [
    ("🎯  Objectif Principal",   ["mse_sum", "n_comps"]),
    ("✂️  Zone de Croisement",   ["crossover", "fc_err"]),
    ("⚡  Sécurité Électrique",  ["impedance", "woofer_attenuation", "thermal"]),
    ("🔊  Sécurité Drivers",     ["tweeter_low", "midrange_low", "midrange_high", "midrange_participation"]),
    ("🧩  Complexité du Filtre", ["components", "resistors"]),
]


class FindMyCrossoverApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FindMyCrossover - Studio")
        self.geometry("720x620")
        self.resizable(False, False)

        # Scan des drivers disponibles
        self.all_woofers   = self.scan_drivers("Woofers")
        self.all_midranges = self.scan_drivers("Midranges")
        self.all_tweeters  = self.scan_drivers("Tweeters")

        # Variables de poids (StringVar par clé)
        self.weight_vars = {
            key: ctk.StringVar(value=str(WEIGHT_DEFS[key][0]))
            for key in WEIGHT_DEFS
        }

        self.build_ui()

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def scan_drivers(self, category):
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

    def get_current_weights_json(self):
        """Construit le JSON des poids depuis les StringVar."""
        out = {}
        for key, var in self.weight_vars.items():
            raw = var.get().strip()
            try:
                typ = WEIGHT_DEFS[key][2]
                out[key] = int(raw) if typ == "int" else float(raw)
            except ValueError:
                out[key] = WEIGHT_DEFS[key][0]  # fallback défaut
        return json.dumps(out)

    # ── Construction de l'UI principale ──────────────────────────────────────

    def build_ui(self):
        # ── En-tête ─────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(15, 5))

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

        # ── Corps principal ──────────────────────────────────────────────────
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="x", padx=20)

        # == Colonne gauche : Haut-parleurs ==================================
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        ctk.CTkLabel(left_frame, text="Haut-Parleurs",
                     font=ctk.CTkFont(weight="bold")).pack(pady=(10, 4))

        # Woofer
        ctk.CTkLabel(left_frame, text="Woofer", text_color="gray").pack(anchor="w", padx=14)
        self.woofer_var = ctk.StringVar(value=self.all_woofers[0] if self.all_woofers else "")
        self.w_menu = ctk.CTkComboBox(
            left_frame, variable=self.woofer_var,
            values=self.all_woofers or ["(vide)"],
            command=self.update_project_name
        )
        self.w_menu.pack(pady=(0, 6), padx=14, fill="x")
        self.woofer_var.trace_add("write", self.filter_woofers)

        # Médium (optionnel)
        ctk.CTkLabel(left_frame, text="Médium  (optionnel — 3 voies)",
                     text_color="gray").pack(anchor="w", padx=14)
        
        # --- MODIFICATION ICI : Valeur par défaut claire ---
        self.midrange_var = ctk.StringVar(value="(Aucun)")
        self.m_menu = ctk.CTkComboBox(
            left_frame, variable=self.midrange_var,
            values=["(Aucun)"] + (self.all_midranges or []),
            command=self._on_midrange_change
        )
        self.m_menu.pack(pady=(0, 6), padx=14, fill="x")
        self.midrange_var.trace_add("write", self.filter_midranges)

        # Tweeter
        ctk.CTkLabel(left_frame, text="Tweeter", text_color="gray").pack(anchor="w", padx=14)
        self.tweeter_var = ctk.StringVar(value=self.all_tweeters[0] if self.all_tweeters else "")
        self.t_menu = ctk.CTkComboBox(
            left_frame, variable=self.tweeter_var,
            values=self.all_tweeters or ["(vide)"],
            command=self.update_project_name
        )
        self.t_menu.pack(pady=(0, 6), padx=14, fill="x")
        self.tweeter_var.trace_add("write", self.filter_tweeters)

        # Nom du projet
        ctk.CTkLabel(left_frame, text="Nom du Projet",
                     font=ctk.CTkFont(weight="bold")).pack(pady=(10, 4))
        self.name_entry = ctk.CTkEntry(left_frame)
        self.name_entry.pack(pady=(0, 14), padx=14, fill="x")
        self.update_project_name()

        # == Colonne droite : Moteur IA ======================================
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(8, 0))

        ctk.CTkLabel(right_frame, text="Moteur IA",
                     font=ctk.CTkFont(weight="bold")).pack(pady=(10, 4))

        # --- NOUVEAU : Checkbox Auto FC ---
        self.auto_fc_var = ctk.BooleanVar(value=True) # Coché par défaut
        self.auto_fc_checkbox = ctk.CTkCheckBox(
            right_frame, text="Fréquences de coupure automatiques",
            variable=self.auto_fc_var,
            command=self._toggle_fc_state,
            text_color="#00A86B" # Petit style vert pour montrer que c'est géré par l'IA
        )
        self.auto_fc_checkbox.pack(anchor="w", padx=14, pady=(0, 12))

        # Fréquence de coupure 1
        self.fc1_label = ctk.CTkLabel(right_frame, text="Fréq. coupure (Hz) :", text_color="gray")
        self.fc1_label.pack(anchor="w", padx=14)
        self.fc_entry = ctk.CTkEntry(right_frame)
        self.fc_entry.insert(0, "2000") # Valeur indicative
        self.fc_entry.pack(pady=(0, 8), padx=14, fill="x")
        self.fc_entry.configure(state="disabled") # Grisé par défaut

        # Fréquence de coupure 2 (3-voies, cachée par défaut)
        self.fc2_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        ctk.CTkLabel(self.fc2_frame, text="Fréq. coupure Médium-Tweeter (Hz) :", text_color="gray").pack(anchor="w")
        self.fc2_entry = ctk.CTkEntry(self.fc2_frame)
        self.fc2_entry.insert(0, "4500") # Valeur indicative
        self.fc2_entry.pack(fill="x")
        self.fc2_entry.configure(state="disabled") # Grisé par défaut
        
        # Générations
        # --- MODIFICATION ICI : On assigne le label à self.gen_label ---
        self.gen_label = ctk.CTkLabel(right_frame, text="Générations :", text_color="gray")
        self.gen_label.pack(anchor="w", padx=14)
        
        self.gen_entry = ctk.CTkEntry(right_frame)
        self.gen_entry.insert(0, "50")
        self.gen_entry.pack(pady=(0, 8), padx=14, fill="x")

        # Taille population
        ctk.CTkLabel(right_frame, text="Taille Population :", text_color="gray").pack(anchor="w", padx=14)
        self.pop_entry = ctk.CTkEntry(right_frame)
        self.pop_entry.insert(0, "500")
        self.pop_entry.pack(pady=(0, 14), padx=14, fill="x")

        # ── Bouton Run ───────────────────────────────────────────────────────
        self.run_btn = ctk.CTkButton(
            self, text="🚀  LANCER L'OPTIMISATION",
            height=42, font=ctk.CTkFont(size=14, weight="bold"),
            command=self.start_optimization
        )
        self.run_btn.pack(pady=14, padx=20, fill="x")

        # ── Console ──────────────────────────────────────────────────────────
        self.console = ctk.CTkTextbox(
            self, height=150,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.console.pack(padx=20, fill="x", pady=(0, 16))
        self.console.insert("0.0", "Prêt. Sélectionnez vos haut-parleurs.\n")
        self.console.configure(state="disabled")

    # ── Filtrage dynamique ────────────────────────────────────────────────────

    def filter_woofers(self, *_):
        typed = self.woofer_var.get().lower()
        filtered = [w for w in self.all_woofers if typed in w.lower()]
        self.w_menu.configure(values=filtered or ["(aucun résultat)"])
        self.update_project_name()

    def filter_midranges(self, *_):
        typed = self.midrange_var.get().lower()
        if typed == "" or typed == "(aucun)":
            filtered = ["(Aucun)"] + self.all_midranges
        else:
            filtered = [m for m in self.all_midranges if typed in m.lower()]
        self.m_menu.configure(values=filtered or ["(aucun résultat)"])
        self.update_project_name()

    def filter_tweeters(self, *_):
        typed = self.tweeter_var.get().lower()
        filtered = [t for t in self.all_tweeters if typed in t.lower()]
        self.t_menu.configure(values=filtered or ["(aucun résultat)"])
        self.update_project_name()

    def _on_midrange_change(self, *_):
        """Affiche/masque le champ fc2 et met à jour le nom."""
        val = self.midrange_var.get().strip()
        
        if val and val != "(Aucun)":
            self.fc1_label.configure(text="Fréq. coupure Woofer-Médium (Hz) :")
            self.fc2_frame.pack(padx=14, fill="x", pady=(0, 8), before=self.gen_label)
        else:
            # --- MODIFICATION : Suppression de "(0=Auto)" ---
            self.fc1_label.configure(text="Fréq. coupure (Hz) :")
            self.fc2_frame.pack_forget()
            
        self.update_project_name()

    def _toggle_fc_state(self):
        """Active ou grise les champs de fréquence selon l'état de la Checkbox."""
        if self.auto_fc_var.get():
            self.fc_entry.configure(state="disabled")
            self.fc2_entry.configure(state="disabled")
        else:
            self.fc_entry.configure(state="normal")
            self.fc2_entry.configure(state="normal")

    def update_project_name(self, *_):
        w = self.woofer_var.get()
        m = self.midrange_var.get().strip()
        t = self.tweeter_var.get()
        
        # --- MODIFICATION ICI : Ne pas inclure "(Aucun)" dans le nom du projet ---
        if m == "(Aucun)":
            m = ""
            
        parts = [p for p in [w, m, t] if p]
        name = "_X_".join(parts)
        self.name_entry.delete(0, "end")
        self.name_entry.insert(0, name)

    # ── Fenêtre Paramètres ────────────────────────────────────────────────────

    def open_settings(self):
        win = ctk.CTkToplevel(self)
        win.title("⚙️  Paramètres Avancés — Poids Fitness")
        win.geometry("500x580")
        win.resizable(False, True)
        win.grab_set()

        ctk.CTkLabel(
            win, text="Poids de la Fonction Fitness",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(pady=(14, 6))
        ctk.CTkLabel(
            win, text="Modifiez uniquement si vous savez ce que vous faites.",
            font=ctk.CTkFont(size=11), text_color="gray"
        ).pack(pady=(0, 10))

        scroll = ctk.CTkScrollableFrame(win, height=420)
        scroll.pack(fill="both", expand=True, padx=16)

        for group_name, keys in WEIGHT_GROUPS:
            # En-tête de groupe
            sep_frame = ctk.CTkFrame(scroll, height=2, fg_color="#444444")
            sep_frame.pack(fill="x", pady=(10, 2))
            ctk.CTkLabel(
                scroll, text=group_name,
                font=ctk.CTkFont(size=12, weight="bold"), text_color="#aaaaaa"
            ).pack(anchor="w", pady=(2, 6))

            for key in keys:
                _, description, _ = WEIGHT_DEFS[key]
                row = ctk.CTkFrame(scroll, fg_color="transparent")
                row.pack(fill="x", pady=2)
                row.columnconfigure(0, weight=1)
                row.columnconfigure(1, weight=0)

                ctk.CTkLabel(
                    row, text=f"{description}",
                    font=ctk.CTkFont(size=12), anchor="w"
                ).grid(row=0, column=0, sticky="w", padx=(0, 10))

                entry = ctk.CTkEntry(row, textvariable=self.weight_vars[key], width=90)
                entry.grid(row=0, column=1, sticky="e")

        # Boutons bas
        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=12)

        ctk.CTkButton(
            btn_frame, text="↩ Réinitialiser",
            fg_color="#555555", hover_color="#666666",
            command=lambda: [
                self.weight_vars[k].set(str(WEIGHT_DEFS[k][0])) for k in WEIGHT_DEFS
            ]
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_frame, text="✔ Appliquer & Fermer",
            command=win.destroy
        ).pack(side="right")

    # ── Lancement de l'optimisation ──────────────────────────────────────────

    def start_optimization(self):
        w  = self.woofer_var.get().strip()
        m  = self.midrange_var.get().strip()
        if m == "(Aucun)":
            m = ""
        t  = self.tweeter_var.get().strip()

        name = self.name_entry.get().strip()
        gen  = self.gen_entry.get().strip()
        pop  = self.pop_entry.get().strip()
        
        # --- MODIFICATION ICI ---
        if self.auto_fc_var.get():
            fc1 = "None"
            fc2 = "None"
        else:
            fc1 = self.fc_entry.get().strip()
            fc2 = self.fc2_entry.get().strip() if m else ""
        # ------------------------

        # Validation basique
        if w not in self.all_woofers:
            self.write_console("[-] Erreur : Woofer introuvable dans la base.")
            return
        if t not in self.all_tweeters:
            self.write_console("[-] Erreur : Tweeter introuvable dans la base.")
            return
        if m and m not in self.all_midranges:
            self.write_console("[-] Erreur : Médium introuvable dans la base.")
            return

        self.run_btn.configure(state="disabled", text="⏳  OPTIMISATION EN COURS...")
        self.console.configure(state="normal")
        self.console.delete("0.0", "end")
        self.console.configure(state="disabled")

        threading.Thread(
            target=self._run_process,
            args=(w, m, t, name, gen, pop, fc1, fc2),
            daemon=True
        ).start()

    def _run_process(self, w, m, t, name, gen, pop, fc1, fc2):
        out_dir = os.path.join("crossovers", name)

        # --- MODIFICATION ICI : Omission intelligente de l'argument ---
        fc_args = []
        if fc1 != "None":
            fc_args = ["--fc", fc1]
            if m and fc2 and fc2 != "None":
                fc_args += [fc2]
        # --------------------------------------------------------------

        cmd = [
            "python", "run.py",
            "--woofer",  w,
            "--tweeter", t,
            "--name",    name,
            "--out_dir", out_dir,
            "--gen",     gen,
            "--pop",     pop,
            "--weights", self.get_current_weights_json(),
            *fc_args,
        ]
        if m:
            cmd += ["--midrange", m]

        custom_env = os.environ.copy()
        custom_env["PYTHONIOENCODING"] = "utf-8"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=custom_env,
            )
            for line in process.stdout:
                self.after(0, self.write_console, line.strip())

            process.wait()

            if process.returncode == 0:
                self.after(0, self.write_console,
                           f"\n✅ MISSION ACCOMPLIE ! Fichiers dans : {out_dir}")
            else:
                self.after(0, self.write_console,
                           f"\n❌ Erreur fatale (Code {process.returncode})")

        except Exception as e:
            self.after(0, self.write_console, f"[-] Erreur de lancement : {e}")

        self.after(0, lambda: self.run_btn.configure(
            state="normal", text="🚀  LANCER L'OPTIMISATION"
        ))


if __name__ == "__main__":
    app = FindMyCrossoverApp()
    app.mainloop()
