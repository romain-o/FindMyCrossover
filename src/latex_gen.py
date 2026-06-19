import os
import subprocess

def sanitize_latex(text):
    """Échappe les caractères spéciaux pour éviter de faire planter LaTeX."""
    if not isinstance(text, str):
        return str(text)
    
    # Remplacements critiques
    text = text.replace('%', '\\%')
    text = text.replace('#', '\\#')
    text = text.replace('&', '\\&')
    text = text.replace('$', '\\$')
    text = text.replace('_', '\\_')
    
    # Optionnel : Remplacement propre du symbole micro
    text = text.replace('µ', '$\\mu$') 
    
    return text

# Exemple d'utilisation dans votre boucle d'écriture :
# safe_description = sanitize_latex(part_info['Description'])
# f.write(f" ... & {safe_description} & ... \\\\\n")

class LatexReportGenerator:
    def __init__(self, project_name, out_dir, woofer_name, tweeter_name, logo_path=None):
        self.project_name = project_name
        self.out_dir = out_dir
        self.woofer_name = woofer_name
        self.tweeter_name = tweeter_name
        if logo_path:
            self.logo_path = os.path.abspath(logo_path).replace("\\", "/")
        else:
            self.logo_path = "logo.png"

    def generate(self, bom_tex_file):
        """Fusionne les résultats dans le template LaTeX et compile le PDF."""
        print("\n" + "="*60)
        print("📄 GÉNÉRATION DU RAPPORT PDF LATEX")
        print("="*60)

        # 1. Récupération du tableau BOM généré précédemment
        bom_content = "% Erreur : BOM introuvable"
        if os.path.exists(bom_tex_file):
            with open(bom_tex_file, 'r', encoding='utf-8') as f:
                bom_content = f.read()

        # Nettoyage des noms pour LaTeX (les '_' et '&' font planter le compilateur)
        safe_drivers = f"{self.woofer_name} \\& {self.tweeter_name}".replace("_", "\\_")

        # Noms des fichiers d'images (relatifs au dossier du projet)
        img_onaxis = f"{self.project_name}_Reponse_SPL.png"
        
        img_offaxis = f"{self.project_name}_Directivity.png"
        img_heatmap = f"{self.project_name}_Directivity_Heatmap.png"
        heatmap_path = os.path.join(self.out_dir, img_heatmap)
        offaxis_path = os.path.join(self.out_dir, img_offaxis)
        # Vérification de l'existence des fichiers d'images
        if not os.path.exists(heatmap_path):
            heatmap_block = ""
            offaxis_block = ""
        else:
            heatmap_block = fr"""\section{{Directivity Heatmap}}
            \begin{{figure}}[H]
                \centering
                \includegraphics[width=1\linewidth]{{{img_heatmap}}}
            \end{{figure}}"""

            offaxis_block = fr"""\section{{Off-Axis Response}}
            \begin{{figure}}[H]
                \centering
                \includegraphics[width=1\linewidth]{{{img_offaxis}}}
            \end{{figure}}"""
            
        img_impedance = f"{self.project_name}_Impedance.png"
        img_schema = f"{self.project_name}_Schema.png"
        img_geometry = f"{self.project_name}_Geometry.png"

        # ==========================================
        # 2. LE TEMPLATE LATEX BRUT
        # ==========================================
        latex_template = r"""\documentclass[11pt,a4paper]{report}

% --- Packages ---
\usepackage[utf8]{inputenc}
\usepackage{float} 
\usepackage[T1]{fontenc}
\usepackage[english]{babel} 
\usepackage[margin=2.5cm, top=3.5cm, bottom=3.5cm]{geometry}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage[skins,breakable]{tcolorbox}
\usepackage{circuitikz}
\usepackage{booktabs}
\usepackage{hyperref}

\usepackage{tocloft}
\setlength{\cftbeforechapskip}{18pt} 
\setlength{\cftbeforesecskip}{8pt}   

\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}

\definecolor{AudioDark}{HTML}{121212}   
\definecolor{AudioAccent}{HTML}{E63946} 
\definecolor{AudioLight}{HTML}{F8F9FA}  

\newcommand{\HeaderBars}{%
  \begin{tikzpicture}[remember picture,overlay]
    \fill[AudioDark] (current page.north west) rectangle ([yshift=-2cm]current page.north east);
    \fill[AudioAccent] ([yshift=-2cm]current page.north west) rectangle ([yshift=-2.1cm]current page.north east);
    \node[anchor=west, text=white, font=\Large\bfseries] at ([xshift=2cm, yshift=-1cm]current page.north west) {AUDIO440 | CROSSOVER};
  \end{tikzpicture}
}

\newcommand{\FooterBars}{%
  \begin{tikzpicture}[remember picture,overlay]
    \fill[AudioDark] (current page.south west) rectangle ([yshift=1.2cm]current page.south east);
    \node[text=white, font=\bfseries] at ([yshift=0.6cm]current page.south) {Page \thepage};
    \node[anchor=east, text=gray, font=\small] at ([xshift=-2cm, yshift=0.6cm]current page.south east) {www.etsy.com/shop/Audio440};
  \end{tikzpicture}
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{\HeaderBars}
\fancyfoot[C]{\FooterBars}
\fancypagestyle{plain}{\fancyhf{}\fancyhead[C]{\HeaderBars}\fancyfoot[C]{\FooterBars}}

\titleformat{\chapter}[block]{\normalfont\Huge\bfseries\color{AudioDark}}{\begin{tikzpicture}[baseline={(0,-0.1)}]\node[fill=AudioAccent, text=white, inner sep=8pt] {\thechapter};\end{tikzpicture}\quad}{0pt}{\Huge}
\titlespacing*{\chapter}{0pt}{-15pt}{30pt}
\titlespacing*{\section}{0pt}{18pt}{5pt}

\begin{document}

\begin{titlepage}
  \thispagestyle{empty}
  \begin{tikzpicture}[remember picture,overlay]
    \fill[AudioDark] (current page.north west) rectangle (current page.south east);
    \fill[AudioAccent] ([xshift=1cm]current page.north west) rectangle ([xshift=1.2cm]current page.south west);
    
    \node[text=white, font=\fontsize{45}{60}\selectfont\bfseries, anchor=west] at ([xshift=2.5cm, yshift=2cm]current page.west) {2-WAY CROSSOVER};
    \node[text=AudioAccent, font=\fontsize{35}{50}\selectfont\bfseries, anchor=west] at ([xshift=2.5cm, yshift=0cm]current page.west) {__DRIVERS__};
    
    \node[text=gray, font=\Large, anchor=west] at ([xshift=2.5cm, yshift=-2cm]current page.west) {Complete Simulation Graphs \& Crossover Design};
    \node[text=white, font=\Large\bfseries, anchor=south east] at ([xshift=-2cm, yshift=2cm]current page.south east) {AUDIO440};
  \end{tikzpicture}
\end{titlepage}

\tableofcontents
\clearpage

\chapter{Acoustic Simulations}
Simulation graphs showing the frequency response of the drivers combined with the crossover. For more information on how to interpret these graphs, please refer to the beginner guide provided with your order.

\section{On-Axis SPL Response}
\begin{figure}[H]
    \centering
    \includegraphics[width=1\linewidth]{__IMG_ONAXIS__}
\end{figure}

__OFFAXIS_BLOCK__

__HEATMAP_BLOCK__

\section{Impedance Response}
\begin{figure}[H]
    \centering
    \includegraphics[width=1\linewidth]{__IMG_IMPEDANCE__}
\end{figure}

\chapter{Bill of Materials (BOM)}
The following table lists all the components required to build one crossover. For a stereo pair, please double the quantities. Note that the provided prices and links are for reference only and may vary over time. Always check the supplier's website for the most up-to-date information.

\vspace{0.5cm}
__BOM_TABLE__

\chapter{Crossover Design}
This section provides the electrical schematic required to build the crossover and achieve the acoustic results presented earlier. If you need help understanding this diagram, please refer to the beginner guide.

\section{Electrical Schematic}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{__IMG_SCHEMA__}
\end{figure}

\section{Baffle Geometry}
All figures are computed using the following baffle layout. We recommend using this layout for optimal acoustic performance, but feel free to experiment with different placements if you have specific constraints or preferences. 
Just keep in mind that drastically changing the geometry may affect the frequency response and directivity of your speakers due to phase shifts.  
\begin{figure}[H]
    \centering
    \includegraphics[width=0.80\linewidth]{__IMG_GEOMETRY__}
\end{figure}

\clearpage
\thispagestyle{empty} 
\begin{tikzpicture}[remember picture,overlay]
    \fill[AudioDark] (current page.north west) rectangle (current page.south east);
    \fill[AudioAccent] ([xshift=1cm]current page.north west) rectangle ([xshift=1.2cm]current page.south west);

    \node[text=white, font=\fontsize{40}{50}\selectfont\bfseries, anchor=west] at ([xshift=2.5cm, yshift=4cm]current page.west) {THANK YOU};
    \node[text=AudioAccent, font=\fontsize{25}{35}\selectfont\bfseries, anchor=west] at ([xshift=2.5cm, yshift=2.5cm]current page.west) {FOR YOUR SUPPORT};

    \node[text=gray, font=\Large, anchor=west, text width=13cm] at ([xshift=2.5cm, yshift=-0.5cm]current page.west) {I hope this guide helps you build the crossover for your DIY audio project. \\ \vspace{0.5cm} If you have any questions, run into issues, or just want to share a picture of your finished build, please feel free to reach out via Etsy messages!};

    \node[text=white, font=\Large\bfseries, anchor=west] at ([xshift=2.5cm, yshift=-4cm]current page.west) {Happy building,};
    \node[text=AudioAccent, font=\Large\bfseries, anchor=west] at ([xshift=2.5cm, yshift=-4.8cm]current page.west) {Audio440};

    % --- INSERTION DU LOGO AVEC CHEMIN ABSOLU ---
    \node[anchor=north west] at ([xshift=2.5cm, yshift=-22cm]current page.north west) {\IfFileExists{__LOGO_PATH__}{\includegraphics[width=4cm]{__LOGO_PATH__}}{}};
    
    \node[text=white, font=\large, anchor=south west] at ([xshift=2.5cm, yshift=2cm]current page.south west) {\href{https://www.etsy.com/shop/Audio440}{www.etsy.com/shop/Audio440}};
\end{tikzpicture}
\end{document}
"""
        # ==========================================
        # 3. INJECTION DES DONNÉES
        # ==========================================
        latex_code = latex_template.replace("__DRIVERS__", safe_drivers)
        latex_code = latex_code.replace("__IMG_ONAXIS__", img_onaxis)
        latex_code = latex_code.replace("__OFFAXIS_BLOCK__", offaxis_block)
        latex_code = latex_code.replace("__IMG_OFFAXIS__", img_offaxis)
        latex_code = latex_code.replace("__HEATMAP_BLOCK__", heatmap_block)
        latex_code = latex_code.replace("__IMG_HEATMAP__", img_heatmap)
        latex_code = latex_code.replace("__IMG_IMPEDANCE__", img_impedance)
        latex_code = latex_code.replace("__IMG_SCHEMA__", img_schema)
        latex_code = latex_code.replace("__IMG_GEOMETRY__", img_geometry)
        latex_code = latex_code.replace("__BOM_TABLE__", bom_content)
        latex_code = latex_code.replace("__LOGO_PATH__", self.logo_path)

        # Sauvegarde du fichier .tex
        tex_path = os.path.join(self.out_dir, f"{self.project_name}_Report.tex")
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_code)
        
        print(f"[+] Fichier LaTeX prêt : {tex_path}")

        # ==========================================
        # 4. TENTATIVE DE COMPILATION AUTOMATIQUE
        # ==========================================
        try:
            # On compile 2 fois pour générer correctement le sommaire (Table of Contents)
            subprocess.run(['pdflatex', '-interaction=nonstopmode', f"{self.project_name}_Report.tex"], cwd=self.out_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['pdflatex', '-interaction=nonstopmode', f"{self.project_name}_Report.tex"], cwd=self.out_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[+] 📗 PDF compilé avec succès ! ({self.project_name}_Report.pdf)")
        except FileNotFoundError:
            print("[-] Note: pdflatex n'est pas installé sur cet ordinateur. ")
            print("    Le fichier .tex a été généré. Vous pouvez l'importer dans Overleaf pour générer le PDF !")
        except Exception as e:
            print(f"[-] Erreur lors de la compilation du PDF : {e}")