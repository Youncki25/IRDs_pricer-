import streamlit as st
from PIL import Image
from pathlib import Path


def show_author_card():
    st.markdown("### 👤 À propos de l'auteur")

    col1, col2 = st.columns([1, 3])

    with col1:
        # On construit un chemin relatif robuste :
        # dossier courant du fichier + dossier assets + fichier younes.jpg
        base_dir = Path(__file__).resolve().parent
        img_path = base_dir / "Photos des créateurs" / "Younes Beldjenna Analyste Senior.jpg"

        if img_path.exists():
            img = Image.open(img_path)
            st.image(img, width=130)
        else:
            st.write("📷 (Photo introuvable – vérifie le chemin `assets/younes.jpg`)")

    with col2:
        st.markdown(
            """
            **Younes Beldjenna**  
            Créateur de l'application *Desk Taux*  
            Réalisée dans le cadre de son mémoire de Master en Banque & Finance.

            📧 Email : [younes.beldjenna@gmail.com](mailto:younes.beldjenna@gmail.com)  
            📞 Téléphone : 06 15 93 64 72
            """,
            unsafe_allow_html=True,
        )
