import streamlit as st
from PIL import Image
from pathlib import Path

def show_author_card():
    st.markdown("### 👤 À propos de l'auteur")

    col1, col2 = st.columns([1, 3])

    with col1:
        # Chemin EXACT de ta photo dans ton repo GitHub
        img_path = Path("Photos des créateurs") / "Younes Beldjenna Analyste Senior.jpg.png"

        try:
            img = Image.open(img_path)
            st.image(img, width=130)
        except Exception as e:
            st.error(f"Impossible de charger la photo : {e}")
            st.write(f"Chemin utilisé : `{img_path}`")

    with col2:
        st.markdown(
            """
            **Younes Beldjenna**  
            Créateur de l'application *Desk Taux*  
            Réalisée dans le cadre de son mémoire de M2.

            📧 : [younes.beldjenna@gmail.com](mailto:younes.beldjenna@gmail.com)  
            📞 : 06 15 93 64 72
            """,
            unsafe_allow_html=True,
        )
