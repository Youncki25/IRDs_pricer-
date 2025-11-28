# Configuration - API keys and app Settings
FRED_API_KEY = "3c8e78b7fbab629ebde0669dc2f41f28"
BF_API_KEY = "fa0f05769356de38c744b31dde3a1736134911a86522dc5713b10a3c"
APP_TITLE = "Desk Taux"
APP_LAYOUT = "wide"
# Author information
from PIL import Image
import streamlit as st

def show_author_card():
    with st.container():
        st.markdown("### 👤 À propos de l'auteur")

        col1, col2 = st.columns([1, 3])

        with col1:
            # Chemin vers les photos des créateurs
            img=Image.open("Photos des créateurs/Younes Beldjenna Analyste Senior.jpg.png")
            st.image(img, width=130, caption="Younes Beldjenna")

        with col2:
            st.markdown(
                """
                **Younes Beldjenna**  
                Développeur de l'application *Desk Taux*  
                Réalisée dans le cadre de son mémoire de Master.

                📧 Email : [younes.beldjenna@gmail.com](mailto:younes.beldjenna@gmail.com)  
                📞 Téléphone : [06 15 93 64 72](tel:+33615936472)
                """,
                unsafe_allow_html=True,
            )
