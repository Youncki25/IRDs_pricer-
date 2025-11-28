from PIL import Image
import streamlit as st

def show_author_card():
    col1, col2 = st.columns([1, 3])

    with col1:
        img = Image.open("assets/younes.jpg")
        st.image(img, width=120)

    with col2:
        st.markdown(
            """
            **Younes Beldjenna**  
            Créateur de l'application *Desk Taux*  
            Réalisée pour son mémoire de Master.

            📧 : [younes.beldjenna@gmail.com](mailto:younes.beldjenna@gmail.com)  
            📞 : 06 15 93 64 72
            """,
            unsafe_allow_html=True,
        )
