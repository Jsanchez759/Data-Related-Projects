import streamlit as st
import os
import sys
sys.path.append(os.getcwd())
from tech_details import tech_implementation
from clustering_action import clustering_app
from streamlit_option_menu import option_menu

def main():

    # Visualize a menu in the sidebar to navigate in the app
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Clustering App", "Technical Implementations"],
            icons=["house", "book"],
            menu_icon="cast",
            default_index=0
        )

    if selected == "Clustering App":
        clustering_app()
    elif selected == "Technical Implementations":
        tech_implementation()

if __name__ == '__main__':
    main()
