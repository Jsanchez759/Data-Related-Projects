import streamlit as st

st.title('Study Material for the quiz :books:')

file_txt = "JobsDriver_ChatBot/app_quiz/pages/study_material.txt"
with open(file_txt, "r", encoding="utf-8") as file:
        material_study = file.read()
        st.markdown(material_study)
