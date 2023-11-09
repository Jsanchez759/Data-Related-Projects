import streamlit as st
import PyPDF2

st.title('Study Material for the quiz :books:')

pdf_file_path = "../Introduction to TheJobsDriver.pdf"
pdf_file = open(pdf_file_path, 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

extracted_text = ""

for page_num in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[page_num]
    extracted_text = page.extract_text()
    st.write(extracted_text)


