import PyPDF2, re

pdf_file_path = "JobsDriver_ChatBot/Introduction to TheJobsDriver.pdf"
pdf_file = open(pdf_file_path, 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)
title_pattern = r'^[A-Z][A-Za-z\s]+'

extracted_text = ""

for page_num in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[page_num]
    extracted_text = page.extract_text()
    match = re.search(title_pattern, extracted_text)
    if match:
        title = match.group().strip()
        print("Title:", title)
    else:
        print("Title not found.")
    print(extracted_text)