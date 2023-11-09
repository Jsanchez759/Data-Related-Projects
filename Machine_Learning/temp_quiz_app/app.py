from test_quiz import questions, send_email
import streamlit as st

st.title("The Jobs Driver Quiz :question: ")
st.sidebar.success('')
st.subheader('Please answer the following multiple choice questions')
name = st.text_input('Please write your name')
# Display the questions and option

score, total_questions = 0, len(questions)
for i, question in enumerate(questions, start=1):
    st.subheader(f"Question {i}/{total_questions}: {question['question']}")
    selected_option = st.radio("Choose an option", question['choices'], index=None)
    
    if selected_option == question['answer']:
        score += 1

st.write(f"You got {score} out of {total_questions} questions correct!")

if st.button('Send Email'):
    message_sent = send_email(name, score, 'trae@accessai.ai')
    if message_sent == 'Email sent successfully.':
        st.success(message_sent)
    else:
        st.error(message_sent)
    #st.balloons()