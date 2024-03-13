import smtplib
from email.message import EmailMessage
import streamlit as st

def send_email(name, score, receiver):
    sender = 'thejobsdriver@outlook.com'  # Replace with your actual email
    password = st.secrets["EMAIL_PASSWORD"]   # This should be handled securely
    smtp_server = 'smtp-mail.outlook.com'
    smtp_port = 587  # Replace with your SMTP port
    
    message = EmailMessage()
    message.set_content(f"{name} completed TheJobsDriver Product Knowledge Test with a score of {score}/10.")
    message['Subject'] = 'Test Completion Notification'
    message['From'] = sender
    message['To'] = receiver
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender, password)
        server.send_message(message)
        server.quit()
        message_sent = 'Email sent successfully.'
    except Exception as e:
        message_sent = f'Failed to send email: {e}'
    
    return message_sent

# Full list of 10 questions
questions = [
    {
        "question": "What is the core strategy of TheJobsDriver in reaching job seekers?",
        "choices": ["Multi-channel approach", "Direct mailing", "Single-platform advertising", "Cold calling techniques"],
        "answer": "Multi-channel approach"
    },
    {
        "question": "What types of job seekers does TheJobsDriver aim to attract for clients?",
        "choices": ["Only active job seekers", "Only passive job seekers", "Both active and passive job seekers", "None of the above"],
        "answer": "Both active and passive job seekers"
    },
    {
        "question": "What kind of contracts does TheJobsDriver offer to clients?",
        "choices": ["Annual contracts", "Fixed long-term contracts", "Month-to-month contracts", "Bi-annual contracts"],
        "answer": "Month-to-month contracts"
    },
    {
        "question": "What does TheJobsDriver prioritize to ensure client satisfaction?",
        "choices": ["Number of job posts", "Volume of website traffic", "Quality of leads generated", "Quantity of clicks on ads"],
        "answer": "Quality of leads generated"
    },
    {
        "question": "How does TheJobsDriver's multi-channel approach benefit clients?",
        "choices": ["Reduces advertising costs", "Increases job post duration", "Mitigates risks from source performance changes", "Limits the pool of applicants"],
        "answer": "Mitigates risks from source performance changes"
    },
    {
        "question": "What is TheJobsDriver's approach to contracts and client retention?",
        "choices": ["Mandatory long-term commitments", "Flexible month-to-month contracts", "One-time project-based contracts", "Prepaid annual subscriptions"],
        "answer": "Flexible month-to-month contracts"
    },
    {
        "question": "How does TheJobsDriver ensure the generation of high-quality leads?",
        "choices": ["Focus on vanity metrics", "Targeting only active candidates", "Continuously optimized algorithms", "Providing free job post templates"],
        "answer": "Continuously optimized algorithms"
    },
    {
        "question": "What differentiates TheJobsDriver from standard job posting services?",
        "choices": ["Lower pricing for job listings", "Exclusive access to a secret job board", "Targeting both active and passive candidates", "Guaranteed job placements"],
        "answer": "Targeting both active and passive candidates"
    },
    {
        "question": "Why are month-to-month contracts advantageous for clients working with TheJobsDriver?",
        "choices": ["Long-term price locking", "Flexibility and control over the recruitment process", "More job posts per month", "Automatic contract renewals"],
        "answer": "Flexibility and control over the recruitment process"
    },
    {
        "question": "What key performance indicator does TheJobsDriver focus on for recruitment advertising?",
        "choices": ["Number of job post views", "Number of potential hires", "Click-through rates of job ads", "Overall client satisfaction scores"],
        "answer": "Number of potential hires"
    }
]
# test send email
# name = input("Please enter your name: ")
# response = send_email('jesus', 2, 'jesussanchezd@hotmail.es')
# print(response)