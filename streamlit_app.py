import streamlit as st
from src.chatbot.response import LegalChatbotResponse
from streamlit_chat import message 


st.set_page_config(page_title="Nyay Mitra - AI Legal Assistant", page_icon="⚖️", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>⚖️ Nyay Mitra - Your AI Legal Assistant</h1>
    <p style='text-align: center; color: #555;'>Ask your legal queries, and get instant AI-generated responses.</p>
    <hr>
    """, unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def generate_response(input_text):
    model = LegalChatbotResponse()
    response = model.generate_answer(input_text)
    return response

user_input = st.text_area("Enter your query:", placeholder="Type your legal question here...")
if st.button("Submit", use_container_width=True):
    if user_input.strip():
        response = generate_response(user_input)
        st.session_state.chat_history.append((user_input, response))


for i, (user_query, bot_response) in enumerate(st.session_state.chat_history):
    message(user_query, is_user=True, key=f"user_{i}")
    message(str(bot_response.content), is_user=False, key=f"bot_{i}")



st.markdown("""
    <hr>
    <p style='text-align: center;'>Made by Ujjwal Sharma (22160) and Malayaj Singh Shekhawat (22138)</p>
    """, unsafe_allow_html=True)
