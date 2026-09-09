from dotenv import load_dotenv
import logging

load_dotenv()  # Load environment variables from .env file
logging.getLogger("google.genai.models").setLevel(logging.ERROR)
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash")

st.title("chatbot using google ")
st.markdown("This is a simple chatbot using Google Gemini 3.6 Flash model. You can ask questions and get responses from the model.")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


history = []
st.markdown(history)




    

query = st.chat_input("Ask a question...")
if query:
    st.session_state.messages.append({"role": "user","content": query})
    history.append({
        "role": "user",
        "content": query
    })
    st.chat_message("user").markdown(query)
    response = llm.invoke(history)
    st.chat_message("assistant").markdown(response.text)
    st.session_state.messages.append({"role":"ai","content":response.text})
    history.append({
        "role": "assistant",
        "content": response.text
    })


