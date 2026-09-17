from dotenv import load_dotenv
load_dotenv()

## laibary

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
##loS SESSION STATES
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded= False

if "agent"  not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store=None
if "messages" not in st.session_state:
    st.session_state.messages=[]


def process_document(path):
##load documet

    loader = PyPDFDirectoryLoader(path)
    docs= loader.load()



    ## splite into chunk


    splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap= 200)
    docs = splitter.split_documents(documents=docs)



    ##embedding and vector db


    embading = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embading
    )



    ##chat model


    llm = ChatOllama(model="gemma4:31b-cloud", temperature=0)

    ## tools

    @tool
    def reterive_context(query:str):
        """reterive document relevant to a query from the knowlwdge base"""

        context =""
        docs = vector_db.similarity_search(query=query,k=3)
        for doc in docs:
            context += doc.page_content +"\n\n"


        return context


    system_prompt = """you are a helpfull assistantae that answer questions using reterive content my knowlwdge based consist of the detail from the uploaded document.ALWAYS use the `reterive_context` tool for questio rquiring external knowledge """



    memory = MemorySaver()

    ##agent 

    agent = create_agent(model=llm,
                        tools=[reterive_context],
                        system_prompt=system_prompt,
                        checkpointer=memory
                        )

    st.session_state.agent = agent
    st.session_state.document_uploaded= True


##upload ui
st.title("📚 DocuChat AI")
st.caption("Your personal AI document assistant")


if not st.session_state.document_uploaded:
    st.markdown("""
    ### Turn your documents into a conversation

    Upload your PDFs and ask questions in your own words.

    **What can you do?**
    - 🔍 Find specific information inside your documents
    - 📝 Summarize lengthy content
    - 💬 Ask follow-up questions with conversation memory

    ### Ready to explore?
    Upload one or more PDFs below to get started.
    """)

    st.info(
        '💡 Try asking: "Summarize this document" '
        'or "What are the key points?"'

    )
    
    uploaded = st.file_uploader(label="Upload your PDF documents",type=["pdf"],accept_multiple_files=True)
    if uploaded :
        with st.spinner("loading"):
            path = "./docs_file/"
            for file in uploaded:
                with open(path +file.name, "wb")as f:
                    f.write(file.getvalue())

            process_document(path)

            st.rerun()
##chat ui  

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)




    quey =st.chat_input("🤖Ask any thing realted to uploaded document")
    if quey:
        st.session_state.messages.append({"role":"user","content":quey})
        st.chat_message("user").markdown(quey)
        respone = st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":quey}]},
            {"configurable":{"thread_id":"1"}}

        )

        answer = respone["messages"][-1].text
        st.chat_message("Ai").markdown(answer)
        st.session_state.messages.append({"role":"AI","content":answer})

    