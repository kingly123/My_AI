import streamlit as st
import sqlite3
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

#Create SQLite database table
def create_db_table():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (user_question TEXT, bot_response TEXT)''')
    conn.commit()
    conn.close()

def insert_into_db(user_question, bot_response):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (user_question, bot_response) VALUES (?, ?)", (user_question, bot_response))

    conn.commit()
    conn.close()

@st.cache_data
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@st.cache_data
def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator ="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_data
def create_vectorstore(text_chunks):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        insert_vectorstore_into_database(vectorstore)
        return vectorstore
    except Exception as e:
        st.error(f"An error occurred: {e}")

def insert_vectorstore_into_database(vectorstore):
    try:
        conn = sqlite3.connect("vectorstore.db")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, vector BLOB)''')
        c.execute('''INSERT INTO vectors (vector) VALUES (?)''', (pickle.dumps(vectorstore),))
        conn.commit()
        conn.close()
        st.success("Vectorstore inserted into database successfully!")
    except Exception as e:
        st.error(f"An error occurred while inserting vectorstore into database: {e}")

#@st.cache_resource
def initialize_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xx1",model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    #st.write (response)
    st.session_state.chat_history = response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            insert_into_db(user_question, message.content)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            insert_into_db('', message.content)

def main():
    load_dotenv()
    st.set_page_config(page_title="Welcome to Chatbot", page_icon=":books:")
    st.write(css,unsafe_allow_html=True)

    create_db_table()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("SUPER CHATBOT")
    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Processing"):
            raw_text = extract_text_from_pdfs(pdf_docs)

            if not raw_text:
                st.error("No text found in the uploaded PDFs.")
            else:
                text_chunks = split_text_into_chunks(raw_text)
                if not text_chunks:
                    st.error("No text chunks found.")
                else:
                    vectorstore = create_vectorstore(text_chunks)
                    if not vectorstore:
                        st.error("Vectorstore is empty.")
                    else:
                        st.session_state.conversation = initialize_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
