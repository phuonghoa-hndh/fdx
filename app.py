import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import json
import os

def get_pdf_text(pdf_paths):
    """
    This function is used to get the text of the pdf files
    :param pdf_paths: path to pdf files
    :return: full text of the pdf files
    """
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    This function is used to get the text chunks, the smaller unit of text
    :param text: get the text in get_pdf_text
    :return: smaller part of text (chunks)
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    This function is used to convert the text chunks into embedding and save it into a vectorstore
    :param text_chunks:
    :return:
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Save the history of chat in a conversation chain
    :param vectorstore:
    :return:
    """
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    save_chat_history(st.session_state.chat_history)  # Save chat history

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def save_chat_history(chat_history):
    history = [{'role': 'user' if i % 2 == 0 else 'bot', 'content': message.content} for i, message in
               enumerate(chat_history)]
    with open('chat_history.json', 'w') as f:
        json.dump(history, f, indent=4)


def main():
    load_dotenv()
    st.set_page_config(page_title="FDX AI Assistant :books:",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("FDX AI Assistant :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Load and process PDF files
    pdf_directory = "C:/Users/Admin/PycharmProjects/FDX/chatbot/data"
    pdf_paths = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith('.pdf')]

    with st.spinner("Processing your documents..."):
        # Get PDF text
        raw_text = get_pdf_text(pdf_paths)

        # Get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # Create vector store
        vectorstore = get_vectorstore(text_chunks)

        # Create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
