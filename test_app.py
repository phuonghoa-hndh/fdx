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
import time
st.image(r"D:\coding\chat_bot\image\back_for_thanh_cong.png")
st.header("Hyundai AI Assistant")
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
    embeddings = OpenAIEmbeddings(openai_api_key= "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")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_conversation_chain(vectorstore):
    """
    Save the history of chat in a conversation chain
    :param vectorstore:
    :return:
    """
    llm = ChatOpenAI(openai_api_key= "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",
                     model="gpt-4o")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),

        memory=memory
    )
    return conversation_chain

def response_generator(text):

    for word in text.strip():
        yield word + ""
        time.sleep(0.01)
def handle_userinput(user_question):
    st.chat_message("user", avatar="images/for_user.jpg").markdown(user_question)
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    save_chat_history(st.session_state.chat_history)  # Save chat history

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:

            st.session_state.messages.append(
                {"role": "user", "content": user_question, "avatar": "images/for_user.jpg"})
        else:
            with st.chat_message("assistant", avatar="images/logo_thanh_cong.jpg"):

                response = st.write_stream(response_generator(message.content))
                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "avatar": "images/logo_thanh_cong.jpg"})

def save_chat_history(chat_history):
    history = [{'role': 'user' if i % 2 == 0 else 'bot', 'content': message.content} for i, message in
               enumerate(chat_history)]
    with open('chat_history.json', 'w') as f:
        json.dump(history, f, indent=4)

def response_generator(text):

    for word in text.strip():
        yield word + ""
        time.sleep(0.01)
def main():
    load_dotenv()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None



    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])


    if prompt := st.chat_input("What is up?"):
        handle_userinput(prompt)



    # Load and process PDF files
    pdf_directory = "data"
    pdf_paths = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith('.pdf')]

    # with st.spinner("Processing your documents..."):
        # Get PDF text
    raw_text = get_pdf_text(pdf_paths)
    # Get the text chunks
    text_chunks = get_text_chunks(raw_text)
    # Create vector store
    vectorstore = get_vectorstore(text_chunks)
    # Create conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

        if st.button("Save Conversation"):
            if st.session_state.chat_history:
                save_chat_history(st.session_state.chat_history)
                st.success("Conversation saved successfully!")
            else:
                st.warning("No conversation history to save.")

if __name__ == "__main__":
    main()

# st.write(st.session_state.chat_history)
# st.write(st.session_state.messages)