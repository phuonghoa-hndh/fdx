import os
import base64
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
import streamlit as st
import base64

# Function to convert image to base64 string
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set your local image path
local_image_path = 'images/FDX_bak.png'  # Replace with your actual image file path

# Get the base64 string of the image
base64_image = get_base64_image(local_image_path)

# Set the background image using CSS with base64
background_image = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/jpg;base64,{base64_image}");
    background-size: cover;  /* Adjust the size as needed */
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

def get_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore_faiss = FAISS.load_local('faiss-db', embeddings=embeddings, allow_dangerous_deserialization=True)
    return vectorstore_faiss
def get_conversation_chain():



    llm = ChatOpenAI(model="gpt-4o")

    PROMPT_TEMPLATE = """bạn là một trợ lý AI thông minh của công ty TNHH FPT Digital hãy cho tôi thông tin dựa trên
    hãy cho tôi nhiều thông tin nhất có thể và nếu như tôi không yêu cầu thì đừng đưa thêm thông tin không cần thiết
    đừng cố tạo ra câu trả lời nếu như bạn không biết, nếu các thông tin không có trong nội dung file tôi cung cấp với bạn thì đừng trả lời:
    {context}

    Chat History:
    {chat_history}
    Answer the following question:
    {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
        memory=memory,
        verbose=False,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return qa

def response_generator(text):

    for word in text.strip():
        yield word + ""
        time.sleep(0.01)

st.header("FPT Digital AI assistant")
def handle_userinput(user_question):
    st.chat_message("user", avatar="images/user.jpg").markdown(user_question)

    st.session_state.messages.append(
        {"role": "user", "content": user_question, "avatar": "images/user.jpg"})

    with st.chat_message("assistant", avatar="images/for_bot.jpg"):
        response = st.write_stream(response_generator(st.session_state.conversation.run(user_question)))
        st.session_state.messages.append(
            {"role": "assistant", "content": response, "avatar": "images/for_bot.jpg"})


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
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    if st.session_state.vectorstore is None:
        with st.spinner(text="Loading your database"):
            st.session_state.vectorstore = get_vector_store()
        st.success('Sucessfully reading your database')
    if prompt := st.chat_input("Tôi có thể giúp gì cho bạn?"):

        handle_userinput(prompt)

    with st.sidebar:

        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain()
        st.subheader("Update knowledge")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            pass
        with st.container(border=True):
            for i in range(10):
                st.button(f'Conversation {i}',key=f'Conversation {i}' ,use_container_width=True)

if __name__ == "__main__":
    main()
