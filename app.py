import os
import base64
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate
from streamlit_float import *
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory

st.set_page_config(
     page_title='FDX_GPT',
     layout="wide",
     initial_sidebar_state="auto",
)
col1, col2, col3 = st.columns([1, 4, 1])
# Load environment variables
load_dotenv()

UPSTASH_URL = 'https://fit-catfish-22726.upstash.io'
UPSTASH_TOKEN = 'AVjGAAIjcDFhYjhiMWMwMDlkMmE0MWNkYmMxNjcxYjgxODZjOTU1YnAxMA'

# Function to convert image to base64 string
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Function to set the background image
def set_background():
    local_image_path = 'images/back_ground_page.png'
    base64_image = get_base64_image(local_image_path)

    background_image = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(background_image, unsafe_allow_html=True)

# Function to load the FAISS vector store
def get_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local('faiss-db', embeddings=embeddings, allow_dangerous_deserialization=True)

# Function to create a conversation chain with the provided prompt



def RAG_Memory_Chain():
    llm = ChatOpenAI(model=st.session_state.model)

    PROMPT_TEMPLATE = """Bạn là một trợ lý AI thông minh của công ty TNHH FPT Digital. 
    Hãy cung cấp cho tôi thông tin một cách đầy đủ và chi tiết nhất có thể. 
    Nếu tôi không yêu cầu rõ ràng về thông tin cụ thể, 
    bạn có thể cung cấp thông tin liên quan nhưng đừng quá sa đà vào các chi tiết không cần thiết. 
    Nếu bạn không chắc chắn về câu trả lời, bạn có thể nói rằng bạn không biết, nhưng nếu có, 
    hãy sử dụng kiến thức và dữ liệu của mình để bổ sung
    và thêm phần tóm gọn nội dung ở cuối mỗi câu trả lời:

    {context}

    Lịch sử cuộc trò chuyện:
    {chat_history}

    Trả lời câu hỏi sau:
    {question}

    Trợ lý:"""

    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=st.session_state.memory,
        verbose=False,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
def Overall_Chain():

    llm = ChatOpenAI(model=st.session_state.model)
    PROMPT_TEMPLATE = """Bạn là trợ lý AI thông minh của công ty TNHH FPT Digital. 
    Hãy cung cấp cho tôi thông tin một cách rõ ràng và chi tiết nhất có thể. 
    Nếu tôi không yêu cầu rõ ràng về một thông tin cụ thể, 
    bạn có thể cung cấp các chi tiết liên quan nhưng không cần lan man vào những điều không cần thiết. 
    Nếu bạn không biết chắc câu trả lời, bạn có thể nói rằng bạn không biết, nhưng hãy sử dụng tất cả kiến thức có sẵn của bạn để hỗ trợ tốt nhất.
    Sau mỗi câu trả lời, tóm tắt lại những ý chính một cách ngắn gọn và súc tích.
    Lịch sử cuộc trò chuyện (để tham khảo):
    {chat_history}
    Câu hỏi hiện tại:
    {question}
    Trợ lý:"""
    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=[ "question", "chat_history"]
    )

    return ConversationChain(
        input_key="question",
        llm=llm,
        prompt=PROMPT,
        memory=st.session_state.memory,
        verbose=False
    )


# Generator function to yield responses with a slight delay for effect
def response_generator(text):
    for word in text.strip():
        yield word + ""
        time.sleep(0.01)

# Main function to handle the user input and generate a response
def handle_userinput(user_question):
    global col2
    with col2:
        st.chat_message("user", avatar="images/user.jpg").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question, "avatar": "images/user.jpg"})
        st.session_state.redis_history.add_user_message(user_question)

        with st.chat_message("assistant", avatar="images/for_bot.jpg"):
            response = st.write_stream(response_generator(st.session_state.conversation.run(user_question)))
            st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "images/for_bot.jpg"})
            st.session_state.redis_history.add_ai_message(response)

# Main function to run the Streamlit app
def main():


    set_background()

    # Sidebar: Upload PDFs and Conversation buttons
    #select your model
    with st.sidebar:
        st.header('History conversation')
        for i in range(10):
            st.button(f'Conversation {i}', key=f'Conversation {i}', use_container_width=True)


    # Initialize session state variables if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True, k=10)

    if 'vectorstore' not in st.session_state:
        with st.spinner("Loading your database..."):
            st.session_state.vectorstore = get_vector_store()
        st.success('Successfully loaded your database')

    if 'model' not in st.session_state:
        st.session_state.model = 'gpt-4o'

    if "conversation" not in st.session_state:
        st.session_state.conversation = Overall_Chain()

    if 'redis_history' not in st.session_state:
        st.session_state.redis_history = UpstashRedisChatMessageHistory(
            url=UPSTASH_URL, token=UPSTASH_TOKEN, session_id="chau_01", ttl=0
        )



    # Load historical messages from Redis into session state
    history = st.session_state.redis_history
    if not st.session_state.messages:
        for i in range(0, len(history.messages), 2):
            if i + 1 < len(history.messages):
                user_input = history.messages[i].content
                assistant_response = history.messages[i + 1].content
                st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "images/user.jpg"})
                st.session_state.messages.append({"role": "assistant", "content": assistant_response, "avatar": "images/for_bot.jpg"})
                st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})

    # Display chat history in the chat window


    with col2:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"])

    # Handle new user input
    float_init()
    with col1:
        st.logo('images/LOGO.png')
        option = st.selectbox(
            "",
            ("gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"),
            index=None,
            placeholder="gpt-4o",
            label_visibility='collapsed'
        )
        if option:
            st.session_state.model = option
    
    with  col3:
        with st.popover("Assistant"):
            chain = st.radio(
                "Select your assistant",
                ["FDX Assistant", "FDX QueryPro"],
                captions=[
                    "FDX Assistant is an intuitive assistant designed to handle everyday tasks and complex requests",
                    "FDX QueryPro taps into FDX’s rich data resources to provide comprehensive answers. While slightly slower than real-time assistants, it compensates with accuracy and depth, ensuring you get high-quality insights every time",
                ],
            )

            if chain == "FDX Assistant":
                st.session_state.conversation = Overall_Chain()
            else:
                st.session_state.conversation = RAG_Memory_Chain()

    col1.float()
    col3.float()

    if prompt := st.chat_input("Tôi có thể giúp gì cho bạn?"):
        handle_userinput(prompt)

# Run the app
main()
