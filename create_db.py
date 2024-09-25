import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import  PdfReader
load_dotenv()



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
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    vectorstore.save_local("./faiss-db")
    return vectorstore



if __name__ == "__main__":
    pdf_directory = "data"
    pdf_paths = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith('.pdf')]
    raw_text = get_pdf_text(pdf_paths)
    # Get the text chunks
    text_chunks = get_text_chunks(raw_text)
    # Create vector store
    vectorstore = get_vectorstore(text_chunks)