import streamlit as st
import openai
import spacy
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores.redis import Redis
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
with st.sidebar:
    st.title('🤗💬 Chat with your Data')
    st.markdown('''
                ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                ''')

def split_text_into_chunks(text, max_chunk_size):
    nlp = spacy.load("en_core_web_sm")
    # Tokenizacija teksta na rečenice
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence        
    chunks.append(current_chunk.strip())
    
    return chunks


def main():
    st.header("Chat with your own PDF")
    pdf = st.file_uploader("Upload your pdf", type="pdf")

    if pdf is not None: #kada korisnik uploaduje pdf fajl, tada ga treba procitati
        pdf_reader = PdfReader(pdf)
        chunks = []

        for page in pdf_reader.pages:
            text = page.extract_text()
            page_chunks = split_text_into_chunks(text,150)
            chunks.extend(page_chunks) 
        
        
        embeddings = OpenAIEmbeddings() #kreiramo embeddings
        VectorStore = Redis.from_texts(chunks, embeddings, redis_url="redis://localhost:6379")
        qa = ConversationalRetrievalChain.from_llm(OpenAI(model = "text-davinci-003",temperature=0), VectorStore.as_retriever()) 
        chat_history = []

        query = st.text_input("Ask questions related to your PDF")
        
        
        if query:
            result = qa({"question": query, "chat_history": chat_history})
            chat_history.append((query, result['answer']))
            st.write(result)

            
        
            
if __name__ == '__main__':
    main()
