import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Setup the UI
st.title("📄 Smart Doc Q&A Agent")
api_key = "YOUR_GEMINI_API_KEY_HERE" # Put your key here

# 2. Upload a File
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and api_key:
    # Save the file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # 3. Process the Document
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    
    # Split the text into small pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    # 4. Create the "Memory" (Vector DB)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_db = Chroma.from_documents(chunks, embeddings)

    # 5. Create the Q&A Logic
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

    # 6. Ask Questions!
    user_query = st.text_input("Ask a question about your document:")
    if user_query:
        response = qa_chain.invoke(user_query)
        st.write(response["result"])