import streamlit as st
import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

st.title("📘 CFA Tutor App")

# Input field for API key
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

# File upload
uploaded_file = st.file_uploader("📄 Upload your CFA PDF (e.g., Ethics)", type="pdf")

# User input for question
query = st.text_input("💬 Ask your CFA question:")

# Process if all inputs are available
if uploaded_file and openai_api_key and query:
    # Save the uploaded PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Initialize embeddings with the API key
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Search for relevant chunks
    results = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    # Prepare and call OpenAI
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a CFA tutor who explains concepts clearly and simply. "
                    "Use plain language and short sentences. Be direct and easy to follow. "
                    "Avoid AI-sounding phrases like 'let’s dive in' or 'unleash your potential.' "
                    "Do not use hype, fluff, or forced friendliness. "
                    "Use analogies if helpful, include exam traps, examples, and a TL;DR summary. "
                    "Stick to only the context provided."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0.4,
    )

    # Display the answer
    st.markdown("### 🤖 Answer")
    st.markdown(response.choices[0].message.content)
