import streamlit as st
import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

st.title("📘 CFA Tutor App")

# API Key input
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

# File upload
uploaded_file = st.file_uploader("📄 Upload your CFA PDF (e.g., Ethics)", type="pdf")

# Question input
query = st.text_input("💬 Ask your CFA question:")

# Submit button
if st.button("✅ Get Answer") and uploaded_file and openai_api_key and query:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Process PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Create vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Search for similar content
    results = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    # Run OpenAI
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

    # Show answer
    st.markdown("### 🤖 Answer")
    st.markdown(response.choices[0].message.content)
