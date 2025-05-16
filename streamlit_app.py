import streamlit as st
import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

st.title("📘 CFA Tutor App")

# Get OpenAI API key
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your CFA PDF (e.g., Ethics)", type="pdf")

# Ask a question
query = st.text_input("💬 Ask your CFA question:")

# Run when button is clicked
if st.button("✅ Get Answer"):
    if not openai_api_key:
        st.error("❌ Please enter your OpenAI API key.")
    elif not uploaded_file:
        st.error("❌ Please upload a PDF file.")
    elif not query:
        st.error("❌ Please enter a CFA question.")
    else:
        try:
            # Save uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            # Load and chunk PDF
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(pages)

            # Embed and create vectorstore
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = FAISS.from_documents(chunks, embeddings)

            # Similarity search
            results = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in results])

            # Run GPT
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

        except openai.AuthenticationError:
            st.error("❌ Invalid OpenAI API Key.")
        except openai.OpenAIError as e:
            st.error(f"❌ OpenAI Error: {str(e)}")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {str(e)}")
