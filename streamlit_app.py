import streamlit as st
import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

st.title("📘 CFA Tutor App")

# API Key
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

# File uploader
uploaded_file = st.file_uploader("📄 Upload your CFA PDF (e.g., Ethics)", type="pdf")

# Question input
query = st.text_input("💬 Ask your CFA question:")

# Submit button
if st.button("✅ Get Answer"):
    if not openai_api_key:
        st.error("❌ Please enter your OpenAI API key.")
    elif not uploaded_file:
        st.error("❌ Please upload a PDF file.")
    elif not query:
        st.error("❌ Please enter a CFA question.")
    else:
        try:
            # Setup
            os.makedirs("cache", exist_ok=True)
            os.makedirs("faiss_store", exist_ok=True)

            # Save PDF to cache
            filename = uploaded_file.name
            pdf_path = os.path.join("cache", filename)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            # Paths for FAISS storage
            index_folder = os.path.join("faiss_store", filename.split(".")[0])

            # Embedding model
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

            # Load or create FAISS index
            if os.path.exists(index_folder):
                vectorstore = FAISS.load_local(index_folder, embeddings)
            else:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(pages)

                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local(index_folder)

            # Search relevant chunks
            results = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in results])

            # Ask OpenAI
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

            st.markdown("### 🤖 Answer")
            st.markdown(response.choices[0].message.content)

        except openai.AuthenticationError:
            st.error("❌ Invalid OpenAI API Key.")
        except openai.OpenAIError as e:
            st.error(f"❌ OpenAI Error: {str(e)}")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {str(e)}")
