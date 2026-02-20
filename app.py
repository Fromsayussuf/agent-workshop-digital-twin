from dotenv import load_dotenv
from openai import OpenAI
import os
import chromadb
import streamlit as st

load_dotenv()

client = OpenAI()

def load_documents():
    docs = []
    folder = "knowledge"

    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            docs.append(f.read())

    return docs

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="can_knowledge")

docs = load_documents()

for doc in docs:
    chunks = chunk_text(doc)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[get_embedding(chunk)],
            ids=[f"id_{i}_{hash(chunk)}"]
        )

def retrieve_context(question):
    embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    return "\n".join(results["documents"][0])

def ask_agent(question, context):
    system_prompt = """
    You are the AI digital twin of Can Özfuttu.
    Answer questions about him using ONLY the provided context.
    Be professional and concise.
    If you don't know, say you don't know.
    """

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )

    return response.choices[0].message.content

st.title("Can's Digital Twin")

question = st.text_input("Ask about Can:")

if question:
    context = retrieve_context(question)
    answer = ask_agent(question, context)
    st.write(answer)
