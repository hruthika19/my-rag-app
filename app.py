import os

import streamlit as st
import pdfplumber
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import ollama

system_prompt=""" 
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If there is any need of mathematical calculation regarding the question relavent to context, answer precisely.
6. If the information in the question is not related to the context, re-read the question carefully and answer contextually.
7. When there is a date given, analyze the calendar of the current year correctly to answer.  

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

def process_documents(file_paths: list[str]) -> list[Document]:
    all_docs = []
    
    for file_path in file_paths:
        with pdfplumber.open(file_path) as pdf:
            extracted_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                tables = page.extract_tables()
                
                if text:
                    extracted_text += text + "\n\n"
                
                if tables:
                    for table in tables:
                        table_text = "\n".join(["\t".join(cell if cell is not None else "" for cell in row) for row in table])
                        extracted_text += "\n[Table Extracted]\n" + table_text + "\n"

            doc = Document(page_content=extracted_text, metadata={"source": file_path})
            all_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(all_docs)


def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space":"cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_names: list[str]):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for file_name in file_names:
        for idx, split in enumerate(all_splits):
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{file_name}_{idx}")
    
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def call_llm(context: str, prompt: str):
    # if "weekday of" in prompt.lower():
    #     date_str = prompt.split("weekday of")[1].strip()
    #     weekday = calculate_weekday(date_str)
    #     return [f"The weekday of {date_str} is {weekday}."]
    
    response = ollama.chat(
        model="llama3.2",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt, 
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    relavent_text = ""
    relavent_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    
    for rank in ranks:
        relavent_text += documents[rank["corpus_id"]]
        relavent_text_ids.append(rank["corpus_id"])
    
    return relavent_text, relavent_text_ids

if __name__ == "__main__":
    st.set_page_config(page_title="Leave_Policies of DevDolphins")
    
    # Define file paths for backend processing
    file_paths = [
        "2025 Calendar.pdf",
        "HOLIDAYS LIST IN DevDolphins-2025.pdf",
        "leave_policy_DevDolphins.pdf"
    ]
    
    # Normalize file names
    normalize_file_names = [os.path.basename(file_path).translate(
        str.maketrans({"-": "_", ".": "_", " ": "_"})
    ) for file_path in file_paths]

    # Process and upload documents
    all_splits = process_documents(file_paths)
    add_to_vector_collection(all_splits, normalize_file_names)

    st.header("Leave Policy Support")
    prompt = st.text_area("**Ask a question related to the documents:**")
    ask = st.button("Ask")

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0]
        relavent_text, relavent_text_id = re_rank_cross_encoders(context)
        response = call_llm(context=context, prompt=prompt)
        st.write("".join(response))

        with st.expander("See retrieved documents"):
            st.write(results)
        with st.expander("See most relevant document ids"):
            st.write(relavent_text_id)
            st.write(relavent_text)