import os
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import re
import fitz
import ollama

def process_documents(file_paths: list[str]) -> list[Document]:
    all_docs = []
    for file_path in file_paths:
        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = str(Path(file_path).resolve())
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Error processing file {file_path}: {str(e)}")
            continue

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
            metadata = split.metadata.copy()
            metadata['source_file'] = str(Path(split.metadata['source']).resolve())
            metadata['page_number'] = split.metadata['page']
            metadatas.append(metadata)
            ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    return collection.query(query_texts=[prompt], n_results=n_results)

def re_rank_cross_encoders(prompt: str, documents: list[str]):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[prompt, doc] for doc in documents]
    scores = model.predict(pairs)
    
    scored_pairs = list(enumerate(scores))
    sorted_pairs = sorted(scored_pairs, key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in sorted_pairs[:3]]
    
    relevant_text = " ".join([documents[idx] for idx in top_indices])
    
    return relevant_text, top_indices

def get_relevant_sentences(text: str, query: str, encoder_model: CrossEncoder, threshold: float = 0.5):
    sentences = [s.strip() for s in re.split('[.!?]', text) if s.strip()]
    pairs = [[query, sentence] for sentence in sentences]
    scores = encoder_model.predict(pairs)
    return [sent for sent, score in zip(sentences, scores) if score > threshold]

def find_sentence_locations(page, sentences: list[str]):
    locations = []
    for sentence in sentences:
        areas = page.search_for(sentence)
        locations.extend(areas)
    return locations

def save_uploaded_file(uploaded_file, save_dir: Path) -> Path:
    save_dir.mkdir(exist_ok=True)
    file_path = save_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def find_text_locations(pdf_path: str, page_number: int, search_text: str):
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    areas = page.search_for(search_text)
    doc.close()
    return areas

def get_pdf_display_content(pdf_path: str, page_number: int, highlight_rect=None):
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    
    if highlight_rect:
        highlight = page.add_highlight_annot(highlight_rect)
        highlight.update()
    
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = pix.tobytes()
    doc.close()
    return img_data

def extract_query_terms(query: str):
    return [term.strip() for term in re.split(r'\W+', query.lower()) if term.strip()]

def call_llm(context: str, prompt: str):
    with open('system_prompt.txt', 'r') as f:
        system_prompt = f.read()
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {prompt}"
        }
    ]
    
    response = ollama.chat(
        model="llama3.2",
        messages=messages,
        stream=True
    )
    
    for chunk in response:
        if 'message' in chunk and 'content' in chunk['message']:
            yield chunk['message']['content']

if __name__ == "__main__":
    st.set_page_config(page_title="AskHR", layout="wide")
    
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    if 'uploaded_files_dict' not in st.session_state:
        st.session_state.uploaded_files_dict = {}
    
    st.title("Leave Policy Support")
    
    st.markdown("""
        <style>
        .highlight-container {
            border: 2px solid #f63366;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .source-text {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }
        .response-text {
            font-size: 1.1em;
            line-height: 1.5;
            padding: 10px;
            background-color: #f0f2f6;
            border-radius: 5px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Choose PDF files", type=['pdf'], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Process Documents"):
                try:
                    for uploaded_file in uploaded_files:
                        temp_path = temp_dir / uploaded_file.name
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.session_state.uploaded_files_dict[uploaded_file.name] = str(temp_path)
                    
                    docs = process_documents([str(p) for p in temp_dir.glob("*.pdf")])
                    if docs:
                        add_to_vector_collection(docs, [f.name for f in uploaded_files])
                        st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    
    user_query = st.text_input("Enter your question:")
    
    if user_query:
        if st.session_state.uploaded_files_dict:
            if st.button("Generate Answer"):
                with st.spinner("Searching and generating response..."):
                    try:
                        results = query_collection(user_query)
                        if results and results['documents'] and results['documents'][0]:
                            relevant_text, relevant_text_ids = re_rank_cross_encoders(user_query, results['documents'][0])
                            
                            encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                            
                            response_container = st.empty()
                            response_text = ""
                            
                            for response_chunk in call_llm(relevant_text, user_query):
                                response_text += response_chunk
                                response_container.markdown(response_text)
                            
                            if not response_text:
                                st.error("No response generated. Please try again.")
                            else:
                                st.write("---")
                                st.subheader("Source Documents")
                                
                                relevant_chunks = [results['documents'][0][idx] for idx in relevant_text_ids]
                                
                                all_relevant_sentences = []
                                chunks_to_highlight = []
                                
                                for chunk in relevant_chunks:
                                    relevant_sentences = get_relevant_sentences(chunk, user_query, encoder_model)
                                    if relevant_sentences:
                                        all_relevant_sentences.extend(relevant_sentences)
                                    else:
                                        chunks_to_highlight.append(chunk)
                                
                                displayed_pages = set()
                                
                                for idx, (doc_text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                                    try:
                                        source_file = metadata.get('source_file', '')
                                        file_name = Path(source_file).name
                                        page_number = metadata.get('page_number', 0)
                                        
                                        page_id = f"{file_name}_{page_number}"
                                        
                                        if page_id in displayed_pages:
                                            continue
                                        
                                        if file_name in st.session_state.uploaded_files_dict:
                                            pdf_path = st.session_state.uploaded_files_dict[file_name]
                                            
                                            st.write(f"Source: {file_name}, Page {page_number + 1}")
                                            
                                            try:
                                                doc = fitz.open(pdf_path)
                                                page = doc[page_number]
                                                
                                                for sentence in all_relevant_sentences:
                                                    areas = page.search_for(sentence.strip())
                                                    for rect in areas:
                                                        highlight = page.add_highlight_annot(rect)
                                                        highlight.update()
                                                
                                                for chunk in chunks_to_highlight:
                                                    areas = page.search_for(chunk.strip())
                                                    for rect in areas:
                                                        highlight = page.add_highlight_annot(rect)
                                                        highlight.update()
                                                
                                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                                                img_bytes = pix.tobytes("png")
                                                
                                                st.image(img_bytes, use_container_width=True)
                                                
                                                doc.close()
                                                
                                                displayed_pages.add(page_id)
                                            except Exception as e:
                                                st.error(f"Error displaying page {page_number + 1} from {file_name}: {str(e)}")
                                    except Exception as e:
                                        st.error(f"Error processing source document {idx + 1}: {str(e)}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please upload and process some documents first.")