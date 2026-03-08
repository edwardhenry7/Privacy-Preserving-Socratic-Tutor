import os
import time

from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# We do NOT import RetrievalQA anymore. We will do it manually.

MODEL_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf" 
PDF_PATH = "textbook.pdf"

SOCRATIC_TEMPLATE = """
[INST] You are a Socratic Physics Tutor. Use the context below.
Context: {context}

Student Question: {question}

Do NOT answer directly. Ask a guiding question based on the context. [/INST]
"""

def main():
    print("--- 1. LOADING KNOWLEDGE BASE ---")
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found. Please add a PDF.")
        return

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    print(f"✔ Indexed {len(texts)} chunks.")

    print("\n--- 2. INITIALIZING VECTOR DATABASE ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings)
    print("✔ Vector Database Ready.")

    print("\n--- 3. LOADING QUANTIZED MODEL ---")
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file '{MODEL_PATH}' not found.")
        return

    llm = CTransformers(
        model=MODEL_PATH,
        model_type="mistral",
        config={'max_new_tokens': 256, 'temperature': 0.1, 'context_length': 512}
    )
    print("✔ Model Loaded.")

    print("\n" + "="*40)
    print("SOCRATIC TUTOR READY (Type 'exit' to quit)")
    print("="*40 + "\n")

    while True:
        query = input("Student: ")
        if query.lower() == 'exit':
            break
        
        print("  [Thinking...]")
        
        try:
            results = db.similarity_search(query, k=2)
            context_text = "\n\n".join([doc.page_content for doc in results])
            
            print(f"  [Evidence Found: ...{context_text[:50]}...]")
            
            full_prompt = SOCRATIC_TEMPLATE.format(context=context_text, question=query)
            
            response = llm.invoke(full_prompt)
            print(f"\nTutor: {response}\n")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":

    main()
