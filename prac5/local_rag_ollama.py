from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader


def load_and_split_docs(file_path):
    print("📄 Loading and splitting documents...")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    print(f"✅ Loaded and split into {len(docs)} chunks.")
    return docs

def create_vectorstore(docs, save_path="faiss_index"):
    print("🔍 Creating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_path)
    print(f" Vector store saved to {save_path}")
    return vectorstore

def load_vectorstore(save_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(save_path, embeddings)

def create_rag_chain(vectorstore, model_name="llama3"):
    print(f" Loading LLM model from Ollama: {model_name} ...")
    llm = Ollama(model=model_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

def main():
    file_path = "ai-project.pdf"

    docs = load_and_split_docs(file_path)

    vectorstore = create_vectorstore(docs)

    rag_chain = create_rag_chain(vectorstore)

    print("\n Ask me anything based on the document. Type 'exit' to quit.")
    while True:
        query = input("\n Your Question: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = rag_chain.run(query)
        print(f" Answer: {answer}")

if __name__ == "__main__":
    main()
