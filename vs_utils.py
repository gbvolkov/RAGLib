import json
import pickle
import os
from typing import List, Dict, Tuple
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging
import config
import requests
from requests.exceptions import JSONDecodeError
import time
from tqdm import tqdm  # For progress tracking
from nltk.tokenize import sent_tokenize
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_string_into_chunks(s, max_length):
    # Use list comprehension to create chunks
    return [s[i:i + max_length] for i in range(0, len(s), max_length)]

def smart_chunk_text(text: str, max_size: int) -> List[str]:
    chunks = []
    current_chunk = ""
    #for line in text.split('\n'):
    for line in split_string_into_chunks(text, max_size):
        if len(current_chunk) + len(line) + 1 > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += '\n'
            current_chunk += line
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def cleanup_ref_content(content: str) -> str:
    pattern = r'##IMAGE##\s+\S+\.(?:jpg|jpeg|png|gif|bmp|svg)'
    cleaned_content = re.sub(pattern, '', content)
    return cleaned_content

def chunk_sentences(sentences, max_chunk_size, overlap_size=0):
    chunks = []
    current_chunk = []
    current_length = 0
    idx = 0

    while idx < len(sentences):
        sentence = sentences[idx]
        sentence_length = len(sentence)
        
        if current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Add the current chunk to the chunks list
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Determine the number of sentences to overlap based on overlap_size
            overlap_sentences = []
            overlap_length = 0
            overlap_idx = idx - 1
            
            while overlap_idx >= 0 and overlap_length + len(sentences[overlap_idx]) <= overlap_size:
                overlap_sentences.insert(0, sentences[overlap_idx])
                overlap_length += len(sentences[overlap_idx])
                overlap_idx -= 1
            
            # Start a new chunk with overlapping sentences
            current_chunk = overlap_sentences.copy()
            current_length = overlap_length
            
            # Avoid infinite loop by not resetting idx beyond a reasonable point
            #if not overlap_sentences:
                # If no overlap is possible, skip the problematic sentence
        idx += 1

    # Add any remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_json_for_indexing(json_file_path: str, max_chunk_size: int = 2000, overlap=0.75) -> List[Dict]:
    try:
        with open(json_file_path, 'r', encoding='utf-8-sig') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        logger.info(f"Error decoding JSON: {e}")
        return []
    except FileNotFoundError:
        logger.info(f"File not found: {json_file_path}")
        return []

    processed_documents = []
    
    for item in data:
        core_content = (
            f"Problem Number: {item.get('problem_number', '')}\n"
            f"Problem Description: {item.get('problem_description', '')}\n"
            f"Systems: {item.get('systems', '')}\n"
            f"Solution Steps: {item.get('solution_steps', '')}\n"
        )
        
        links = item.get('links', '')
        #additional_content = f"Links:\n{links}\n\nReferences:\n{references}"
        if max_chunk_size > 0:
            references = cleanup_ref_content(item.get('references', ''))
        else:
            references = item.get('references', '')
        additional_content = f"Links:\n \n\nReferences:\n{references}"

        if max_chunk_size > 0 and len(additional_content) >= max_chunk_size:
            sentences = sent_tokenize(additional_content, language='russian')
            additional_chunks = chunk_sentences(sentences, max_chunk_size=max_chunk_size, overlap_size=max_chunk_size * overlap)    
        else:
            additional_chunks = [additional_content]       #pno = item.get('problem_number', '')
        #if str(pno) == '61':
        #    pass

        #additional_chunks = smart_chunk_text(additional_content, max_chunk_size)
        
        for i, chunk in enumerate(additional_chunks):
            full_content = f"{core_content}\nAdditional Information (Part {i+1}/{len(additional_chunks)}):\n{chunk}"
            processed_documents.append({
                'content': full_content,
                'metadata': {
                    'problem_number': item.get('problem_number', ''),
                    'chunk_number': i+1,
                    'total_chunks': len(additional_chunks),
                    'actual_chunk_size': len(full_content)
                }
            })
    
    return processed_documents

def create_vectorstore_old(json_file_path: str, embedding_model_name: str) -> FAISS:
    processed_docs = process_json_for_indexing(json_file_path)
    with open('processed_docs.txt', 'w', encoding='utf-8-sig') as f:
        f.write('\n\n======================\n'.join([doc['content'] for doc in processed_docs]))
    logger.info(f"Documents processed from {json_file_path}. {len(processed_docs)} documents found.")
    #embedding_model_name = '../models/models--sentence-transformers--distiluse-base-multilingual-cased-v1'
    #embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    embeddings = JinaEmbeddings(jina_api_key=config.JINA_API_KEY, model_name="jina-embeddings-v3")
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=2500)

    logger.info(f"Embeddings loaded from {embedding_model_name}.")
    documents = [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in processed_docs]
    logger.info("Documents built.")
    vectorstore = FAISS.from_documents(documents, embeddings)
    logger.info("Vectore store ready.")
    return vectorstore


def get_documents(
        json_file_path: str,
        max_chunk_size: int = 4000,
        overlap: int = 0.5
) -> List[Document]:
    # Step 1: Process the JSON file
    processed_docs = process_json_for_indexing(json_file_path, max_chunk_size=max_chunk_size, overlap=overlap)
    logger.info(f"Documents processed from {json_file_path}. {len(processed_docs)} documents found.")
    if not processed_docs:
        logger.error("No documents to process. Exiting vector store creation.")
        return []
    
    if logger.isEnabledFor(logging.DEBUG):
        # Save processed documents to a text file for debugging
        with open('./logs/processed_docs.txt', 'w', encoding='utf-8-sig') as f:
            f.write('\n\n======================\n'.join([doc['content'] for doc in processed_docs]))
        logger.info("Processed documents saved to 'processed_docs.txt'.")
    # Step 2: Convert processed docs to Document objects
    documents = [
        Document(page_content=doc['content'], metadata=doc.get('metadata', {}))
        for doc in processed_docs
    ]
    logger.info("Documents converted to LangChain Document objects.")
    return documents
    
def create_vectorstore(
    json_file_path: str,
    embedding_model_name: str,
    batch_size: int = 500,  # Adjust based on API limits
    max_retries: int = 3,
    max_chunk_size: int = 4000,
    overlap: int = 0.5
) -> Tuple[FAISS, List[Document]]:
    """
    Creates a FAISS vectorstore from a JSON file using the specified embedding model.

    Args:
        json_file_path (str): Path to the JSON file containing documents.
        embedding_model_name (str): Name or path of the embedding model.
        batch_size (int, optional): Number of documents to process in each batch. Defaults to 500.
        max_retries (int, optional): Maximum number of retry attempts for failed batches. Defaults to 3.

    Returns:
        FAISS: The created FAISS vectorstore.
    """
    try:
        # Step 1: Convert jsons to chunked list of Document objects
        documents = get_documents(json_file_path, max_chunk_size=max_chunk_size, overlap=overlap)
        if len(documents) == 0:
            logger.error("No documents to process. Exiting vector store creation.")
            return None
        
        # Step 2: Initialize the embedding model
        #embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        embeddings = JinaEmbeddings(
            jina_api_key=config.JINA_API_KEY,
            model_name="jina-embeddings-v3"  # e.g., "jina-embeddings-v3"
        )
        logger.info(f"Embeddings loaded from model: {embedding_model_name}.")


        # Step 4: Initialize FAISS vectorstore with the first batch
        first_batch = documents[:batch_size]
        logger.info(f"Initializing FAISS vectorstore with the first batch of {len(first_batch)} documents.")
        vectorstore = FAISS.from_documents(
            documents=first_batch,
            embedding=embeddings
        )
        logger.info("FAISS vectorstore initialized successfully with the first batch.")

        # Step 5: Process and add remaining documents in batches
        remaining_documents = documents[batch_size:]
        total_batches = (len(remaining_documents) + batch_size - 1) // batch_size
        logger.info(f"Adding remaining documents in {total_batches} batches of size {batch_size}.")

        for batch_num, i in enumerate(tqdm(range(0, len(remaining_documents), batch_size), desc="Adding Batches"), start=1):
            batch = remaining_documents[i:i + batch_size]
            attempt = 0
            while attempt < max_retries:
                try:
                    logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} documents.")
                    vectorstore.add_documents(batch)
                    logger.info(f"Batch {batch_num}/{total_batches} added successfully.")
                    break  # Exit retry loop upon success
                except (JSONDecodeError, requests.exceptions.RequestException, ValueError) as e:
                    attempt += 1
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed for batch {batch_num}: {e}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        logger.error(f"Batch {batch_num} failed after {max_retries} attempts. Skipping this batch.")
                        break  # Optionally, handle skipped batches here

        logger.info("All batches processed. Vector store is ready.")
        full_documents = get_documents(json_file_path, max_chunk_size=-1, overlap=0.5)
        logger.info("Full documents store processed. Vector store and doc strore are ready.")
        return (vectorstore, full_documents)

    except Exception as general_e:
        logger.exception(f"An unexpected error occurred: {general_e}")
        raise  # Re-raise the exception after logging

def save_vectorstore(vectorstore: FAISS, docstore: List[Document], file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    logger.info(f"Storing vectore store to {file_path}.")
    try:
        vectorstore.save_local(file_path)
        with open(f'{file_path}/docstore.pkl', 'wb') as file:
            pickle.dump(docstore, file)
    except Exception as e:
        logger.error(f"Unexpected error while storing vector store: {str(e)}")
        raise
    logger.info(f"Vectorstore saved to {file_path}")

def load_vectorstore(file_path: str, embedding_model_name: str) -> FAISS:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No vectorstore found at {file_path}")
    
    #with open(file_path, "rb") as f:
    #    vectorstore = pickle.load(f)
    
    # Reinitialize the embedding function
    logger.info(f"Loading vectorstore  from {file_path}")
    #embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    embeddings = JinaEmbeddings(jina_api_key=config.JINA_API_KEY, model_name="jina-embeddings-v3")
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=2500)

    vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
    logger.info(f"Vectorstore loaded from {file_path}")

    # Load docstore
    with open(f'{file_path}/docstore.pkl', 'rb') as file:
        documents = pickle.load(file)
    logger.info(f"Documentstore loaded from {file_path}/docstore.pkl")

    return (vectorstore, documents)


from langchain_openai.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore

if __name__ == "__main__":
    json_file = "knowledgebase/kb_test_small.json"
    embedding_model = "jina-embeddings-v3"  # Replace with your actual model name or path
    store_path = "data/vectorstore_jina_multivector"

    try:
        #1. Create vectorstore for smaller chunks (400 chars)
        vectorstore = create_vectorstore(json_file, embedding_model, batch_size=500, max_chunk_size=400, overlap=0.75)
        #2. process_json_for_indexing with bigger chunks (docs)
        #docs = process_json_for_indexing(json_file, max_chunk_size=4000, overlap=0)
        #3. get list of problem_nos from docs
        #4. set docstore at retriever retriever.docstore.mset(list(zip(doc_ids, docs)))


        query = "Что такое осень?"
        embeddings = JinaEmbeddings(
            jina_api_key=config.JINA_API_KEY,
            model_name="jina-embeddings-v3"  # e.g., "jina-embeddings-v3"
        )

        vectorstore = load_vectorstore()
        docs_and_scores: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(query, k="5")

        llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get('OPENAI_API_KEY'))
        base_retriever = vectorstore.as_retriever()
        retriever = MultiQueryRetriever.from_llm(base_retriever , llm, include_original=True, verbose=True)
        docs2 = retriever.invoke(query)

        for doc in docs2:
            print(doc.page_content[:128])
            print("==============================================")

    except Exception as e:
        logger.error(f"Failed to create vectorstore: {e}")
