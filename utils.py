import json
from schemas import ClaimInfo
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import MetadataMode
from config import INDEX_PATH
import os
from llama_index.core.vector_stores.types import MetadataFilters
from typing import List
from llama_index.core import SimpleDirectoryReader
import logging


logger = logging.getLogger(__name__)


def parse_claim(file_path: str) -> ClaimInfo:
    with open(file_path, "r") as f:
        data = json.load(f)
    return ClaimInfo.model_validate(data)

def load_index(index_path, name):
    """Load the index from disk."""
    persist_dir = os.path.join(index_path, name)
    if not os.path.exists(persist_dir):
        logger.info(f"Index {name} does not exist in {persist_dir}, creating new one.")
        return None
    logger.info(f"Loading index {name} from {persist_dir}.")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return VectorStoreIndex.from_vector_store(
        vector_store=storage_context.vector_store,
        storage_context=storage_context
    )
def save_index(index: VectorStoreIndex, index_path, name):
    """Persist index to disk."""
    persist_dir = os.path.join(index_path, name)
    logger.info(f"Saving index {name} to {persist_dir}.")
    index.storage_context.persist(persist_dir=persist_dir)


def create_index(docs: List[Document], index_path: str, index_name:str) -> VectorStoreIndex:
        """Create and persist an index from documents."""
        logger.info(f"Creating new index {index_name}")
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
             documents=docs,
             storage_context=storage_context
         )
        save_index(index, index_path, index_name)
        return index

def get_declarations_docs(index:VectorStoreIndex, policy_number: str, top_k: int = 1):
    """Get declarations retriever."""
    logger.info(f"Getting declarations doc for policy {policy_number}")
    # build retriever and query engine
    filters = MetadataFilters.from_dicts([
        {"key": "policy_number", "value": policy_number}
    ])
    retriever = index.as_retriever(
        rerank_top_n=top_k,
        filters=filters
    )
    # semantic query matters less here
    return retriever.retrieve(f"declarations page for {policy_number}")


def load_documents(input_dir, metadata_extractor=None):
    """Loads documents from the input directory using SimpleDirectoryReader, optionally extracts metadata."""
    logger.info(f"Loading documents from {input_dir}")
    documents = []
    reader = SimpleDirectoryReader(input_dir=input_dir, file_exts=[".pdf"])
    loaded_docs= reader.load_data()

    if metadata_extractor:
        for doc in loaded_docs:
          doc.metadata.update(metadata_extractor(doc.metadata['file_path']))

    documents.extend(loaded_docs)
    logger.info(f"Loaded {len(documents)} documents from {input_dir}")


    return documents

def _extract_policy_number_from_filename(filename):
    logger.debug(f"Extracting policy number from filename: {filename}")
    p=filename.replace('-declarations.md',"").split('/')[1] # data/john-declarations.md
    claim_info = parse_claim(f"data/{p}.json")
    policy_num = claim_info.policy_number
    return {
        "policy_number": policy_num,
            "file_name": filename
        }