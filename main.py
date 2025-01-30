import nest_asyncio
nest_asyncio.apply()

from llama_index.core import Document
#from llama_index.core.utils import draw_all_possible_flows
from config import  LLM_MODEL_NAME, INDEX_PATH, LOG_LEVEL
from workflow import AutoInsuranceWorkflow, LogEvent
from utils import create_index, load_index, load_documents, _extract_policy_number_from_filename
import asyncio
import os
import logging

logger = logging.getLogger(__name__)

async def stream_workflow(workflow, **workflow_kwargs):
    handler = workflow.run(**workflow_kwargs)
    async for event in handler.stream_events():
        if isinstance(event, LogEvent):
            if event.delta:
                print(event.msg, end="")
            else:
                print(event.msg)

    return await handler

async def main():
    # Load policy documents
    policy_dir = "data"  # Assuming policy doc is in data, we will filter inside
    logger.info("Loading policy documents")
    policy_docs = load_documents(policy_dir, metadata_extractor=lambda filename: {"file_name": filename})

    policy_docs = [doc for doc in policy_docs if os.path.basename(doc.metadata['file_name']) == "CAIP400_03012006_CA.pdf"]
    logger.info(f"Found {len(policy_docs)} policy documents")


    # Load declaration documents, and set metadata
    declarations_dir = "data"  # Assuming declarations are also in the data folder
    logger.info("Loading declarations documents")
    declarations_docs = load_documents(declarations_dir, metadata_extractor=_extract_policy_number_from_filename)
    logger.info(f"Found {len(declarations_docs)} declarations documents")


    # Load or create indexes
    policy_index = load_index(INDEX_PATH, "auto_insurance_policies_0")
    if policy_index is None:
      policy_index=create_index(policy_docs, INDEX_PATH, "auto_insurance_policies_0")

    declarations_index = load_index(INDEX_PATH, "auto_insurance_declarations_0")
    if declarations_index is None:
       declarations_index = create_index(declarations_docs, INDEX_PATH, "auto_insurance_declarations_0")


    workflow = AutoInsuranceWorkflow(
        policy_index=policy_index,
        declarations_index=declarations_index,
        verbose=True,
        timeout=None  # don't worry about timeout to make sure it completes
    )

    draw_all_possible_flows(AutoInsuranceWorkflow, filename="auto_insurance_workflow.html")

    # Run workflow for john's claim
    logger.info("Running the workflow for John's claim")
    response_dict = await stream_workflow(workflow, claim_json_path="data/john.json")
    print(str(response_dict["decision"]))
    
    # Run workflow for alice's claim
    logger.info("Running the workflow for Alice's claim")
    response_dict = await stream_workflow(workflow, claim_json_path="data/alice.json")
    print(str(response_dict["decision"]))
    logger.info("Finished running the application")


if __name__ == "__main__":
    asyncio.run(main())