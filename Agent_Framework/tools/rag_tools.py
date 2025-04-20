import logging
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from ..tools.base_tool import BaseTool
from ..config.settings import llm, gemini_embedding_model, PERSIST_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class QueryTool(BaseTool):
    """Base class for Query Tool"""

    def __init__(self, name, description):
        super().__init__(name, description)
        self.service = None


class GetQueryTool(QueryTool):
    """Tool for query docs"""

    def __init__(self):
        super().__init__(
            name="GetQuery",
            description="Searches locally indexed documents (like previously uploaded PDFs or text files) to answer questions based only on their content. Use this tool when the user asksto find information 'in the document', 'from the uploaded files', 'in my notes', 'what the record says about X', 'based on the provided context', or refers to 'past records' or the 'knowledge base' associated with this session. This searches the specific documents that have been processed and stored locally. Do NOT use for general web searches, code execution, or real-time information not present in the indexed files."
        )


    async def execute(self, query):
        if not self.service:
            if not await self.initialize():
                return {
                    "status": "error",
                    "message": "Failed to get the query service"
                }

        try:
            Settings.llm = llm
            Settings.embed_model = gemini_embedding_model
            # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            logger.info("Index successfully built.")

            query_engine = index.as_query_engine()
            logger.info("Query engine initialized.")

            response = query_engine.query(query)
            response_text = str(response)

            logger.info("Query executed successfully.")
            logger.debug(f"RAG response: {response_text}")

            return {
                "status": "success",
                "RAG response": response_text
            }

        except Exception as e:
            logger.error(f"Error during document query: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }


