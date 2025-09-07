import os
import psycopg2
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel # Using preview for now
import json
from typing import Optional, List, Dict, Any
from google.genai import types

# Assuming ADK's BaseTool and ToolContext are available
# If not, you might need to adjust imports based on your ADK setup
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.utils.model_name_utils import is_gemini_model, is_gemini_1_model # Assuming these are available

# --- Configuration ---
# Vertex AI Project and Location
# These should ideally be dynamically loaded or passed via ToolContext/environment


PROJECT_ID = ""  
LOCATION = "us-central1"          
# AlloyDB Connection Details
DB_HOST = ""
DB_PORT = "5432"
DB_NAME = ""
DB_USER = ""
DB_PASSWORD = ""
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072

# PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id") # Use env var if available
# LOCATION = "us-central1" # Or load from config/context

# # AlloyDB Connection Details
# # These should ideally be dynamically loaded or passed via ToolContext/environment
# DB_HOST = os.environ.get("ALLOYDB_HOST", "your_alloydb_host")
# DB_PORT = os.environ.get("ALLOYDB_PORT", "5432")
# DB_NAME = os.environ.get("ALLOYDB_DB_NAME", "your_db_name")
# DB_USER = os.environ.get("ALLOYDB_DB_USER", "your_db_user")
# DB_PASSWORD = os.environ.get("ALLOYDB_DB_PASSWORD", "your_db_password")

# # Embedding Model
# EMBEDDING_MODEL_NAME = "gemini-embedding-001"
# EMBEDDING_DIMENSION = 3072 # Confirmed dimension

# --- Helper function to get Gemini embeddings ---
def _get_gemini_embeddings(texts: list[str], model_name: str, project_id: str, location: str) -> list[list[float]]:
    """Generates embeddings for a list of texts using the specified Gemini model."""
    try:
        aiplatform.init(project=project_id, location=location)
        model = TextEmbeddingModel.from_pretrained(model_name)
        embeddings = model.get_embeddings(texts)
        return [embedding.values for embedding in embeddings]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

# --- Helper function to draft a report ---
def _draft_report(query_text: str, task_context: str, guideline_snippets: list[dict], report_format: str) -> str:
    """Drafts a report based on query, context, and retrieved guidelines."""
    if report_format == "discovery_questionnaire_report":
        report_parts = []
        report_parts.append(f"## Discovery Questionnaire Report: {task_context if task_context else 'Task'}")
        report_parts.append(f"\n**Query:** \"{query_text}\"")
        report_parts.append("\n**Guidelines Consulted:**\n")

        if not guideline_snippets:
            report_parts.append("No relevant guidelines found.")
        else:
            for snippet in guideline_snippets:
                report_parts.append(f"*   **{snippet['document_name']}**:")
                # Truncate content for brevity in the report
                content_preview = snippet['text_content'][:200] + "..." if len(snippet['text_content']) > 200 else snippet['text_content']
                report_parts.append(f"    *   **Guideline:** \"{content_preview}\"")
                report_parts.append(f"    *   **Relevance Score:** {snippet['distance']:.4f}")
                report_parts.append("") # Newline for spacing

        report_parts.append("\n**Assessment:**")
        # Placeholder for more sophisticated AI-driven assessment.
        report_parts.append("Consulted guidelines provide context for the query. Further analysis may be required to form a definitive assessment.")

        report_parts.append("\n**Recommendation:**")
        # Placeholder for recommendations.
        report_parts.append("Adhere to the retrieved guidelines. Specific actions depend on the task context and detailed analysis of the guidelines.")

        return "\n".join(report_parts)
    else:
        # Default or unsupported format
        return f"Report format '{report_format}' not supported. Retrieved {len(guideline_snippets)} snippets for query: '{query_text}'."

# --- The main tool class adhering to ADK's BaseTool ---
class GuidelineConsultantTool(BaseTool):
    """
    A custom tool for consulting platform guidelines and discovery questionnaires
    stored in AlloyDB, and drafting reports.
    """

    # Tool attributes as per BaseTool
    name: str = 'guideline_consultant'
    description: str = (
        'Consults platform guidelines and discovery questionnaires stored in AlloyDB. '
        'Can retrieve relevant snippets and draft reports based on the findings.'
    )

    # Tool-specific configuration (can be passed during initialization)
    # These are made instance attributes to be accessible within process_llm_request
    # and potentially configured dynamically via ToolContext if ADK supports it.
    _project_id: str
    _location: str
    _db_host: str
    _db_port: str
    _db_name: str
    _db_user: str
    _db_password: str
    _embedding_model_name: str
    _embedding_dimension: int

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        db_host: Optional[str] = None,
        db_port: Optional[str] = None,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        # You might also pass top_k, report_format defaults here if desired
    ):
        """
        Initializes the GuidelineConsultant tool.

        Args:
            project_id: Google Cloud project ID. Defaults to env var GOOGLE_CLOUD_PROJECT.
            location: Vertex AI region.
            db_host: AlloyDB host.
            db_port: AlloyDB port.
            db_name: AlloyDB database name.
            db_user: AlloyDB username.
            db_password: AlloyDB password.
            embedding_model_name: Name of the embedding model to use.
            embedding_dimension: Dimension of the embeddings.
        """
        super().__init__(name=self.name, description=self.description)

        # Use provided values or fall back to environment variables/defaults
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
        self._location = location or LOCATION
        self._db_host = db_host or os.environ.get("ALLOYDB_HOST", DB_HOST)
        self._db_port = db_port or os.environ.get("ALLOYDB_PORT", DB_PORT)
        self._db_name = db_name or os.environ.get("ALLOYDB_DB_NAME", DB_NAME)
        self._db_user = db_user or os.environ.get("ALLOYDB_DB_USER", DB_USER)
        self._db_password = db_password or os.environ.get("ALLOYDB_DB_PASSWORD", DB_PASSWORD)
        self._embedding_model_name = embedding_model_name
        self._embedding_dimension = embedding_dimension

        # Basic validation for essential parameters
        if not self._project_id or self._project_id == "your-gcp-project-id":
            print("Warning: PROJECT_ID is not set. Ensure it's configured via env var or tool initialization.")
        if not self._db_host or self._db_host == "your_alloydb_host":
            print("Warning: AlloyDB host is not set. Ensure it's configured via env var or tool initialization.")
        # Add more validation as needed

    # ADK requires process_llm_request for tools that interact with the LLM or external services
    # This method is called by the agent when it decides to use this tool.
    # It receives the LLM request and the tool context.
    # The tool's job is to modify the LLM request or perform actions based on it.
    # For a tool that *retrieves* data and *then* provides it to the LLM,
    # we'll return the data as the tool's output, which the agent will then use.
    # ADK's mechanism for returning data to the LLM might involve adding it to the prompt
    # or using a specific output channel. Here, we'll return it as a dictionary.
    # The exact mechanism of how the LLM consumes this output depends on ADK's design.
    # For simplicity, we'll assume the agent can access the tool's return value.

    # Note: The original VertexAiSearchTool modified the LLM request directly.
    # For a data retrieval tool, it's more common to return the data *as the tool's output*.
    # We'll simulate this by returning a dictionary.
    # If ADK expects the tool to modify LLMRequest.config.tools, that's a different pattern.
    # Assuming here the tool *executes* and returns results.

    # The signature for process_llm_request in ADK might vary.
    # Based on the VertexAiSearchTool example, it modifies LLMRequest.
    # However, for a data retrieval tool, it's more common to return data.
    # Let's assume for this example that the tool's *return value* is what the agent uses.
    # If ADK expects modification of LLMRequest, this would need adjustment.

    # For now, we'll define a method that the agent can call directly,
    # and assume ADK handles the "tool calling" mechanism.
    # If ADK requires a specific `process_llm_request` signature that returns
    # data to the LLM, we'll adapt.

    # Let's define a method that the agent can call directly,
    # and assume ADK's tool registration handles making this callable.
    # If ADK expects a specific `process_llm_request` that modifies the LLMRequest object,
    # this structure would need to change.

    # For now, we'll make the core logic a public method that can be called.
    # The ADK integration would then map an agent's intent to calling this method.
    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query_text": types.Schema(type=types.Type.STRING, description="The query to search guidelines for"),
                    "task_context": types.Schema(type=types.Type.STRING, description="Optional context for the task"),
                    "report_format": types.Schema(type=types.Type.STRING, description="Optional report format"),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="Number of top results to return")
                },
                required=["query_text"]
            )
        )
    # async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
    #     return self.execute(**args). #kiruthika
    
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
    # Debug: show that the tool was invoked by an agent and what args were requested.
        try:
            use_db = tool_context.state.get("all_db_settings", {}).get("use_database")
        except Exception:
            use_db = None
        print(f"[Kiru GuidelineConsultantTool.run_async] called with args keys={list(args.keys())}, use_database={use_db}")
        # optionally print the query_text if present (non-secret)
        if isinstance(args, dict) and "query_text" in args:
            q_preview = args["query_text"][:200] if isinstance(args["query_text"], str) else str(args["query_text"])
            print(f"[Kiru GuidelineConsultantTool.run_async] query_text_preview={q_preview!r}")
        return self.execute(**args)
    
    
    def execute(
        self,
        query_text: str,
        task_context: Optional[str] = None,
        report_format: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes the guideline consultation and report drafting.

        Args:
            query_text (str): The natural language query describing the task or information needed.
            task_context (str, optional): Context about the agent's current task. Defaults to None.
            report_format (str, optional): Desired output format (e.g., "discovery_questionnaire_report"). Defaults to None.
            top_k (int, optional): Number of most relevant guideline snippets to retrieve. Defaults to 3.

        Returns:
            Dict[str, Any]: A dictionary containing 'guideline_snippets' and optionally 'drafted_report' or 'error'.
        """
        
        print(f"[ Kiru GuidelineConsultantTool.execute] Executing guideline search for query_text preview: {query_text[:200]!r}, top_k={top_k}")
        
        guideline_snippets = []
        drafted_report = None
        conn = None
        cur = None

        try:
            # 1. Generate embedding for the query
            query_embedding = _get_gemini_embeddings(
                [query_text],
                self._embedding_model_name,
                self._project_id,
                self._location
            )
            if not query_embedding:
                return {
                    "error": "Failed to generate embedding for the query.",
                    "guideline_snippets": [],
                    "drafted_report": None
                }
            query_embedding = query_embedding[0] # Get the single embedding list

            # 2. Connect to AlloyDB
            conn = psycopg2.connect(
                host=self._db_host,
                database=self._db_name,
                user=self._db_user,
                password=self._db_password,
                port=self._db_port
            )
            cur = conn.cursor()

            # 3. Execute similarity search query
            query_sql = """
            SELECT
                document_name,
                text_content,
                embedding <-> CAST(%s AS VECTOR) AS distance
            FROM
                product_guidelines
            ORDER BY
                embedding <-> CAST(%s AS VECTOR)
            LIMIT %s;
            """

            cur.execute(query_sql, (query_embedding, query_embedding, top_k))

            results = cur.fetchall()

            # 4. Process results into guideline_snippets
            for row in results:
                guideline_snippets.append({
                    "document_name": row[0],
                    "text_content": row[1],
                    "distance": row[2]
                })

            # 5. Draft report if requested
            if report_format:
                drafted_report = _draft_report(query_text, task_context, guideline_snippets, report_format)

            # 6. Prepare the tool's output
            tool_output = {
                "guideline_snippets": guideline_snippets,
            }
            if drafted_report:
                tool_output["drafted_report"] = drafted_report

            return tool_output

        except (Exception, psycopg2.Error) as error:
            print(f"Error in GuidelineConsultantTool.execute: {error}")
            return {
                "error": f"An error occurred: {error}",
                "guideline_snippets": guideline_snippets, # Return partial results if available
                "drafted_report": drafted_report # Return partial report if available
            }
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    # ADK's process_llm_request is typically for tools that modify the LLM request
    # or directly interact with the LLM's generation process.
    # For a data retrieval tool like this, the agent would likely call the tool's
    # 'execute' method directly and then use the returned data.
    # If ADK requires a specific `process_llm_request` that returns data to the LLM,
    # this would need to be adapted. For now, we assume the agent calls `execute`.
    # If ADK expects the tool to be callable via `tool_context.call_tool()`,
    # then the `execute` method would be the target.

    # If ADK requires a `process_llm_request` that returns data to the LLM,
    # it might look something like this (this is a hypothetical ADK pattern):
    # @override
    # async def process_llm_request(
    #     self,
    #     tool_context: ToolContext,
    #     llm_request: LlmRequest, # Assuming LlmRequest has a way to receive tool output
    # ) -> None:
    #     # This is a placeholder. The actual mechanism depends on ADK's design.
    #     # If the agent's prompt includes a placeholder for tool output,
    #     # you might populate that here.
    #     # For example, if the agent expects the tool to return a string:
    #     # tool_output_data = self.execute(
    #     #     query_text=llm_request.prompt, # Assuming prompt is the query
    #     #     task_context=getattr(llm_request, 'task_context', None), # If context is passed
    #     #     report_format=getattr(llm_request, 'report_format', None) # If format is passed
    #     # )
    #     # llm_request.tool_output = json.dumps(tool_output_data) # Or a specific field
    #     pass # Placeholder for actual ADK integration logic


# --- Example Usage (for testing the tool class directly) ---
# if __name__ == "__main__":
#     # --- IMPORTANT: Fill in your actual configuration details above ---
#     # Ensure you have run the CREATE TABLE and INSERT statements previously.

#     print("--- Testing GuidelineConsultantTool Class ---")

#     # Instantiate the tool
#     # You can override defaults here if needed, e.g.,
#     # tool = GuidelineConsultantTool(project_id="my-specific-project", db_host="my-db.example.com")
#     tool = GuidelineConsultantTool()

#     # Test Case 1: Get relevant guidelines for a security query with report
#     print("\n--- Test Case 1: Security Query with Report ---")
#     security_query = "What are the security principles?"
#     security_context = "Evaluating network security for a new service."
#     security_results = tool.execute(
#         query_text=security_query,
#         task_context=security_context,
#         report_format="discovery_questionnaire_report",
#         top_k=3
#     )
#     print(json.dumps(security_results, indent=2))

#     # Test Case 2: Get relevant guidelines for a Terraform query with report
#     print("\n--- Test Case 2: Terraform Query with Report ---")
#     terraform_query = "Does the product support Terraform?"
#     terraform_context = "Assessing infrastructure provisioning for Product Y."
#     terraform_results = tool.execute(
#         query_text=terraform_query,
#         task_context=terraform_context,
#         report_format="discovery_questionnaire_report",
#         top_k=3
#     )
#     print(json.dumps(terraform_results, indent=2))

#     # Test Case 3: Query with no report format requested
#     print("\n--- Test Case 3: No Report Format ---")
#     networking_query = "What about private networking?"
#     networking_results = tool.execute(
#         query_text=networking_query,
#         task_context="Checking network setup for a new deployment.",
#         top_k=2
#     )
#     print(json.dumps(networking_results, indent=2))

#     # Test Case 4: Query that might not yield results (or very low relevance)
#     print("\n--- Test Case 4: Less Relevant Query ---")
#     irrelevant_query = "What is the capital of France?"
#     irrelevant_results = tool.execute(
#         query_text=irrelevant_query,
#         task_context="General knowledge check.",
#         top_k=3
#     )
#     print(json.dumps(irrelevant_results, indent=2))
