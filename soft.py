# PLAN
# 1. Single Tool: `run_gcp_logging_script`.
# 2. Logic:
#    - Receive Python code string from LLM.
#    - Parse with AST (Abstract Syntax Tree) to ensure safety (no os/subprocess).
#    - Execute code in a restricted scope (`exec`).
#    - Inject a pre-authenticated `client` object into that scope.
#    - Capture the `result` variable defined by the LLM.
# 3. Security:
#    - PII Redaction on the output.
#    - Network Egress restrictions (handled by Cloud Run/GKE).
#    - IAM restrictions (Service Account).

import ast
import json
import datetime
import re
import logging
from typing import Any, Dict, Optional

# Third-party imports
try:
    from mcp.server.fastmcp import FastMCP
    from google.cloud import logging as gcp_logging
    from google.api_core import exceptions as google_exceptions
except ImportError:
    raise ImportError("Missing dependencies. Run: pip install 'mcp[cli]' google-cloud-logging")

# --- SECURITY CONFIGURATION ---

# Allowed modules that the LLM can import inside the script
ALLOWED_IMPORTS = {'json', 'datetime', 'math', 'google.cloud.logging'}

# PII Redaction (Same as before)
REDACTION_PATTERNS = {
    "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "API_KEY": r'(?i)(?:key|token|secret)[=:\s"]+([a-zA-Z0-9_\-]{16,})',
}

mcp = FastMCP("gcp-logging-repl", stateless_http=True, json_response=True)

# --- Helper Functions ---

def _redact_output(text: str) -> str:
    """Scrub PII from the final result."""
    if not text: return ""
    processed = text
    for _, pattern in REDACTION_PATTERNS.items():
        processed = re.sub(pattern, "[REDACTED]", processed)
    return processed

def _validate_code_safety(code: str):
    """
    Static Analysis: Parses code to ensure no dangerous imports or calls.
    """
    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        # 1. Check Imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names = [alias.name.split('.')[0] for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                module_names.append(node.module.split('.')[0])
            
            for mod in module_names:
                if mod not in ALLOWED_IMPORTS:
                    raise ValueError(f"Security Violation: Import of '{mod}' is not allowed.")

        # 2. Check for dangerous function calls (exec, eval, open)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['exec', 'eval', 'open', 'compile']:
                    raise ValueError(f"Security Violation: Function '{node.func.id}' is forbidden.")

def _get_safe_globals(project_id: Optional[str] = None):
    """
    Creates the restricted execution environment.
    We inject the 'client' so the LLM doesn't need to authenticate.
    """
    try:
        # Initialize Client
        client = gcp_logging.Client(project=project_id)
    except Exception as e:
        raise RuntimeError(f"Failed to init GCP Client: {e}")

    return {
        "client": client,  # <--- The Magic: LLM uses this directly
        "datetime": datetime,
        "json": json,
        "print": print, # Allowed for debugging, but we capture 'result' variable
        "__builtins__": {
            "list": list, "dict": dict, "str": str, "int": int, 
            "len": len, "range": range, "enumerate": enumerate,
            "filter": filter, "map": map, "sorted": sorted,
            "min": min, "max": max, "sum": sum, "bool": bool
        }
    }

# --- The Single Tool ---

@mcp.tool()
def run_gcp_logging_script(script: str, project_id: Optional[str] = None) -> str:
    """
    Executes a Python script to query Google Cloud Logging.
    
    PRE-DEFINED VARIABLES:
    - `client`: An authenticated google.cloud.logging.Client object.
    
    INSTRUCTIONS:
    1. Use the `client` object to list_entries, list_logs, etc.
    2. Perform filtering and data extraction in Python.
    3. Assign your final answer (list or dict) to a variable named `result`.
    4. Do NOT attempt to import os, sys, or subprocess.
    
    EXAMPLE:
    ```python
    # Find errors in the last hour
    import datetime
    one_hour_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
    
    entries = client.list_entries(
        filter_='severity="ERROR"',
        order_by="timestamp desc",
        max_results=5
    )
    
    # Extract just the messages
    result = [e.payload for e in entries]
    ```
    """
    # 1. Static Safety Check
    try:
        _validate_code_safety(script)
    except ValueError as e:
        return f"Code Safety Error: {str(e)}"
    except SyntaxError as e:
        return f"Syntax Error: {str(e)}"

    # 2. Prepare Scope
    try:
        safe_globals = _get_safe_globals(project_id)
        local_scope = {}
    except Exception as e:
        return f"Initialization Error: {str(e)}"

    # 3. Execute
    try:
        exec(script, safe_globals, local_scope)
    except Exception as e:
        # Return the error so the LLM can fix its own code
        return f"Runtime Error: {str(e)}"

    # 4. Extract Result
    if "result" not in local_scope:
        return "Error: The script executed successfully, but did not define a 'result' variable."
    
    output_data = local_scope["result"]

    # 5. Serialize and Redact
    try:
        # Handle non-JSON serializable objects (like GCP LogEntry objects)
        # We rely on a simple default serializer or expect the LLM to convert to dicts
        json_output = json.dumps(output_data, default=str, indent=2)
        return _redact_output(json_output)
    except Exception as e:
        return f"Serialization Error: {str(e)}. Please ensure 'result' contains basic types (dict, list, str)."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
