# PLAN
# 1. Security Imports: re (regex) for PII scrubbing.
# 2. Define Redaction Logic: Create patterns for Emails, IP addresses, and generic API Keys.
# 3. Hardened Tools:
#    - Apply redaction to all log payloads before returning to the LLM.
#    - Strict type checking and regex validation for Project IDs (prevent command injection).
#    - Generic error messages (prevent stack trace leakage).
# 4. Server Config: Keep stateless_http=True for security (no session storage).

import json
import datetime
import re
import logging
from typing import List, Optional, Any, Dict

# Third-party imports
try:
    from mcp.server.fastmcp import FastMCP
    from google.cloud import logging as gcp_logging
    from google.api_core import exceptions as google_exceptions
    # Resource manager is optional but useful
    from google.cloud import resourcemanager_v3
    HAS_RESOURCE_MANAGER = True
except ImportError:
    # Fail safe: If dependencies are missing, we crash early rather than running insecurely
    raise ImportError("Critical dependencies missing. Run: pip install 'mcp[cli]' google-cloud-logging google-cloud-resource-manager uvicorn")

# --- SECURITY CONFIGURATION ---

# 1. PII Redaction Patterns
# We mask these patterns to prevent sensitive data from entering the LLM context.
REDACTION_PATTERNS = {
    "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "IPV4": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "API_KEY": r'(?i)(?:key|token|secret|password|auth)[=:\s"]+([a-zA-Z0-9_\-]{16,})',
    "CREDIT_CARD": r'\b(?:\d[ -]*?){13,16}\b'
}

# 2. Input Validation Patterns
# Prevent shell injection or malformed project IDs
PROJECT_ID_PATTERN = re.compile(r'^[a-z][a-z0-9-]{5,29}$')

# Initialize Server
mcp = FastMCP("gcp-logging-server", stateless_http=True, json_response=True)

# --- Helper Functions ---

def _validate_project_id(project_id: Optional[str]) -> None:
    """
    Security: Validates project_id format to prevent injection attacks.
    """
    if project_id and not PROJECT_ID_PATTERN.match(project_id):
        raise ValueError("Invalid Project ID format. Must be 6-30 lowercase letters, digits, or hyphens.")

def _redact_text(text: str) -> str:
    """
    Security: Scrubs PII and Secrets from text strings.
    """
    if not text: return ""
    
    processed_text = text
    for label, pattern in REDACTION_PATTERNS.items():
        # Replace found patterns with [REDACTED_LABEL]
        processed_text = re.sub(pattern, f"[REDACTED_{label}]", processed_text)
    return processed_text

def _redact_payload(payload: Any) -> Any:
    """
    Security: Recursively redact PII from JSON/Dict payloads.
    """
    if isinstance(payload, str):
        return _redact_text(payload)
    elif isinstance(payload, dict):
        return {k: _redact_payload(v) for k, v in payload.items()}
    elif isinstance(payload, list):
        return [_redact_payload(item) for item in payload]
    else:
        return payload

def _get_logging_client(project_id: Optional[str] = None) -> gcp_logging.Client:
    """
    Securely creates a Google Cloud Logging client using ADC.
    """
    _validate_project_id(project_id)
    try:
        return gcp_logging.Client(project=project_id)
    except google_exceptions.GoogleAPICallError as e:
        # Log the real error internally, return generic error to user
        logging.error(f"GCP Client Init Failed: {e}")
        raise RuntimeError("Failed to initialize GCP Client. Check internal logs.")

def _format_entry(entry: Any) -> Dict[str, Any]:
    """
    Sanitizes, Formats, and Redacts a GCP LogEntry.
    """
    payload = entry.payload
    payload_data = payload if isinstance(payload, dict) else {"message": str(payload)}

    # SECURITY: Apply Redaction before returning data
    safe_payload = _redact_payload(payload_data)

    return {
        "insert_id": entry.insert_id,
        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
        "severity": entry.severity,
        "resource_type": entry.resource.type if entry.resource else "unknown",
        "log_name": entry.log_name,
        "payload": safe_payload, # <--- REDACTED DATA
        # Labels often contain environment metadata, redact them too just in case
        "labels": _redact_payload(entry.labels) if entry.labels else {},
    }

def _json_serializer(obj: Any) -> Any:
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    return str(obj)

# --- Tools ---

@mcp.tool()
def search_logs(
    filter_expression: str,
    project_id: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    Search logs with strict PII redaction.
    Args:
        filter_expression: GCP filter string.
        project_id: GCP Project ID.
        limit: Max logs (Hard cap: 20 to prevent context flooding).
    """
    # Security: Enforce strict limit to prevent Denial of Service (DoS) on the LLM context
    if limit > 20:
        return "Error: Limit exceeds maximum allowed (20) for security reasons."

    try:
        client = _get_logging_client(project_id)
        
        # Security: We do not allow arbitrary sorting to prevent potential query performance attacks
        # We enforce timestamp descending.
        entries = client.list_entries(
            filter_=filter_expression,
            order_by="timestamp desc",
            max_results=limit
        )
        
        results = [_format_entry(entry) for entry in entries]
        
        if not results:
            return "No logs found matching the criteria."
            
        return json.dumps(results, default=_json_serializer, indent=2)

    except ValueError as ve:
        return f"Validation Error: {str(ve)}"
    except Exception as e:
        # Security: Do not return raw stack traces
        logging.error(f"Search Logs Error: {str(e)}")
        return "An internal error occurred while searching logs."

@mcp.tool()
def list_projects() -> str:
    """Lists accessible projects."""
    if not HAS_RESOURCE_MANAGER:
        return "Tool unavailable."
    
    try:
        client = resourcemanager_v3.ProjectsClient()
        request = resourcemanager_v3.ListProjectsRequest()
        page_result = client.list_projects(request=request)
        
        projects = []
        for project in page_result:
            # Only return Active projects
            if project.state.name == "ACTIVE":
                projects.append({"project_id": project.project_id})
            if len(projects) >= 20: break
            
        return json.dumps(projects, indent=2)
    except Exception:
        return "Error listing projects."

if __name__ == "__main__":
    # Security: Bind to 0.0.0.0 is necessary for containerization, 
    # but Ingress/Firewall rules must restrict access.
    mcp.run(transport="streamable-http")
