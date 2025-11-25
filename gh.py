# mcp_github_server.py

import os
import base64
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

app = FastAPI()

# Load configuration from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")        # GitHub personal access token or app token
GITHUB_ORG = os.getenv("GITHUB_ORG")            # GitHub organization name
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")  # Expected audience for Google ID tokens

if not GITHUB_TOKEN or not GITHUB_ORG or not GOOGLE_CLIENT_ID:
    raise RuntimeError("Environment variables GITHUB_TOKEN, GITHUB_ORG, and GOOGLE_CLIENT_ID must be set.")

# Header for GitHub API requests
GH_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

# Definition of available tools for tools/list
TOOLS = [
    {
        "name": "searchTerraformModules",
        "title": "Search Terraform Modules",
        "description": "Search for Terraform modules in the GitHub organization by keyword.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for Terraform modules"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "getBackstageDocs",
        "title": "Get Backstage Documentation",
        "description": "Retrieve Markdown or YAML documentation for a Backstage service from GitHub.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "serviceName": {
                    "type": "string",
                    "description": "Name of the Backstage service"
                }
            },
            "required": ["serviceName"]
        }
    }
]

def verify_google_token(auth_header: str):
    """
    Verify Google OAuth2 bearer token (ID token) in the Authorization header.
    Raises HTTPException(401) if invalid.
    """
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ", 1)[1]
    try:
        # Verify the token's signature and audience
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
        # Optionally, you can check idinfo['hd'] for hosted domain restrictions
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

def search_terraform_modules(query: str):
    """
    Call GitHub Search API to find repositories in the org matching the query.
    Returns a string listing name, description, and URL of each repo.
    """
    # Construct GitHub search query: include org filter
    q = f"{query} org:{GITHUB_ORG} in:name,description"
    url = "https://api.github.com/search/repositories"
    resp = requests.get(url, headers=GH_HEADERS, params={"q": q, "per_page": 5})
    if resp.status_code != 200:
        return f"GitHub search error: {resp.text}"
    data = resp.json()
    items = data.get("items", [])
    if not items:
        return f"No Terraform modules found matching '{query}'."
    # Format results as bullet list
    lines = []
    for item in items:
        name = item.get("name", "")
        desc = item.get("description", "").strip() if item.get("description") else ""
        url = item.get("html_url", "")
        lines.append(f"- **{name}**: {desc} ({url})")
    return "\n".join(lines)

def get_backstage_docs(service_name: str):
    """
    Search GitHub for a file named {service_name}.md or .yaml in the org,
    and return its contents (decoded from Base64).
    """
    search_url = "https://api.github.com/search/code"
    file_content = None
    found_path = None
    found_repo = None

    # Try markdown first, then yaml
    for ext in ["md", "yaml", "yml"]:
        query = f"filename:{service_name}.{ext} org:{GITHUB_ORG}"
        resp = requests.get(search_url, headers=GH_HEADERS, params={"q": query, "per_page": 1})
        if resp.status_code == 200:
            results = resp.json().get("items", [])
            if results:
                item = results[0]
                repo = item["repository"]
                found_repo = repo["full_name"]  # "owner/repo"
                found_path = item["path"]
                break

    if not found_path:
        return f"No documentation file found for service '{service_name}'."

    # Fetch the file contents
    owner, repo = found_repo.split("/", 1)
    contents_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{found_path}"
    resp = requests.get(contents_url, headers=GH_HEADERS)
    if resp.status_code != 200:
        return f"Failed to fetch file from GitHub: {resp.text}"
    content_data = resp.json()
    # GitHub returns file content in Base64
    encoded = content_data.get("content", "")
    try:
        decoded_bytes = base64.b64decode(encoded, validate=False)
        file_content = decoded_bytes.decode('utf-8', errors='ignore')
    except Exception:
        return "Error decoding file content."
    header = f"**File:** `{found_path}` in `{owner}/{repo}`\n\n"
    return header + file_content

@app.post("/")
async def handle_rpc(request: Request):
    # Parse JSON-RPC request
    body = await request.json()
    # Validate basic structure
    if body.get("jsonrpc") != "2.0" or "method" not in body or "id" not in body:
        return JSONResponse(status_code=400, content={"error": {"code": -32600, "message": "Invalid JSON-RPC request"}})
    # OAuth2 token validation
    try:
        verify_google_token(request.headers.get("Authorization"))
    except HTTPException as auth_err:
        return JSONResponse(status_code=401, content={"jsonrpc": "2.0", "id": body["id"],
                                                     "error": {"code": -32001, "message": auth_err.detail}})
    method = body["method"]
    # tools/list: return metadata for each tool
    if method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "tools": TOOLS,
                "nextCursor": None
            }
        })

    # tools/call: execute a named tool
    if method == "tools/call":
        params = body.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "searchTerraformModules":
            query = args.get("query", "").strip()
            if not query:
                return JSONResponse({"jsonrpc": "2.0", "id": body["id"],
                                     "error": {"code": -32602, "message": "Missing 'query' parameter"}})
            output_text = search_terraform_modules(query)
        elif name == "getBackstageDocs":
            service = args.get("serviceName", "").strip()
            if not service:
                return JSONResponse({"jsonrpc": "2.0", "id": body["id"],
                                     "error": {"code": -32602, "message": "Missing 'serviceName' parameter"}})
            output_text = get_backstage_docs(service)
        else:
            # Method not found
            return JSONResponse({"jsonrpc": "2.0", "id": body["id"],
                                 "error": {"code": -32601, "message": f"Tool '{name}' not found"}})
        # Return the tool result as text content
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": output_text
                    }
                ],
                "isError": False
            }
        })

    # Unknown method
    return JSONResponse({"jsonrpc": "2.0", "id": body["id"],
                         "error": {"code": -32601, "message": f"Method '{method}' not supported"}})
