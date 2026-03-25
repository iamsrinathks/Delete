# mcp_github_server.py
import os
import base64
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Load config from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_ORG = os.getenv("GITHUB_ORG")

if not GITHUB_TOKEN or not GITHUB_ORG:
    raise RuntimeError("Environment vars GITHUB_TOKEN and GITHUB_ORG must be set.")

GH_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

TOOLS = [
    {
        "name": "searchTerraformModules",
        "title": "Search Terraform Modules",
        "description": "Search GitHub org for Terraform modules",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keyword"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "getBackstageDocs",
        "title": "Get Backstage Docs",
        "description": "Find Markdown/YAML docs for a Backstage service",
        "inputSchema": {
            "type": "object",
            "properties": {
                "serviceName": {
                    "type": "string",
                    "description": "Backstage service name"
                }
            },
            "required": ["serviceName"]
        }
    }
]


# ---------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------

def search_terraform_modules(query: str):
    """
    Search GitHub for Terraform modules in your org.
    Only returns repos starting with 'terraform-google-'
    """
    url = "https://api.github.com/search/repositories"
    # broad search first
    q = f"{query} org:{GITHUB_ORG} in:name"
    
    resp = requests.get(url, headers=GH_HEADERS, params={"q": q, "per_page": 20})
    
    if resp.status_code != 200:
        return f"GitHub search error: {resp.text}"

    all_items = resp.json().get("items", [])

    # filter by naming convention
    modules = [
        item for item in all_items
        if item["name"].startswith("terraform-google-")
    ]

    if not modules:
        return f"No Terraform modules found matching pattern 'terraform-google-*' with query '{query}'."

    # Format results
    lines = []
    for item in modules:
        name = item["name"]
        desc = item.get("description", "") or ""
        url = item["html_url"]
        lines.append(f"- **{name}**: {desc} ({url})")

    return "\n".join(lines)


def get_backstage_docs(service_name: str):
    """
    Search ONLY inside the curation-catalogue repo for Backstage docs.
    Looks inside docs/ and catalog/ folders for .md/.yaml/.yml files.
    """
    BACKSTAGE_REPO = "curation-catalogue"

    search_url = "https://api.github.com/search/code"

    # possible folder locations
    search_paths = ["docs", "catalog", "backstage", "services"]

    # file extensions supported
    extensions = ["md", "yaml", "yml"]

    found = None

    for path in search_paths:
        for ext in extensions:
            query = (
                f"repo:{GITHUB_ORG}/{BACKSTAGE_REPO} "
                f"path:{path} "
                f"filename:{service_name}.{ext}"
            )

            resp = requests.get(search_url, headers=GH_HEADERS, params={"q": query, "per_page": 1})

            if resp.status_code == 200 and resp.json().get("items"):
                found = resp.json()["items"][0]
                break
        if found:
            break

    if not found:
        return (
            f"No Backstage documentation found for '{service_name}' "
            f"in repo '{BACKSTAGE_REPO}'."
        )

    # fetch actual content
    file_path = found["path"]
    repo_full = found["repository"]["full_name"]  # ORG/curation-catalogue
    owner, repo = repo_full.split("/", 1)

    contents_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    resp = requests.get(contents_url, headers=GH_HEADERS)

    if resp.status_code != 200:
        return f"Failed to fetch documentation: {resp.text}"

    encoded = resp.json().get("content", "")
    decoded = base64.b64decode(encoded).decode("utf-8", errors="ignore")

    return f"**File:** `{file_path}` in `{repo_full}`\n\n{decoded}"


# ---------------------------
# MCP ENDPOINT
# ---------------------------

@app.post("/")
async def mcp_entrypoint(request: Request):
    body = await request.json()

    if body.get("jsonrpc") != "2.0":
        return JSONResponse({"error": {"code": -32600, "message": "Invalid JSON-RPC format"}})

    method = body.get("method")
    req_id = body.get("id")

    # === tools/list ===
    if method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS,
                "nextCursor": None
            }
        })

    # === tools/call ===
    if method == "tools/call":
        params = body.get("params", {})
        tool = params.get("name")
        args = params.get("arguments", {})

        if tool == "searchTerraformModules":
            q = args.get("query", "")
            result = search_terraform_modules(q)

        elif tool == "getBackstageDocs":
            svc = args.get("serviceName", "")
            result = get_backstage_docs(svc)

        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool}"}
            })

        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": result}],
                "isError": False
            }
        })

    # Unknown method
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"}
    })
