from mcp.server.fastmcp import FastMCP
from rag.agent_rag import repondre   # voir point b)

mcp = FastMCP("manuels", host="127.0.0.1", port=8000)

@mcp.tool()
def rag_multimodal(question: str, image: str | None = None) -> dict:
    """Recherche dans les manuels (passages et figures) et retourne une
    reponse sourcee. `image` est une reference (ex: img_px200_01)."""
    return repondre(question, image)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")   # http://127.0.0.1:8000/mcp