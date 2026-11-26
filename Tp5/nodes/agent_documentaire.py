# tp5/nodes/agent_documentaire.py
import json
from functools import lru_cache
from typing import Literal

from langchain.agents import create_agent
from langgraph.types import Command
from langchain_mcp_adapters.client import MultiServerMCPClient
from tp5.llm import get_llm
from tp5.state import EtatSAV
from tp5.tools.rag import rag_multimodal


PROMPT =(
            "Tu es l'agent de depannage d'un SAV. Utilise l'outil rag_multimodal "
            "pour chercher la reponse dans les manuels, puis reponds de facon claire "
            "en t'appuyant uniquement sur le resultat de l'outil. "
            "Si l'outil ne trouve rien de pertinent, dis-le sans inventer."
        )


client = MultiServerMCPClient({
    "manuels": {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable_http"}
})
_react = None

async def _get_react():
    global _react
    if _react is None:
        tools = await client.get_tools()
        _react = create_agent(model=get_llm(), tools=tools, system_prompt=PROMPT)
    return _react


async def agent_documentaire(etat: EtatSAV) -> Command[Literal["orchestrateur"]]:
    msg = etat["question"]
    if etat.get("image"):
        msg += f"\n(Photo de l'ecran jointe : {etat['image']})"

    react = await _get_react()
    resultat =await react.invoke({"messages": [{"role": "user", "content": msg}]})
    messages = resultat["messages"]

    # Récupère les sources depuis le retour de l'outil (ToolMessage)
    sources: list[str] = []
    for m in messages:
        if m.type == "tool":
            try:
                sources = json.loads(m.content).get("sources", [])
            except (ValueError, AttributeError, TypeError):
                pass

    return Command(
        update={"reponse_finale": messages[-1].content, "sources": sources},
        goto="orchestrateur",
    )