from typing import Literal
from langchain.agents import create_agent
from langgraph.types import Command
from tp5.llm import get_llm
from tp5.state import EtatSAV
from tp5.tools.garantie import consulter_garantie

react = create_agent(
    model=get_llm(),
    tools=[consulter_garantie],
    system_prompt=(
        "Tu es l'agent garantie d'un SAV. Utilise l'outil consulter_garantie "
        "pour verifier la garantie, puis reponds de facon concise."
    ),
)


def agent_garantie(etat: EtatSAV) -> Command[Literal["orchestrateur"]]:
    msg = f"{etat['question']} (numero de serie : {etat.get('numero_serie')})"
    resultat = react.invoke({"messages": [{"role": "user", "content": msg}]})
    reponse = resultat["messages"][-1].content
    return Command(update={"reponse_finale": reponse}, goto="orchestrateur")