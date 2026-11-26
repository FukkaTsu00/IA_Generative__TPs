# tp5/nodes/orchestrateur.py
from typing import Literal

from langgraph.graph import END
from pydantic import BaseModel

from tp5.llm import get_llm
from tp5.state import EtatSAV

PROMPT = """Tu es l'orchestrateur du SAV d'un distributeur d'appareils electroniques.
Classe la demande dans UNE seule categorie :
- "documentation" : demande d'information, de documentation, de support (avec ou sans photo d'ecran).
- "garantie" : verification de la garantie a partir d'un numero de serie."""


class Decision(BaseModel):
    categorie: Literal["documentation", "garantie"]


def orchestrateur(etat: EtatSAV) -> dict:
    # Retour d'un agent : rien a decider
    if etat.get("categorie"):
        return {}

    llm = get_llm().with_structured_output(Decision, method="function_calling")
    contexte = (
        f"Question : {etat['question']}\n"
        f"Numero de serie fourni : {'oui' if etat.get('numero_serie') else 'non'}\n"
        f"Photo jointe : {'oui' if etat.get('image') else 'non'}"
    )
    decision = llm.invoke(
        [{"role": "system", "content": PROMPT}, {"role": "user", "content": contexte}]
    )
    return {"categorie": decision.categorie}


def router(etat: EtatSAV) -> str:
    if etat.get("reponse_finale"):
        return END
    return "agent_garantie" if etat["categorie"] == "garantie" else "agent_documentaire"