# tp5/graph.py
from langgraph.graph import StateGraph, START, END

from tp5.state import EtatSAV
from tp5.nodes.orchestrateur import orchestrateur, router
from tp5.nodes.agent_documentaire import agent_documentaire
from tp5.nodes.agent_garantie import agent_garantie


def construire_graphe():
    g = StateGraph(EtatSAV)
    g.add_node("orchestrateur", orchestrateur)
    g.add_node("agent_documentaire", agent_documentaire)
    g.add_node("agent_garantie", agent_garantie)

    g.add_edge(START, "orchestrateur")
    g.add_conditional_edges(
        "orchestrateur",
        router,
        ["agent_documentaire", "agent_garantie", END],
    )
    # Retour obligatoire à l'orchestrateur après chaque agent
    g.add_edge("agent_documentaire", "orchestrateur")
    g.add_edge("agent_garantie", "orchestrateur")
    return g.compile()


if __name__ == "__main__":
    app = construire_graphe()
    config = {"recursion_limit": 10}  # protection contre les boucles
    #res = app.invoke({"question": "Ma garantie est-elle valide ?", "numero_serie": "SN-123456"}, config)
    #print(res)