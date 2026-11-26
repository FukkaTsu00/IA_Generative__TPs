# tp5/rag/agent_rag.py
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool

from tp5.llm import get_llm
from tp5.mcp_server.rag.indexer import get_collection, DATA

DOSSIER_IMAGES = DATA / "images"
EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

SEUIL_DISTANCE = 0.65   # distance cosinus, a regler
N_TEXTE, N_FIGURES = 4, 2

PROMPT_AGENT = """Tu es un agent RAG de depannage pour un SAV.
Procedure :
1. Appelle rechercher_passages avec une requete precise (modele, symptome, message).
2. Appelle rechercher_figures pour trouver les pictogrammes/figures lies a la question.
3. Tu peux relancer une recherche reformulee si les resultats sont insuffisants.
4. Reponds UNIQUEMENT a partir des extraits retournes, en etapes claires si besoin,
   et cite les extraits par leur numero, par exemple [1], [2].
Si rien de pertinent n'est trouve, dis-le sans inventer."""


# ---------- Image ----------

def _resoudre_image(ref: str) -> Path | None:
    """Transforme une reference (ex: img_px200_01) en fichier local cote serveur."""
    ref = Path(ref).name  # evite la traversee de repertoire (../../etc)
    for ext in ("",) + EXTENSIONS:
        chemin = DOSSIER_IMAGES / f"{ref}{ext}"
        if chemin.is_file():
            return chemin
    return None


def _decrire_image(ref: str) -> str:
    import base64, mimetypes

    chemin = _resoudre_image(ref)
    if chemin is None:
        return f"(Image '{ref}' introuvable sur le serveur.)"

    mime = mimetypes.guess_type(chemin.name)[0] or "image/png"
    data = base64.b64encode(chemin.read_bytes()).decode()
    rep = get_llm().invoke([{
        "role": "user",
        "content": [
            {"type": "text", "text": "Decris factuellement cette image d'appareil : "
                                     "voyants (couleur, clignotement), pictogramme, "
                                     "texte ou code d'erreur visible."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ],
    }])
    return rep.content


# ---------- Outils internes de l'agent RAG ----------

def _etiquette(meta: dict) -> str:
    base = f"{meta['document']}, p.{meta['page']}"
    if meta["type"] == "figure":
        return f"{base}, figure {meta['figure']} ({meta['chemin']})"
    return base


def _creer_outils(trouves: list[dict]):
    """Cree les outils lies a une liste `trouves` qui accumule les extraits
    (permet de retourner les sources sans etat global)."""
    col = get_collection()

    def _chercher(requete: str, type_: str, n: int) -> str:
        res = col.query(query_texts=[requete], n_results=n, where={"type": type_})
        lignes = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if dist > SEUIL_DISTANCE:
                continue
            trouves.append({"texte": doc, "meta": meta})
            lignes.append(f"[{len(trouves)}] ({_etiquette(meta)})\n{doc}")
        return "\n\n".join(lignes) or "Aucun resultat pertinent."

    @tool
    def rechercher_passages(requete: str) -> str:
        """Recherche des passages de texte dans les manuels. Retourne des extraits numerotes."""
        return _chercher(requete, "texte", N_TEXTE)

    @tool
    def rechercher_figures(requete: str) -> str:
        """Recherche des figures/pictogrammes dans les manuels (via leur legende).
        Retourne des extraits numerotes."""
        return _chercher(requete, "figure", N_FIGURES)

    return [rechercher_passages, rechercher_figures]


# ---------- Point d'entree appele par le serveur MCP ----------

def repondre(question: str, image: str | None = None) -> dict:
    col = get_collection()
    if col.count() == 0:
        return {"reponse": "Base documentaire vide : lancer l'indexation.", "sources": []}

    message = question
    if image:
        try:
            message += f"\n\nDescription de l'image jointe ({image}) :\n{_decrire_image(image)}"
        except Exception as e:
            message += f"\n\n(Image '{image}' non analysable : {type(e).__name__}.)"

    trouves: list[dict] = []
    agent = create_agent(
        model=get_llm(),
        tools=_creer_outils(trouves),
        system_prompt=PROMPT_AGENT,
    )
    resultat = agent.invoke({"messages": [{"role": "user", "content": message}]})

    contenu = resultat["messages"][-1].content
    if isinstance(contenu, list):  # blocs de contenu
        contenu = "".join(b.get("text", "") for b in contenu if isinstance(b, dict))

    sources = list(dict.fromkeys(_etiquette(e["meta"]) for e in trouves))  # sans doublons
    return {"reponse": contenu, "sources": sources}