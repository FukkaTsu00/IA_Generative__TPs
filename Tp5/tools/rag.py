import base64
import mimetypes
from pathlib import Path

from langchain_core.tools import tool

from tp5.llm import get_llm
from tp5.mcp_server.rag.index import get_collection

SEUIL_DISTANCE = 0.65   # distance cosinus : plus petit = plus proche. A regler.
N_TEXTE, N_FIGURES = 4, 2

PROMPT_REPONSE = """Tu es un assistant de depannage. Reponds a la question en t'appuyant
UNIQUEMENT sur les extraits fournis, de facon claire et en etapes si besoin.
Si les extraits ne permettent pas de repondre, dis-le sans inventer.
Cite les extraits utilises par leur numero, par exemple [1], [2]."""


def _url_image(image: str) -> str:
    """URL http, data URI, ou chemin local converti en data URI."""
    if image.startswith(("http://", "https://", "data:")):
        return image
    chemin = Path(image)
    mime = mimetypes.guess_type(chemin.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(chemin.read_bytes()).decode()}"


def _decrire_image(image: str) -> str:
    """Un LLM vision decrit l'ecran, pour enrichir la requete de recherche."""
    rep = get_llm().invoke([{
        "role": "user",
        "content": [
            {"type": "text", "text": "Decris factuellement cet ecran d'appareil : "
                                     "texte visible, code ou message d'erreur, voyants, menu affiche."},
            {"type": "image_url", "image_url": {"url": _url_image(image)}},
        ],
    }])
    return rep.content


def _chercher(col, requete: str, type_: str, n: int) -> list[dict]:
    res = col.query(query_texts=[requete], n_results=n, where={"type": type_})
    trouves = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= SEUIL_DISTANCE:
            trouves.append({"texte": doc, "meta": meta})
    return trouves


def _etiquette(meta: dict) -> str:
    base = f"{meta['document']}, p.{meta['page']}"
    return f"{base}, figure {meta['figure']}" if meta["type"] == "figure" else base


@tool
def rag_multimodal(question: str, image: str | None = None) -> dict:
    """Recherche une reponse a la question dans les manuels (passages et figures).
    Si une image d'ecran est fournie (URL ou chemin), elle est decrite puis utilisee
    pour enrichir la recherche. Retourne un dictionnaire avec la reponse sourcee
    et la liste des sources (document, page, figure)."""
    col = get_collection()
    if col.count() == 0:
        return {"reponse": "Base documentaire vide : lancer l'indexation.", "sources": []}

    requete = question
    if image:
        try:
            requete += "\n" + _decrire_image(image)
        except Exception:
            pass  # image illisible : on continue avec la question seule

    extraits = _chercher(col, requete, "texte", N_TEXTE) + _chercher(col, requete, "figure", N_FIGURES)
    if not extraits:
        return {"reponse": "Aucun passage pertinent trouve dans les manuels.", "sources": []}

    contexte = "\n\n".join(
        f"[{i}] ({_etiquette(e['meta'])})\n{e['texte']}" for i, e in enumerate(extraits, start=1)
    )
    reponse = get_llm().invoke([
        {"role": "system", "content": PROMPT_REPONSE},
        {"role": "user", "content": f"Question : {question}\n\nExtraits :\n{contexte}"},
    ]).content

    sources = list(dict.fromkeys(_etiquette(e["meta"]) for e in extraits))  # sans doublons
    return {"reponse": reponse, "sources": sources}