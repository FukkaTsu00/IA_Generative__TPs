import base64
import hashlib
from pathlib import Path

import chromadb
from pypdf import PdfReader

from llm import get_llm

DATA = Path(__file__).resolve().parent.parent / "data"
MANUELS = DATA / "manuels"
FIGURES = DATA / "figures"
CHROMA_DIR = DATA / "chroma"
TAILLE_MIN = 64


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection("manuels", metadata={"hnsw:space": "cosine"})


def hash_pdf(chemin: Path) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest()[:12]


def decouper(texte: str, taille: int = 800, chevauchement: int = 100) -> list[str]:
    texte = " ".join(texte.split())
    return [texte[i:i + taille] for i in range(0, len(texte), taille - chevauchement)]


def decrire_figure(chemin: Path) -> str:
    mime = "image/png" if chemin.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(chemin.read_bytes()).decode()
    return get_llm().invoke([{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "Image d'un manuel d'appareil electronique. Decris-la factuellement "
                "en francais : pictogramme (forme, couleur), voyants et leur etat, "
                "texte ou code visible, ce que la figure illustre.")},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ],
    }]).content


def indexer():
    col = get_collection()
    FIGURES.mkdir(parents=True, exist_ok=True)
    ids, docs, metas = [], [], []

    for pdf in sorted(MANUELS.glob("*.pdf")):
        doc_id = hash_pdf(pdf)
        if col.get(where={"doc_id": doc_id}, limit=1)["ids"]:
            print(f"{pdf.name} deja indexe.")
            continue

        for p, page in enumerate(PdfReader(pdf).pages, start=1):
            meta = {"doc_id": doc_id, "document": pdf.name, "page": p}

            for i, morceau in enumerate(decouper(page.extract_text() or "")):
                ids.append(f"{doc_id}-p{p}-c{i}")
                docs.append(morceau)
                metas.append({**meta, "type": "texte"})

            for k, img in enumerate(page.images):
                if img.image.width < TAILLE_MIN or img.image.height < TAILLE_MIN:
                    continue
                fig_id = f"{doc_id}-p{p}-img{k}"
                chemin = FIGURES / f"{fig_id}.png"
                img.image.convert("RGB").save(chemin)
                chemin.write_bytes(img.data)

                ids.append(fig_id)
                docs.append(decrire_figure(chemin))
                metas.append({**meta, "type": "figure"})

    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"{len(ids)} elements indexes ({col.count()} au total).")


def supprimer_pdf(doc_id: str) -> None:
    get_collection().delete(where={"doc_id": doc_id})
    for f in FIGURES.glob(f"{doc_id}-*"):
        f.unlink()


def lister_pdf() -> dict[str, str]:
    return {m["doc_id"]: m["document"] for m in get_collection().get(include=["metadatas"])["metadatas"]}


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "indexer"
    if cmd == "indexer":
        indexer()
    elif cmd == "supprimer":
        supprimer_pdf(sys.argv[2])
    elif cmd == "lister":
        print(lister_pdf())