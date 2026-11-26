import re
import sqlite3
from datetime import date
from pathlib import Path

from langchain_core.tools import tool

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "registre.db"
FORMAT_SERIE = re.compile(r"^SN-\d{6}$")  # adapte a ton vrai format


def _connexion_lecture_seule() -> sqlite3.Connection:
    # mode=ro : toute tentative d'ecriture leve une erreur SQLite
    return sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)


@tool
def consulter_garantie(numero_serie: str) -> dict:
    """Consulte le registre de garantie (lecture seule) a partir d'un numero de serie
    au format SN-123456. Retourne un dictionnaire avec le statut (active, expiree,
    introuvable, format_invalide ou erreur_registre), le produit, la date de fin
    et la couverture."""
    numero = numero_serie.strip().upper()

    if not FORMAT_SERIE.match(numero):
        return {
            "statut": "format_invalide",
            "numero_serie": numero_serie,
            "detail": "Le numero de serie doit etre au format SN-123456.",
        }

    try:
        with _connexion_lecture_seule() as con:
            ligne = con.execute(
                "SELECT produit, date_fin, couverture FROM garanties WHERE numero_serie = ?",
                (numero,),
            ).fetchone()
    except sqlite3.Error:
        return {"statut": "erreur_registre", "numero_serie": numero,
                "detail": "Registre indisponible, reessayer plus tard."}

    if ligne is None:
        return {"statut": "introuvable", "numero_serie": numero}

    produit, date_fin, couverture = ligne
    statut = "active" if date.fromisoformat(date_fin) >= date.today() else "expiree"
    return {
        "statut": statut,
        "numero_serie": numero,
        "produit": produit,
        "date_fin": date_fin,
        "couverture": couverture,
    }