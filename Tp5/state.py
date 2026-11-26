from typing import Literal, Optional, TypedDict

class EtatSAV(TypedDict, total=False):
    question: str
    image: Optional[str]          # base64, URL ou chemin
    numero_serie: Optional[str]
    categorie: Optional[Literal["depannage", "garantie"]]
    reponse_finale: Optional[str]
    sources: list[str]