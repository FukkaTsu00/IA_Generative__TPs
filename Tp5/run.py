from tp5.graph import construire_graphe

app = construire_graphe()

SCENARIOS = {
    "depannage_question": {"question": "mon port hdmi ne fonctionne plus, que faire ?"},
    #"depannage_photo": {"question": "Que veut dire cette erreur ?", "image": "photo_ecran.jpg"},
    #"garantie": {"question": "Ma garantie est-elle valide ?", "numero_serie": "SN-000002"},
}

if __name__ == "__main__":
    for nom, entree in SCENARIOS.items():
        res = app.invoke(
            entree,
            config={
                "recursion_limit": 10,
                "run_name": nom,
                "tags": ["tp5", "manuel"],
            },
        )
        print(nom, "->", res.get("categorie"), "|", res.get("reponse_finale"))