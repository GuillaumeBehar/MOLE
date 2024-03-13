# IMPORTATION
import json
import random
import os


def get_main_dir(depth: int = 0):  # nopep8
    """Get the main directory of the project."""
    import os
    import sys
    from os.path import dirname as up
    main_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(depth):
        sys.path.append(up(main_dir))
        main_dir = up(main_dir)
    return main_dir


MAIN_DIR_PATH = get_main_dir(1)  # nopep8

from utils.lecture_xml import pmid_to_pmcid, get_data

# Obtenez le chemin absolu du répertoire du script Python
repertoire_script = os.path.dirname(os.path.abspath(__file__))

# Construisez le chemin complet pour le fichier JSON
chemin_fichier_json = os.path.join(repertoire_script, "pubID_list.json")

with open(chemin_fichier_json, "r") as file:
    # Charger les données JSON depuis le fichier
    data = json.load(file)

# Construction de la liste des choix, où tout les articles sont dans PMC et sont récupérables

# liste_id = data['pubid_list']
# choix = []
# i = 0
# bon = 0

# while len(choix) < 50:
#     article = random.choice(liste_id)
#     print("PMID :", article)
#     liste_id.remove(article)
#     if ((id := pmid_to_pmcid(article)) != None):
#         bon += 1
#         print("PMCID :", id)
#         data = get_data(id, api=True, show=True)
#         if (data != None):
#             choix.append(article)
#     i += 1
#     if i == 10:
#         print(bon)
#         print(len(choix))
#         i = 0

# print(choix)

# Retrouver les id de PMC à partir de ceux de PM

id_test_pm = [22022005, 21639875, 20852029, 25844699, 25379003, 26817669, 22563393, 20659337, 27643685, 26693009, 19888227, 20878146, 26337974, 23355459, 25495800, 22640485, 24059973, 24409166, 22909062, 24447369, 19180231, 22569336, 23231769, 23557178,
              21617180, 24958351, 27500275, 19933996, 24330812, 26227965, 27574676, 27473420, 22709483, 26289293, 23949151, 27336604, 26460750, 18575589, 24884655, 18493326, 23015864, 26175775, 26418562, 26418133, 21696606, 25036418, 24847033, 26295946, 27595989, 21981946]

id_test_pmc = [3195132, 3218944, 2992784, 4467582, 4213846, 4729033, 3338521, 2918533, 5028067, 4676151, 2788266, 3020317, 4558969, 3721168, 4279816, 3511263, 3851161, 3873600, 3441385, 3903047, 2628736, 3446336, 3551651, 3651302,
               3121440, 4230643, 4975560, 2828649, 4029378, 4521469, 4964010, 4966758, 3416570, 4895690, 3776978, 4947703, 4603675, 2424173, 4041915, 2375056, 3445072, 4499589, 4587925, 4587927, 3144447, 4103805, 4309068, 4546589, 5011830, 3199253]
