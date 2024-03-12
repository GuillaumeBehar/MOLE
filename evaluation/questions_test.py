# IMPORTATION
from utils.lecture_xml import pmid_to_pmcid
import json
import random
import os

# Obtenez le chemin absolu du répertoire du script Python
repertoire_script = os.path.dirname(os.path.abspath(__file__))

# Construisez le chemin complet pour le fichier JSON
chemin_fichier_json = os.path.join(repertoire_script, "pubID_list.json")

with open(chemin_fichier_json, "r") as file:
    # Charger les données JSON depuis le fichier
    data = json.load(file)


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


# Construction de la liste des choix, où tout les articles sont dans PMC

# liste_id = data['pubid_list']
# choix = []
# i = 0

# while len(choix) < 50:
#     article = random.choice(liste_id)
#     liste_id.remove(article)
#     if (pmid_to_pmcid(article)[0] != 'P'):
#         choix.append(article)
#     i += 1
#     if i == 10:
#         print(len(choix))
#         i = 0

# print(choix)

# Retrouver les id de PMC à partir de ceux de PM

id_test_pm = [22315282, 22417809, 26573152, 23587438, 25485089, 23453038, 24887092, 23194649, 21854558, 20413710, 21139995, 27366677, 24485404, 17114189, 19728867, 22739735, 19547701, 10486375, 22682150, 26663142, 22696140, 27648007, 21092132, 26912052,
              25409881, 23472169, 24156704, 23008026, 16517573, 27160188, 25617223, 23035975, 16503976, 19885211, 15538945, 20011543, 23181353, 26264094, 27350760, 19417601, 23029280, 22002811, 21039601, 27136446, 24217166, 20548040, 24321741, 22679365, 23369247, 22909162]
# pmc = []

# for id in id_test_pm:
#     pmc.append(pmid_to_pmcid(id))

# print(pmc)

id_test_pmc = ['5238935', '3446379', '4647573', '3637408', '4258008', '10282314', '4041655', '3556302', '3176244', '2981024', '2997332', '4916132', '3922640', '1954639', '2745358', '3560784', '2695784', '1727662', '3407485', '5193136', '3373004', '5020395', '2994813', '4765141',
               '4667864', '3589265', '3874777', '3528904', '2111217', '4862165', '4494983', '3552177', '1395302', '2769721', '534803', '2789413', '3538519', '5096553', '4902146', '2727066', '3448698', '5823004', '3058253', '4895376', '3748456', '2905894', '4029300', '3367495', '3552685', '3428737']

id_test_pmc = [int(num) for num in id_test_pmc]
