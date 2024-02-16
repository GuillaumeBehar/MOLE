import xml.etree.ElementTree as ET

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

MAIN_DIR_PATH = up(up(os.path.abspath(__file__)))


def recup_abstract(element, indent=0, abstract_list=None):
    if abstract_list is None:
        abstract_list = []

    # Vérifier si element.text est défini
    if element.tag == "text":
        abstract_list.append(element.text.strip())

    for key, value in element.attrib.items():
        if value == "section_type":
            if element.text and element.text.strip():
                if element.text.strip() == "INTRO":
                    abstract_list.append(element.text.strip())

    for child in element:
        recup_abstract(child, indent + 1, abstract_list)

    # Trouver l'index de 'INTRODUCTION' dans la liste
    introduction_index = next(
        (i for i, text in enumerate(abstract_list) if "INTRO" in text), None
    )

    # Si 'INTRODUCTION' est trouvé, retourner la partie avant 'INTRODUCTION'
    if introduction_index is not None:
        return concatener_courtes(abstract_list[:introduction_index])

    # Sinon, retourner toute la liste
    return concatener_courtes(abstract_list)


def concatener_courtes(abstract_list):
    nouvelle_liste = []

    # Parcourir la liste
    i = 0
    while i < len(abstract_list):
        # Si la longueur de la chaîne actuelle est inférieure à 20
        if len(abstract_list[i]) < 20:
            # Concaténer avec la chaîne suivante si elle existe
            if i + 1 < len(abstract_list):
                nouvelle_liste.append(abstract_list[i] + " " + abstract_list[i + 1])
                i += 2  # Passer à l'élément suivant après la concaténation
            else:
                nouvelle_liste.append(
                    abstract_list[i]
                )  # Ajouter la dernière chaîne si elle n'a pas de suivante
                i += 1  # Passer à l'élément suivant
        else:
            nouvelle_liste.append(abstract_list[i])  # Ajouter la chaîne telle quelle
            i += 1  # Passer à l'élément suivant

    return nouvelle_liste


def recup_tout(element, indent=0, abstract_list=None):
    if abstract_list is None:
        abstract_list = []

    # Vérifier si element.text est défini
    if element.tag == "text":
        abstract_list.append(element.text.strip())

    for child in element:
        recup_abstract(child, indent + 1, abstract_list)

    # Sinon, retourner toute la liste
    return " ".join(abstract_list)


if __name__ == "__main__":

    xml_path = MAIN_DIR_PATH + "./data/PMC10500001.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()

    import urllib3

    url = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:10500000&metadataPrefix=pmc"

    http = urllib3.PoolManager()
    response = http.request("GET", url)
    xml_content = response.data.decode("utf-8")

    # tree = ET.ElementTree(ET.fromstring(xml_content))
    # print(tree)
    # root = tree.getroot()
    # print(root[0])
    # toto = root.findall('GetRecord')
    # print(toto)

    # xml_bytes = ET.tostring(root, encoding="utf-8")
    # print(root[0])

    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()
    print(root)

    def print_elements(element, indent=0):
        # Print the current element with appropriate indentation
        print(" " * indent + element.tag)

        # Recursively print children elements
        for child in element:
            print_elements(child, indent + 4)

    # Afficher tous les attributs de la racine
    attributs_racine = root.attrib
    print("Attributs de la racine :")
    for attribut, valeur in attributs_racine.items():
        print(f"{attribut}: {valeur}")

    print(root.get("xmlns"))
