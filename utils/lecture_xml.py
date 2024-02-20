import xml.etree.ElementTree as ET
import urllib3

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


def recursive_print(element, indent=0):
    print("  " * indent + element.tag + ":", element.text)
    for child in element:
        recursive_print(child, indent + 1)


def recursive_get_text(element):
    # Initialize an empty string to store text content
    all_text = ""

    # Check if the node has text content
    if element.text and len(element.text) > 10:
        all_text += element.text + " "

    # Recursively traverse child nodes
    for child in element:
        all_text += recursive_get_text(child) + " "

    # Check if the node has tail content
    if element.tail and len(element.tail) > 10:
        all_text += element.tail + " "

    return all_text


def get_data(id: int, api: bool = False):
    document = {"metadata": {"abstract": ""}, "text": ""}

    if api == True:
        try:
            url = f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{id}&metadataPrefix=pmc"
            http = urllib3.PoolManager()
            response = http.request("GET", url)
            xml_content = response.data.decode("utf-8")
            root = ET.fromstring(xml_content)
            namespace = "{https://jats.nlm.nih.gov/ns/archiving/1.3/}"

            # Get the id
            document["metadata"]["id"] = id

            # Get the title
            document["metadata"]["title"] = root.find(
                ".//" + namespace + "article-title"
            ).text

            # Get the abstract
            abstracts = root.findall(".//" + namespace + "abstract")
            abstract_text = " ".join(
                [recursive_get_text(abstract) for abstract in abstracts]
            )
            document["metadata"]["abstract"] = abstract_text.strip()

            # Get the body
            body = root.findall(".//" + namespace + "body")
            body_text = " ".join([recursive_get_text(b) for b in body])
            document["text"] = body_text.strip()
            return document
        except:
            raise Exception(f"PMC{id} not found")
    else:
        try:
            xml_path = MAIN_DIR_PATH + f"./data/PMC{id}.xml"
            tree = ET.parse(xml_path)
            root = tree.getroot()

            document["metadata"]["id"] = id

            for child in root:
                if child.tag == "document":
                    for passage in child:
                        if passage.tag == "passage":
                            tag = "body"
                            text = ""
                            for infon in passage:
                                if infon.tag == "infon":
                                    if infon.text == "ABSTRACT":
                                        tag = "abstract"
                                    elif infon.text == "TITLE":
                                        tag = "title"
                                if infon.tag == "text":
                                    text = infon.text
                            if tag == "abstract":
                                document["metadata"]["abstract"] += text + " "
                            elif tag == "title":
                                document["metadata"]["title"] = text
                            else:
                                document["text"] += text + " "
            return document
        except:
            raise Exception(f"PMC{id} not found")


if __name__ == "__main__":

    for k in range(10):
        print(get_data(10500000 + k, False))
