import xml.etree.ElementTree as ET


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
    if element.tag == 'text':
        abstract_list.append(element.text.strip())


    for child in element:
        recup_abstract(child, indent + 1, abstract_list)

    # Sinon, retourner toute la liste
    return ' '.join(abstract_list)
