import ast
import json
import re

import requests
from bs4 import BeautifulSoup


# if True:
#     url = "https://www.idealwine.com/fr/prix-vin/4-1997-Bouteille-Bordeaux-Saint-Emilion-Grand-Cru-Chateau-Angelus-1er-Classe-A-Rouge.jsp"
#     response = requests.get(url)
#     data = response.text
# else:
#     with open("data/Cote Château Angélus 2018 Saint Émilion Grand Cru Rouge.html", "r") as file:
#         data = file.read()
#
# soup = BeautifulSoup(data, features="html.parser")

def extract1(soup):
    info = {}
    for a in soup.select("section.cote-step-2 article.info-cote-new ul.property li"):
        info[a.select_one("span").text.replace(" :", "")] = a.select_one("strong").text.strip()

    return info

def extract2(soup):
    info = {
        "nb_amateurs": soup.select_one("section.table-perf article.a3 aside p strong").text.strip().split(" ")[0],
    }

    note = soup.select_one("section.table-note article.bg-stone aside.a2 p strong")
    if note:
        info["note"] = note.text.strip()

    prix = soup.select_one("section.table-perf article.a2 li:first-child p strong")

    if prix:
        info["prix_primeur"]= prix.text.strip().split(" ")[0]

    return info

def extract3(data):
    index = data.find("/* Radar note */")

    start_config = data[index:].find("{")
    end_script = data[index:].find("</script>")

    end_config = data[index:index+end_script].rfind("}")

    config_raw = data[index+start_config:index+end_config+1]

    result = re.search(r"data: (\[[^\]]*\])", config_raw.replace("\n", ""))

    if not result:
        return {}

    datapoint = ast.literal_eval(result.group(1))

    result = re.search(r"labels: (\[[^\]]*\])", config_raw.replace("\n", ""))
    labels = ast.literal_eval(result.group(1))

    return {
        "degustation": dict(zip(labels, datapoint))
    }

def extract4(data):
    index = data.find("/* CHART JS*//* Graphique de cote */")

    filter_data = data[index:]
    result = re.search(r"data: (\[[^\]]*\])", filter_data.replace("\n", ""))
    datapoint = ast.literal_eval(result.group(1))

    result = re.search(r"labels: (\[[^\]]*\])", filter_data.replace("\n", ""))
    labels = ast.literal_eval(result.group(1))

    return {
        "cote": dict(zip(labels, datapoint))
    }


def extract_information(path):
    print(path)
    with open(path, "r",encoding="utf-8") as file:
        data = file.read()

    soup = BeautifulSoup(data, features="html.parser")

    all_information = {
        **extract1(soup), **extract2(soup), **extract3(data), **extract4(data)
    }

    return all_information


#print(extract_information("data/Cote Château Angélus 2018 Saint Émilion Grand Cru Rouge.html"))