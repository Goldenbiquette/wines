import json
import os
from os import listdir
from os.path import isfile, join

from extract_information import extract_information

def parse_all_files():
    data_path = "data"
    data_formated = "data_formated"

    os.makedirs(data_formated, exist_ok=True)

    for f in listdir(data_path):
        if isfile(join(data_path, f)) and f.endswith(".jsp"):

            with open(join(data_formated, f"{f[:-4]}.json"), "w") as file:
                formatted = extract_information(join(data_path, f))

                file.write(json.dumps(formatted))
