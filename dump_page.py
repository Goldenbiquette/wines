from bs4 import BeautifulSoup
import requests
import os

force_redownload = False


def dump_root_page(url):
    print(f"process root page {url}")
    response = requests.get(url)
    data = response.text

    urlparts = url.split("/")
    with open(f"data/{urlparts[-1]}", "w",encoding="utf-8") as file:
        file.write(data)

    soup = BeautifulSoup(data, features="html.parser")

    for link in soup.select("section.cote-step-1 article.vintage-table li a"):
        url = link.get("href")
        print(url)

        if not force_redownload and os.path.exists(url):
            continue

        response = requests.get("https://www.idealwine.com"+url)

        urlparts = url.split("/")
        with open(f"data/{urlparts[-1]}", "w",encoding="utf-8") as file:
            file.write(response.text)


def dump_all_page(file_urls):
    os.makedirs("data", exist_ok=True)
    with open(file_urls) as file:
        for url in file.readlines():
            dump_root_page(url.strip())


# dump_all_page("list_of_link_wine.txt")