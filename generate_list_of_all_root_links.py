import requests
from bs4 import BeautifulSoup


def dump_list_of_wine_type(wine_type):
    base_url = f"https://www.idealwine.com/fr/cote/{wine_type}.jsp"

    url = base_url
    with open("list_of_link_wine_"+wine_type+".txt", "w") as list_of_link_wine:
        page = 1
        while True:
            print(f"Page {page}")
            next_page = extract_page(url, list_of_link_wine)

            if not next_page:
                break

            url = next_page

            if page > 100:
                break

            page += 1


already_seen = set()


def extract_page(base_url, list_of_link_wine):
    response = requests.get(base_url)
    data = response.text

    soup = BeautifulSoup(data, features="html.parser")

    for wine in soup.select("section.section.group.wrapper.top-wine section.section.group.wrapper.post"):
        url = wine.select_one("a").get("href")

        print(url)
        if url not in already_seen:
            already_seen.add(url)
            list_of_link_wine.write(f"https://www.idealwine.com/{url}\n")

    next = soup.select_one("a.next")

    if next:
        return f"https://www.idealwine.com/{next.get('href')}"

    return None


#dump_list_of_wine_type("bordeaux")
