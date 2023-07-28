import requests
from bs4 import BeautifulSoup

def get_image_links(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Vérifier si la requête a réussi
        soup = BeautifulSoup(response.content, 'html.parser')
        image_links = []
        for img_tag in soup.find_all('img'):
            src = img_tag.get('src')
            if src and src.startswith('http'):  # Filtrer les liens absolus des images
                image_links.append(src)
        return image_links
    except requests.exceptions.RequestException as e:
        print(f"Une erreur s'est produite : {e}")
        return []

# Remplacez l'URL ci-dessous par celle du lien que vous souhaitez analyser
url_to_scrape = 'https://permalink.weproov.com/RNiGvWuR6-XH9lY7uRDaUakVUfqAfTK4uvJHAtrBOFndj5ox6xIeXXMBkvQnp8peHI-siecZV-wwwzWrMajFKBL-P68Cm97ZWBwZ9L0IhwJW32Kbf35QZqXdgG-BQ1dd'
image_links = get_image_links(url_to_scrape)

if image_links:
    print("Liens des images trouvés sur la page :")
    for link in image_links:
        print(link)
else:
    print("Aucun lien d'image trouvé sur la page.")

