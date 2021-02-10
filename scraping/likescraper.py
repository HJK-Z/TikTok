import requests
from bs4 import BeautifulSoup

url = "https://www.tiktok.com/@hjk.z?lang=en"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('strong[title="Likes"]')[0]

likes = select.contents[0]

print("I have " + likes + " likes on TikTok")
