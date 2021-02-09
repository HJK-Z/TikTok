import requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
