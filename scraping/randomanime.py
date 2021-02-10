import requests
from bs4 import BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()