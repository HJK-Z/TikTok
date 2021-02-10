from requests_html import HTMLSession
from bs4 import BeautifulSoup
import numpy as np
import cv2

url = "https://tiktok.com"
session = HTMLSession()
raw = session.get(url)
raw.html.render(sleep=5)

soup = BeautifulSoup(raw.html.raw_html, "html.parser")


wraps = soup.select('span[class="lazyload-wrapper"]')
tags = set([])
for wrap in wraps:
    print(wrap.prettify())
    caption = wrap.find('div[class="tt-video-meta-caption"]')
    for capchild in caption.children:
        if capchild.name=='a':
            txt = capchild.select('strong')[0].content
            if txt[0]=='#':
                tags.add(txt)
                print(txt)


print()
print(list(tags))


