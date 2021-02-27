import PIL.Image

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex
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


import os, time
from PIL import Image, ImageSequence

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str

def main():
    path = os.getcwd()
    fpath = path + "/media/nya.gif"

    gif = Image.open(fpath)

    ind = 1
    for frame in ImageSequence.Iterator(gif):
        image = resize(frame, 200)
        image = to_greyscale(image)

        ascii_str = pixel_to_ascii(image)
        img_width = image.width
        ascii_str_len = len(ascii_str)
        ascii_img = ""

        for i in range(0, ascii_str_len, img_width):
            ascii_img += ascii_str[i : i + img_width] + "\n"

        with open(path + "/media/nya/gif" + str(ind) + ".txt", "w") as f:
            f.write(ascii_img)
        
        ind += 1


main()

import os
import librosa
import numpy as np
import pygame
import time
import cv2


def normalize(val, omn, omx):
    return (val - omn) import librosa
import numpy as np
import pygame


def clamp(min_value, max_value, value):
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


class AudioBar:
    def __init__(
        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.height = clamp(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (self.x, self.y + self.max_height - self.height, self.width, self.height),
        )


filename = "audio/somebs.wav"
time_series, sample_rate = librosa.load(filename)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitu
def fs()
    return Truede_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]


pygame.init()
pygame.display.set_caption('uhhh yeah')
infoObject = pygame.display.Info()

screen_w = int(infoObject.current_w / 2.5)
screen_h = int(infoObject.current_w / 4)
screen = pygame.display.set_mode([screen_w, screen_h])

bars = []
frequencies = np.arange(100, 8000, 100)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

mx = np.max(frequencies)
mn = np.min(frequencies)
rng = mx-mn

for c in frequencies:
    color = 255*(mx-c)/rng
    bars.append(AudioBar(x, 0, c, (0, 0, color), max_height=400, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(filename)
pygame.mixer.music.play(0)

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        b.update(deltaTime, get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq))
        b.render(screen)

    pygame.display.flip()

pygame.quit()
/ (omx - omn)


def getColorNorm(height, mn, mx):
    return (0, 0, np.clip(255 * normalize(height, mn, mx), 0, 255))


class AudioBar:
    def __init__(
        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(selfport BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Image

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plport BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):ep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
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


import os, time
from PIL import Image, ImageSequence

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = os.getcwd()
    fpath = path + "/media/nya.gif"

    gif = Image.open(fpath)

    ind = 1
    for frame in ImageSequence.Iterator(gif):
        image = resize(frame, 200)
        image = to_greyscale(image)

        ascii_str = pixel_to_ascii(image)
        img_width = image.width
        ascii_str_len = len(ascii_str)
        ascii_img = ""

        for i in range(0, ascii_str_len, img_width):
            ascii_img += ascii_str[i : i + img_width] + "\n"

        with open(path + "/media/nya/gif" + str(ind) + ".txt", "w") as f:
            f.write(ascii_img)
        
        ind += 1


main()

import os
import librosa
import numpy as np
import pygame
import time
import cv2


def normalize(val, omn, omx):
    return (val - omn) / (omx - omn)


def getColorNorm(height, mn, mx):
    return (0, 0, np.clip(255 * normalize(height, mn, mx), 0, 255))


class AudioBar:
    def __init__(
        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(tarimport requests
from bs4 import BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, port BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")
port BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
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


import os, time
from PIL import Image, ImageSequence

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def 
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Image

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
soup = BeautifulSoup(raw.html.raw_html, "html.parser")


wraps = soup.select('span[class="lazyload-wrapper"]')
tags = set([])
for wrapimport PIL.Image

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
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


import os, time
from PIL import Image, ImageSequence

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = os.getcwd()
    fpath = path + "/media/nya.gif"

    gif = Image.open(fpath)

    ind = 1
    for frame in ImageSequence.Iterator(gif):
        image = resize(frame, 200)
        image = to_greyscale(image)

        ascii_str = pixel_to_ascii(image)
        img_width = image.width
        ascii_str_len = len(ascii_str)
        ascii_img = ""

        for i in range(0, ascii_str_len, img_width):
            ascii_img += ascii_str[i : i + img_width] + "\n"

        with open(path + "/media/nya/gif" + str(ind) + ".txt", "w") as f:
            f.write(ascii_img)
        
        ind += 1


main()

import os
import librosa
import numpy as np
import pygame
import time
import cv2


def normalize(val, omn, omx):
    return (val - omn) import librosa
import numpy as np
import pygame


def clamp(min_value, max_value, value):
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


class AudioBar:
    def __init__(
        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.height = clamp(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (self.x, self.y + self.max_height - self.height, self.width, self.height),
        )


filename = "audio/somebs.wav"
time_series, sample_rate = librosa.load(filename)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]


pygame.init()
pygame.display.set_caption('uhhh yeah')
infoObject = pygame.display.Info()

screen_w = int(infoObject.current_w / 2.5)
screen_h = int(infoObject.current_w / 4)
screen = pygame.display.set_mode([screen_w, screen_h])

bars = []
frequencies = np.arange(100, 8000, 100)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

mx = np.max(frequencies)
mn = np.min(frequencies)
rng = mx-mn

for c in frequencies:
    color = 255*(mx-c)/rng
    bars.append(AudioBar(x, 0, c, (0, 0, color), max_height=400, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(filename)
pygame.mixer.music.play(0)

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        b.update(deltaTime, get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq))
        b.render(screen)

    pygame.display.flip()

pygame.quit()
/ (omx - omn)


def getColorNorm(height, mn, mx):
    return (0, 0, np.clip(255 * normalize(height, mn, mx), 0, 255))


class AudioBar:
    def __init__(
        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(selfport BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Image

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plport BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):ep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
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


import os, time
from PIL import Image, ImageSequence

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = os.getcwd()
    fpath = path + "/media/nya.gif"

    gif = Image.open(fpath)

    ind = 1
    for frame in ImageSequence.Iterator(gif):
        image = resize(frame, 200)
        image = to_greyscale(image)

        ascii_str = pixel_to_ascii(image)
        img_width = image.width
        ascii_str_len = len(ascii_str)
        ascii_img = ""

        for i in range(0, ascii_str_len, img_width):
            ascii_img += ascii_str[i : i + img_width] + "\n"

        with open(path + "/media/nya/gif" + str(ind) + ".txt", "w") as f:
            f.write(ascii_img)
        
        ind += 1


main()

import os
import librosa
import numpy as np
import pygame
import time
import cv2


def normalize(val, omn, omx):
    return (val - omn) / (omx - omn)


def getColorNorm(height, mn, mx):
    return (0, 0, np.clip(255 * normalize(height, mn, mx), 0, 255))


class AudioBar:
    def __init__(
        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(tarimport requests
from bs4 import BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, port BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")
port BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)
import PIL.Image

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
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


import os, time
from PIL import Image, ImageSequence

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = os.getcwd()
    fpath = path + "/media/nya.gif"

    gif = Image.open(fpath)

    ind = 1
    for frame in ImageSequence.Iterator(gif):
        image = resize(frame, 200)
        image = to_greyscale(image)

        ascii_str = pixel_to_ascii(image)
        img_width = image.width
        ascii_str_len = len(ascii_str)
        ascii_img = ""

        for i in range(0, ascii_str_len, img_width):
            ascii_img += ascii_str[i : i + img_width] + "\n"

        with open(path + "/media/nya/gif" + str(ind) + ".txt", "w") as f:
            f.write(ascii_img)
        
        ind += 1


main()

import os
import librosa
import numpy as np
import pygame
import time
import cv2


def normalize(val, omn, omx):
    return (val - omn) import librosa
import numpy as np
import pygame


def clamp(min_value, max_value, value):
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


class AudioBar:
    def __init__(
        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.height = clamp(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (self.x, self.y + self.max_height - self.height, self.width, self.height),
        )


filename = "audio/somebs.wav"
time_series, sample_rate = librosa.load(filename)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]


pygame.init()
pygame.display.set_caption('uhhh yeah')
infoObject = pygame.display.Info()

screen_w = int(infoObject.current_w / 2.5)
screen_h = int(infoObject.current_w / 4)
screen = pygame.display.set_mode([screen_w, screen_h])

bars = []
frequencies = np.arange(100, 8000, 100)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

mx = np.max(frequencies)
mn = np.min(frequencies)
rng = mx-mn

for c in frequencies:
    color = 255*(mx-c)/rng
    bars.append(AudioBar(x, 0, c, (0, 0, color), max_height=400, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(filename)
pygame.mixer.music.play(0)

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        b.update(deltaTime, get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq))
        b.render(screen)

    pygame.display.flip()

pygame.quit()
/ (omx - omn)


def getColorNorm(height, mn, mx):
    return (0, 0, np.clip(255 * normalize(height, mn, mx), 0, 255))


class AudioBar:
    def __init__(
        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(selfport BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Image

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plport BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):ep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
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


import os, time
from PIL import Image, ImageSequence

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = os.getcwd()
    fpath = path + "/media/nya.gif"

    gif = Image.open(fpath)

    ind = 1
    for frame in ImageSequence.Iterator(gif):
        image = resize(frame, 200)
        image = to_greyscale(image)

        ascii_str = pixel_to_ascii(image)
        img_width = image.width
        ascii_str_len = len(ascii_str)
        ascii_img = ""

        for i in range(0, ascii_str_len, img_width):
            ascii_img += ascii_str[i : i + img_width] + "\n"

        with open(path + "/media/nya/gif" + str(ind) + ".txt", "w") as f:
            f.write(ascii_img)
        
        ind += 1


main()

import os
import librosa
import numpy as np
import pygame
import time
import cv2


def normalize(val, omn, omx):
    return (val - omn) / (omx - omn)


def getColorNorm(height, mn, mx):
    return (0, 0, np.clip(255 * normalize(height, mn, mx), 0, 255))


class AudioBar:
    def __init__(
        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(tarimport requests
from bs4 import BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, port BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")
port BeautifulSoup
import numpy as np
import cv2

url = "https://safebooru.org/index.php?page=post&s=random"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('img[id="image"]')[0]

imgurl = select['src']
print(imgurl)c
print(imgurl)c

resp = requests.get(imgurl, stream=True).raw

img = np.asarray(bytearray(resp.read()))
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)der(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()


cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)
spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)der(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()


cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygam
spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)der(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()


cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05ebe-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import numpy as np
import cv2

url = "https://tiktok.com"e.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)der(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()


cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05ebe-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import numpy as np
import cv2

url = "https://tiktok.com"

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05ebe-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import numpy as np
import cv2

url = "https://tiktok.com"
soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import numpy as np
import cv2

url = "https://tiktok.com"
    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
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


import os, time
from PIL import Image, ImageSequence

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def 
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)der(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()


cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05ebe-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
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
            if txt[0]=='#':, frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Image

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
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

        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()


cv2.imshow('random anime', img)

cv2.waitKey(0)
cv2.destroyAllWindows()get_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygamed(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLasimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Iimport requests
from bs4 import BeautifulSoup

url = "https://weather.com/weather/today/l/781efc69df8aaa7d9cec5351e10e05eb42f39bce6e2da953d2b56b8a28b118b6"
raw = requests.get(url)

soup = BeautifulSoup(raw.content, "html.parser")

select = soup.select('span[data-testid="TemperatureValue"]')[0]

temp = select.contents[0]

print(temp)
mage

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
<html xml:lang="en" lang="en">\n\
  <head>\n\
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n\
    <style>\n\
      body {letter-spacing: -3px; line-height: 1px;  font-size: 8px;}\n\
      large-body   {letter-spacing: -4px; line-height: 14px; font-size: 20px;}\n\
    </style>\n\
  </head>\n\
  <body>\n\
  %s\n\
  </body>\n\
</html>\n' % hex

print html
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
  %s\n\
  </body>\n\
</html>\n' % hex

print html
# See http://en.wikipedia.org/wiki/Braille_ASCII
# " A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)="

# ⠀ (blank)
# ⠁ A or 1
# ⠂ comma
# ⠃ B or 2
# ⠄ apostrophe
# ⠅ K
# ⠆ semicolon
# ⠇ L
# ⠈
# ⠉ C or 3
# ⠊ I or 9
# ⠋ F or 6
# ⠌ (grade 2) the letters ST
# ⠍ M
# ⠎ S
# ⠏ P
# ⠐
# ⠑ E or 5
# ⠒
# ⠓ H or 8
# ⠔
# ⠕ O
# ⠖ exclamation point
# ⠗ R
# ⠘
# ⠙ D or 4
# ⠚ J or 0
# ⠛ G or 7
# ⠜
# ⠝ N
# ⠞ T
# ⠟ Q
# ⠠ capital letter follows
# ⠡ (grade 2) the letters CH
# ⠢
# ⠣
# ⠤ hyphen
# ⠥ U
# ⠦ opening question/quotation mark
# ⠧ V
# ⠨
# ⠩ (grade 2) the letters SH
# ⠪
# ⠫
# ⠬
# ⠭ X
# ⠮
# ⠯ (grade 2) the word AND
# ⠰
# ⠱
# ⠲ full stop
# ⠳
# ⠴ closing question/quotation mark
# ⠵ Z
# ⠶ bracket (parentheses)
# ⠷
# ⠸
# ⠹ (grade 2) the letters TH
# ⠺ W
# ⠻


import os, time
from PIL import Image, ImageSequence

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def 
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

import PIL.Image

ASCII_CHARS = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]


def resize(image, new_width=100):
    width, height = image.size
    new_height = int(new_width * height / width)
    return image.resize((new_width, new_height))


def to_greyscale(image):
    return image.convert("L")


def pixel_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]
    return ascii_str


def main():
    path = "media\plep.jpg"
    try:
        image = PIL.Image.open(path)
    except:
        print(path, "Unable to find image ")

    # resize image
    image = resize(image)
    # convert image to greyscale image
    greyscale_image = to_greyscale(image)
    # convert greyscale image to ascii characters
    ascii_str = pixel_to_ascii(greyscale_image)
    img_width = greyscale_image.width
    ascii_str_len = len(ascii_str)
    ascii_img = ""
    # Split the string based on width  of the image
    for i in range(0, ascii_str_len, img_width):
        ascii_img += ascii_str[i : i + img_width] + "\n"
    # save the string to a file
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_img)


main()

from PIL import Image

im = Image.open("./redlogo.png")

bin = im.convert("L").point(lambda px: px < 148) # I might extract the mean

lines = [[[bin.getpixel((x,y)) for x in range(i, i+2) for y in range(j, j+3)] for i in range(0,bin.size[0]-2,2)] for j in range(0,bin.size[1]-3,3)]

values = [[b[0] + 2*b[1] + 4*b[2] + 8*b[3] + 16*b[4] + 32*b[5] for b in line] for line in lines]

hex = '<br />\n'.join([''.join(["&#%s;" % (10240 + v) for v in value]) for value in values])

html = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"\n\
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n\
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

        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

 in wraps:
    print(wrap.prettify())
    caption = wrap.find('div[class="tt-video-meta-caption"]')
    for capchild in caption.children:
        if capchild.name=='a':
            txt = capchild.select('strong')[0].content
            if txt[0]=='#':
                tags.add(txt)
                print(txt)

        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


audiopath = "media\zerotwo.wav"
videopath = "media\zerotwo.mp4"

time_series, sample_rate = librosa.load(audiopath)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]

os.environ['SDL_VIDEO_WINDOW_POS']="%d,%d"%(2200,500)

pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=500, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(audiopath)
pygame.mixer.music.play(0)

cap = cv2.VideoCapture(videopath)
vidrefresh = time.time()

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    curtime = time.time()
    if curtime-vidrefresh>.0155 and cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("zero two ;)", frame)
        vidrefresh = curtime

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()

