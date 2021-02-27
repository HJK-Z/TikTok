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