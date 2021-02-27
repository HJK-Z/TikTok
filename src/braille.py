from PIL import Image

# Colors for both irc and terminal. Adjust the rgb value to match better if needed
COLORS = [
    #   [renderIRC    RGB             Term    Name]
    [0, (255, 255, 255), "97", "white"],
    [1, (0, 0, 0), "30", "black"],
    [2, (0, 0, 127), "34", "blue"],
    [3, (0, 147, 0), "32", "green"],
    [4, (255, 0, 0), "91", "light red"],
    [5, (127, 0, 0), "31", "brown"],
    [6, (156, 0, 156), "35", "purple"],
    [7, (252, 127, 0), "33", "orange"],
    [8, (255, 255, 0), "93", "yellow"],
    [9, (0, 252, 0), "92", "light green"],
    [10, (0, 147, 147), "36", "cyan"],
    [11, (0, 255, 255), "96", "light cyan"],
    [12, (0, 0, 252), "94", "light blue"],
    [13, (255, 0, 255), "95", "pink"],
    [14, (127, 127, 127), "90", "grey"],
    [15, (210, 210, 210), "37", "light grey"],
]


def convert(
    img,
    doColor=False,
    renderIRC=True,
    cutoff=50,
    size=1.0,
    invert=False,
    alphaColor=(0, 0, 0),
):
    i = Image.open(img)

    WIDTH = int(300 * size)
    HIGHT = int(700 * size)

    # Resize the image to fix bounds
    s = i.size
    if (
        s[0] == 0
        or s[1] == 0
        or (float(s[0]) / float(WIDTH)) == 0
        or (float(s[1]) / float(HIGHT)) == 0
    ):
        return []
    ns = (WIDTH, int(s[1] / (float(s[0]) / float(WIDTH))))
    if ns[1] > HIGHT:
        ns = (int(s[0] / (float(s[1]) / float(HIGHT))), HIGHT)

    i2 = i.resize(ns)

    bimg = []

    for r in range(0, i2.size[1], 4):
        line = u""
        lastCol = -1
        for c in range(0, i2.size[0], 2):
            val = 0
            i = 0
            cavg = [0, 0, 0]
            pc = 0

            for ci in range(0, 4):
                for ri in range(0, 3 if ci < 2 else 1):
                    # Convert back for the last two pixels
                    if ci >= 2:
                        ci -= 2
                        ri = 3

                    # Retrieve the pixel data
                    if c + ci < i2.size[0] and r + ri < i2.size[1]:
                        p = i2.getpixel((c + ci, r + ri))
                        alpha = p[3] if len(p) > 3 else 1
                        if invert and alpha > 0:
                            p = map(lambda x: 255 - x, p)
                        elif alpha == 0:
                            p = alphaColor
                    else:
                        p = (0, 0, 0)

                    # Check the cutoff value and add to unicode value if it passes
                    luma = (
                        0.2126 * float(p[0])
                        + 0.7152 * float(p[1])
                        + 0.0722 * float(p[2])
                    )
                    pv = sum(p[:3])
                    if luma > cutoff:
                        val += 1 << i
                        cavg = map(sum, zip(cavg, p))
                        pc += 1
                    i += 1

            if doColor and pc > 0:
                # Get the average of the 8 pixels
                cavg = map(lambda x: x / pc, cavg)

                # Find the closest color with geometric distances
                colorDist = lambda c: sum(
                    map(lambda x: (x[0] - x[1]) ** 2, zip(cavg, c[1]))
                )
                closest = min(COLORS, key=colorDist)

                if closest[0] == 1 or lastCol == closest[0]:
                    # Check if we need to reset the color code
                    if lastCol != closest[0] and lastCol != -1:
                        line += "\x03" if renderIRC else "\033[0m"
                    line += chr(0x2800 + val)
                else:
                    # Add the color escape to the first character in a set of colors
                    if renderIRC:
                        line += ("\x03%u" % closest[0]) + chr(0x2800 + val)
                    else:
                        line += ("\033[%sm" % closest[2]) + chr(0x2800 + val)
                lastCol = closest[0]
            else:
                # Add the offset from the base braille character
                line += chr(0x2800 + val)
        bimg.append(line)
    return bimg


alphaColor = (0, 0, 0)
for c in COLORS:
    if c[3].lower() == 1:
        alphaColor = c[1]
        break

for u in convert("media/astolfokawaii.png"):
    print(str(u))
