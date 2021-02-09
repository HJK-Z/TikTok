import requests
url = "https://archiveofourown.org/downloads/8337607/Yesterday%20Upon%20The%20Stair.pdf?updated_at=1612726972"
bts = requests.get(url).content

with open('tmp/ffbytes.pdf', 'wb') as f:
    f.write(bts)

"https://archiveofourown.org/downloads/8337607/Yesterday%20Upon%20The%20Stair.pdf?updated_at=1612726972"