import requests
from bs4 import BeautifulSoup
import re

base = 'https://archiveofourown.org'
searchurl = '/works?utf8=%E2%9C%93&work_search%5Bsort_column%5D=kudos_count&work_search%5Bother_tag_names%5D=&work_search%5Bexcluded_tag_names%5D=&work_search%5Bcrossover%5D=&work_search%5Bcomplete%5D=T&work_search%5Bwords_from%5D=&work_search%5Bwords_to%5D=&work_search%5Bdate_from%5D=&work_search%5Bdate_to%5D=&work_search%5Bquery%5D=&work_search%5Blanguage_id%5D=en&commit=Sort+and+Filter&tag_id=%E5%83%95%E3%81%AE%E3%83%92%E3%83%BC%E3%83%AD%E3%83%BC%E3%82%A2%E3%82%AB%E3%83%87%E3%83%9F%E3%82%A2+%7C+Boku+no+Hero+Academia+%7C+My+Hero+Academia'
search = BeautifulSoup(requests.get(base + searchurl).content, 'html.parser')

ls = search.select('li[role="article"]')

ffcnt = 0
for li in ls:
    a = li.find('a')
    ffurl = a['href']
    name = a.contents[0]
    ff = BeautifulSoup(requests.get(base + ffurl).content, 'html.parser')
    dls = ff.select('li[class="download"]')[0].find('ul')
    dlurl = dls.find('a', text=re.compile('PDF'))['href']

    pdf = requests.get(base + dlurl).content

    with open('fanfics/ff' + str(ffcnt) + '.pdf', 'wb') as f:
        f.write(pdf)

    print('Downloaded ' + str(name))
    ffcnt += 1