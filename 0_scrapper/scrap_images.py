import requests
from bs4 import BeautifulSoup

import os

try:
    os.system("clear")
except:
    pass

URL = "https://www.google.com/search?q=people+from+different+backgrounds&sxsrf=ALiCzsbTMISNtxbfzon2vv3P0_nBNM_eIg:1666088339309&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiktIPdxun6AhW4R_EDHS5BAMIQ_AUoAXoECAIQAw&biw=960&bih=871&dpr=1"  # Replace this with the website's URL
getURL = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
# print(getURL.status_code)

soup = BeautifulSoup(getURL.text, "html.parser")

images = soup.find_all("img")
# print(images)

imageSources = []

for image in images:
    imageSources.append(image.get("src"))

# print(imageSources)

for image in imageSources:
    webs = requests.get(image)
    open(
        "/home/ibad/Desktop/friend_detection_app/scrapper/images/"
        + image.split("/")[-1],
        "wb",
    ).write(webs.content)
