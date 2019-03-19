import requests
import sys

url = sys.argv[1]
file = sys.argv[2]

with open(file, "rb") as img_file:
    response = requests.post(url, files={"payload": img_file})
    print(response.content)
