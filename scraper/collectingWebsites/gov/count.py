from bs4 import BeautifulSoup
import os

pwd = os.listdir('.')
input1 = 0

for filename in pwd:
  opened = None
  try:
    opened = open(filename)
    openedText = opened.read()
    html = BeautifulSoup(openedText)
    allInput = html.find_all("input")
    if len(allInput) <= 1:
      if allInput[0]["type"] == "hidden":
        os.remove(allInput[0])
    opened.close()
  except:
    if opened is not None:
       opened.close()
    continue

print(input1)
