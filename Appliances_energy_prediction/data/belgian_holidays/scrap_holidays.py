from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.timeanddate.com/holidays/belgium/2016?hol=795"
headers = {"Accept-Language": "en-US,en;q=0.5"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")
table = soup.find("table")

data = []
rows = table.find_all("tr")
for row in rows:
    date_col = row.find("th")
    cols = row.find_all("td")
    if date_col and len(cols) >= 2:
        date = date_col.text.strip()
        weekday = cols[0].text.strip()
        name = cols[1].text.strip()
        htype = cols[2].text.strip()
        data.append([date, weekday, name, htype])

df = pd.DataFrame(data, columns=["Date", "Weekday", "Holiday Name", "Holiday Type"])

# date in format: 1 Jan, 6 Jan... 
df.Date = df.Date.apply(lambda x: datetime.strptime(x + " 2016", "%d %b %Y"))
df.to_csv("belgian_holidays_2016.csv", index=False)