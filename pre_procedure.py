import bs4
import csv



class Watch:
    videoTitle = ""
    Channel = ""
    date = ""

FILE_NAME = "watch-history.html"
DATA_PATH = "./Data/" + FILE_NAME

# 入力データ
soup = bs4.BeautifulSoup(open(DATA_PATH), 'html.parser')
print("ok")
div_inside0 = soup.find_all("div", class_="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1")


with open('./CSVs/output.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(["video_title", "video_id", "channel_name", "channel_id", "watch_date"])
        f.close()

for div_inside in div_inside0:
    div_inside2 = div_inside.find_all("a")
    print("ok")
    csvlist = []
    i = 0
    isSkip = False
    isNone = True
    for a in div_inside2:
        isNone = False
        if a.text.find('https://www.youtube.com/watch?v=') == 0:
            isSkip = True
            break
        sample_txt = a.text
        csvlist.append(sample_txt)
        if i == 0:
            sample_id = a.get('href')
            sample_id = sample_id.replace('https://www.youtube.com/watch?v=', "")
            csvlist.append(sample_id)
            i = i + 1
            continue
        if i == 1:
            sample_id = a.get('href')
            sample_id = sample_id.replace('https://www.youtube.com/channel/', "")
            csvlist.append(sample_id)
            i = i + 1
        if i == 2:
            time = div_inside.text
            time = time.replace(csvlist[0], '')
            time = time.replace(csvlist[2], '')
            time = time.replace('Watched\xa0', '')
            #time = time.replace('\xa0を視聴しました', '')
            csvlist.append(time)
            i = i + 1
    if isSkip:
        continue
    if isNone:
        continue
    with open('./CSVs/output.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(csvlist)
        f.close()

#-----------------------------------------------------------------------------------------------------------------

