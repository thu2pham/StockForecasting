import json
from textblob import TextBlob 
import csv

def analyze_json(year, month):
    file_str = 'jsons/' + str(year) + '-' + '{:02}'.format(month) + '.json'
    with open(file_str) as data_file:    
        NYTimes_data = json.load(data_file)

    positive_count = 0.0
    negative_count = 0.0
    total_count = 0.0

    for i in range(len(NYTimes_data["response"]["docs"][:])):
        try:
            headline = NYTimes_data["response"]["docs"][:][i]['headline']['main']
            analysis = TextBlob(headline) 
            total_count += 1
            # set sentiment 
            if analysis.sentiment.polarity > 0: 
                positive_count += 1
            elif analysis.sentiment.polarity == 0: 
                positive_count += 0.5
                negative_count += 0.5
            else: 
                negative_count += 1
        except:
            pass

    return positive_count / total_count

data_points = []

def addDataPoint(year, month):
    positive = analyze_json(year, month)
    print(year, month, positive)
    data_points.append((year, month, positive))


for year in range(2000, 2019, 1):
    for month in range(1, 13, 1):
        addDataPoint(year, month)

addDataPoint(2019, 1)
addDataPoint(2019, 2)
addDataPoint(2019, 3)
addDataPoint(2019, 4)
addDataPoint(2019, 5)

with open('sentimental_data.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['year', 'month', 'positive'])
    for data_point in data_points:
        filewriter.writerow([data_point[0], data_point[1], data_point[2]])
