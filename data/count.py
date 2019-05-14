import json
from textblob import TextBlob 
import csv

def analyze_json(year, month):
    file_str = 'jsons/' + str(year) + '-' + '{:02}'.format(month) + '.json'
    with open(file_str) as data_file:    
        NYTimes_data = json.load(data_file)

    return len(NYTimes_data["response"]["docs"][:])

data_points = []

def addDataPoint(year, month):
    count = analyze_json(year, month)
    print(year, month, count)
    data_points.append((year, month, count))


for year in range(2000, 2019, 1):
    for month in range(1, 13, 1):
        addDataPoint(year, month)

addDataPoint(2019, 1)
addDataPoint(2019, 2)
addDataPoint(2019, 3)
addDataPoint(2019, 4)
addDataPoint(2019, 5)

with open('count_data.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['year', 'month', 'count'])
    for data_point in data_points:
        filewriter.writerow([data_point[0], data_point[1], data_point[2]])
