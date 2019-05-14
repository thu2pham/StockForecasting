import json
from textblob import TextBlob 
import csv

keywords = ['microsoft', 'msft']

def stringContainsKeywords(str, lst):
    loweredStr = str.lower()
    for lstItem in lst:
        if lstItem in loweredStr:
            return True
    return False

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
            # set sentiment 
            if stringContainsKeywords(headline, keywords): 
                total_count += 1
                if analysis.sentiment.polarity > 0: 
                    positive_count += 1
                elif analysis.sentiment.polarity == 0: 
                    positive_count += 0.5
                    negative_count += 0.5
                else: 
                    negative_count += 1
        except:
            pass


    result = 0.0
    try:
        result = positive_count / total_count
    except:
        pass

    return (int(total_count), result)

data_points = []

def addDataPoint(year, month):
    (appearance, positive2) = analyze_json(year, month)
    print(year, month, appearance, positive2)
    data_points.append((year, month, appearance, positive2))


for year in range(2000, 2019, 1):
    for month in range(1, 13, 1):
        addDataPoint(year, month)

addDataPoint(2019, 1)
addDataPoint(2019, 2)
addDataPoint(2019, 3)
addDataPoint(2019, 4)
addDataPoint(2019, 5)

with open('sentimental_data2.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['year', 'month', 'appearance', 'positive2'])
    for data_point in data_points:
        filewriter.writerow([data_point[0], data_point[1], data_point[2], data_point[3]])
