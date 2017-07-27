#!/usr/bin/python
## Finds all cities in Orange County and creates .csv
# CityName,ForecastPctChange
# Irvine,0.5

import re
# regex needed to ignore delimeters in between double quotes

INPUTFILENAME = "AllRegionsForePublic.csv"
OUTFILENAME = "forecastGeoHierarchy.csv"

## Header ["Region","RegionName","State","ForecastYoYPctChange","MSAName","CountyName","CityName"]
## localHierarchy = {"MSA":"Los Angeles-Long Beach-Anaheim, CA","County":"Orange","City":"Lake Forest","Zipcode":"92610"}

rawData = []
csvData = []

header = "City,Forecast"

csvData.append(header)

def findLargerWholes():
    with open(INPUTFILENAME, 'r') as f:
        header = f.next()
        for line in f:
            ## regex needed to split line correctly
            ## MSA names typically include ,"name,STATE",   when ',' is the delimeter
            entry = re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', line)

            if ("MSA" in entry[0] and "Los Angeles-Long Beach-Anaheim" in entry[1]):        
                print line
                csvData.append(entry[1].replace('"', "") + ',' + entry[3])
            elif ("County" in entry[0] and "Orange" in entry[1] and "Los Angeles-Long Beach-Anaheim" in entry[4]):
                print line
                csvData.append(entry[1].replace('"',"") + ',' + entry[3])
            elif ("City" in entry[0] and "Lake Forest" in entry[1] and "Los Angeles-Long Beach-Anaheim" in entry[4]):
                print line
                csvData.append(entry[1].replace('"',"") + ',' + entry[3])
            elif ("ZipCode" in entry[0] and "92610" in entry[1] and "Los Angeles-Long Beach-Anaheim" in entry[4]):
                print line
                csvData.append(entry[1].replace('"',"") + ',' + entry[3])        
    f.close()

    return csvData

def findCitiesInCounty(county, msa):
    with open(INPUTFILENAME, 'r') as f:
        header = f.next()        
        for line in f:
            ## regex needed to split line correctly
            ## MSA names typically include ,"name,STATE",   when ',' is the delimeter
            entry = re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', line)
            if ("City" in entry[0] and county in entry[5] and msa in entry[4]):
                csvData.append(entry[1].replace('"',"") + ',' + entry[3])     
    f.close()

    return csvData        


##csvData = findLargerWholes()
csvData = findCitiesInCounty("Orange" , "Los Angeles-Long Beach-Anaheim")

w = open(OUTFILENAME, 'w')
for line in csvData:
    w.write(line + '\n')

for line in csvData:
    print line
