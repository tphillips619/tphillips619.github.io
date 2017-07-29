#!/usr/bin/python
import re

INPUTFILENAME = "combined.csv"
OUTPUTFILENAME = "geoHierarchyTimeSeries.csv"

lineList = []
outList = []

outList.append("area,date,price\n")
dl = ','

with open(INPUTFILENAME, 'r') as f:
    keyLine = f.readline().rstrip() # Remove any trailing whitespace or endline
    keyLine = re.sub('["]', '', keyLine) # Remove all surrounding " chars
    keys = keyLine.split(',')

##    outList.append(keyLine)

##    print "Number of keys = " + str(len(keys))
##    print keys
    
    for line in f:
        values = re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', line.rstrip()) # Allows parsing "Los Angeles-Long Beach-Anaheim, CA"
        lineList.append(values)
        areaName = values[0]
##        print "Number of values for " + values[0] + " = " + str(len(values))
        outLine = ""
        for i in range(1, len(values)): #Start at second item, index1
##            print "i = " + str(i)
            outLine = areaName + dl + keys[i] + dl + str(values[i]) + '\n'
##            print(outLine)
            outList.append(outLine)
    f.close()

with open(OUTPUTFILENAME,'w') as w:
    for line in outList:
        w.write(line)
    w.close()
