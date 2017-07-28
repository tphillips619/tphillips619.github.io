#!/usr/bin/python
## Takes TSV taken from FredieMac.com "30yrfixedTimeSeries.tsv"
#Empty third column and Year value only listed when Year value changes from previous Year
#Year  Week         Rate    Points
#1972  01/05        7.46    1.0
#      01/12        7.46    0.9

## Tranforms to CSV "30yrfixed.csv"
#date,rate,points
#1972-01-07,7.46,1.0
#1972-01-14,7.46,0.9

INPUTFILENAME = "30yrfixedTimeSeries.tsv"
OUTPUTFILENAME = "30yrfixed.csv"

OutHeader = "date,rate,points\n"
OutDelimeter = ','

outLineList = []

print ("Reading from " + INPUTFILENAME)
f = open(INPUTFILENAME, 'r')

with open(INPUTFILENAME,'r') as f:
    f.next() # skip header line
    for line in f:
        entry = line.split("\t")
        
        #Clear any whitespace
        for value in entry:
            value = value.strip()
            
        if entry[0].strip() != "":
            currentYear = entry[0]
        else:
            entry[0] = currentYear

        outLine = ""
        for index in range(0,len(entry)):
            if index == 0:
                outLine += entry[index] + "-"
            elif index == 1:
                date = entry[index].split('/')
                if len(date) == 2:
                    outLine += date[0] + "-"
                    outLine += date[1] + OutDelimeter
            elif index == 2:
                continue
            else:
                outLine += entry[index] + OutDelimeter

        # Slice off last OutDelimeter
        outLineList.append(outLine[:-1])
    f.close()

print ("Writing to " + OUTPUTFILENAME)
w = open(OUTPUTFILENAME, 'w')

w.write(OutHeader)
for line in outLineList:
    w.write(line)
w.close()
