
##  Combines two data files with input format
# date,Zip  AND  date,County

## Into one data file with output format
# area,date,price
# where area is either Zip or County

OUTFILENAME = "data2.csv"
OutHeader = 'area,date,price'

File1Name = "zip.csv"
File1Key = "Zip 92610"
File2Name = "county.csv"
File2Key = "Orange County"

outLineList = []

f = open(File1Name,'r')
for line in f:
    entry = line.split(',')
    if (entry[0] != 'date'):   # Remove .csv header
        outLineList.append(File1Key + ',' + entry[0] + ',' + entry[1])
f.close()

f = open(File2Name, 'r')
for line in f:
    entry = line.split(',')
    if (entry[0] != 'date'):   # Remove .csv header
        outLineList.append(File2Key + ',' + entry[0] + ',' + entry[1]) 

print "Writing to file: " + OUTFILENAME
w = open(OUTFILENAME,'w')
w.write(OutHeader+'\n')
for line in outLineList:
    w.write(line)
w.close()
