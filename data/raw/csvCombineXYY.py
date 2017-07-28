OUTFILENAME = "data.csv"
fileList = ["zip.csv","county.csv"]


outDict = {}

## date is the first value and the primary key for the set
header = 'date'


for f in fileList:
    print "Opening file: " + f
    f = open(f,'r')
    for line in f:
        entry = line.split(',')
        if entry[0] == 'date':
            header += ',' + entry[1].strip()
        elif (entry[0] in outDict):
            outDict[entry[0]] += ',' + str(entry[1]).strip()
        else:
            outDict[entry[0]] = str(entry[1]).strip()
    f.close()

print "Writing to file: " + OUTFILENAME
w = open(OUTFILENAME,'w')
w.write(header+'\n')
for key in sorted(outDict.keys()):
    w.write(key + ',' + outDict[key] + '\n')
w.close()

print outDict
