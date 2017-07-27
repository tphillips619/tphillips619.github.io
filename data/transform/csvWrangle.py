INPUTFILENAME = "zipTest.csv"
OUTPUTFILENAME = "zip.csv"
CSVHEADER = "Zip"

f = open(INPUTFILENAME,'r')

dates = []
values = []

line = f.readline()
dates = line.split(',') 

line = f.readline()
values = line.split(',')
f.close()

w = open(OUTPUTFILENAME,'w')

if ( len(dates) == len(values) ):
    w.write("date,"+CSVHEADER + "\n")
    for i in range (0,len(dates)):
# Add [1:-1] slice to remove surrounding " " to dates
        lineString = dates[i].strip()[1:-1] + "," + values[i].strip()
        print lineString
        w.write(lineString + '\n')
    w.close()
else:
    print "Dates and Values are not of equal length"

