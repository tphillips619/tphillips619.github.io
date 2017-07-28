#!/usr/bin/python
import os
import re

# Input
# 

# Incomplete
# 1) Remove " or ' chars from each column
# 2) Identify only yyyy-mm date values in the key
# 3) Pivot data to the following format
#   area,date,price

areas = ["92610","Orange","Los Angeles-Long Beach-Anaheim, CA","United States"]

## Compressed Zillow Time Series for Zip, County, and Metro (including Nation) are large (250MB, 34MB, and 75KB respectively)
## Script assumes user has pulled relevant lines from each zipped time series and placed them in their own .csv files within this directory
files = [f for f in os.listdir('.') if os.path.isfile(f)]
# Removing list items while iterating through list is problematic
# Assign results to new list
csvfiles = []
for f in files:
    if '.csv' in f:
        csvfiles.append(f)

# Assumes "Region Name" is the second item per line
for fn in csvfiles:
    with open(fn, 'r') as f:
        keys = re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', f.next())
        # Collect index for "Region Name" use as "area" for output
        # List indicies for any date yyyy-mm
        
        values = re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', f.next())  #Avoids splitting on , chars between two quotes; e.g. capture of "Los Angeles-Long Beach-Anaheim, CA"
        # Collect values associated with each index
        # Append output as
        #  area,date,price
        
        f.close()
                

            
            
        
