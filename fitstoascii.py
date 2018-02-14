import numpy as np
import matplotlib.pyplot as pl

from astropy.table import Table

import csv

with open(sys.argv[1]) as f:			# opening .txt or .dat file to read list of .fits files
	files = f.readlines()
	
files = [x.strip() for x in files]

for name in files:
	t = Table.read(name+".txt", format = "fits")
	t.write(name+".txt", format="ascii.fixed_width")
