import pandas as pd

import re #regular expression matching for removing unwanted columns by name
import natsort as ns #3rd party package for natural sorting


#import the data
data0_raw = pd.read_csv("5G_counts.tsv", sep = "\t")

#removing all QC data
data1_noQC = data0_raw.select(lambda x: not re.search("QC", x), axis = 1)

#removing all non FM40 data
data2_FM40only = data1_noQC.select(lambda x: re.search("FM40", x), axis = 1)

#naturally sorting FM40 data by columns
cols = list(ns.natsorted(data2_FM40only.columns))
data3_sorted=data2_FM40only[cols]

#adding the descriptor columns back to FM40
qualitative = data0_raw.loc[:,"locus_tag":"translation"]
data4_sorted = pd.concat([qualitative, data3_sorted], axis = 1)

