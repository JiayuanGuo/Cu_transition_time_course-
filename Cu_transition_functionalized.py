
import os.path
import pandas as pd

import re #regular expression matching for removing unwanted columns by name
import natsort as ns #3rd party package for natural sorting


def raw_data_cleanup(filename):
    """
    Imports RNAseq .csv file and does basic clean up of "FM40"
        -sorts FM40 timecourse sequence chronologically
        -removes all QC data and non FM40 columns
        -returns dataframe with locus tag set as index

    """

    if os.path.isfile(filename):
        print("{} was located in the directory".format(filename))

        # import the data
        data0_raw = pd.read_csv(filename, sep="\t")
        print("{} was imported into dataframe".format(filename))

        # removing all QC data
        data1_noQC = data0_raw.select(lambda x: not re.search("QC", x), axis=1)
        print("QC columns were removed from dataframe")

        # removing all non FM40 data
        data2_FM40only = data1_noQC.select(lambda x: re.search("FM40", x), axis=1)
        print("All non FM40 data were removed from dataframe")

        # naturally sorting FM40 data by columns
        cols = list(ns.natsorted(data2_FM40only.columns))
        data3_sorted = data2_FM40only[cols]
        print("All FM40 columns were sorted by timecourse sequence")

        # adding the descriptor columns back to FM40
        qualitative = data0_raw.loc[:, "locus_tag":"translation"]
        data4_sorted = pd.concat([qualitative, data3_sorted], axis=1)

        # setting locus tag to be the index
        data5_index = data4_sorted.set_index("locus_tag")

        print("Clean-up of raw data complete")
        return data5_index

    else:
        print("{} does not exist in directory. Function was not complete.".format(filename))
        return
