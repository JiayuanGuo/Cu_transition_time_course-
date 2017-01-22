
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


def TPM_counts(dataframe,
               gene_start,
               gene_stop,
               columns):
    """
    TPM_counts(dataframe, gene_start, gene_stop, columns):

    Parameters
    ----------
    daraframe = dataframe object variable
    gene_start = string with column name containing gene start coordinate
    gene_stop = string with column name containing gene stop coordinate
    columns = list of strings of column names to be converted to TPM
    """

    # create empty dataframe
    gene_length = pd.DataFrame()

    # gene length in kilo base pairs as new column
    gene_length["gene_length"] = (dataframe[gene_stop] - dataframe[gene_start] + 1) / 1000

    # normalize read counts by gene length in kilo base pairs
    RPK = dataframe.loc[:, columns].div(gene_length.gene_length, axis=0)

    # creating a series with the sums of each FM40 column / 1,000,000
    norm_sum = RPK.sum(axis=0) / 1000000
    norm_sum1 = pd.Series.to_frame(norm_sum)
    norm_sum2 = norm_sum1.T

    # dividing by the the total transcript counts in each repicate
    TPM = RPK.div(norm_sum2.ix[0])

    dataframe.loc[:, columns] = TPM

    return dataframe

