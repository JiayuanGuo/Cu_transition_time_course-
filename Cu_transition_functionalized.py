
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

    returns a dataframe with TPM instead of reads

    Parameters
    ----------
    daraframe = dataframe object variable
    gene_start = string with column name containing gene start coordinate
    gene_stop = string with column name containing gene stop coordinate
    columns = list of strings of column names to be converted to TPM


    Run the following lines to execute the function for my data

    columns = ['5GB1_FM40_T0m_TR2', '5GB1_FM40_T10m_TR3', '5GB1_FM40_T20m_TR2', '5GB1_FM40_T40m_TR1',
           '5GB1_FM40_T60m_TR1', '5GB1_FM40_T90m_TR2', '5GB1_FM40_T150m_TR1_remake', '5GB1_FM40_T180m_TR1']

    TPM_counts(df,"start_coord","end_coord",columns)
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

    print("TPM transform complete")

    dataframe.to_csv("TPM_counts.csv")

    return dataframe


def congruency_table(df,
                     data_clm_strt,
                     data_clm_stop,
                     step=len(df.columns),
                     mask_diagonal=False):
    """

    corr(df, data_clm_strt, data_clm_stop, step = len(df.columns), mask_diagonal=False)

    returns a new datafram - congruency table - a pairwise pearson correlation matrix for every row pair

    Parameters
    ----------

    df - dataframe argument - recommended to use TPM counts for RNAseq datasets.
    data_clm_strt = first column that contains data to be processed
    data_clm_stop = last column that contains data to be processed
    step = length of dataset
    mask_diagonal = mask diagonal values which shoud come out as 1


    Run the following lines to execute the function for my data

    congruency_table(df1, "5GB1_FM40_T0m_TR2" , "5GB1_FM40_T180m_TR1")

    """

    df = df.loc[:, data_clm_strt: data_clm_stop]  # isolating the rows that are relavent to us.
    df = df.T

    n = df.shape[0]

    def corr_closure(df):
        d = df.values
        sums = d.sum(0, keepdims=True)
        stds = d.std(0, keepdims=True)

        def corr_(k=0, l=10):
            d2 = d.T.dot(d[:, k:l])
            sums2 = sums.T.dot(sums[:, k:l])
            stds2 = stds.T.dot(stds[:, k:l])

            return pd.DataFrame((d2 - sums2 / n) / stds2 / n,
                                df.columns, df.columns[k:l])

        return corr_

    c = corr_closure(df)

    step = min(step, df.shape[1])

    tups = zip(range(0, n, step), range(step, n + step, step))

    corr_table = pd.concat([c(*t) for t in tups], axis=1)

    if mask_diagonal:
        np.fill_diagonal(corr_table.values, np.nan)

    return corr_table

