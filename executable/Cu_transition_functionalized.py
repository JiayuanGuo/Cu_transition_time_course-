
import os.path
import pandas as pd
import numpy as np

import re #regular expression matching for removing unwanted columns by name
import natsort as ns #3rd party package for natural sorting

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


def raw_data_cleanup_v1(filename):
    """
    Imports RNAseq .tsv file and does basic clean up"
        -enter filename with extension as str, file should be in subdirectory raw_data
        -Several columns are empty, and several transcriptomics datasets are not appropriate for this analysis.
        -sorts the data columns naturally
        -returns dataframe with locus tag set as index
    """
    # identifying path to raw_data directory and checking for data file

    PATH = "./raw_data/" + str(filename)

    if os.path.isfile(PATH):
        print("{} was located in the raw_data directory".format(filename))

        # import the data
        df0_raw = pd.read_csv(PATH, sep="\t")
        print("{} was imported into dataframe".format(filename))

        # identified columns to be removed. Dropping these columns and reorganizing the dataframe
        to_remove = ["5GB1_ferm_WT_QC", "5GB1_ferm_Ack_QC", "5GB1C_latelog_vial_TR2_QC",
                     "5GB1_FM58_Td20h_TR1_QC", "5GB1_FM58_Td32h_TR3_QC", "5GB1_LTrecycle_TR1_QC", "5GB1_LTrecycle_TR1",
                     "5GB1_vial_wLa_TR3", "5GB1_vial_woLa_TR2"]

        # manually removing some unwanted columns
        df1_raw_filtered = df0_raw.drop(df0_raw.loc[:, to_remove].columns, axis=1)
        # Isolating FM34: Cu transition 3+ hours to be reinserted later
        df1a_FM34_only = df1_raw_filtered.select(lambda x: re.search("FM34", x), axis=1)
        # Removing all QC runs
        df1_raw_filtered = df1_raw_filtered.select(lambda x: not re.search("QC", x), axis=1)

        # naturally sorting the filtered columns
        to_sort = df1_raw_filtered.loc[:, "5GB1_FM69_t2_TR1":]

        cols = list(ns.natsorted(to_sort.columns))
        cols_sorted = to_sort[cols]

        # adding descriptive columns
        qualitative = df1_raw_filtered.loc[:, "locus_tag":"translation"]
        df1b_filtered_sorted = pd.concat([qualitative, cols_sorted], axis=1)

        # isolating FM40 (to be added back later for particular order )
        df1b_FM40_only = df1b_filtered_sorted.select(lambda x: re.search("FM40", x), axis=1)

        # removing FM40 (to be added right back at the end)
        df1b_filtered_sorted = df1b_filtered_sorted.select(lambda x: not re.search("FM40", x), axis=1)

        # Adding FM40
        cleaned_up = pd.concat([df1b_filtered_sorted, df1b_FM40_only], axis=1)

        # adding FM34
        df2_cleaned_up = pd.concat([cleaned_up, df1a_FM34_only], axis=1)

        # setting locus tag as index
        df2_cleaned_up = df2_cleaned_up.set_index("locus_tag")

        # new column names in the same order as df2_cleaned_up (except for FM23)
        df_new_columns = pd.DataFrame({"New_columns":
                                           ['FM12_CH4-lim_3.0/day',
                                            'FM12_CH4-lim_3.0/day_R1',
                                            'FM14_CH4-lim_3.0/day',
                                            'FM14_CH4-lim_3.0/day_R1',
                                            'FM18_CH3OH_4.1/day',
                                            'FM18_CH3OH_4.1/day_R1',
                                            'FM19_O2-lim_3.5.day',
                                            'FM19_O2-lim_3.5.day_R1',
                                            'FM19_O2-lim_3.5.day_R2',
                                            'FM20_no-lim_5.2/day',
                                            'FM20_no-lim_5.2/day_R1',
                                            'FM21_no-lim_5.4/day',
                                            'FM21_no-lim_5.4/day_R1',
                                            'FM21_no-lim_5.4/day_R2',
                                            'FM22_O2-lim_4.2/day',
                                            'FM22_O2-lim_4.2/day_R1',
                                            'FM22_O2-lim_4.2/day_R2',
                                            '5GB1_FM23_TR3',
                                            'FM69_O2-lim+_0.7/day_t2',
                                            'FM69_?-lim_0.7/day_t3',
                                            'FM69_?-lim_0.7/day_t3_R1',
                                            'FM69_?-lim_0.7/day_t4',
                                            'FM69_?-lim_0.7/day_t4_R2',
                                            'FM80_O2-lim_0.7/day_t2',
                                            'FM80_O2-lim_0.7/day_t4',
                                            'FM81_O2-lim+_0.7/day_t1',
                                            'FM81_O2-lim_0.7/day_t2',
                                            'FM40_-Cu_2.9/day_t0m',
                                            'FM40_+Cu_2.9/day_t10m',
                                            'FM40_+Cu_2.9/day_t20m',
                                            'FM40_+Cu_2.9/day_t40m',
                                            'FM40_+Cu_2.9/day_t60m',
                                            'FM40_+Cu_2.9/day_t90m',
                                            'FM40_+Cu_2.9/day_t150m',
                                            'FM40_+Cu_2.9/day_t180m',
                                            'FM34_-Cu_2.7/day_t0',
                                            'FM34_+Cu_2.7/day_t180m',
                                            'FM34_+Cu_2.7/day_t240m',
                                            'FM34_+Cu_2.7/day_t300m',
                                            'FM34_+Cu_2.7/day_t360m',
                                            'FM34_+Cu_2.7/day_t420m',
                                            'FM34_+Cu_2.7/day_t480m']})

        # resetting the column names to be more descriptive
        df2_cleaned_up.columns = list(df2_cleaned_up.iloc[:, :8].columns) + list(df_new_columns.New_columns)

        columns_ordered = ['product',
                           'type',
                           'gene_symbol',
                           'locus',
                           'start_coord',
                           'end_coord',
                           'note',
                           'translation',
                           '5GB1_FM23_TR3',
                           'FM18_CH3OH_4.1/day',
                           'FM18_CH3OH_4.1/day_R1',
                           'FM20_no-lim_5.2/day',
                           'FM20_no-lim_5.2/day_R1',
                           'FM21_no-lim_5.4/day',
                           'FM21_no-lim_5.4/day_R1',
                           'FM21_no-lim_5.4/day_R2',
                           'FM12_CH4-lim_3.0/day',
                           'FM12_CH4-lim_3.0/day_R1',
                           'FM14_CH4-lim_3.0/day',
                           'FM14_CH4-lim_3.0/day_R1',
                           'FM19_O2-lim_3.5.day',
                           'FM19_O2-lim_3.5.day_R1',
                           'FM19_O2-lim_3.5.day_R2',
                           'FM22_O2-lim_4.2/day',
                           'FM22_O2-lim_4.2/day_R1',
                           'FM22_O2-lim_4.2/day_R2',
                           'FM34_-Cu_2.7/day_t0',
                           'FM40_-Cu_2.9/day_t0m',
                           'FM40_+Cu_2.9/day_t10m',
                           'FM40_+Cu_2.9/day_t20m',
                           'FM40_+Cu_2.9/day_t40m',
                           'FM40_+Cu_2.9/day_t60m',
                           'FM40_+Cu_2.9/day_t90m',
                           'FM40_+Cu_2.9/day_t150m',
                           'FM40_+Cu_2.9/day_t180m',
                           'FM34_+Cu_2.7/day_t180m',
                           'FM34_+Cu_2.7/day_t240m',
                           'FM34_+Cu_2.7/day_t300m',
                           'FM34_+Cu_2.7/day_t360m',
                           'FM34_+Cu_2.7/day_t420m',
                           'FM34_+Cu_2.7/day_t480m',
                           'FM69_O2-lim+_0.7/day_t2',
                           'FM69_?-lim_0.7/day_t3',
                           'FM69_?-lim_0.7/day_t3_R1',
                           'FM69_?-lim_0.7/day_t4',
                           'FM69_?-lim_0.7/day_t4_R2',
                           'FM80_O2-lim_0.7/day_t2',
                           'FM80_O2-lim_0.7/day_t4',
                           'FM81_O2-lim+_0.7/day_t1',
                           'FM81_O2-lim_0.7/day_t2']

        df2_cleaned_up = df2_cleaned_up[columns_ordered]

        return df2_cleaned_up

    else:
        print("{} does not exist in raw_data directory. Function was not complete.".format(filename))
        return


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
              columns,
              remove_zero = True):

    """
    TPM_counts(dataframe, gene_start, gene_stop, columns):

    returns a dataframe with TPM instead of reads

    Parameters
    ----------
    daraframe = dataframe object variable
    gene_start = string with column name containing gene start coordinate
    gene_stop = string with column name containing gene stop coordinate
    columns = list of strings of column names to be converted to TPM
    remove_zero = if True, will remove rows containing zero expression


    Run the following two lines to properly execute this function:

    columns = ['5GB1_FM40_T0m_TR2', '5GB1_FM40_T10m_TR3', '5GB1_FM40_T20m_TR2', '5GB1_FM40_T40m_TR1',
           '5GB1_FM40_T60m_TR1', '5GB1_FM40_T90m_TR2', '5GB1_FM40_T150m_TR1_remake', '5GB1_FM40_T180m_TR1']

    TPM_counts(df,"start_coord","end_coord",columns)
    """

    #create empty dataframe
    gene_length = pd.DataFrame()

    #gene length in kilo base pairs as new column
    gene_length["gene_length"] = (dataframe[gene_stop]- dataframe[gene_start] + 1)/1000

    #normalize read counts by gene length in kilo base pairs
    RPK = dataframe.loc[:,columns].div(gene_length.gene_length, axis=0)

    #creating a series with the sums of each FM40 column / 1,000,000
    norm_sum = RPK.sum(axis=0)/1000000
    norm_sum1 = pd.Series.to_frame(norm_sum)
    norm_sum2 = norm_sum1.T

    #dividing by the the total transcript counts in each repicate
    TPM = RPK.div(norm_sum2.ix[0])

    dataframe.loc[:,columns] = TPM

    if remove_zero:
        dataframe_values = dataframe.loc[:,columns]
        remove_index = dataframe_values[dataframe_values.isin([0]).any(axis=1)].index

        dataframe = dataframe.drop(remove_index)

    return dataframe


def mean_center(df, first_data_column, last_data_column):
    """
    mean_center(dataframe,
                first_data_column,
                last_data_column)

    Return a new dataframe with the range of data columns log2 transformed.

    Parameters
    ----------
    daraframe = dataframe object variable
    first_data_column = first column that contains actual data (first non categorical)
    last_data_column = last column taht contains actual data (last non categorigal column)

    Run the following to execute the function for Cu transition dataset.

    mean_center(df, "5GB1_FM40_T0m_TR2", "5GB1_FM40_T180m_TR1")

    """

    df2_TPM_values = df.loc[:, first_data_column:last_data_column]  # isolating the data values
    df2_TPM_values_T = df2_TPM_values.T  # transposing the data

    standard_scaler = StandardScaler(with_std=False)
    TPM_counts_mean_centered = standard_scaler.fit_transform(df2_TPM_values_T)  # mean centering the data

    TPM_counts_mean_centered = pd.DataFrame(TPM_counts_mean_centered)  # back to Dataframe

    # transposing back to original form and reincerting indeces and columns
    my_index = df2_TPM_values.index
    my_columns = df2_TPM_values.columns

    TPM_counts_mean_centered = TPM_counts_mean_centered.T
    TPM_counts_mean_centered.set_index(my_index, inplace=True)
    TPM_counts_mean_centered.columns = my_columns

    return TPM_counts_mean_centered


def euclidean_distance(dataframe, first_data_column, last_data_column):
    """
    euclidean_distance(dataframe,
                first_data_column,
                last_data_column)

    Return a new dataframe - pairwise distance metric table, euclidean distance between every pair of rows.

    Parameters
    ----------
    daraframe = dataframe object variable
    first_data_column = first column that contains actual data (first non categorical)
    last_data_column = last column taht contains actual data (last non categorigal column)

    Run the following to execute the function for Cu transition dataset.

    euclidean_distance(df, "5GB1_FM40_T0m_TR2", "5GB1_FM40_T180m_TR1")

    """

    df_values = dataframe.loc[:, first_data_column:last_data_column]  # isolating the data values

    df_euclidean_distance = pd.DataFrame(euclidean_distances(df_values))

    my_index = dataframe.index

    df_euclidean_distance = df_euclidean_distance.set_index(my_index)
    df_euclidean_distance.columns = my_index

    return df_euclidean_distance

def log_2_transform(dataframe,
                    first_data_column,
                    last_data_column):
    """
    log_2_transform(dataframe,
                    first_data_column,
                    last_data_column)

    Return a new dataframe with the range of data columns log2 transformed.
    *all zero values are changed to 1 (yield 0 after transform)
    *all values less than 1 are changed to 1 (yield 0 after transform)

    Parameters
    ----------
    daraframe = dataframe object variable
    first_data_column = first column that contains actual data (first non categorical)
    last_data_column = last column taht contains actual data (last non categorigal column)

    Run the following to execute the function for Cu transition dataset.

    log_2_transform(df, "5GB1_FM40_T0m_TR2", "5GB1_FM40_T180m_TR1")

    """

    df_data = dataframe.loc[:, first_data_column:last_data_column]  # isolate the data

    df_data = df_data.replace(0, 1)  # replace all zeros with 1s

    df_data[df_data < 1] = 1  # replace all values less than 1 with 1

    df_data_log2 = df_data.apply(np.log2)

    return df_data_log2


def congruency_table(df,
                     data_clm_strt,
                     data_clm_stop,
                     step,
                     mask_diagonal=False):
    """

    congruency_table(df, data_clm_strt, data_clm_stop, step = len(df.columns), mask_diagonal=False)

    returns a new datafram - congruency table - a pairwise pearson correlation matrix for every row pair

    Parameters
    ----------

    df - dataframe argument - recommended to use TPM counts for RNAseq datasets.
    data_clm_strt = first column that contains data to be processed
    data_clm_stop = last column that contains data to be processed
    step = length of dataset (integer - number of rows in the dataset)
    mask_diagonal = mask diagonal values which shoud come out as 1


    Run the following lines to execute the function for my data

    congruency_table(df1, "5GB1_FM40_T0m_TR2" , "5GB1_FM40_T180m_TR1", step = len(df.columns))

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
