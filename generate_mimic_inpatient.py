import sys
from pathlib import Path
import os
import importlib
import pandas as pd
from pandas.api.types import union_categoricals
from sklearn.model_selection import train_test_split
import pickle
from functools import partial
from tqdm.auto import tqdm
tqdm.pandas()

module_path='preprocessing/day_intervals_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)

module_path='utils'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='preprocessing/hosp_module_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='model'
if module_path not in sys.path:
    sys.path.append(module_path)
#print(sys.path)
import day_intervals_cohort
from day_intervals_cohort import *

import day_intervals_cohort_v2
from day_intervals_cohort_v2 import *

import data_generation_icu

import data_generation
import evaluation

import feature_selection_hosp
from feature_selection_hosp import *
from feature_selection_hosp import feature_nonicu, preprocess_features_hosp, generate_summary_hosp, features_selection_hosp

# import train
# from train import *


import ml_models
from ml_models import *

import dl_train
from dl_train import *

import tokenization
from tokenization import *


import behrt_train
from behrt_train import *

import feature_selection_icu
from feature_selection_icu import *
from feature_selection_icu import feature_icu

import fairness
import callibrate_output


# version = 'Version 2'
# radio_input4 = "ALL"  # or "mortality"
# radio_input3 = "No Disease Filter" # add columns for each disease for encounters dataframe

data_type = "Non-ICU" # or "Non-ICU"
icd_code = 'No Disease Filter'
time = 0
root_dir = os.path.dirname(os.path.abspath(__file__))

version_path = 'mimiciv/2.0'
disease_label = ""
label = "ALL"

# uncomment the next line to regenerate cohort (takes 2 hours to process readmissions label)
# cohort_output = day_intervals_cohort_v2.extract_data(data_type, label, time, icd_code, root_dir, disease_label)

cohort_output = "cohort_" + data_type.lower() + "_" + label.lower().replace(" ", "_") + "_" + str(time) + "_" + disease_label

diag_flag, out_flag, chart_flag, proc_flag, med_flag = True, True, True, True, True
lab_flag = True

# uncomment the next lines to rerun feature selection -- takes around 1 hour
# feature_nonicu(cohort_output, version_path, diag_flag, lab_flag, proc_flag, med_flag)

# This runs super fast -- convert diagnoses to ICD-10 and Converts drug names into NDC non-proprietary drug name
# group_diag = "Convert ICD-9 to ICD-10 codes"
# group_med = "Yes"
# group_proc = "ICD-9 and ICD-10"
# preprocess_features_hosp(cohort_output, diag_flag, proc_flag, med_flag, False, group_diag, group_med, group_proc, False, False, 0, 0)

def concatenate(dfs, **kwargs):
    """Concatenate while preserving categorical columns.

    NB: We change the categories in-place for the input dataframes"""
    from pandas.api.types import union_categoricals
    import pandas as pd
    # Iterate on categorical columns common to all dfs
    for col in set.intersection(
        *[
            set(df.select_dtypes(include='category').columns)
            for df in dfs
        ]
    ):
        # Generate the union category across dfs for this column
        # for df in dfs:
        #     # Convert all categories to be string dtype
        #     if not df[col].cat.categories.dtype == 'object':
        #         df[col] = df[col].cat.rename_categories({each: str(each) for each in df[col].cat.categories})
        uc = union_categoricals([df[col] for df in dfs])
        # Change to union category for all dataframes
        for df in dfs:
            df[col] = pd.Categorical(df[col].values, categories=uc.categories)
    return pd.concat(dfs, **kwargs)

def generate_inpatient_triplets():
    print("[GENERATING INPATIENT TRIPLETS]")
    diag = pd.read_csv("./data/features/preproc_diag.csv.gz", compression='gzip',header=0)
    med = pd.read_csv("./data/features/preproc_med.csv.gz", compression='gzip',header=0)
    proc = pd.read_csv("./data/features/preproc_proc.csv.gz", compression='gzip',header=0)
    labs = pd.read_csv("./data/features/preproc_labs.csv.gz", compression='gzip',header=0)
    cohort = pd.read_csv("/storage/nassim/projects/MIMIC-IV-Data-Pipeline/data/cohort/cohort_non-icu_all_0_.csv.gz", compression='gzip',header=0)

    print("processing labs")
    admittime = pd.to_datetime(labs.admittime)
    lab_time_from_admit = pd.to_timedelta(labs.lab_time_from_admit, errors='coerce')
    labs['date'] = admittime + lab_time_from_admit
    labs.rename(columns={'valuenum': "value", 'itemid': "variable"}, inplace=True)
    labs = labs[["hadm_id", "date", 'variable', "value"]]
    labs['is_cat'] = 0
    labs['variable'] = labs.variable.astype(str).astype('category')
    
    
    # triplet encode diagnosis
    print('processing diagnosis')
    diag = diag.merge(cohort[['hadm_id', 'admittime']], on='hadm_id', how='left')
    diag.rename(columns={'admittime': 'date'}, inplace=True)
    diag['variable'] = "diagnosis"
    diag = diag[["hadm_id", "date", 'variable', "new_icd_code"]].rename(columns={"new_icd_code": "value"})
    diag['date'] = pd.to_datetime(diag.date)
    diag['variable'] = diag.variable.astype('category')
    diag['value'] = diag.value.astype('category')
    diag['is_cat'] = 1

    # triplet encode meds
    # med end date
    print('processing med end dates')
    med_end = med[['hadm_id', 'drug_name', 'stoptime']]
    med_end['stoptime'] = pd.to_datetime(med_end['stoptime'])
    med_end.rename(columns={'stoptime': 'date', 'drug_name': 'value'}, inplace=True)
    med_end['value'] = med_end.value.astype(str).astype('category')
    med_end['variable'] = "med_end"
    med_end['variable'] = med_end['variable'].astype('category')
    med_end = med_end[["hadm_id", "date", 'variable', "value"]]
    med_end['is_cat'] = 1

    # med rate
    print('processing med rates')
    med_rate = med[['hadm_id', 'drug_name', 'starttime', 'dose_val_rx']]
    med_rate['starttime'] = pd.to_datetime(med_rate['starttime'])
    med_rate['drug_name'] = med_rate['drug_name'].astype('category')
    med_rate.rename(columns={'starttime': 'date', 'dose_val_rx': 'value', 'drug_name': 'variable'}, inplace=True)
    med_rate = med_rate[["hadm_id", "date", 'variable', "value"]]
    med_rate['is_cat'] = 0

    # triplet encode procedures
    print('processing procedures')
    proc = proc[['hadm_id', 'admittime', 'icd_code']].rename(columns={'admittime': 'date', 'icd_code': 'value'})
    proc['date'] = pd.to_datetime(proc.date)
    proc['variable'] = "procedure"
    proc['variable'] = proc.variable.astype('category')
    proc['value'] = proc.value.astype(str).astype('category')
    proc['is_cat'] = 1

    # triplet encode encounters
    cohort['admittime'] = pd.to_datetime(cohort.admittime)
    cohort['dischtime'] = pd.to_datetime(cohort.dischtime)
    enc = cohort[['hadm_id', 'admittime', 'dischtime']].head().melt(id_vars=['hadm_id'], value_vars=['admittime', 'dischtime'])
    enc.rename(columns={'value': 'date', 'variable': 'value'}, inplace=True)
    enc['variable'] = "encounter"
    enc['variable'] = enc.variable.astype('category')
    enc['value'] = enc.value.astype('category')
    enc['is_cat'] = 1
    enc = enc[["hadm_id", "date", 'variable', "value"]]

    # concatenate dfs
    print("concatenating dfs")
    cat_df = concatenate([diag, med_end, proc, enc], ignore_index=True)
    cont_df = concatenate([labs, med_rate], ignore_index=True)
    cat_df = cat_df.reset_index(drop=True)
    cont_df = cont_df.reset_index(drop=True)
    # store raw dataframes
    cat_df['raw_index'] = cat_df.index
    cont_df['raw_index'] = cont_df.index + cat_df.shape[0]
    hf_subtype_output_dir = "/storage/shared/hf_subtype/datasets/MimicAdmission"
    all_variables = union_categoricals([cat_df.variable, cont_df.variable])
    cat_df['variable'] = pd.Categorical(cat_df.variable.values, categories=all_variables.categories)
    cont_df['variable'] = pd.Categorical(cont_df.variable.values, categories=all_variables.categories)
    cat_df.to_pickle(os.path.join(hf_subtype_output_dir, "raw_cat_timeseries.pkl"))
    cont_df.to_pickle(os.path.join(hf_subtype_output_dir, "raw_cont_timeseries.pkl"))

    print("[SUCCESSFULLY SAVED INPATIENT TRIPLETS]")


def generate_encounters_and_splits():
    print("[GENERATING ENCOUNTERS AND SPLITS]")
    cohort = pd.read_csv("/storage/nassim/projects/MIMIC-IV-Data-Pipeline/data/cohort/cohort_non-icu_all_0_.csv.gz", compression='gzip',header=0)
    cohort['admittime'] = pd.to_datetime(cohort.admittime)
    cohort['dischtime'] = pd.to_datetime(cohort.dischtime)
    enc = cohort.rename(
        columns={'subject_id': 'patient_id', 'admittime': "admit_date", 'dischtime': "discharge_date"})
    hf_subtype_output_dir = "/storage/shared/hf_subtype/datasets/MimicAdmission"
    enc.to_pickle(os.path.join(hf_subtype_output_dir, "encounters.pkl"))

    # generate splits
    train, non_train = train_test_split(enc.patient_id.unique().tolist(), test_size=0.2, random_state=87541)
    test, val = train_test_split(non_train, test_size=0.2, random_state=190847)
    pickle.dump(train, open(os.path.join(hf_subtype_output_dir, "train_patient_ids.pkl"), "wb"))
    pickle.dump(test, open(os.path.join(hf_subtype_output_dir, "test_patient_ids.pkl"), "wb"))
    pickle.dump(val, open(os.path.join(hf_subtype_output_dir, "val_patient_ids.pkl"), "wb"))


def normalize(df, train_mrns, patient_id="mgh_mrn"):
    # Filter out training data using MRNs
    train_df = df[df[patient_id].isin(train_mrns)]

    # Calculate mean and std for each 'variable' in the training set
    means = train_df.groupby("variable")["value"].mean().fillna(0)
    stds = train_df.groupby("variable")["value"].std().fillna(1).replace(0, 1)
    print("mean and std computed, merging")
    mapping = dict(means=means.to_dict(), stds=stds.to_dict())

    # Merge means and stds with the original DataFrame based on the 'variable' column
    df = df.merge(
        means.rename("mean"), how="left", left_on="variable", right_index=True
    )
    df = df.merge(stds.rename("std"), how="left", left_on="variable", right_index=True)

    # Perform the normalization
    df["value"] = (df["value"] - df["mean"]).div(df["std"]).fillna(0)

    # Drop the mean and std columns as they are not needed anymore
    df.drop(columns=["mean", "std"], inplace=True)

    return df, mapping


def make_numeric(x):
    try:
        return float(x)
    except:
        return None


def filter_data():
    print('[FILTERING DATA]')
    hf_subtype_output_dir = "/storage/shared/hf_subtype/datasets/MimicAdmission"
    cat_df = pd.read_pickle(os.path.join(hf_subtype_output_dir, "raw_cat_timeseries.pkl"))
    cont_df = pd.read_pickle(os.path.join(hf_subtype_output_dir, "raw_cont_timeseries.pkl"))
    cohort = pd.read_csv("/storage/nassim/projects/MIMIC-IV-Data-Pipeline/data/cohort/cohort_non-icu_all_0_.csv.gz", compression='gzip',header=0)
    cat_df = cat_df.merge(cohort[['hadm_id', 'subject_id']], on='hadm_id', how='left')
    cat_df = cat_df[["subject_id", "date", "variable", "value", "is_cat", "raw_index"]].rename(columns={"subject_id": "patient_id"})
    assert cat_df.patient_id.isna().sum() == 0
    cont_df = cont_df.merge(cohort[['hadm_id', 'subject_id']], on='hadm_id', how='left')
    cont_df = cont_df[["subject_id", "date", "variable", "value", "is_cat", "raw_index"]].rename(columns={"subject_id": "patient_id"})
    assert cont_df.patient_id.isna().sum() == 0
    # group rare cat_df values and label encode
    val_counts = cat_df.value.value_counts()
    rare_values = val_counts.index[val_counts < 1000]
    cat_df['value'] = cat_df['value'].cat.add_categories(['rare_value'])
    cat_df.loc[cat_df['value'].isin(rare_values), 'value'] = 'rare_value'
    cat_df['value'] = cat_df['value'].cat.remove_unused_categories()
    cat_df['value'] = cat_df['value'].cat.codes
    
    # group rare cat_df variables and label encode
    var_counts = cat_df.variable.value_counts()
    rare_vars = var_counts.index[var_counts < 1000]
    cat_df['variable'] = cat_df['variable'].cat.add_categories(['rare_cat_variable'])
    cat_df.loc[cat_df['variable'].isin(rare_vars), 'variable'] = 'rare_cat_variable'
    cat_df['variable'] = cat_df['variable'].cat.remove_unused_categories()
    cat_df['variable'] = cat_df['variable'].cat.codes
    
    # group rare cont_df variables and label encode
    var_counts = cont_df.variable.value_counts()
    rare_vars = var_counts.index[var_counts < 1000]
    cont_df['variable'] = cont_df['variable'].cat.add_categories(['rare_cont_variable'])
    cont_df.loc[cont_df['variable'].isin(rare_vars), 'variable'] = 'rare_cont_variable'
    cont_df['variable'] = cont_df['variable'].astype('category')
    cont_df['variable'] = cont_df['variable'].cat.remove_unused_categories()
    cont_df['variable'] = cont_df['variable'].cat.codes + cat_df['variable'].max() + 1

    train_patient_ids = pickle.load(open(os.path.join(hf_subtype_output_dir, "train_patient_ids.pkl"), "rb"))
    cont_df['value'] = cont_df['value'].map(make_numeric)
    cont_df.dropna(subset=["value"], inplace=True)
    cont_df = normalize(cont_df, train_mrns=train_patient_ids, patient_id="patient_id")[0]

    print(f"cat_df.variable.max(): {cat_df.variable.max()}")
    print(f"cat_df.value.max(): {cat_df.value.max()}")
    print(f"cont_df.variable.max(): {cont_df.variable.max()}")

    df = concatenate([cat_df, cont_df])

    df.sort_values(by=["patient_id", "date", "variable"], inplace=True)
    df.dropna(subset=['patient_id', 'date', 'variable', 'value'], inplace=True)
    df.to_pickle(os.path.join(hf_subtype_output_dir, "timeseries.pkl"))
    data_dict = dict(list(df.groupby("patient_id")))
    pickle.dump(data_dict, open(os.path.join(hf_subtype_output_dir, "timeseries_dict.pkl"), "wb"))


def check_sufficient(data_dict, enc):
    patient_id = enc.patient_id
    data = data_dict[patient_id]
    left_len = (data.date <= enc.date).sum()
    right_len = ((data.date > enc.date) & (data.date <= enc.discharge_date)).sum()
    sufficient = (left_len >= 16) and (right_len >= 16)
    return sufficient


def get_sufficient_encounters():
    print("[GET SUFFICIENT ENCOUNTERS]")
    hf_subtype_output_dir = "/storage/shared/hf_subtype/datasets/MimicAdmission/"
    cohort = pd.read_csv("/storage/nassim/projects/MIMIC-IV-Data-Pipeline/data/cohort/cohort_non-icu_all_0_.csv.gz", compression='gzip',header=0)
    cohort.rename(columns={"admittime": "date", "dischtime": "discharge_date", "subject_id": "patient_id"}, inplace=True)
    encounters = cohort[['patient_id', 'date', 'discharge_date', 'mortality_label', 'readmission_label', 'los_label', 'dod']]
    data_dict = pickle.load(open(os.path.join(hf_subtype_output_dir, "timeseries_dict.pkl"), "rb"))
    print('processing sufficient encounters')
    encounters = encounters[encounters.patient_id.isin(data_dict.keys())]
    encounters['date'] = pd.to_datetime(encounters.date)
    encounters['discharge_date'] = pd.to_datetime(encounters.discharge_date)
    encounters = encounters[(((encounters.discharge_date - encounters.date) / pd.Timedelta(days=1)) >= 4/24)] # LOS must be at least 4 hours
    encounters.to_pickle(os.path.join(hf_subtype_output_dir, "processed_encounter.pkl"))
    print(f"Number of encounters: {encounters.shape[0]}")
    sufficient_encounters = encounters[encounters.progress_apply(partial(check_sufficient, data_dict), axis=1)]
    sufficient_encounters.to_pickle(os.path.join(hf_subtype_output_dir, "sufficient_encounters.pkl"))
    print(f"Number of sufficient encounters: {sufficient_encounters.shape[0]}")


# generate_inpatient_triplets()
# generate_encounters_and_splits()
# filter_data()
get_sufficient_encounters()
