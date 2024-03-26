import sys
from pathlib import Path
import os
import importlib
import pandas as pd
from pandas.api.types import union_categoricals
from sklearn.model_selection import train_test_split
import pickle

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
from feature_selection_hosp import feature_nonicu

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
from feature_selection_icu import feature_icu, preprocess_features_icu, generate_summary_icu, features_selection_icu

import fairness
import callibrate_output


# version = 'Version 2'
# radio_input4 = "ALL"  # or "mortality"
# radio_input3 = "No Disease Filter" # add columns for each disease for encounters dataframe

data_type = "ICU" # or "Non-ICU"
icd_code = 'No Disease Filter'
time = 0
root_dir = os.path.dirname(os.path.abspath(__file__))

version_path = 'mimiciv/2.0'
disease_label = ""
label = "ALL"

# uncomment the next line to regenerate cohort (takes 10 hours to process readmissions label)
# cohort_output = day_intervals_cohort_v2.extract_data(data_type, label, time, icd_code, root_dir, disease_label)

cohort_output = "cohort_" + data_type.lower() + "_" + label.lower().replace(" ", "_") + "_" + str(time) + "_" + disease_label

diag_flag, out_flag, chart_flag, proc_flag, med_flag = True, True, True, True, True

# uncomment the next line to rerun some processing -- takes around 30 min
# feature_icu(cohort_output, version_path, diag_flag, out_flag, chart_flag, proc_flag, med_flag)
# This runs super fast -- convert diagnoses to ICD-10 and Converts drug names into NDC non-proprietary drug name
# group_diag = "Convert ICD-9 to ICD-10 codes"
# preprocess_features_icu(cohort_output, diag_flag, group_diag, False, False, False, 0, 0)

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

def generate_icu_triplets():
    print("[GENERATING FEATURE SUMMARY]")
    diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
    med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip',header=0)
    proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip',header=0) 
    out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip',header=0)
    chart = pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0)
    cohort = pd.read_csv("/storage/nassim/projects/MIMIC-IV-Data-Pipeline/data/cohort/cohort_icu_all_0_.csv.gz", compression='gzip',header=0)

    # triplet encode charts print('processing charts')
    chart = chart.merge(cohort[['stay_id', 'intime']], on='stay_id', how='left')
    intime = pd.to_datetime(chart.intime)
    time_from_admit = pd.to_timedelta(chart.event_time_from_admit, errors='coerce')
    chart['date'] = intime + time_from_admit
    chart.rename(columns={'valuenum': "value", 'itemid': "variable"}, inplace=True)
    chart = chart[["stay_id", "date", 'variable', "value"]]
    chart['is_cat'] = 0
    chart['variable'] = chart.variable.astype(str).astype('category')

    # triplet encode diagnosis
    print('processing diagnosis')
    diag = diag.merge(cohort[['stay_id', 'intime']], on='stay_id', how='left')
    diag.rename(columns={'intime': 'date'}, inplace=True)
    diag['variable'] = "diagnosis"
    diag = diag[["stay_id", "date", 'variable', "new_icd_code"]].rename(columns={"new_icd_code": "value"})
    diag['date'] = pd.to_datetime(diag.date)
    diag['variable'] = diag.variable.astype('category')
    diag['value'] = diag.value.astype('category')
    diag['is_cat'] = 1

    # triplet encode meds
    # med start date
    print('processing med start dates')
    med_start = med[['stay_id', 'itemid', 'starttime']]
    med_start['starttime'] = pd.to_datetime(med_start['starttime'])
    med_start.rename(columns={'starttime': 'date', 'itemid': 'variable'}, inplace=True)
    med_start['value'] = "med_start"
    med_start['value'] = med_start.value.astype('category')
    med_start['variable'] = med_start['variable'].astype(str).astype('category')
    med_start = med_start[["stay_id", "date", 'variable', "value"]]
    med_start['is_cat'] = 1

    # med end date
    print('processing med end dates')
    med_end = med[['stay_id', 'itemid', 'endtime']]
    med_end['endtime'] = pd.to_datetime(med_end['endtime'])
    med_end.rename(columns={'endtime': 'date', 'itemid': 'variable'}, inplace=True)
    med_end['value'] = "med_end"
    med_end['value'] = med_end.value.astype('category')
    med_end['variable'] = med_end['variable'].astype(str).astype('category')
    med_end = med_end[["stay_id", "date", 'variable', "value"]]
    med_end['is_cat'] = 1

    # med rate
    print('processing med rates')
    med_rate = med[['stay_id', 'itemid', 'starttime', 'rate']]
    med_rate['starttime'] = pd.to_datetime(med_rate['starttime'])
    med_rate['itemid'] = med_rate['itemid'].astype(str).astype('category')
    med_rate.rename(columns={'starttime': 'date', 'rate': 'value', 'itemid': 'variable'}, inplace=True)
    med_rate = med_rate[["stay_id", "date", 'variable', "value"]]
    med_rate['is_cat'] = 0

    # triplet encode procedures
    print('processing procedures')
    proc = proc[['stay_id', 'starttime', 'itemid']].rename(columns={'starttime': 'date', 'itemid': 'value'})
    proc['date'] = pd.to_datetime(proc.date)
    proc['variable'] = "procedure"
    proc['variable'] = proc.variable.astype('category')
    proc['value'] = proc.value.astype(str).astype('category')
    proc['is_cat'] = 1
    
    # triplet encode out
    print('processing out')
    out = out[['stay_id', 'charttime', 'itemid']]
    out.rename(columns={'charttime': 'date', "itemid": "value"}, inplace=True)
    out['date'] = pd.to_datetime(proc.date)
    out['variable'] = "chart"
    out['variable'] = out.variable.astype('category')
    out['value'] = out.value.astype(str).astype('category')
    out['is_cat'] = 1

    # concatenating
    print("concatenating dfs")
    cat_df = concatenate([diag, med_start, med_end, proc, out], ignore_index=True)
    cont_df = concatenate([chart, med_rate], ignore_index=True)
    cat_df = cat_df.reset_index(drop=True)
    cont_df = cont_df.reset_index(drop=True)
    # store raw dataframes
    cat_df['raw_index'] = cat_df.index
    cont_df['raw_index'] = cont_df.index + cat_df.shape[0]
    hf_subtype_output_dir = "/storage/shared/hf_subtype/datasets/MimicIcu"
    all_variables = union_categoricals([cat_df.variable, cont_df.variable])
    cat_df['variable'] = pd.Categorical(cat_df.variable.values, categories=all_variables.categories)
    cont_df['variable'] = pd.Categorical(cont_df.variable.values, categories=all_variables.categories)
    cat_df.value = cat_df.value.astype('category')
    cat_df.to_pickle(os.path.join(hf_subtype_output_dir, "raw_cat_timeseries.pkl"))
    cont_df.to_pickle(os.path.join(hf_subtype_output_dir, "raw_cont_timeseries.pkl"))


def generate_encounters_and_splits():
    print("[GENERATING ENCOUNTERS AND SPLITS]")
    cohort = pd.read_csv("/storage/nassim/projects/MIMIC-IV-Data-Pipeline/data/cohort/cohort_icu_all_0_.csv.gz", compression='gzip',header=0)
    cohort['intime'] = pd.to_datetime(cohort.intime)
    cohort['outtime'] = pd.to_datetime(cohort.outtime)
    enc = cohort.rename(
        columns={'subject_id': 'patient_id', 'intime': "admit_date", 'outtime': "discharge_date"})
    hf_subtype_output_dir = "/storage/shared/hf_subtype/datasets/MimicIcu"
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

def filter_data():
    hf_subtype_output_dir = "/storage/shared/hf_subtype/datasets/MimicIcu"
    cat_df = pd.read_pickle(os.path.join(hf_subtype_output_dir, "raw_cat_timeseries.pkl"))
    cont_df = pd.read_pickle(os.path.join(hf_subtype_output_dir, "raw_cont_timeseries.pkl"))
    cohort = pd.read_csv("/storage/nassim/projects/MIMIC-IV-Data-Pipeline/data/cohort/cohort_icu_all_0_.csv.gz", compression='gzip',header=0)
    cat_df = cat_df.merge(cohort[['stay_id', 'subject_id']], on='stay_id', how='left')
    cat_df = cat_df[["stay_id", "date", "variable", "value", "is_cat", "raw_index"]].rename(columns={"stay_id": "patient_id"})
    assert cat_df.patient_id.isna().sum() == 0
    cont_df = cont_df.merge(cohort[['stay_id', 'subject_id']], on='stay_id', how='left')
    cont_df = cont_df[["stay_id", "date", "variable", "value", "is_cat", "raw_index"]].rename(columns={"stay_id": "patient_id"})
    assert cont_df.patient_id.isna().sum() == 0
    val_counts = cat_df.value.value_counts()
    rare_values = val_counts.index[val_counts < 1000]
    cat_df['value'] = cat_df['value'].cat.add_categories(['rare_value'])
    cat_df.loc[cat_df['value'].isin(rare_values), 'value'] = 'rare_value'
    cat_df['value'] = cat_df['value'].cat.remove_unused_categories()
    cat_df['value'] = cat_df['value'].cat.codes

    train_patient_ids = pickle.load(open(os.path.join(hf_subtype_output_dir, "train_patient_ids.pkl"), "rb"))
    cont_df = normalize(cont_df, train_mrns=train_patient_ids, patient_id="patient_id")[0]

    df = concatenate([cat_df, cont_df])

    var_counts = df.variable.value_counts()
    rare_vars = var_counts.index[var_counts < 1000]
    df['variable'] = df['variable'].cat.add_categories(['rare_variable'])
    df.loc[df['variable'].isin(rare_vars), 'variable'] = 'rare_variable'
    df['variable'] = df['variable'].cat.remove_unused_categories()
    df['variable'] = df['variable'].cat.codes
    df.sort_values(by=["patient_id", "date"], inplace=True)
    df.to_pickle(os.path.join(hf_subtype_output_dir, "timeseries.pkl"))
    data_dict = dict(list(df.groupby("patient_id")))
    pickle.dump(data_dict, open(os.path.join(hf_subtype_output_dir, "timeseries_dict.pkl"), "wb"))


# generate_icu_triplets()
# generate_encounters_and_splits()
# filter_data()


def generate_events():
    hf_subtype_output_dir = "/storage/shared/hf_subtype/datasets/MimicIcu"
    cat_df = pd.read_pickle(os.path.join(hf_subtype_output_dir, "raw_cat_timeseries.pkl"))
    cont_df = pd.read_pickle(os.path.join(hf_subtype_output_dir, "raw_cont_timeseries.pkl"))
    cohort = pd.read_csv("/storage/nassim/projects/MIMIC-IV-Data-Pipeline/data/cohort/cohort_icu_all_0_.csv.gz", compression='gzip',header=0)
    map_itemids = ['220052', '220181', '224322', '225312', '229827']
    mask_ventilation_itemids = ['225303', '225792', '225794', '226260']
    pao2_itemids = ['220224']
    sao2_itemids = ['220227']
    map_events = cont_df[cont_df.variable.isin(map_itemids)]
    pao2_events = cont_df[cont_df.variable.isin(pao2_itemids)]
    sao2_events = cont_df[cont_df.variable.isin(sao2_itemids)]

    hypotension_events = map_events[map_events.value <= 60]
    hypoxia_events = pd.concat([pao2_events[pao2_events.value <= 90], sao2_events[sao2_events.value <= 90]])
    mechanical_vent_events = cat_df[cat_df.value.isin(mask_ventilation_itemids)]

    events = pd.concat([hypotension_events, hypoxia_events, mechanical_vent_events])
    print(hypotension_events.stay_id.nunique())
    print(hypoxia_events.stay_id.nunique())
    print(mechanical_vent_events.stay_id.nunique())
    print(events.stay_id.nunique())
    import pdb; pdb.set_trace()
    

generate_events()
