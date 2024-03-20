import glob
import os
import pandas as pd

def get_csv_filesin_dir(path):
    """ Returns a dictionary of csv files in a directory 
    Args:
        path (str): path to directory
    Returns:
        dict: dictionary of csv files"""
    csv = {}
    for path in glob.glob(str(path) + "/*.csv"):
        csv[os.path.basename(path).split('.')[0]] = pd.read_csv(path, encoding='cp1252')
    return csv

def build_df(history, matchups, avg):
    df = pd.merge(matchups, history, on=['TeamID', 'OTeamID'], how='left')
    df = df.fillna(avg).fillna(0)

    if 'FirstD1Season' in df.columns:
        df['FirstD1Season'] = df['FirstD1Season'].astype(int)
        df['LastD1Season'] = df['LastD1Season'].astype(int)
    
    return df