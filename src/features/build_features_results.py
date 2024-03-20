import pandas as pd

def build_results(gender, csvs):
    """ Build results dataframe from csvs
    Args:
        gender (str): M or W
        csvs (dict): dictionary of csv files
    Returns:
        pd.DataFrame: dataframe of results
    """
    csv_names = ['NCAATourneyCompactResults', 'RegularSeasonCompactResults']
    csv_names = list(map(lambda x: gender + x, csv_names))
    csvs      = list(map(lambda x: csvs[x], csv_names))

    return pd.concat(csvs)

def _winner(ids):
    id, wId, lId = ids

    return int(id == wId)

def _opponent(x):
    winInt, wId, lId = x
    win = not winInt
    
    return wId if win else lId

def _score_diff(x):
    winInt, wScore, lScore = x
    win = not winInt
    
    return (wScore - lScore) if win else (lScore - wScore)

def build_season_results(df):
    season_results = df
    season_results['TeamID'] = season_results[['WTeamID', 'LTeamID']].values.tolist()
    season_results = season_results.explode('TeamID')
    season_results['Win'] = season_results[['TeamID', 'WTeamID', 'LTeamID']].apply(_winner, axis=1)
    season_results['Defeat'] = season_results['Win'].apply(lambda x: 1 - x)
    season_results['Games'] = season_results['Win'] + season_results['Defeat']
    season_results['ScoreDiff'] = season_results[['Win', 'WScore', 'LScore']].apply(_score_diff, axis=1)
    season_results['OTeamID'] = season_results[['Win', 'WTeamID', 'LTeamID']].apply(_opponent, axis=1)
    season_results['Home'] = season_results['WLoc'].apply(lambda x: int(x[0] == 'H'))
    season_results = season_results.drop(['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT'], axis=1)
    season_results = season_results.groupby(by=['TeamID', 'OTeamID']).sum()
    season_results['WinRatio'] = season_results['Win'] / season_results['Games']
    season_results = season_results.drop(['Win', 'Defeat'], axis=1)

    return season_results



