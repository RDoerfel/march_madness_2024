import pandas as pd

def build_history(season_results, seeds, teams, elo, rankings=None):
    history = season_results.join(teams, on='TeamID').join(seeds, on='TeamID').join(elo, on='TeamID')
    history = history.reset_index()
    history = pd.merge(history, seeds, left_on='OTeamID', right_on='TeamID', suffixes=('_T', '_O'))
    history['SeedDiff'] = history['Seed_T'] - history['Seed_O']
    history = history.drop(['Seed_T', 'Seed_O'], axis=1)

    if rankings is not None:
        history = history.join(rankings, on='TeamID')
        history = pd.merge(history, rankings, left_on='OTeamID', right_on='TeamID', suffixes=('_T', '_O'))
        history['RankingsDiff'] = history['OrdinalRank_T'] - history['OrdinalRank_O']
        history = history.drop(['OrdinalRank_T', 'OrdinalRank_O'], axis=1)
    
    return history.set_index(['TeamID', 'OTeamID']).fillna(0)

def build_avg(history):
    agg = {}
    for col in history.columns:
        if col == 'Games' or col == 'Home':
            agg[col] = 'sum'
        else:
            agg[col] = 'mean'
    
    avg = history.groupby('TeamID').agg(agg)
    
    return avg