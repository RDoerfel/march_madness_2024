#%%
import pandas as pd
from pathlib import Path
import glob
import os
from tqdm import tqdm
import numpy as np 
from scipy.stats import linregress
from sklearn.model_selection import cross_val_score, train_test_split
import lightgbm as lgb
import optuna as op

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATADIR = PROJECT_DIR / 'data' / 'raw' / 'mmlm24'

# gatter all CSVs
CSV = {}
for path in glob.glob(str(DATADIR) + "/*.csv"):
    CSV[os.path.basename(path).split('.')[0]] = pd.read_csv(path, encoding='cp1252')

## Build results
def build_results(gender):
    csv_names = ['NCAATourneyCompactResults', 'RegularSeasonCompactResults']
    csv_names = list(map(lambda x: gender + x, csv_names))
    csvs      = list(map(lambda x: CSV[x], csv_names))
    
    return pd.concat(csvs)

results_m = build_results('M')
results_w = build_results('W')

print(results_m.head())
print(results_w.head())

## Build teams
def build_teams(gender):
    teams = CSV["{}Teams".format(gender)].copy()
    teams = teams.drop('TeamName', axis=1)
    teams = teams.set_index('TeamID')
    
    return teams

teams_m = build_teams('M')
teams_w = build_teams('W')

print(teams_m.head())
print(teams_w.head())

## Build elo
def calculate_elo(teams, data, initial_rating=2000, k=140, alpha=None):
    '''
    Calculate Elo ratings for each team based on match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000).
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).
    - alpha (float or None): Tuning parameter for the multiplier for the margin of victory. No multiplier if None.

    Returns: 
    - list: Historical ratings of the winning team (WTeam).
    - list: Historical ratings of the losing team (LTeam).
    '''
    
    # Dictionary to keep track of current ratings for each team
    team_dict = {}
    for team in teams:
        team_dict[team] = initial_rating
        
    # Lists to store ratings for each team in each game
    r1, r2 = [], []
    margin_of_victory = 1

    # Iterate through the game data
    for wteam, lteam, ws, ls  in tqdm(zip(data.WTeamID, data.LTeamID, data.WScore, data.LScore), total=len(data)):
        # Append current ratings for teams to lists
        r1.append(team_dict[wteam])
        r2.append(team_dict[lteam])

        # Calculate expected outcomes based on Elo ratings
        rateW = 1 / (1 + 10 ** ((team_dict[lteam] - team_dict[wteam]) / initial_rating))
        rateL = 1 / (1 + 10 ** ((team_dict[wteam] - team_dict[lteam]) / initial_rating))
        
        if alpha:
            margin_of_victory = (ws - ls)/alpha

        # Update ratings for winning and losing teams
        team_dict[wteam] += k * margin_of_victory * (1 - rateW)
        team_dict[lteam] += k * margin_of_victory * (0 - rateL)

        # Ensure that ratings do not go below 1
        if team_dict[lteam] < 1:
            team_dict[lteam] = 1
        
    return r1, r2

def create_elo_data(teams, data, initial_rating=2000, k=140, alpha=None):
    '''
    Create a DataFrame with summary statistics of Elo ratings for teams based on historical match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000).
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).

    Returns: 
    - DataFrame: Summary statistics of Elo ratings for teams throughout a season.
    '''
    
    r1, r2 = calculate_elo(teams, data, initial_rating, k, alpha)
    
    # Concatenate arrays vertically
    seasons = np.concatenate([data.Season, data.Season])
    days = np.concatenate([data.DayNum, data.DayNum])
    teams = np.concatenate([data.WTeamID, data.LTeamID])
    tourney = np.concatenate([data.tourney, data.tourney])
    ratings = np.concatenate([r1, r2])
    # Create a DataFrame
    rating_df = pd.DataFrame({
        'Season': seasons,
        'DayNum': days,
        'TeamID': teams,
        'Rating': ratings,
        'Tourney': tourney
    })

    # Sort DataFrame and remove tournament data
    rating_df.sort_values(['TeamID', 'Season', 'DayNum'], inplace=True)
    rating_df = rating_df[rating_df['Tourney'] == 0]
    grouped = rating_df.groupby(['TeamID', 'Season'])
    results = grouped['Rating'].agg(['mean', 'median', 'std', 'min', 'max', 'last'])
    results.columns = ['Rating_Mean', 'Rating_Median', 'Rating_Std', 'Rating_Min', 'Rating_Max', 'Rating_Last']
    results['Rating_Trend'] = grouped.apply(lambda x: linregress(range(len(x)), x['Rating']).slope, include_groups=False)
    results.reset_index(inplace=True)
    
    return results

def build_elo(gender, results, teams):
    csv_names = ['NCAATourneyCompactResults', 'RegularSeasonCompactResults']
    csv_names = list(map(lambda x: gender + x, csv_names))
    csvs      = list(map(lambda x: CSV[x], csv_names))

    tourneys = results.copy()
    tourneys['tourney'] = 0
    tourneys.loc[len(csvs[0]):, 'tourney'] = 1
    tourneys = tourneys.sort_values(['Season', 'DayNum'])
    
    return create_elo_data(teams.reset_index().TeamID, tourneys).drop('Season', axis=1).groupby('TeamID').mean()


elo_m = build_elo('M', results_m, teams_m)
elo_w = build_elo('W', results_w, teams_w)

print(elo_m.head())
print(elo_w.head())

# build game
def winner(ids):
    id, wId, lId = ids

    return int(id == wId)

def opponent(x):
    winInt, wId, lId = x
    win = not winInt
    
    return wId if win else lId

def score_diff(x):
    winInt, wScore, lScore = x
    win = not winInt
    
    return (wScore - lScore) if win else (lScore - wScore)

def build_season_results(df):
    season_results = df
    season_results['TeamID'] = season_results[['WTeamID', 'LTeamID']].values.tolist()
    season_results = season_results.explode('TeamID')
    season_results['Win'] = season_results[['TeamID', 'WTeamID', 'LTeamID']].apply(winner, axis=1)
    season_results['Defeat'] = season_results['Win'].apply(lambda x: 1 - x)
    season_results['Games'] = season_results['Win'] + season_results['Defeat']
    season_results['ScoreDiff'] = season_results[['Win', 'WScore', 'LScore']].apply(score_diff, axis=1)
    season_results['OTeamID'] = season_results[['Win', 'WTeamID', 'LTeamID']].apply(opponent, axis=1)
    season_results['Home'] = season_results['WLoc'].apply(lambda x: int(x[0] == 'H'))
    season_results = season_results.drop(['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT'], axis=1)
    season_results = season_results.groupby(by=['TeamID', 'OTeamID']).sum()
    season_results['WinRatio'] = season_results['Win'] / season_results['Games']
    season_results = season_results.drop(['Win', 'Defeat'], axis=1)

    return season_results

season_results_m = build_season_results(results_m)
season_results_w = build_season_results(results_w)

print(season_results_m.head())
print(season_results_w.head())

## Seeds
def clean_seeds(seed):
    res = seed[1:]

    if len(res) > 2:
        res = res[:-1]

    return int(res)

def build_seeds(gender):
    seeds = CSV["{}NCAATourneySeeds".format(gender)] 
    seeds['Seed'] = seeds['Seed'].apply(clean_seeds)
    seeds = seeds.drop('Season', axis=1)
    seeds = seeds.groupby(by='TeamID').mean()
    
    return seeds

seeds_m = build_seeds('M')
seeds_w = build_seeds('W')

print(seeds_m.head())
print(seeds_w.head())

def build_rankings(gender):
    rankings = CSV["{}MasseyOrdinals".format(gender)]
    rankings = rankings.drop(['SystemName', 'RankingDayNum'], axis=1)
    rankings = rankings.groupby(by='TeamID').mean()
    rankings = rankings.drop('Season', axis=1)

    return rankings

rankings_m = build_rankings('M')

print(rankings_m.head())

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

history_m = build_history(season_results_m, seeds_m, teams_m, elo_m, rankings_m)
history_w = build_history(season_results_w, seeds_w, teams_w, elo_w)

print(history_m.head())
print(history_w.head())

def build_avg(history):
    agg = {}
    for col in history.columns:
        if col == 'Games' or col == 'Home':
            agg[col] = 'sum'
        else:
            agg[col] = 'mean'
    
    avg = history.groupby('TeamID').agg(agg)
    
    return avg

avg_m = build_avg(history_m)
avg_w = build_avg(history_w)

print(avg_m.head())
print(avg_w.head())

def build_matchups(gender):
    teams = CSV["{}Teams".format(gender)].copy()
    teams = teams[['TeamID']]
    teams = pd.merge(teams, teams, how='cross')
    teams = teams.rename(columns={'TeamID_x': 'TeamID', 'TeamID_y': 'OTeamID'})
    teams = teams[teams['TeamID'] != teams['OTeamID']]
    teams = teams.set_index(['TeamID', 'OTeamID'])

    return teams

matchups_m = build_matchups('M')
matchups_w = build_matchups('W')

print(matchups_m.head())
print(matchups_w.head())

## Features
def build_df(history, matchups, avg):
    df = pd.merge(matchups, history, on=['TeamID', 'OTeamID'], how='left')
    df = df.fillna(avg).fillna(0)

    if 'FirstD1Season' in df.columns:
        df['FirstD1Season'] = df['FirstD1Season'].astype(int)
        df['LastD1Season'] = df['LastD1Season'].astype(int)
    
    return df

df_m = build_df(history_m, matchups_m, avg_m)
df_w = build_df(history_w, matchups_w, avg_w)

print(df_m.head())
print(df_w.head())

corr_m = df_m.corr()
corr_m.style.background_gradient(cmap='coolwarm')

corr_w = df_w.corr()
corr_w.style.background_gradient(cmap='coolwarm')

corr_m = df_m.corr()['WinRatio'].sort_values(ascending=False)
high_corr_m = corr_m[[abs(corr_m) > 0.1 for corr_m in corr_m]]

corr_w = df_w.corr()['WinRatio'].sort_values(ascending=False)
high_corr_w = corr_w[[abs(corr_w) > 0.1 for corr_w in corr_w]]

print(high_corr_m)
print(high_corr_w)

# training
def score_dataset(lgbm_params, X, y):
    reg   = lgb.LGBMRegressor(**lgbm_params)
    score = cross_val_score(reg, X, y)
    score = -1 * score.mean() + score.std()

    return score

def objective(trial, X, y):
    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'num_leaves': trial.suggest_int('num_leaves', 5, 31),
        'n_estimators': trial.suggest_int('n_estimators', 1, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 300),
        'device_type': 'cpu',
        'verbose': -1
    }

    return score_dataset(params, X, y)

def study(X, y):
    study = op.create_study()
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100, n_jobs=-1, show_progress_bar=True)

    return study.best_params

def build_x_y(df):
    target_column = 'WinRatio'
    feature_columns = df.columns.tolist()
    feature_columns.remove(target_column)
    
    return df[feature_columns], df[target_column]

X_m, y_m = build_x_y(df_m)
X_w, y_w = build_x_y(df_w)

params_m = study(X_m, y_m)
params_w = study(X_w, y_w)

def accuracy(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    reg_test = lgb.LGBMRegressor(**params)
    reg_test.fit(X_train, y_train)

    print('LightGBM Model accuracy score: {0:0.4f}'.format(reg_test.score(X_test, y_test)))
    print('LightGBM Model accuracy score [train]: {0:0.4f}'.format(reg_test.score(X_train, y_train)))

accuracy(X_m, y_m, params_m)
accuracy(X_w, y_w, params_w)

# Predictions
def build_wins(X, y, params):
    reg = lgb.LGBMRegressor(**params)
    reg.fit(X, y)

    wins = X
    wins['WinRatio'] = reg.predict(X)
    wins = wins[['WinRatio']]

    return wins

wins_m = build_wins(X_m, y_m, params_m)
wins_w = build_wins(X_w, y_w, params_w)

print(wins_m.head())
print(wins_w.head())

def build_slots(gender):
    slots = CSV["{}NCAATourneySlots".format(gender)]
    slots = slots[slots['Season'] == 2023]
    slots = slots[slots['Slot'].str.contains('R')] 

    return slots

slots_m = build_slots('M')  
slots_w = build_slots('W')

print(slots_m.head())
print(slots_w.head())

def build_seeds_2024():
    seeds_2024 = CSV['2024_tourney_seeds']

    return seeds_2024[seeds_2024['Tournament'] == 'M'], seeds_2024[seeds_2024['Tournament'] == 'W']

seeds_2024_m, seeds_2024_w = build_seeds_2024()

print(seeds_2024_m.head())
print(seeds_2024_w.head())

def prepare_data(seeds):
    seed_dict = seeds.set_index('Seed')['TeamID'].to_dict()
    inverted_seed_dict = {value: key for key, value in seed_dict.items()}

    return seed_dict, inverted_seed_dict

def simulate(round_slots, seeds, inverted_seeds, wins):
    '''
    Simulates each round of the tournament.

    Parameters:
    - round_slots: DataFrame containing information on who is playing in each round.
    - seeds (dict): Dictionary mapping seed values to team IDs.
    - inverted_seeds (dict): Dictionary mapping team IDs to seed values.
    - wins (DF): DF that includes wins prediction per matchup.
    Returns:
    - list: List with winning team IDs for each match.
    - list: List with corresponding slot names for each match.
    '''
    winners = []
    slots = []

    for slot, strong, weak in zip(round_slots.Slot, round_slots.StrongSeed, round_slots.WeakSeed):
        team_1, team_2 = seeds[strong], seeds[weak]

        team_1_prob = wins.loc[team_1, team_2].WinRatio
        winner = np.random.choice([team_1, team_2], p=[team_1_prob, 1 - team_1_prob])

        # Append the winner and corresponding slot to the lists
        winners.append(winner)
        slots.append(slot)

        seeds[slot] = winner

    return [inverted_seeds[w] for w in winners], slots

def run_simulation(seeds, round_slots, wins, brackets):
    '''
    Runs a simulation of bracket tournaments.

    Parameters:
    - seeds (pd.DataFrame): DataFrame containing seed information.
    - round_slots (pd.DataFrame): DataFrame containing information about the tournament rounds.
    - wins (DF): DF that includes wins prediction per matchup.
    - brackets (int): Number of brackets to simulate.
    Returns:
    - pd.DataFrame: DataFrame with simulation results.
    '''
    # Get relevant data for the simulation
    seed_dict, inverted_seed_dict = prepare_data(seeds)
    # Lists to store simulation results
    results = []
    bracket = []
    slots = []

    # Iterate through the specified number of brackets
    for b in tqdm(range(1, brackets + 1)):
        # Run single simulation
        r, s = simulate(round_slots, seed_dict, inverted_seed_dict, wins)
        
        # Update results
        results.extend(r)
        bracket.extend([b] * len(r))
        slots.extend(s)

    # Create final DataFrame
    result_df = pd.DataFrame({'Bracket': bracket, 'Slot': slots, 'Team': results})

    return result_df

n_brackets = 100000
result_m = run_simulation(seeds_2024_m, slots_m, wins_m, n_brackets)
result_m.insert(0, 'Tournament', 'M')
result_w = run_simulation(seeds_2024_w, slots_w, wins_w, n_brackets)
result_w.insert(0, 'Tournament', 'W')


# %%
