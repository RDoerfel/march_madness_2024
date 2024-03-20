#%%
import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress

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

def build_elo(gender, results, teams, csvs):
    """ Build Elo dataframe from 
    Args:
        gender (str): M or W
        results (pd.DataFrame): dataframe of results
        teams (pd.DataFrame): dataframe of teams
        csvs (dict): dictionary of csv files
    Returns:
        pd.DataFrame: dataframe of Elo ratings
    """
    csv_names = ['NCAATourneyCompactResults', 'RegularSeasonCompactResults']
    csv_names = list(map(lambda x: gender + x, csv_names))
    csv      = list(map(lambda x: csvs[x], csv_names))

    tourneys = results.copy()
    tourneys['tourney'] = 0
    tourneys.loc[len(csv[0]):, 'tourney'] = 1
    tourneys = tourneys.sort_values(['Season', 'DayNum'])
    
    return create_elo_data(teams.reset_index().TeamID, tourneys).drop('Season', axis=1).groupby('TeamID').mean()

