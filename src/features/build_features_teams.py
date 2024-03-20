import pandas as pd

def build_teams(gender, csvs):
    """ Build teams dataframe from csv
    Args:
        gender (str): M or W
        csvs (dict): dictionary of csv files
    Returns:
        pd.DataFrame: dataframe of teams
    """
    teams = csvs["{}Teams".format(gender)].copy()
    teams = teams.drop('TeamName', axis=1)
    teams = teams.set_index('TeamID')

    return teams

def build_matchups(gender, csvs):
    teams = csvs["{}Teams".format(gender)].copy()
    teams = teams[['TeamID']]
    teams = pd.merge(teams, teams, how='cross')
    teams = teams.rename(columns={'TeamID_x': 'TeamID', 'TeamID_y': 'OTeamID'})
    teams = teams[teams['TeamID'] != teams['OTeamID']]
    teams = teams.set_index(['TeamID', 'OTeamID'])

    return teams