import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.features.build_features import get_csv_filesin_dir, build_df
from src.features.build_features_teams import build_teams, build_matchups
from src.features.build_features_elo import build_elo
from src.features.build_features_history import build_history, build_avg
from src.features.build_features_results import build_results, build_season_results
from src.features.build_features_seeds import build_seeds, build_rankings


def _build_df(gender,csvs):
    results= build_results(gender, csvs)
    teams = build_teams(gender, csvs)
    results = build_results(gender, csvs)
    teams = build_teams(gender, csvs)
    elo = build_elo(gender, results, teams, csvs)
    season_results = build_season_results(results)
    seeds = build_seeds(gender, csvs)
    if gender == 'M':
        rankings = build_rankings(gender, csvs) #only M has rankings
    else: 
        rankings = None
    history = build_history(season_results, seeds, teams, elo, rankings)
    avg = build_avg(history)
    matchups = build_matchups(gender, csvs)
    df = build_df(history, matchups, avg)
    return df

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    csvs = get_csv_filesin_dir(input_filepath)
    for gender in ['M', 'W']:
        df = _build_df(gender, csvs)
        output_name = "features_{}.csv".format(gender)
        df.to_csv(Path(output_filepath) / output_name)

if __name__ == '__main__':
    main()



