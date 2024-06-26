# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data import load_kaggle

def _make_path(path):
    if isinstance(path, str):
        return Path(path)
    elif isinstance(path, Path):
        return path
    else:
        raise ValueError
        
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))   
def main(input_filepath: Path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    input_filepath = _make_path(input_filepath)
    load_kaggle.load_kaggle(input_filepath, dataset='march-machine-learning-mania-2024')
    load_kaggle.unzip_kaggle(input_filepath / 'march-machine-learning-mania-2024.zip', input_filepath / 'mmlm24')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
