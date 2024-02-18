#%%
import os
import logging
from pathlib import Path
from shutil import unpack_archive

def load_kaggle(datadir, dataset):
    cmd = f"kaggle competitions download -c {dataset} -p {datadir}"
    os.system(cmd)

def unzip_kaggle(datadir, outdir):
    unpack_archive(datadir, extract_dir=outdir)

def main(datadir):  
    load_kaggle(datadir)
    unpack_archive(datadir / 'march-machine-learning-mania-2023.zip', datadir / 'mmlm23')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(datadir=project_dir / 'data' / 'raw')
