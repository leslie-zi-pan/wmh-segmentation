from enum import StrEnum
import os


class DataDict(StrEnum):
    ImageT1 = 'img_t1'
    ImageFlair = 'img_flair'
    Image = 'image'
    Label = 'label'
    Id = 'subj_id'
    CountryDirType = 'dir_type'
    DepthZ = 'depth_z'
    Prediction = 'prediction'

class CountryDirType(StrEnum):
    Singapore = 'Singapore'
    Utrecht = 'Utretcht'
    Amsterdam = 'Amsterdam'


RAW_TRAINING_DIR = 'data/raw/training/'
RAW_TESTING_DIR = 'data/raw/test/'
INTERIM_TRAINING_DIR = 'data/interim/training/'
INTERIM_TESTING_DIR = 'data/interim/test/'

# Raw training dir paths
SINGAPORE_RAW_TRAINING_DIR = os.path.join(RAW_TRAINING_DIR, 'Singapore/')
UTRECHT_RAW_TRAINING_DIR = os.path.join(RAW_TRAINING_DIR, 'Utrecht/')
AMSTERDAM_RAW_TRAINING_DIR = os.path.join(RAW_TRAINING_DIR, 'Amsterdam/')

SINGAPORE_RAW_VOLUME_TRAINING_IMG_PATHS = [f.path for f in os.scandir(SINGAPORE_RAW_TRAINING_DIR) if f.is_dir()] 
UTRECHT_RAW_VOLUME_TRAINING_IMG_PATHS = [f.path for f in os.scandir(UTRECHT_RAW_TRAINING_DIR) if f.is_dir()] 
AMSTERDAM_RAW_VOLUME_TRAINING_IMG_PATHS = [f.path for f in os.scandir(AMSTERDAM_RAW_TRAINING_DIR) if f.is_dir()] 

# Raw testing dir paths
SINGAPORE_RAW_TESTING_DIR = os.path.join(RAW_TESTING_DIR, 'Singapore/')
UTRECHT_RAW_TESTING_DIR = os.path.join(RAW_TESTING_DIR, 'Utrecht/')
AMSTERDAM_RAW_TESTING_DIR = os.path.join(RAW_TESTING_DIR, 'Amsterdam/')

SINGAPORE_RAW_VOLUME_TESTING_IMG_PATHS = [f.path for f in os.scandir(SINGAPORE_RAW_TESTING_DIR) if f.is_dir()] 
UTRECHT_RAW_VOLUME_TESTING_IMG_PATHS = [f.path for f in os.scandir(UTRECHT_RAW_TESTING_DIR) if f.is_dir()] 
AMSTERDAM_RAW_VOLUME_TESTING_IMG_PATHS = [f.path for f in os.scandir(AMSTERDAM_RAW_TESTING_DIR) if f.is_dir()] 


# INTERIM PATHS
SINGAPORE_INTERIM_TRAINING_DIR = os.path.join(INTERIM_TRAINING_DIR, 'Singapore/')
UTRECHT_INTERIM_TRAINING_DIR = os.path.join(INTERIM_TRAINING_DIR, 'Utrecht/')
AMSTERDAM_INTERIM_TRAINING_DIR = os.path.join(INTERIM_TRAINING_DIR, 'Amsterdam/')


SINGAPORE_INTERIM_VOLUME_TRAINING_IMG_PATHS = [f.path for f in os.scandir(SINGAPORE_INTERIM_TRAINING_DIR) if f.is_dir()] 
UTRECHT_INTERIM_VOLUME_TRAINING_IMG_PATHS = [f.path for f in os.scandir(UTRECHT_INTERIM_TRAINING_DIR) if f.is_dir()] 
AMSTERDAM_INTERIM_VOLUME_TRAINING_IMG_PATHS = [f.path for f in os.scandir(AMSTERDAM_INTERIM_TRAINING_DIR) if f.is_dir()] 

INTERIM_TRAIN_DIR_DICT = {
    CountryDirType.Singapore: SINGAPORE_INTERIM_TRAINING_DIR,
    CountryDirType.Utrecht: UTRECHT_INTERIM_TRAINING_DIR,
    CountryDirType.Amsterdam: AMSTERDAM_INTERIM_TRAINING_DIR
}


SINGAPORE_INTERIM_TESTING_DIR = os.path.join(INTERIM_TESTING_DIR, 'Singapore/')
UTRECHT_INTERIM_TESTING_DIR = os.path.join(INTERIM_TESTING_DIR, 'Utrecht/')
AMSTERDAM_INTERIM_TESTING_DIR = os.path.join(INTERIM_TESTING_DIR, 'Amsterdam/')
SINGAPORE_INTERIM_VOLUME_TESTING_IMG_PATHS = [f.path for f in os.scandir(SINGAPORE_INTERIM_TESTING_DIR) if f.is_dir()] 
UTRECHT_INTERIM_VOLUME_TESTING_IMG_PATHS = [f.path for f in os.scandir(UTRECHT_INTERIM_TESTING_DIR) if f.is_dir()] 
AMSTERDAM_INTERIM_VOLUME_TESTING_IMG_PATHS = [f.path for f in os.scandir(AMSTERDAM_INTERIM_TESTING_DIR) if f.is_dir()] 
INTERIM_TEST_DIR_DICT = {
    CountryDirType.Singapore: SINGAPORE_INTERIM_TESTING_DIR,
    CountryDirType.Utrecht: UTRECHT_INTERIM_TESTING_DIR,
    CountryDirType.Amsterdam: AMSTERDAM_INTERIM_TESTING_DIR
}

INTERIM_TRAINING_DATA_PATHS = SINGAPORE_INTERIM_VOLUME_TRAINING_IMG_PATHS + UTRECHT_INTERIM_VOLUME_TRAINING_IMG_PATHS
INTERIM_TESTING_DATA_PATHS = SINGAPORE_INTERIM_VOLUME_TESTING_IMG_PATHS + UTRECHT_INTERIM_VOLUME_TESTING_IMG_PATHS