# -*- coding: utf-8 -*-
import logging
import os
from os import path
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import click

from src.enums import (AMSTERDAM_RAW_VOLUME_TESTING_IMG_PATHS,
                       AMSTERDAM_RAW_VOLUME_TRAINING_IMG_PATHS,
                       SINGAPORE_RAW_VOLUME_TESTING_IMG_PATHS,
                       SINGAPORE_RAW_VOLUME_TRAINING_IMG_PATHS,
                       UTRECHT_RAW_VOLUME_TESTING_IMG_PATHS,
                       UTRECHT_RAW_VOLUME_TRAINING_IMG_PATHS,
                       INTERIM_TESTING_DIR,
                       INTERIM_TRAINING_DIR,
                       INTERIM_TRAIN_DIR_DICT,
                       INTERIM_TEST_DIR_DICT,
                       CountryDirType, DataDict)
from src.utils import image_directories_handler, read_image_volume, slice_and_save_volume_image

logger = logging.getLogger(__name__)

def get_train_raw_paths():
    raw_train_paths = image_directories_handler(SINGAPORE_RAW_VOLUME_TRAINING_IMG_PATHS, CountryDirType.Singapore)
    raw_train_paths += image_directories_handler(UTRECHT_RAW_VOLUME_TRAINING_IMG_PATHS, CountryDirType.Utrecht)
    raw_train_paths += image_directories_handler(AMSTERDAM_RAW_VOLUME_TRAINING_IMG_PATHS, CountryDirType.Amsterdam)

    return raw_train_paths

def get_test_raw_paths():
    raw_test_paths = image_directories_handler(SINGAPORE_RAW_VOLUME_TESTING_IMG_PATHS, CountryDirType.Singapore)
    raw_test_paths += image_directories_handler(UTRECHT_RAW_VOLUME_TESTING_IMG_PATHS, CountryDirType.Utrecht)
    raw_test_paths += image_directories_handler(AMSTERDAM_RAW_VOLUME_TESTING_IMG_PATHS, CountryDirType.Amsterdam)

    return raw_test_paths

def slice_volume_handler(raw_data_paths, interim_dir_dict):
    for index, filedict in enumerate(raw_data_paths):
        # Collect image and subject id
        img_t1 = read_image_volume(filedict[DataDict.ImageT1], True)
        img_flair = read_image_volume(filedict[DataDict.ImageFlair], True)
        label = read_image_volume(filedict[DataDict.Label], False)
        subj_id = filedict[DataDict.Id]

        country_dir_type = filedict[DataDict.CountryDirType]
        output_country_dir = interim_dir_dict[country_dir_type]
        subject_slice_dir = os.path.join(output_country_dir, f'{subj_id}/')
        
        if not path.exists(output_country_dir):
            os.mkdir(output_country_dir)

        if not path.exists(subject_slice_dir):
            os.mkdir(subject_slice_dir)
            os.mkdir(subject_slice_dir + 'Image/')
            os.mkdir(subject_slice_dir + 'Image/T1/')
            os.mkdir(subject_slice_dir + 'Image/Flair/')
            os.mkdir(subject_slice_dir + 'Label/')

        out_dir_image = os.path.join(output_country_dir, f'{subj_id}/Image/')
        out_dir_image_t1 = os.path.join(output_country_dir, f'{subj_id}/Image/T1')
        out_dir_image_flair = os.path.join(output_country_dir, f'{subj_id}/Image/Flair')
        out_dir_label = os.path.join(output_country_dir, f'{subj_id}/Label/')

        num_slices_image_t1 = slice_and_save_volume_image(img_t1, f'brain_{str(subj_id)}', out_dir_image_t1)
        num_slices_image_flair = slice_and_save_volume_image(img_flair, f'brain_{str(subj_id)}', out_dir_image_flair)
        num_slices_label = slice_and_save_volume_image(label, f'brain_{str(subj_id)}', out_dir_label)

        if num_slices_image_t1 != num_slices_label & num_slices_image_flair != num_slices_label:
            raise ValueError(f'Image and Label dim not match: {num_slices_image_t1.shape}, {label.shape}')

        logger.info(f'\n{filedict}, {num_slices_image_t1} slices created \n')


@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('>> Creating slice images from volume images')
    # Create interim training and testing dir if not exist
    if not path.exists(INTERIM_TRAINING_DIR):
        os.mkdir(INTERIM_TRAINING_DIR)
    
    if not path.exists(INTERIM_TESTING_DIR):
        os.mkdir(INTERIM_TESTING_DIR)

    raw_train_paths = get_train_raw_paths()
    raw_test_paths = get_test_raw_paths()

    slice_volume_handler(raw_train_paths, INTERIM_TRAIN_DIR_DICT)
    slice_volume_handler(raw_test_paths, INTERIM_TEST_DIR_DICT)
    logger.info('>> Successfully sliced images')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
