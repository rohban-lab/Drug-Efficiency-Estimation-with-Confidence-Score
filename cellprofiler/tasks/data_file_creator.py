import os
import luigi
import pathlib
import argparse
import pandas as pd

from pathlib import Path

import data_paths
import config
import tasks.parameters as parameters


class PlateDataFileCreator(luigi.Task):
    input_dir = luigi.Parameter(default=data_paths.IMAGES_PATH)
    output_dir = luigi.Parameter(default=data_paths.CELLPROFILER_FEATURES_PATH)
    expr = luigi.Parameter()
    plate = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(
            parameters.data_file_directory(self.output_dir, self.expr, self.plate), 
            '.SUCCESS'))


    def run(self):
        image_dir = parameters.images_directory(self.input_dir, self.expr, self.plate)

        well_names  = [image_name[:-10] for image_name in os.listdir(image_dir) 
                        if image_name.endswith('_s1_w1.png')] 
        for well_number, well_name in enumerate(well_names):
            well_dir = os.path.join(
                parameters.data_file_directory(self.output_dir, self.expr, self.plate), 
                well_name)
            well_data_file = []

            for site_number in range(1, config.NUMBER_OF_SITES+1):
                row = {'Metadata_Site' : site_number , 'Metadata_Well' : well_name, 
                        'Metadata_Plate' : self.plate, 
                        'ImageNumber' : config.NUMBER_OF_SITES*well_number+site_number}

                for illum_suffix in config.ILLUMINATION_PATTERN_FILES.keys():
                    row['PathName_' + illum_suffix] = parameters.illumination_function_directory(
                        self.output_dir, self.expr, self.plate) 

                    row['FileName_' + illum_suffix] = illum_suffix + '.npy'

                for channel_number, channel in enumerate(config.CHANNEL_NAMES, start=1):
                    row['FileName_' + channel] = '{}_s{}_w{}.png'.format(
                        well_name, site_number, channel_number)
                    row['PathName_' + channel] = image_dir

                well_data_file.append(row)
        
            pathlib.Path(well_dir).mkdir(parents=True, exist_ok=True)
            well_data_file_pdf = pd.DataFrame(well_data_file)
            well_data_file_pdf.to_csv(os.path.join(well_dir, 'loadData.csv'), index=False)

        Path(self.output().path).touch()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plate Data Files Creator')
    parser.add_argument('--input', required=False, default=data_paths.IMAGES_PATH, type=str, 
                        help='path to images')
    parser.add_argument('--output', required=False, default=data_paths.CELLPROFILER_FEATURES_PATH, 
                        type=str, help='path to output folder')
    parser.add_argument('--expr', required=True, type=str, help='experiment')
    parser.add_argument('--plate', required=True, type=str, help='experiment')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    input_dir = os.path.abspath(args.input)
    expr = args.expr
    plate = args.plate

    luigi.build([PlateDataFileCreator(
        input_dir=input_dir,
        output_dir=output_dir,
        expr=expr,
        plate=plate
    )], local_scheduler=True)