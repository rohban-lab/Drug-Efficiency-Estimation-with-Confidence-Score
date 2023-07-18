import os
import luigi
import argparse
import subprocess
import multiprocessing

from pathlib import Path

import tasks.parameters as parameters
import data_paths

from tasks.illumination_function_calculator import PlatelluminationFunctionCalculator
from tasks.data_file_creator import PlateDataFileCreator


class WellFeatureCalculator(luigi.Task):
    input_dir = luigi.Parameter(default=data_paths.IMAGES_PATH)
    output_dir = luigi.Parameter(default=data_paths.CELLPROFILER_FEATURES_PATH)
    expr = luigi.Parameter()
    plate = luigi.IntParameter()
    well = luigi.Parameter()

    def requires(self):
        return {
            'illumination_function': PlatelluminationFunctionCalculator(
                input_dir=self.input_dir, 
                output_dir=self.output_dir, 
                expr=self.expr, 
                plate=self.plate),
            'data_file': PlateDataFileCreator(
                input_dir=self.input_dir, 
                output_dir=self.output_dir, 
                expr=self.expr, 
                plate=self.plate)
        }

    def output(self):
        return luigi.LocalTarget(os.path.join(
            parameters.well_cellprofiler_feature_directory(
                self.output_dir, self.expr, self.plate, self.well
            ), '.SUCCESS'))

    def run(self):
        command =[
                'cellprofiler', '-c', '-p', data_paths.CELLPROFILER_ANALYSIS_CPPIPE,
                '--data-file', parameters.well_data_file(
                    self.output_dir, self.expr, self.plate, self.well),
                '-i', parameters.images_directory(self.input_dir, self.expr, self.plate),
                '-o', parameters.well_cellprofiler_feature_directory(
                    self.output_dir, self.expr, self.plate, self.well)
            ]
        completed_process = subprocess.run(command)
        if completed_process.returncode == 0:
            Path(self.output().path).touch()
        else:
            print('well feature calculator failed due to {}'.format(completed_process.stderr))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plate Feature Calculator')
    parser.add_argument('--input', required=False, default=data_paths.IMAGES_PATH, type=str, 
                        help='path to images')
    parser.add_argument('--output', required=False, default=data_paths.CELLPROFILER_FEATURES_PATH, 
                        type=str, help='path to output folder')
    parser.add_argument('--expr', required=True, type=str, help='experiment')
    parser.add_argument('--plate', required=True, type=str, help='plate')
    parser.add_argument('--proc', required=False, type=int, default=None, help='number of proc')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    input_dir = os.path.abspath(args.input)
    expr = args.expr
    plate = args.plate
    
    num_proc = args.proc
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()-1


    images_directory = parameters.images_directory(input_dir, expr, plate)
    well_names = [image_name[:-10] for image_name in os.listdir(images_directory) 
                    if image_name.endswith('_s1_w1.png')]
    tasks = []
    for well in well_names:
        tasks.append( 
            WellFeatureCalculator(
                input_dir=input_dir,
                output_dir=output_dir,
                expr=expr,
                plate=plate,
                well=well
            )
        )
    print(tasks)
    luigi.build(tasks, workers=num_proc, local_scheduler=True)
