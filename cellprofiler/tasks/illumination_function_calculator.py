import os
import luigi
import pathlib
import argparse
import subprocess

from os.path import join
from pathlib import Path

import data_paths

import tasks.parameters as parameters


class PlatelluminationFunctionCalculator(luigi.Task):
    input_dir = luigi.Parameter(default=data_paths.IMAGES_PATH)
    output_dir = luigi.Parameter(default=data_paths.CELLPROFILER_FEATURES_PATH)
    expr = luigi.Parameter()
    plate = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(join(
            parameters.illumination_function_directory(self.output_dir, self.expr, self.plate), 
            '.SUCCESS'))

    def run(self):
        pathlib.Path(parameters.illumination_function_directory(
            self.output_dir, self.expr, self.plate)).mkdir(parents=True, exist_ok=True)
        command =[
            'cellprofiler', '-c', '-p', data_paths.CELLPROFILER_ILLUM_CPPIPE,
            '-i', parameters.images_directory(self.input_dir, self.expr, self.plate),
            '-o', parameters.illumination_function_directory(self.output_dir, self.expr, self.plate)
        ]
        returncode = subprocess.call(command)
        if returncode == 0:
            Path(self.output().path).touch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plate Illumination Function Calculator')
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

    luigi.build([PlatelluminationFunctionCalculator(
        input_dir=input_dir,
        output_dir=output_dir,
        expr=expr,
        plate=plate
    )], local_scheduler=True)
        