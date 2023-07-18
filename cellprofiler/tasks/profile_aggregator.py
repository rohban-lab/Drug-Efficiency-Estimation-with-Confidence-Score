import os
import luigi
import argparse
import pandas as pd
import multiprocessing

import config
import data_paths
import tasks.parameters as parameters

from tasks.feature_calculator import WellFeatureCalculator


class PlateProfileAggregator(luigi.Task):
    input_dir = luigi.Parameter(default=data_paths.IMAGES_PATH)
    output_dir = luigi.Parameter(default=data_paths.CELLPROFILER_FEATURES_PATH)
    expr = luigi.Parameter()
    plate = luigi.IntParameter()

    
    def requires(self):
        images_directory = parameters.images_directory(self.input_dir, self.expr, self.plate)
        well_names = [image_name[:-10] for image_name in os.listdir(images_directory) 
                        if image_name.endswith('_s1_w1.png')]
        tasks = []
        for well in well_names:
            tasks.append( 
                WellFeatureCalculator(
                    input_dir=self.input_dir,
                    output_dir=self.output_dir,
                    expr=self.expr,
                    plate=self.plate,
                    well=well
                )
            )
        yield tasks

    def output(self):
        return luigi.LocalTarget(os.path.join(
            parameters.output_directory(self.output_dir, self.expr, self.plate), 
            'profile_{}_{}.csv'.format(config.AGGREGATE_METHOD, config.AGGREGATE_LEVEL)))

    def run(self):
        output_directory = parameters.cellprofiler_feature_directory(
            self.output_dir, self.expr, self.plate)
        wells = [ well for well in os.listdir(output_directory) 
                    if os.path.isdir(
                        parameters.well_cellprofiler_feature_directory(
                            self.output_dir, self.expr, self.plate, well))]

        well_profiles = []
        for well in wells:
            data_path = parameters.well_cellprofiler_feature_directory(
                            self.output_dir, self.expr, self.plate, well)
            cells_path = os.path.join(data_path, 'filteredCells.csv')
            nuclei_path = os.path.join(data_path, 'filteredNuclei.csv')
            cytoplasm_path = os.path.join(data_path, 'Cytoplasm.csv')
            image_path = os.path.join(data_path, 'Image.csv')

            cells = pd.read_csv(cells_path)
            nucleis = pd.read_csv(nuclei_path)
            cytoplasm= pd.read_csv(cytoplasm_path)
            image = pd.read_csv(image_path)[['ImageNumber', 'Metadata_Site']]

            well_pdf = cells.merge(
                nucleis.merge(cytoplasm, on='ObjectNumber'),
                on='ObjectNumber'
            ).astype(float)
            
            well_pdf = well_pdf.assign(cell_count=len(well_pdf))    

            if config.AGGREGATE_LEVEL == 'site':
                grouped_well_pdf = well_pdf.groupby('ImageNumber')
                if config.AGGREGATE_METHOD == 'mean': well_pdf = grouped_well_pdf.mean()
                else: well_pdf = grouped_well_pdf.median()

                well_pdf = well_pdf.drop('ObjectNumber', axis=1).merge(image, on='ImageNumber')
                well_pdf['site_id'] = well_pdf.Metadata_Site.apply(
                    lambda x : self.expr + '_' + str(self.plate)+ '_' + well + '_' + str(x))
                well_profile = well_pdf.drop(['Metadata_Site', 'ImageNumber'], axis=1)
            
            else:
                well_pdf = well_pdf.assign(well_id = self.expr + '_' + str(self.plate) + '_' + well)
                grouped_well_pdf = well_pdf.groupby('well_id', as_index=False)
                if config.AGGREGATE_METHOD == 'mean': well_pdf = grouped_well_pdf.mean()
                else: well_pdf = grouped_well_pdf.median()

                well_profile = well_pdf.drop(['ObjectNumber', 'ImageNumber'], axis=1)

            well_profiles.append(well_profile)
        
        profile = pd.concat(well_profiles, ignore_index=True)
        profile.to_csv(self.output().path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plate Aggregator')
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

    luigi.build([PlateProfileAggregator(
        input_dir=input_dir,
        output_dir=output_dir,
        expr=expr,
        plate=plate
    )], workers=num_proc, local_scheduler=True)