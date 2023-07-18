import luigi
import argparse
import multiprocessing

from os.path import abspath

import data_paths
import config

from tasks.profile_aggregator import PlateProfileAggregator
from tasks.illumination_function_calculator import PlatelluminationFunctionCalculator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Pipeline')
    parser.add_argument('--input', required=False, default=data_paths.IMAGES_PATH, type=str, 
                        help='path to images')
    parser.add_argument('--output', required=False, default=data_paths.CELLPROFILER_FEATURES_PATH, 
                        type=str, help='path to output folder')
    parser.add_argument('--expr', required=True, type=str, help='experiment')
    parser.add_argument('--proc', required=False, type=int, default=None, help='number of proc')
    parser.add_argument('--verbose', help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    output_dir = abspath(args.output)
    input_dir = abspath(args.input)
    expr = args.expr
    verbose = args.verbose 
    num_proc = args.proc
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()-1

    num_plates = config.EXPERIMENT_PLATES[expr]

    if verbose: print('Initialization experiment {} pipeline ...'.format(expr))

    if verbose: print('Illumination correction calculator is calling for {} plates.'.format(
                        num_plates))
    tasks = []
    for plate in range(1, num_plates+1):
        tasks.append(
            PlatelluminationFunctionCalculator(
                input_dir=input_dir,
                output_dir=output_dir,
                expr=expr,
                plate=plate
            )
        )
    luigi_run_result = luigi.build(tasks, workers=num_proc, detailed_summary=True)
    print(luigi_run_result.summary_text)

    if verbose: 
        print('Well level cellprofiler features calculator is calling for'+ 
                '{} plates and {} workers.'.format(num_plates, num_proc))

    for plate in range(1, num_plates+1):
        task = PlateProfileAggregator(
            input_dir=input_dir,
            output_dir=output_dir,
            expr=expr,
            plate=plate
        )
        luigi_run_result = luigi.build([task], workers=num_proc, detailed_summary=True)
        print(luigi_run_result.summary_text)
        if verbose: print('Plate {} finished.'.format(plate))

