import pathlib
from os.path import join


ROOT = join(pathlib.Path(__file__).parent.absolute(), '..')

DATA_PATH = join(ROOT, 'data')
PIPELINE_PATH = join(ROOT, 'pipeline')
CELLPROFILER_FEATURES_PATH = join(ROOT, 'features')

METADATA_PATH = join(DATA_PATH, 'metadata.csv')
IMAGES_PATH = join(DATA_PATH, 'images')

CELLPROFILER_PIPELINES_PATH = join(ROOT, 'cppipes')
CELLPROFILER_ILLUM_CPPIPE = join(CELLPROFILER_PIPELINES_PATH, 'gaussian_illum.cppipe')
CELLPROFILER_ANALYSIS_CPPIPE = join(CELLPROFILER_PIPELINES_PATH, 'hrce_analysis.cppipe')
