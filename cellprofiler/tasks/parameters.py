import os
     
def plate_name(plate):
    return 'Plate' + str(plate)

def images_directory(input_dir, expr, plate):
    return os.path.join(input_dir, expr, plate_name(plate))

def output_directory(output_dir, expr, plate):
    return os.path.join(output_dir, expr, plate_name(plate))

def data_file_directory(output_dir, expr, plate):
    return os.path.join(output_directory(output_dir, expr, plate), 'data_files')

def illumination_function_directory(output_dir, expr, plate):
    return os.path.join(output_directory(output_dir, expr, plate), 'illumination_functions')

def cellprofiler_feature_directory(output_dir, expr, plate):
    return os.path.join(output_directory(output_dir, expr, plate), 'cellprofiler_features')

def well_cellprofiler_feature_directory(output_dir, expr, plate, well):
    return os.path.join(cellprofiler_feature_directory(output_dir, expr, plate), well)

def well_data_file(output_dir, expr, plate, well):
        return  os.path.join(data_file_directory(output_dir, expr, plate), well, 'loadData.csv')

