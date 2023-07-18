CHANNEL_NAMES = ['OrigDNA', 'OrigER', 'OrigActinCyto', 'OrigRNA', 'OrigGolgi']

ILLUMINATION_PATTERN_FILES = {
    'illumAC':'illumAC.npy', 
    'illumDNA':'illumDNA.npy', 
    'illumER': 'illumER.npy', 
    'illumGAP': 'illumGAP.npy', 
    'illumRNA': 'illumRNA.npy'
}

NUMBER_OF_SITES = 1 #4

EXPERIMENT_PLATES = {
    'HRCE-1': 26,
    'HRCE-2': 27,
    'VERO-1': 2,
    'VERO-1': 2
}

AGGREGATE_METHOD = 'median'
AGGREGATE_LEVEL = 'site'