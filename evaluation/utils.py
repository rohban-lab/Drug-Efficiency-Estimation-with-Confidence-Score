import os
import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA



def similarity(table):
    sigma = table.T.corr(method='pearson')
    return sigma

def random_sample(profile, metadata, sample_size):
    group_ids = metadata.groupid.unique()
    sub_sample_ids = random.sample(list(group_ids), min(len(group_ids),sample_size))
    imagnumbers = []
    for sample_id in sub_sample_ids:
        row = metadata[metadata.groupid == sample_id].sample()
        imagnumbers.append(row.index.tolist()[0])
    sample = profile[profile.index.isin(imagnumbers)]
    return sample

def background_distribution(profile, metadata, group_size, num_samples):
    background_dist = []
    for i in range(num_samples):
        sample = random_sample(profile, metadata, group_size)
        sigma = similarity(sample)
        up_tri = np.asarray(sigma)[np.triu_indices(sigma.shape[0], k=1)]
        mcc = up_tri.mean()
        background_dist.append(mcc)
    return background_dist

def find_threshold(distribution, coef):
    confidence_interval = pd.Series(sorted(distribution)).quantile([coef])
    return float(confidence_interval)

def normalize(profile, control):
    control_means = control.median(axis=0)
    control_mads = control.mad(axis=0) 
    profile = pd.concat([profile, control])
    for col_name in profile.columns:
        if control_mads[col_name] == 0: 
            profile = profile.drop(col_name, axis=1)
            continue
        profile[col_name] = (profile[col_name] - control_means[col_name])/(control_mads[col_name])
    return profile

def test_expriment(profile, metadata):
    result = []
    group_ids = list(metadata.groupid.unique())
    for group_id  in group_ids:
        if 'negctrl' in list(metadata.loc[metadata.groupid == group_id, 'label']): continue
        test_group = profile[profile.index.isin(list(metadata[metadata.groupid == group_id].index))]
        if len(test_group) < 2: continue
        sigma = similarity(test_group)
        up_tri = np.asarray(sigma)[np.triu_indices(sigma.shape[0], k=1)]
        mcc = up_tri.mean()
        result.append(mcc)
    return result

def get_treatment_condition_metadata(metadata, experiment, plate, level):
    metadata = metadata[(metadata.experiment == experiment) & (metadata.plate == plate)]
    metadata = metadata[['treatment', 'treatment_conc' , level+'_id']]
    metadata = metadata.assign(groupid = metadata.groupby(['treatment', 'treatment_conc']).ngroup())
    metadata = metadata.assign(label=np.where(metadata['groupid'] == -1, 'negctrl','posctrl'))
    metadata.index = metadata[level+'_id']
    return metadata

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def calculate_on_score(sample, origin, axis):
    sample_vector = np.array(sample) - np.array(origin)
    axis_vector = np.array(axis) - np.array(origin)

    result = np.dot(sample_vector, unit_vector(axis_vector).T)
    return result

def calculate_off_score(sample, origin, axis):
    sample_vector = np.array(sample) - np.array(origin)
    axis_vector = np.array(axis) - np.array(origin)
    axis_image = np.dot(sample_vector, unit_vector(axis_vector).T)
    result = np.linalg.norm(sample_vector - unit_vector(axis_vector)*axis_image)
    return result

def create_contingency_table(top_similarity, down_similarity):
    c_table = [
        [sum(top_similarity), len(top_similarity) - sum(top_similarity)], 
        [sum(down_similarity), len(down_similarity) - sum(down_similarity)]
    ]
    return c_table

def moa_similarity(moa_list1, moa_list2):
    if str(moa_list1) == 'nan' or str(moa_list2) == 'nan': return 0
    if set(moa_list1) & set(moa_list2):
        return 1
    return 0

def perform_treatment_condition_correlation_histogram_test(
    features, labels, group_size, num_samples):
    print('Calculating Back Ground Distribution ...')
    bgdist = background_distribution(
        features, 
        labels, 
        group_size=group_size, 
        num_samples=num_samples, 
    )
    threshold = find_threshold(bgdist, 0.95)
    print('Calculating Experiment Distribution ...')
    exp = test_expriment(features, labels)
    samples_above_threshold = sum([sample >= threshold for sample in exp])
    ratio = samples_above_threshold/len(exp)
    print('Ratio is {:.2f}.'.format(ratio))
    return ratio, exp, bgdist

def whiten(features, dim=1024):
    pca = PCA(n_components=dim, whiten=True)
    pcas = pca.fit_transform(features)
    data = pd.DataFrame(data=pcas, columns = ['f'+str(i) for i in range(dim)])
    data.index = features.index
    return data