import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

#Given a target variable and a list of features, this returns a correlation chart and graphs
def create_correl_charts(data, target, feature_list, max_width=2):
    cor = pd.DataFrame()
    for f in feature_list:
        cor.set_value(target, f, np.corrcoef(x=data[f], y=data[target])[0,1])
    print cor

    num_features = len(feature_list)
    w = min(max_width, num_features)
    h = num_features/max_width
    if num_features % max_width != 0:
        h += 1

    y=data[target]
    plt.figure(figsize=(20,10))
    idx=1
    for f in feature_list:
        x=data[f]
        plt.subplot(h, w, idx)
        plt.title('%s vs %s' %(f, target))
        plt.xlabel(f)
        plt.ylabel(target)
        plt.scatter(x,y,alpha=0.5)
        idx += 1
    plt.tight_layout()

#Given a dictionary of segments, explores relatinship between target and features
def compare_segments_charts(data_dict, target, feature, dict_keys=None, max_width=2):
    cor = pd.DataFrame()
    if dict_keys is None:
        dict_keys = data_dict.keys()
    for key in dict_keys:
        d = data_dict[key]
        cor.set_value(target, key, np.corrcoef(x=d[feature], y=d[target])[0,1])
    print cor

    num_data = len(data_dict)
    w = min(max_width, num_data)
    h = num_data/max_width
    if num_data % max_width != 0:
        h += 1

    plt.figure(figsize=(20,10))
    idx=1
    for key in dict_keys:
        d = data_dict[key]
        x=d[feature]
        y = d[target]
        plt.subplot(h,w,idx)
        plt.title(key)
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.scatter(x,y,alpha=0.5)
        idx += 1
    plt.tight_layout()

#Given dictionary of segments, looks at differences of target variable
def desc_stats(data_dict, target, dict_keys=None):
    df = pd.DataFrame()
    if dict_keys is None:
        dict_keys = data_dict.keys()
    for key in dict_keys:
        target_data = data_dict[key][target]
        df.set_value('Min', key, min(target_data))
        df.set_value('1Q', key, np.percentile(target_data, 25))
        df.set_value('2Q', key, np.percentile(target_data, 50))
        df.set_value('Mean', key, np.mean(target_data))
        df.set_value('3Q', key, np.percentile(target_data, 75))
        df.set_value('Max', key, max(target_data))
        df.set_value('Variance', key, np.var(target_data))
        df.set_value('Skew', key, skew(target_data))
        df.set_value('n', key, sum(target_data))
    print df.round(2)

    data = [data_dict[x][target] for x in dict_keys]
    plt.boxplot(data, labels=dict_keys, showmeans=True)
    plt.title(target)
