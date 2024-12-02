import numpy as np
import pandas as pd
import time
from collections import Counter
from operator import itemgetter
from itertools import groupby
from sklearn.base import BaseEstimator, TransformerMixin

class EqualWidthTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self, attr_list, num_intervals):
        self.attr_list = attr_list
        self.num_intervals = num_intervals
        
    #Return self, nothing else to do here
    def fit( self, X, y = None ):
        return self 
    
    def transform(self, X, y = None):
        X_ = X.copy()
        
        for attr in self.attr_list:
            _, intervals = np.histogram(X_[attr],bins=self.num_intervals)
            X_[attr] = pd.cut(X_[attr],bins=intervals, right=True, include_lowest=True, labels=[i for i in range(1,len(intervals))])
            
        return X_
    
class EqualFreqTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self, attr_list, quantile):
        self.attr_list = attr_list
        self.quantile = quantile
        
    #Return self, nothing else to do here
    def fit( self, X, y = None ):
        return self 
    
    def transform(self, X, y = None):
        X_ = X.copy()
        
        for attr in self.attr_list:
            _, intervals = pd.qcut(X_[attr], q=self.quantile, retbins=True)
            X_[attr] = pd.cut(X_[attr],bins=intervals, right=True, include_lowest=True, labels=[i for i in range(1,len(intervals))])
            
        return X_
    
class ChiMergeTransformer(BaseEstimator, TransformerMixin):
#Class Constructor
    def __init__( self, attr_list, min_intervals, hot_start_num_intervals, allow_early_stop):
        self.attr_list = attr_list
        self.min_intervals = min_intervals
        self.hot_start_num_intervals = hot_start_num_intervals
        self.allow_early_stop = allow_early_stop

    def chi_merge_4(self, data,attr,label,min_intervals,chi_thresh,hot_start_num_intervals=None,allow_early_stop=True):

        with np.errstate(divide='ignore',invalid='ignore'):
            if hot_start_num_intervals is None:
                intervals = np.unique(data[attr]).tolist()
            else:
                out,bins=pd.qcut(data[attr],hot_start_num_intervals,retbins=True,duplicates='drop')
                intervals = bins.tolist()
            
            data['bin'] = pd.cut(data[attr], bins=intervals,labels=[i for i in range(len(intervals)-1)],include_lowest=True,right=False).values.add_categories(len(intervals)-1)
            data['bin'] = data['bin'].fillna(len(intervals)-1)
            
            # print(data.head())

            counts_arr = data.groupby(['bin',label]).count().iloc[:,0].to_numpy().reshape((len(intervals),len(np.unique(data[label]))))
            count_total_arr = np.array([counts_arr[i] + counts_arr[i+1] for i in range(len(counts_arr)-1)])
            total_arr = np.sum(count_total_arr,axis=1)

            exp0_arr = np.array([sum(counts_arr[i,:]) * count_total_arr[i,:] / total_arr[i] for i in range(len(counts_arr)-1)]).reshape(len(total_arr),len(np.unique(data[label])))
            exp1_arr = np.array([sum(counts_arr[i,:]) * count_total_arr[i-1,:] / total_arr[i-1] for i in range(1,len(counts_arr))]).reshape(len(total_arr),len(np.unique(data[label])))

            chi_arr = (counts_arr[:-1,:] - exp0_arr)**2 / exp0_arr + (counts_arr[1:,:] - exp1_arr)**2 / exp1_arr 
            chi_arr = np.nan_to_num(chi_arr)
            chi_total_arr = np.sum(chi_arr,axis=1)
            
            while len(intervals) >= min_intervals:

                min_chi_ls = np.where(chi_total_arr == np.amin(chi_total_arr))[0].tolist()
                
                for n in range(len(min_chi_ls)):
                    
                    min_chi_idx = min_chi_ls[n]
                    
                    # merge bin
                    counts_arr[min_chi_idx,:] += counts_arr[min_chi_idx + 1,:]
                    counts_arr = np.delete(counts_arr, min_chi_idx + 1, axis=0)
                    
                    # delete chi value between merged intervals
                    chi_total_arr = np.delete(chi_total_arr, min_chi_idx, axis=0)
                    
                    for i in [min_chi_idx-1, min_chi_idx]: #lower and upper chi values to be updated
                        
                        if i == -1 or i == len(chi_total_arr):
                            continue
                        
                        class_total_slice = counts_arr[[i],:] + counts_arr[[i + 1],:]

                        total_slice = np.sum(class_total_slice, axis=1)
                        
                        interval_total_slice = np.sum(counts_arr[i:i+2,:], axis = 1, keepdims = True)
                        
                        exp_ij = (class_total_slice * interval_total_slice) / total_slice
                        
                        new_chi = (counts_arr[i,:] - exp_ij[0,:])**2 / exp0_arr[i] + (counts_arr[i+1,:] - exp_ij[1,:])**2 / exp_ij[1,:]
                        new_chi = np.nan_to_num(new_chi)
                        new_chi_total = np.sum(new_chi)
                        
                        chi_total_arr[i] = new_chi_total
                        
                    min_chi_ls = [n-1 for n in min_chi_ls]
                
                new_intervals = [intervals[i] for i in range(len(intervals)) if i not in np.add(min_chi_ls,1).tolist()]
                intervals = new_intervals
                
                # early stopping
                if allow_early_stop == True:
                    if all(i > chi_thresh for i in chi_total_arr):
                        # print('early stop with ', len(intervals), 'intervals')
                        break
                
        return chi_total_arr, intervals

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        
        if hasattr(self, "attr_intervals"):
            print('before fitting, had intervals!')
            del self.attr_intervals

    #Return self, nothing else to do here
    def fit( self, X, y = None ):
        X_ = X.copy()
        
        self._reset()
        # print('before fitting X max:', X['meaninten1_frob'].max())
        # print('before fitting X_ max:', X_['meaninten1_frob'].max())
        
        self.attr_intervals = {}
        for attr in self.attr_list:
            # print('Chimerge learning', attr)
            _, intervals = self.chi_merge_4(data=X_, attr=attr, label='Label', 
                                            min_intervals=self.min_intervals, 
                                            chi_thresh= 3.84,
                                            hot_start_num_intervals=self.hot_start_num_intervals,
                                            allow_early_stop=self.allow_early_stop)
            
            # if intervals[-1] != X_[attr].max():
            #     intervals.append(X_[attr].max())
            if intervals[0] != 0:
                intervals = [0] + intervals
            intervals[-1] = np.inf
                
            self.attr_intervals[attr] = intervals
        
        return self 
    
    def transform(self, X, y = None):
        X_ = X.copy()
        
        for attr in self.attr_list:
            # print('Chimerge discretizing', attr)
            
            # print('discrete transform cols', X_.columns)
            # print('min max', X_.min(), X_.max())
            
            intervals = self.attr_intervals[attr]
            X_[attr] = pd.cut(X_[attr],bins=intervals, right=True, include_lowest=True, labels=[i for i in range(1,len(intervals))])
            # print('hot start', self.hot_start_num_intervals, self.allow_early_stop,'feature', attr, str(len(intervals)))
            
        return X_
    
class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns
        
    def fit( self, X, y = None ):
        return self 

    def transform(self,X,y=None):
        X_ = X.copy()
        X_ = X_.drop(self.columns,axis=1)
        # print(X_.head())
        return X_

