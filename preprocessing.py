import os
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

def normalize(data, scaler):
    """ normalize data to mean = 0 and std = 1 """
    return scaler.transform(data)

def denormalize(normalized_data, scaler):
    """ denormalize data back to original mean and std"""
    return scaler.inverse_transform(normalized_data)

def cut_off(time, readings, start_time=24):
    """ cut off data at starting point """
    
    idx = np.argmin(np.abs(time - start_time))
    
    return time[idx:]-start_time, readings[idx:]

def create_sequences(readings, times, seq_length=10, pred_length=1):
    """ slice data into rolling windows and split windows into inputs and labels
    
    readings: (multivariate) time series values 
    times: uniform time values
    seq_length: length of (multivariate) input sequence
    pred_length: length of (multivariate) output sequence
    
    """
    sequences = []
    t_sequences = []
    labels = []
    t_labels = []
    
    for i in range(len(readings) - seq_length - pred_length + 1):
        sequences.append(readings[i : i+seq_length])
        t_sequences.append(times[i : i+seq_length])
        
        labels.append(readings[i+seq_length : i+seq_length+pred_length])
        t_labels.append(times[i+seq_length : i+seq_length+pred_length])
        
    return np.array(sequences), np.array(labels), np.array(t_sequences), np.array(t_labels)

def create_sequences_many_to_one(readings, times, output_idx = 0, seq_length=10, pred_length=1):
    """ slice data into rolling windows and split windows into inputs and labels
    
    readings: time series values 
    times: uniform time values
    output_idx: index to determine which output variable to predict on
    seq_length: length of (multivariate) input sequence
    pred_length: length of (univariate) output sequence
    
    """
    sequences = []
    t_sequences = []
    labels = []
    t_labels = []
    
    for i in range(len(readings) - seq_length - pred_length + 1):
        sequences.append(readings[i : i+seq_length])
        t_sequences.append(times[i : i+seq_length])
        
        labels.append(readings[i+seq_length : i+seq_length+pred_length, output_idx])
        t_labels.append(times[i+seq_length : i+seq_length+pred_length, output_idx])
        
    return np.array(sequences), np.array(labels), np.array(t_sequences), np.array(t_labels)

def interpolate(readings, time, t_max, t_step, kind="linear"):
    """all time steps are given in hours""" 
    
    uniform_time = np.arange(0, t_max+t_step, step=t_step)
    uniform_readings = np.zeros((len(uniform_time), readings.shape[1])) 
    
    for i in range(readings.shape[1]): 
        interpolator = interp1d(time, readings[:,i], kind=kind)
        uniform_readings[:,i] = interpolator(uniform_time)
    
    return uniform_time, uniform_readings

