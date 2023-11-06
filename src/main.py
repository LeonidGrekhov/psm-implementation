import logging
import logging.config
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.util import FileProvider as FP
from src.datamodel.Column import DataDictionary as dd
import src.util.randomForestPSMModel as randomForestPSMModel
import src.util.methods as methods
import src.util.LogReg as LogReg
import src.util.neuralNetwork as neuralNetwork
import src.util.SvmPsmModel as SvmPsmModel
import src.util.DataGenerator as DataGenerator
import src.util.matchPatients as matching
# Set a random seed for reproducibility
np.random.seed(42)

def main():
    logger = logging.getLogger('Main')
    logger.debug('======Start======')
    matching.matchPatients(path='src/data/psm_sample_data.csv', label_columns = [dd.sex, dd.race, dd.ethnicity], one_hot_columns = [dd.race, dd.ethnicity], target_column = [dd.treatment])
    logger.debug('======Finish======')
    
