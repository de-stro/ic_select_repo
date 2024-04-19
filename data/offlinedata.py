"""
    TODO Description
"""
import os

from brainflow.data_filter import DataFilter
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.data_filter import FilterTypes
from brainflow.data_filter import DetrendOperations

board_id = BoardIds.CYTON_BOARD
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)[:-1] # drop last channel (as it is ECG)
ecg_channel = [BoardShim.get_ecg_channels(board_id).pop()] # last channel is ECG


def getFilepaths():

    filepaths = []

    
    filepaths.append(r'SessionRecordings\\Adele_2024-02-01_19-31-18\\BrainFlow-RAW_2024-02-01_19-31-18_0.csv')
    filepaths.append(r'SessionRecordings\\Dennis_2024-02-02_12-11-33\\BrainFlow-RAW_2024-02-02_12-11-33_0.csv')
    filepaths.append(r'SessionRecordings\\Linn_2024-02-03_18-07-41\\BrainFlow-RAW_2024-02-03_18-07-41_0.csv')
    filepaths.append(r'SessionRecordings\\Felix_2024-02-05_21-16-18\\BrainFlow-RAW_2024-02-05_21-16-18_0.csv')
    filepaths.append(r'SessionRecordings\\Gabriel_2024-02-09_11-30-48\\BrainFlow-RAW_2024-02-09_11-30-48_0.csv')
    
    return filepaths

class RecordingData:
    def __init__(self, filepath):
        self.filepath = filepath
        # load OpenBCI recording (CSV-file in BrainFlow Format)
        self.loaded_data =  DataFilter.read_file(filepath)
        self.eeg_data = self.loaded_data[eeg_channels, :]
        self.ecg_data = self.loaded_data[ecg_channel, :]

    def getECG(self):
         # BrainFlow returns uV!!
        return self.ecg_data
    
    def getEEG(self):
         # BrainFlow returns uV!!
        return self.eeg_data
