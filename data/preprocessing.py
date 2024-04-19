"""
    TODO Description
"""

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.data_filter import FilterTypes
from brainflow.data_filter import DetrendOperations


board_id = BoardIds.CYTON_BOARD
sampling_rate = BoardShim.get_sampling_rate(board_id)


""" _summary_ 
    TODO description
"""
def preprocessData(data_2D):

    # perform filtering for each row ( = channel) of 2D data array
    for channel in range(data_2D.shape[0]):
        # filters work in-place
        DataFilter.detrend(data_2D[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(data_2D[channel], sampling_rate, 3.0, 45.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(data_2D[channel], sampling_rate, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(data_2D[channel], sampling_rate, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
    
    return data_2D