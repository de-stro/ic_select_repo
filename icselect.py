import numpy as np
import scipy.signal as scisig
import sklearn.preprocessing as sklprep
import matplotlib.pyplot as plt
import time

from brainflow.board_shim import BoardShim, BoardIds

import neurokit2 as nk
import mne

import data.offlinedata as offD
import data.preprocessing as prepro

import ica as icascript

board_id = BoardIds.CYTON_BOARD
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)[:-1] # drop last channel (as it is ECG)
ecg_channel = [BoardShim.get_ecg_channels(board_id).pop()] # last channel is ECG


def calculate_snr_correlation(clean_signal, noisy_signal):
    correlation = np.corrcoef(clean_signal, noisy_signal)[0, 1]
    correlation = np.correlate(clean_signal, noisy_signal)
    correlation = correlation ** 2
    return correlation


def compute_cross_correlation(clean_signal, noisy_signal):
    # Compute cross-correlation
    cross_corr = scisig.correlate(clean_signal, noisy_signal, mode='full')

    # Normalize cross-correlation (using L2 Norm)
    # norm_cross_corr = cross_corr / (np.linalg.norm(clean_signal) * np.linalg.norm(noisy_signal))

    norm_cross_corr = sklprep.minmax_scale(cross_corr, feature_range= (-1, 1))
    akf_noisy = scisig.correlate(noisy_signal, noisy_signal, mode='same')
    fft_series = np.fft.fft(akf_noisy)
    fft_series = np.fft.fftshift(fft_series)
    fft_series = np.square(np.abs(fft_series))


    fig, axs = plt.subplots(5, 1)
    axs[0].plot(clean_signal, label = 'Clean Signal')
    axs[1].plot(noisy_signal, label = 'Noisy Signal')
    axs[2].plot(cross_corr, label = 'Cross-Correlation of Noisy')
    axs[3].plot(scisig.correlate(cross_corr, cross_corr, mode='same'), label = 'AKF of Cross-Corr')

    #axs[3].plot(np.square(np.abs(np.fft.fftshift(np.fft.fft(cross_corr)))), label = "FFT Cross-Corr")
    #axs[3].plot(akf_noisy, label = 'AKF of Noisy')
    #axs[3].plot(fft_series, label = 'FFT of akf noisy_signal')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    #plt.show()
    
    #return norm_cross_corr
    return np.max(np.abs(norm_cross_corr))
    #return np.max(norm_cross_corr)


def main():

    ############################ get data and perform ICA on it (analogous to ICA script)
    # TODO Outsource this completely to ICA script (Interface for ICA-ing needed)
    
    # load and select sample / window from test recordings (eeg and ecg channels seperately)
    # initiate test recording data objects as list
    dataRecordings = []
    filepaths = offD.getFilepaths()
    '''
    for path in filepaths:
        dataRecordings.append(RecordingData(path))
    '''
    dataRecordings.append(offD.RecordingData(filepaths[1]))
    ########################################
    
    ### perform offline ICA Analysis on loaded and (pre-) filtered (BrainFlow) EEG data of Test Recording No.1
    rawEEG = icascript.createObjectMNE('eeg', prepro.preprocessData(dataRecordings[0].getEEG()))
    rawECG = icascript.createObjectMNE('ecg', prepro.preprocessData(dataRecordings[0].getECG()))

    # get user template from ECG recording (not simulated)
    userTemplate = mne.io.Raw.copy(rawECG).crop(tmin=360.0, tmax=420.0)


    # Here we'll crop to 'tmax- tmin' seconds (240-180 seconds)
    # TODO Rausfinden warum float_NaN to int conversion Fehler, wenn tmin 300 und tmax 360
    # TODO Rausfinden warum broadcast / shape error wenn 
    rawEEG.crop(tmin = 900.0, tmax=960.0)
    rawECG.crop(tmin = 900.0, tmax=960.0)
    
    # apply bandpass filter on EEG data (before ICA) to prevent slow downward shift (?)
    #filt_rawEEG = rawEEG.copy().filter(l_freq=1.0, h_freq=50.0)


    # perform ICA comparing the different algorithmic performance on the data (regarding pTp SNR of ECG-related IC)
    method_key, ica_dicts = icascript.choseBestICA(rawEEG, rawECG, maxIter = "auto")
    bestICA = ica_dicts[method_key]

    
    rawResult = bestICA["sources"]
    rawECG.add_channels([rawResult])
    # TEST plot results (fusedECG) to visually check results (time series of ICs) of best ICA chosen
    rawECG.plot(title = "rawECG (fused)")

    # extract components to perform cross-correlation on
    components = []
    timeseries = rawResult.get_data()
    for component in timeseries:
        components.append(component.flatten())

    ecg_related_index = bestICA["ECG_related_index_from_0"]
    ecg_ref = (rawECG.get_data()).flatten()

    ############## Create Templates for Cross-Correlation ###############

    # simulate ideal ECG signal for (ideal) template
    ideal_template = nk.data(dataset="ecg_1000hz")
    #plt.figure()
    #plt.plot(np.square(np.abs(np.fft.fftshift(np.fft.fft(ideal_template)))), label = "FFT Template")
    ideal_template_resamp = nk.signal_resample(ideal_template, sampling_rate = 1000, desired_sampling_rate = sampling_rate)
    # TODO extract exactly one beat / two beats ... ACURATELY using neurokit methods (instead of magic numbers)
    ideal_template_1B = ideal_template_resamp[400:700]
    ideal_template_2B = ideal_template_resamp[400:950]
    ideal_template_3B = ideal_template_resamp[400:1150]
    ideal_template_4B = ideal_template_resamp[400:1450]
    #nk.signal_plot(ideal_template_1B, sampling_rate = sampling_rate)
    #plt.show()

    # TODO use ecg_ref to create individual user template?
    # -> sinnvoll, wenn letzten endes eh ein anderer Herzschlag beim template matching verwendet wird?
    '''
    # create user ecg_template from copied Raw_ecg object
    templateData, resultTimes = userTemplate[:]
    user_ecg_template = templateData[0][:]
    #user_ecg_template = user_ecg_template[500:750]
    plt.plot(user_ecg_template)
    plt.show()
    '''
    #######################################################################

    ############## Compute Cross-Correlation with Ideal Template (simulated) #############################
    
    print("####### RESULTS: Ideal_TEMPL_CROSSCORR_CGPT ###############")
    ideal_temp_crosscorr_CGPT = []
    it = 0 
    ideal_temp_crosscorr_CGPT_timeBegin = time.time()
    for curr_component in components:
         crosscorr = compute_cross_correlation(ideal_template_1B, curr_component)
         print("(Ideal_Temp_CGPT) Cross-Correlation of ", it, "-th component (from 0!) is: ", crosscorr)
         ideal_temp_crosscorr_CGPT.append(crosscorr)
         it += 1
    ideal_temp_crosscorr_CGPT_timeTotal = time.time() - ideal_temp_crosscorr_CGPT_timeBegin
    print("Component with max crosscorr is: ", np.argmax(ideal_temp_crosscorr_CGPT), " component (from 0!), with crosscorr of ", np.max(ideal_temp_crosscorr_CGPT))
    print("Ideal_TEMPL_CROSSCORR_CGPT took ", ideal_temp_crosscorr_CGPT_timeTotal, " seconds in total")
    
    
    ############## Compute Cross-Correlation with User Template (from ecg_ref) #############################
    # TODO
    ########################################################
    
    ############### AUTOCORRELATION APPROACH #################
    # TODO
    ########################################################

    plt.show()
    
if __name__ == "__main__":
    main()