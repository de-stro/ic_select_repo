"""
    script to perform offline ICA Analysis in MNE on local (earlier recorded) EEG-Session (with parallel chest-ECG on
    channel 8, in BrainFlow Format (CSV)
"""

import math 
import numpy as np
import time 
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BoardIds

import neurokit2 as nk
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

import data.offlinedata as offD
import data.preprocessing as prepro

board_id = BoardIds.CYTON_BOARD
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)[:-1] # drop last channel (as it is ECG)
ecg_channel = [BoardShim.get_ecg_channels(board_id).pop()] # last channel is ECG

ecg_related_index = 0
componentAmountConsidered = len(eeg_channels)


# Creating MNE objects from brainflow data array
def createObjectMNE(type: str, data_2D):

    data_2D = data_2D / 1000000  # BrainFlow returns uV, convert to V for MNE!1!1!1!

    if type == 'eeg':
        ch_types = ['eeg'] * len(eeg_channels)
        info = mne.create_info(ch_names=len(eeg_channels), sfreq=sampling_rate, ch_types=ch_types)
        return mne.io.RawArray(data_2D, info)
    else:
        ch_types = ['ecg']
        info = mne.create_info(ch_names=len(ecg_channel), sfreq=sampling_rate, ch_types=ch_types)
        return mne.io.RawArray(data_2D, info)



def calculate_corr_sign(ref_signal, noisy_signal):
    mittelwert = (np.max(ref_signal) + np.min(ref_signal)) / 2
    normalized_ref = ref_signal - mittelwert
    normalized_ref = normalized_ref / np.max(normalized_ref)

    mittelwert = (np.max(noisy_signal) + np.min(noisy_signal)) / 2
    normalized_noisy = noisy_signal - mittelwert
    normalized_noisy = normalized_noisy / np.max(normalized_noisy) 

    similarity = normalized_noisy * normalized_ref

    return np.sign(np.mean(similarity))



''' Correlation-based Method: Compute the correlation coefficient between the noisy signal and the clean reference signal. 
    Return Pearson product-moment correlation coefficients
    The closer the correlation coefficient is to 1, the higher the SNR. 
'''
def calculate_snr_correlation_wR(ref_signal, noisy_signal):
    correlation = np.corrcoef(ref_signal, noisy_signal)[0, 1]
    snr = correlation ** 2
    return snr



''' segment the data into epochs around QRS complexes detected 
    TODO peak_finding methode mit in methoden kopf nehmen!
'''
def epoch_for_QRS_and_noise(noisy_signal, reference_signal = None):

    # if passed, use reference signal for delineation (to mark QRS-complexes etc) 
    if reference_signal is not None:
        # detect R-peaks in reference signal
        ref_peaks_signal, ref_peaks_info = nk.ecg_peaks(reference_signal, sampling_rate = sampling_rate, 
                                                        method = 'neurokit', correct_artifacts = False, show = False)
        # Delineate cardiac cycle of reference signal
        ref_signals, waves = nk.ecg_delineate(reference_signal, ref_peaks_info["ECG_R_Peaks"], sampling_rate = sampling_rate,
                                                  method='dwt', show=False, show_type = "bounds_R")
    else:
        # detect R-peaks in noisy signal
        noisy_peaks_signal, noisy_peaks_info = nk.ecg_peaks(noisy_signal, sampling_rate = sampling_rate, 
                                                        method ='neurokit', correct_artifacts = False, show = False)
        # Delineate cardiac cycle of noisy signal
        noisy_signals, waves = nk.ecg_delineate(noisy_signal, noisy_peaks_info, sampling_rate = sampling_rate,
                                                  method = 'dwt', show = False, show_type = "bounds_R")
        
    onsets_R_list = waves["ECG_R_Onsets"]
    offsets_R_list = waves["ECG_R_Offsets"]

    # ensure that epochs beginn with first R_Offset (hence, a noise_segment) and end with last R_Offset,
    # therefore, last detected QRS complex is included, while first QRS complex might be omitted!
    if onsets_R_list[0] < offsets_R_list[0]:
        onsets_R_list = onsets_R_list[1:]
    if onsets_R_list[-1] > offsets_R_list[-1]:
        onsets_R_list = onsets_R_list[:-1]
    # now size offsets_R_list must be equal or 1 greater than size onsets_R_list!
    #print("TEST:: Size of onset_R_list ", len(onsets_R_list), " and of offset_R_list: ", len(offsets_R_list))
    #print("TEST:: OFFSET_R_LIST MUST BE EQUAL OR 1 GREATER THAN THE SIZE OF ONSET_R_LIST !!")

    # calculate duration between R_offset and R_onset in seconds (omit last R_Offset for noise epochs)
    noise_epochs_duration = np.array(onsets_R_list) - np.array(offsets_R_list[:-1])
    noise_epochs_duration = noise_epochs_duration / sampling_rate

    # calculate duration between R_onset and R_offset in seconds (omit first R_Offset for QRS epochs)
    qrs_epochs_duration = np.array(offsets_R_list[1:]) - np.array(onsets_R_list)
    qrs_epochs_duration = qrs_epochs_duration / sampling_rate
    #print("TEST:: Length of epochs_duration arrays must be the same!!!")
    #print("TEST::Length of qrs_epochs_duration: ", qrs_epochs_duration.size, " and of noise_epochs_duration: ", noise_epochs_duration.size)
    # TEST
    #print("mean duration for QRS complex (in sec): ", np.mean(epochs_duration))
    #print("duration of 10th QRS complex (in sec): ", ((ref_waves["ECG_R_Offsets"])[9] - (ref_waves["ECG_R_Onsets"])[9]) / sampling_rate)

    # signal delineation to create epochs of noisy signal around QRS-complexes found 
    qrs_epochs = nk.epochs_create(noisy_signal, events = onsets_R_list, sampling_rate = sampling_rate, epochs_start = 0, 
                                epochs_end = qrs_epochs_duration.tolist(), event_labels = None, event_conditions = None, baseline_correction = False)
    
    # do the same epoching for epochs of noise parts of noisy signal (all except for QRS complexes)
    noise_epochs = nk.epochs_create(noisy_signal, events = offsets_R_list, sampling_rate = sampling_rate, epochs_start = 0, 
                                epochs_end = noise_epochs_duration.tolist(), event_labels = None, event_conditions = None, baseline_correction = False)
        
    ''' TEST
    # Iterate through epoch data
    i = 0
    for qrs_epoch in qrs_epochs.values():
        if i == 10:
            # Plot scaled signals
            nk.signal_plot(qrs_epoch['Signal']) 
        i = i + 1
    
    i = 0
    for noise_epoch in noise_epochs.values():
        if i == 10:
            # Plot scaled signals
            nk.signal_plot(noise_epoch['Signal']) 
        i = i + 1

    plt.show()
    '''
    return (qrs_epochs, noise_epochs)
    


def epoch_peak_to_peak_amplitude(epoch):
    signal = epoch['Signal']
    max_peak = np.max(signal)
    min_peak = np.min(signal)
    return max_peak - min_peak



def epoch_RSSQ(epoch):
    signal = epoch['Signal']
    RSSQ = math.sqrt(np.sum(np.square(signal)))
    return RSSQ



''' Calculate SNR of noisy signal epoched around heart beats / R-peaks detected (in noisy signal, no reference)
'''
def calculate_snr_PeakToPeak_from_epochs(qrs_epochs, noise_epochs):
    # get avg peak-to-peak-amplitude of signal epochs (qrs)
    qrs_pTp_amplitude_sum = 0
    for qrs_epoch in qrs_epochs.values():
        qrs_pTp_amplitude_sum = qrs_pTp_amplitude_sum + epoch_peak_to_peak_amplitude(qrs_epoch)
    qrs_pTp_amplitude_avg = qrs_pTp_amplitude_sum / len(qrs_epochs)

    # get avg peak-to-peak-amplitude of noise epochs (noise)
    noise_pTp_amplitude_sum = 0
    for noise_epoch in noise_epochs.values():
        noise_pTp_amplitude_sum = noise_pTp_amplitude_sum + epoch_peak_to_peak_amplitude(noise_epoch)
    noise_pTp_amplitude_avg = noise_pTp_amplitude_sum / len(noise_epochs)

    # calculate SNR in dB
    # TODO genaue Formel (insb Vorfaktor) für milliVolts???! (siehe unten)
    # when signal and noise are measured in volts (V) or amperes (A) (measures of amplitude) 
    # they must first be squared to obtain a quantity proportional to power
    snr = 20 * np.log10(qrs_pTp_amplitude_avg / noise_pTp_amplitude_avg)
    return snr



def calc_snr_RSSQ_from_epochs(qrs_epochs, noise_epochs):
    # get avg RSSQ of signal epochs (qrs)
    qrs_RSSQ_sum = 0
    for qrs_epoch in qrs_epochs.values():
        qrs_RSSQ_sum = qrs_RSSQ_sum + epoch_RSSQ(qrs_epoch)
    qrs_RSSQ_avg = qrs_RSSQ_sum / len(qrs_epochs)

    noise_RSSQ_sum = 0
    for noise_epoch in noise_epochs.values():
        noise_RSSQ_sum = noise_RSSQ_sum + epoch_RSSQ(noise_epoch)
    noise_RSSQ_avg = noise_RSSQ_sum / len(noise_epochs)

    # calculate SNR in dB
    # TODO genaue Formel (insb Vorfaktor) für milliVolts???! (siehe unten)
    # when signal and noise are measured in volts (V) or amperes (A) (measures of amplitude) 
    # they must first be squared to obtain a quantity proportional to power
    snr = 20 * np.log10(qrs_RSSQ_avg / noise_RSSQ_avg)
    return snr



def choseBestICA(eeg_data:mne.io.Raw, ecg_ref:mne.io.Raw, amountICs = componentAmountConsidered, maxIter = 'auto'):
    """Given an eeg_signal, this methods computes ICA using the best performing algorithm based on (pTp) SNR and 
    returns dictionary with the results after fitting (key and dict of ICA dicts)
    """
    
    ''' TODO assume the data is already preprocessed and as (cropped) MNE Object
    # perform preprocessing on raw data
    prep_eeg = prepro.preprocessData(eeg_data)
    prep_ecg = prepro.preprocessData(ecg_ref)

    # create MNE Raw Objects to perform ICA on
    mne_eeg = createObjectMNE("eeg", prep_eeg)
    mne_ecg = createObjectMNE("ecg", prep_ecg)
    '''
    # TODO was this already done (in preprocessing)??
    # apply bandpass filter on EEG data (before ICA) to prevent slow downward shift (?)
    filtBP_eeg = eeg_data.copy().filter(l_freq=1.0, h_freq=50.0)

    # compute and compare the different MNE ICA algorithm implementations (comp time and SNR of ECG-IC)
    
    # configure the ica objects for fitting
    picard_dict = {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='picard')}
    infomax_dict = {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='infomax')}
    fastica_dict = {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='fastica')}
    ica_dicts = {"picard": picard_dict, "infomax": infomax_dict, "fastica": fastica_dict}

    # fit the ICA objects and time the fitting
    for methodName, ica_dict in ica_dicts.items():
        ica_dict["rawCopy"] = eeg_data.copy()
        ica_dict["filterCopy"] = filtBP_eeg.copy()

        startTime = time.time()
        ica_dict["ica"].fit(ica_dict["filterCopy"])
        fitTime = time.time() - startTime

        ica_dict["fitTime"] = fitTime
        ica_dict["sources"] = ica_dict["ica"].get_sources(ica_dict["rawCopy"]) # TODO can we also *not* use Raw Obj Instance here?

        # identify ECG-related IC (using ecg_ref) and compute the resulting SNR
        timeseries, times = (ica_dict["sources"]).get_data(return_times=True)
        timeseries.flatten()
        snr_correlations_CGPT = []
        ecg_ref_series = (ecg_ref.get_data()).flatten()
        
        for component in timeseries:
            snr_correlations_CGPT.append(calculate_snr_correlation_wR(ecg_ref_series, component))

        ecg_related_index = snr_correlations_CGPT.index(max(snr_correlations_CGPT))
        ica_dict["ECG_related_index_from_0"] = ecg_related_index

        print("ChoseBest_TEST: ECG_RELATED INDEX")
        print("Method used: ",  methodName)
        print("ECG-related Component calulated: ", ecg_related_index, "-th component (starting from 0!)")
        print("with SNR_corr value of: ", snr_correlations_CGPT[ecg_related_index])

        # TODO Plot the sources / results to visually control calculated ECG_component

        # calulate peak-to-peak SNR of ECG-related component
        ecg_related_timeseries = timeseries[ecg_related_index]
        # flip ecg-related component if negatively correlated with ref_ecg
        ecg_related_timeseries *= calculate_corr_sign(ecg_ref_series, ecg_related_timeseries)

        ica_dict["ECG_related_Timeseries"] = ecg_related_timeseries

        # segment the signal into indivdual heartbeats to calculate pTp SNR / RSSQ SNR
        epoch = epoch_for_QRS_and_noise(ecg_related_timeseries, ecg_ref_series)
        #epoch = epoch_for_QRS_and_noise(nk_cleaned_ecg)

        snr_ptp = calculate_snr_PeakToPeak_from_epochs(epoch[0], epoch[1])
        print("SNR (avg peak-to-peak-amplitude) is ", snr_ptp, "dB")
        ica_dict["pTp_SNR_dB"] = snr_ptp

        snr_rssq = calc_snr_RSSQ_from_epochs(epoch[0], epoch[1])
        print("SNR (avg RSSQ) is ", snr_rssq, "dB")
        ica_dict["rssq_SNR_dB"] = snr_rssq
    
    # chose best performing ICA algorithm and return results (based on SNR alone)
    # TODO calculate metric (optimal time/SNR ratio) to rank algorithm performance
    maxSNR_pTp = (ica_dicts["picard"])["pTp_SNR_dB"]
    bestMethod_pTp = "picard"
    maxSNR_rssq = (ica_dicts["picard"])["rssq_SNR_dB"]
    bestMethod_rssq = "picard"

    for methodName, ica_dict in ica_dicts.items():
        if  ica_dict["pTp_SNR_dB"] > maxSNR_pTp:
            maxSNR_pTp = ica_dict["pTp_SNR_dB"]
            bestMethod_pTp = methodName
        if ica_dict["rssq_SNR_dB"] > maxSNR_rssq:
            maxSNR_rssq = ica_dict["rssq_SNR_dB"]
            bestMethod_rssq = methodName
    
    print("########## TEST_choseBestICA: RESULTS ###################")
    print("Best method regarding pTp_SNR: ", bestMethod_pTp, "with ", maxSNR_pTp, " dB")
    print("with ECG_related component being the ", (ica_dicts[bestMethod_pTp])["ECG_related_index_from_0"], "-th component (from 0!)")
    print("Took in sec: ", (ica_dicts[bestMethod_pTp])["fitTime"], "with amount iterations: ", (ica_dicts[bestMethod_pTp])["ica"].n_iter_)
    print("###")
    print("Best method regarding rssq_SNR: ", bestMethod_rssq, "with ", maxSNR_rssq, " dB")
    print("with ECG_related component being the ", (ica_dicts[bestMethod_rssq])["ECG_related_index_from_0"], "-th component (from 0!)")
    print("Took in sec: ", (ica_dicts[bestMethod_rssq])["fitTime"], "with amount iterations: ", (ica_dicts[bestMethod_rssq])["ica"].n_iter_)

    return (bestMethod_pTp, ica_dicts)





def main():

    '''

    global componentAmountConsidered
    
    ########## load and select sample / window from test recordings (eeg and ecg channels seperately)

    # initiate test recording data objects as list
    dataRecordings = []
    filepaths = offD.getFilepaths()

    # TODO Clean up
    #for path in filepaths:
    #    dataRecordings.append(RecordingData(path))
    
    dataRecordings.append(offD.RecordingData(filepaths[1]))
    ########################################
    

    ### perform offline ICA Analysis on loaded and (pre-) filtered (BrainFlow) EEG data of Test Recording No.1
    startTime_prepro = time.time()
    rawEEG = createObjectMNE('eeg', prepro.preprocessData(dataRecordings[0].getEEG()))
    rawECG = createObjectMNE('ecg', prepro.preprocessData(dataRecordings[0].getECG()))
    endTime_prepro = time.time() - startTime_prepro
    

    # Here we'll crop to 'tmax' seconds 
    # Note that the new tmin is assumed to be t=0 for all subsequently called functions
    # TODO Data Cropping auslagern wenn verschiebung der sample label ein problem
    #clean data
    rawEEG.crop(tmin = 180.0, tmax=240.0)
    rawECG.crop(tmin = 180.0, tmax=240.0)
    # noisy data (channel 6 dies in the end)
    #rawEEG.crop(tmin = 630.0, tmax=690.0)
    #rawECG.crop(tmin = 630.0, tmax=690.0)

    choseBestICA(rawEEG, rawECG, componentAmountConsidered, "auto")

    startTime_filter = time.time()
    # TODO Check if this is already done by BrainFlow Filtering
    # TODO Note down that this is necessary (bandpass filter EEG before ICA to prevent slow downward shift) (?)
    # apply bandpass filter(?) on EEG data (before ICA) to prevent slow downward(?) shift (?)
    filt_rawEEG = rawEEG.copy().filter(l_freq=1.0, h_freq=50.0)
    endTime_filter = time.time() - startTime_filter

    print("Preprocessing (BrainFlow Filter + uV-V Conversion + RawObj Creation + additional MNE_filter) took (in sec): ", endTime_prepro + endTime_filter)


    # apply ICA on the filtered raw EEG data

    startTime_ICA = time.time()
    # using picard
    #ica = ICA(n_components=componentAmountConsidered, max_iter=1, random_state=97, method='picard')

    # using fastICA
    #ica = ICA(n_components=componentAmountConsidered, max_iter=1, random_state=97, method='fastica')

    # using infoMax
    ica = ICA(n_components=componentAmountConsidered, max_iter="auto", random_state=97, method='infomax')
    
    ica.fit(filt_rawEEG)
    ica

    icaTime = time.time() - startTime_ICA

    print("ICA took (in sec): ", icaTime)
    print("ICA took in total:", ica.n_iter_, "iterations")

    # retrieve the fraction of variance in the original data 
    # that is explained by our ICA components in the form of a dictionary
    explained_var_ratio = ica.get_explained_variance_ratio(filt_rawEEG) 
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
        )

    # plot_sources will show the time series of the ICs. Note that in our call to plot_sources 
    # we can use the original, unfiltered Raw object
    filt_rawEEG.load_data()
    #ica.plot_sources(rawEEG, title = "ICA Sources" )

    
    rawResult = ica.get_sources(rawEEG)
    print("len (timepoints) of rawResult: ", len(rawResult))
    #rawResult.plot()
    rawECG.add_channels([rawResult])
    rawECG.plot(title = "rawECG (fused)")
    
    #rawComponents.plot()
    #rawResult.plot() 

    #comps = ica.get_sources(raw)
    #comps.plot()

    #ica.plot_components()

    # get all IC components for SNR Evaluation (each)
    resultData, resultTimes = rawECG[:]
    new_ecg = resultData[0][:]
    components = []
    for i in range(1, componentAmountConsidered + 1):
            components.append(resultData[i][:])
 

    # print SNR values for ECG-related component only (compared to reference ECG)
    correlations = []
    for curr_component in components:
            correlations.append(calculate_snr_correlation_wR_CGPT(new_ecg, curr_component))
    
    global ecg_related_index
    ecg_related_index = correlations.index(max(correlations))

    print("##########################")
    print("SNR results for the", (ecg_related_index+1), "th component")
    print("SNR:", calculate_snr_wR(new_ecg, components[ecg_related_index]), "dB")
    #print("SNR (RMS):", calculate_snr_RMS(components[0], components[i]), "dB")
    #print("SNR (peak-to-peak):", calculate_snr_peak_to_peak(components[i]))
    #print("SNR (wavelet):", calculate_snr_wavelet(components[i]))
    print("SNR (correlation with REF ECG):", calculate_snr_correlation_wR_CGPT(new_ecg, components[ecg_related_index]))
    #print("SNR (IDUN) for 1 comp:", calculate_snr_Idun(components[i]))
    print("##########################")


    
    # plot the extracted ecg related component
    fig, axs = plt.subplots(componentAmountConsidered + 1, 1)
    axs[0].plot(new_ecg)
    axs[0].set_title('clean ecg')

    for j in range(componentAmountConsidered):
        axs[j+1].plot(components[j])
        axs[j+1].set_title(str(j+1) + "-th component")

    plt.show()

    

    # employ NeuroKit Heartbeat Visualization and Quality Assessment
    # TODO über den plot mit markierten R-Peaks nochmal Herzrate drüberlegen für visuelle Kontrolle
    # TODO mit individual heartbeats plot spielen um guten Wert für epoch segmentierung zu finden (bzgl. SNR Formel Güler/PluxBio)

    ecg_related_comp = components[ecg_related_index] * calculate_corr_sign(new_ecg, components[ecg_related_index])
    nk_signals, nk_info = nk.ecg_process(ecg_related_comp, sampling_rate=sampling_rate)

    # Extract ECG-related IC and R-peaks location
    nk_rpeaks = nk_info["ECG_R_Peaks"]
    nk_cleaned_ecg = ecg_related_comp #nk_signals["ECG_Clean"]
    
    # assess ECG signal quality using nk_function ecg_quality()
    print("NK2. Signal Quality Assessment (Zhao2018, simple approach):", nk.ecg_quality(nk_cleaned_ecg, sampling_rate=sampling_rate, method = "zhao2018",  approach="simple"))
    print("NK2. Signal Quality Assessment (Zhao2018, fuzzy approach):", nk.ecg_quality(nk_cleaned_ecg, sampling_rate=sampling_rate, method = "zhao2018",  approach="fuzzy"))
    quality = nk.ecg_quality(nk_cleaned_ecg, sampling_rate=sampling_rate, method = "averageQRS")
    print("NK2. Signal Quality Assessment (QRS Avg):", np.mean(quality))

    #nk.signal_plot([nk_cleaned_ecg, quality], sampling_rate = sampling_rate, subplots = True, standardize=False)


    # Visualize R-peaks in ECG signal
    nk_plot = nk.events_plot(nk_rpeaks, nk_cleaned_ecg)
    
    # Plotting all the heart beats
    #nk_epochs = nk.ecg_segment(nk_cleaned_ecg, rpeaks=None, sampling_rate=sampling_rate, show=True)

    ###################
    
    heartbeats = extract_heartbeats(nk_cleaned_ecg, peaks=nk_rpeaks, sampling_rate=sampling_rate)
    heartbeats.head()

    heartbeats_pivoted = heartbeats.pivot(index='Time', columns='Label', values='Signal')
    heartbeats_pivoted.head()
    
    #TODO Clean up
        # Prepare figure
        nk_fig, ax = plt.subplots()

        ax.set_title("Individual Heart Beats")
        ax.set_xlabel("Time (seconds)")

        # Aesthetics
        labels = list(heartbeats_pivoted)
        labels = ['Channel ' + x for x in labels] # Set labels for each signal
        cmap = iter(plt.cm.YlOrRd(np.linspace(0,1, int(heartbeats["Label"].nunique())))) # Get color map
        lines = [] # Create empty list to contain the plot of each signal

        for i, x, color in zip(labels, heartbeats_pivoted, cmap):
            line, = ax.plot(heartbeats_pivoted[x], label='%s' % i, color=color)
            lines.append(line)

        plt.show()
    
    
    
    epoch = epoch_for_QRS_and_noise(nk_cleaned_ecg, new_ecg)
    #epoch = epoch_for_QRS_and_noise(nk_cleaned_ecg)

    snr_ptp = calculate_snr_PeakToPeak_from_epochs(epoch[0], epoch[1])
    print("SNR (avg peak-to-peak-amplitude) is ", snr_ptp, "dB")

    snr_rssq = calc_snr_RSSQ_from_epochs(epoch[0], epoch[1])
    print("SNR (avg RSSQ) is ", snr_rssq, "dB")
    
    '''

if __name__ == "__main__":
    main()