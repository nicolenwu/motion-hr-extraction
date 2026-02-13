# importing libraries
# %matplotlib inline
import csv
import numpy as np 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join
from scipy import stats
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from scipy.signal import freqz
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# lists for folders and files
sample_list = os.listdir ('data')
dataset_list = ['accelerometer', 'gyroscope_uncalibrated', 'gyroscope', 'linear_acceleration']

# create folders for results
if not os.path.exists ('Results'):
    os.mkdir('Results')

for sample in sample_list:
    if not os.path.exists ('Results/%s' % sample):
        os.mkdir('Results/%s' % sample)
    
        for dataset in dataset_list:
            os.mkdir('Results/%s/%s' % (sample, dataset))


# lists for error values for each sensor
accel_error = []
gyro_uncalib_error = []
gyro_error = []
linear_accel_error = []


# iterate through datasets and analyze 
for sample in sample_list:

    # iterate through files to extract the ground truth for the sample
    ground_truth = ""
    ground_truth_retrieved = False
    files = os.listdir ('data/' + sample)

    for dataset in files:

        # check for file containing ground truth
        if dataset.startswith ("stereo") and dataset.endswith ("rec.wav"):

            # extract ground truth from the file name
            ground_truth = dataset.split("_") [2]
            ground_truth = float (ground_truth.replace ("bpm", ""))
            ground_truth_retrieved = True
            break

    
    for dataset in dataset_list:

        print ("\n------------------  Sample %s: %s  ------------------" % (sample, dataset))

        filename = sample + '/' + dataset

        # read raw data file
        columns = ['timestamp', 'unix timestamp', 'x-axis', 'y-axis', 'z-axis', 'user', 'activity']
        df = pd.read_csv('data/%s.csv' % filename, header = None, names = columns)

        # fill in columns with dummy values
        df['user'] = df['user'].fillna (1)
        df['activity'] = df['activity'].fillna ('standing')

        # subtract by min & divide by 10000 to get the time in seconds
        df ['timestamp'] = df ['timestamp'] - df ['timestamp'].min()
        df ['timestamp'] = df ['timestamp'] / 10000

        # drop starting and ending values
        n = 25
        df = df [n : -n]

        # arrange data in ascending order of the user and timestamp
        df = df.sort_values (by = ['user', 'timestamp'], ignore_index = True)

        
        # lineplot - raw data
        plt.clf()
        sns.lineplot (data = df, x = 'timestamp', y = 'x-axis')
        sns.lineplot (data = df, x = 'timestamp', y = 'y-axis')
        sns.lineplot (data = df, x = 'timestamp', y = 'z-axis')
        plt.legend (['x-axis', 'y-axis', 'z-axis'])
        plt.title ("%s: Raw Data" % dataset)
        plt.savefig('Results/%s/%s/Raw Data.png' % (sample, dataset))
        
        # activity distribution - raw data
        plt.clf()
        sns.FacetGrid (df, hue = 'activity').map(sns.histplot, 'x-axis', kde = True, stat = "density").add_legend()#
        plt.title ("%s: X Activity Distribution" % dataset)
        plt.savefig('Results/%s/%s/X Activity Distribution.png' % (sample, dataset))

        plt.clf()
        sns.FacetGrid (df, hue = 'activity').map(sns.histplot, 'y-axis', kde = True, stat = "density").add_legend()
        plt.title ("%s: Y Activity Distribution" % dataset)
        plt.savefig('Results/%s/%s/Y Activity Distribution.png' % (sample, dataset))

        plt.clf()
        sns.FacetGrid (df, hue = 'activity').map(sns.histplot, 'z-axis', kde = True, stat = "density").add_legend()
        plt.title ("%s: Z Activity Distribution" % dataset)
        plt.savefig('Results/%s/%s/Z Activity Distribution.png' % (sample, dataset))
        

        # lists for moving averages data
        x_moving_ave = []
        y_moving_ave = []
        z_moving_ave = []
        train_labels = []

        # variables for calculating moving averages
        moving_ave = []
        window_size = 3
        i = 0

        # calculate moving averages
        while i < df.shape[0] - window_size + 1:

            # window of data
            x_window = df['x-axis'].values[i : i + window_size]
            y_window = df['y-axis'].values[i : i + window_size]
            z_window = df['z-axis'].values[i : i + window_size]
            label = df['activity'][i : i + window_size].mode()[0]

            # calculate window averages
            x_window_ave = round (sum(x_window) / window_size, 2)
            y_window_ave = round (sum(y_window) / window_size, 2)
            z_window_ave = round (sum(z_window) / window_size, 2)

            # append window averages to moving averages
            x_moving_ave.append (x_window_ave)
            y_moving_ave.append (y_window_ave)
            z_moving_ave.append (z_window_ave)
            train_labels.append (label)

            i += 1

        # time values for moving averages data
        time = df['timestamp'].values[0 : len(x_moving_ave)]

        
        # lineplot - moving averages data
        plt.clf()
        plt.plot (time, x_moving_ave, color = 'r')
        plt.plot (time, y_moving_ave, color = 'b')
        plt.plot (time, z_moving_ave, color = 'g')
        plt.xlabel ('Time (seconds)')
        plt.grid (True)
        plt.axis ('tight')
        plt.legend (['x-axis', 'y-axis', 'z-axis'], loc = 'upper right')
        plt.title ("%s: Moving Averages Data" % dataset)
        plt.savefig('Results/%s/%s/Moving Averages (MA) Data.png' % (sample, dataset))

        # activity distribution - moving averages data
        plt.clf()
        sns.histplot (x_moving_ave, kde = True, stat = "density")
        plt.title ("%s: X Moving Average Activity Distribution" % dataset)
        plt.savefig('Results/%s/%s/X Moving Ave. Activity Dist.png' % (sample, dataset))

        plt.clf()
        sns.histplot (y_moving_ave, kde = True, stat = "density")
        plt.title ("%s: Y Moving Average Activity Distribution" % dataset)
        plt.savefig('Results/%s/%s/Y Moving Ave. Activity Dist.png' % (sample, dataset))

        plt.clf()
        sns.histplot (z_moving_ave, kde = True, stat = "density")
        plt.title ("%s: Z Moving Average Activity Distribution" % dataset)
        plt.savefig('Results/%s/%s/Z Moving Ave. Activity Dist.png' % (sample, dataset))


        # band-pass Butterworth
        def butter_bandpass (lowcut, highcut, fs, order):
            return butter (order, [lowcut, highcut], fs = fs, btype = 'band')

        # band-pass Butterworth filter
        def butter_bandpass_filter (data, lowcut, highcut, fs, order):
            b, a = butter_bandpass (lowcut, highcut, fs, order = order)
            y = lfilter (b, a, data)
            return y

        # filter the data with band-pass butterworth filter
        if __name__ == "__main__":

            # define sample rate & desired cutoff frequencies of the filter
            fs = 50.0
            lowcut = 10.0
            highcut = 13.0
            
            # filter the data using band-pass butterworth filter - order 4
            filtered_x = butter_bandpass_filter (x_moving_ave, lowcut, highcut, fs, order = 4)
            filtered_y = butter_bandpass_filter (y_moving_ave, lowcut, highcut, fs, order = 4)
            filtered_z = butter_bandpass_filter (z_moving_ave, lowcut, highcut, fs, order = 4)

            # plot the 4th order band-pass butterworth filter data
            plt.figure (1)
            plt.clf()
            plt.plot (time, filtered_x, color = 'r')
            plt.plot (time, filtered_y, color = 'b')
            plt.plot (time, filtered_z, color = 'g')
            plt.xlabel ('time (seconds)')
            plt.grid (True)
            plt.axis ('tight')
            plt.legend (['x-axis', 'y-axis', 'z-axis'], loc = 'upper right')
            plt.title ("%s: Band-Pass Filtered Moving Ave. Data" % dataset)
            plt.savefig('Results/%s/%s/Filtered MA Data.png' % (sample, dataset))


        # calculate L2 norm 
        L2_norm = []

        for i in range (len(filtered_x)):
            x_temp = filtered_x[i]
            y_temp = filtered_y[i]
            z_temp = filtered_z[i]
            l2_temp = np.sqrt (x_temp**2 + y_temp**2 + z_temp**2)
            L2_norm.append (l2_temp)

        # plot L2 norm
        plt.clf()
        plt.plot (time, L2_norm, color = 'r')
        plt.xlabel ('time (seconds)')
        plt.grid (True)
        plt.axis ('tight')
        plt.legend (['L2 Norm'], loc = 'upper right')
        plt.title ("%s: L2 Norm" % dataset)
        plt.savefig('Results/%s/%s/L2 Norm.png' % (sample, dataset))


        # filter the L2 norm using band-pass butterworth filter
        if __name__ == "__main__":

            # sample rate & desired cutoff frequencies of the filter
            fs = 50.0
            lowcut = 0.75
            highcut = 2.5
            
            # filter the data using butterworth bandpass filter - order 2
            filtered_L2 = butter_bandpass_filter (L2_norm, lowcut, highcut, fs, order = 2)

            # plot the filtered L2 norm
            plt.clf()
            plt.plot (time, filtered_L2, color = 'r')
            plt.xlabel ('time (seconds)')
            plt.grid (True)
            plt.axis ('tight')
            plt.legend (['L2 norm'], loc = 'upper right')
            plt.title ("%s: Band-Pass Filtered L2 Norm" % dataset)
            plt.savefig('Results/%s/%s/Filtered L2 Norm.png' % (sample, dataset))


        # convert L2 norm to frequency domain using FFT
        L2_fft = np.abs(np.fft.fft(filtered_L2))

        # calculate frequency values
        timestep = 1 / fs
        freq = np.fft.fftfreq (len(L2_fft), d = timestep)

        # plot the FFT of L2 norm
        plt.clf()
        plt.plot (freq, L2_fft, color = 'r')
        plt.xlim ([0, 10])
        plt.xlabel ('frequency (Hz)')
        plt.grid (True)
        plt.legend (['FFT L2 norm'], loc = 'upper right')
        plt.title ("%s: FFT L2 Norm" % dataset)
        plt.savefig('Results/%s/%s/FFT L2 Norm.png' % (sample, dataset))


        # find peaks in the FFT of L2 norm
        peaks_index, _ = find_peaks (L2_fft, height = 0)

        # remove peaks with negative frequencies
        peaks_index = peaks_index [freq [peaks_index] > 0]

        # sort the indices of the peaks by descending order of height of peaks
        descending_peaks_index = np.argsort (L2_fft[peaks_index]) [::-1]


        # compare predicted heart rate with ground truth
        if ground_truth_retrieved:

            # calculate potential heart rates indicated by the highest peaks in the FFT of L2 norm
            potential_bpm = []
            num_potential_bpm = 3

            for i in range (num_potential_bpm):
                
                # calculate the potential heart rates
                temp_index = peaks_index [descending_peaks_index [i]]
                potential_bpm.append (float (freq [temp_index] * 60))
            

            # compare the potential heart rates with the ground truth
            predicted_bpm = float ('-inf')
            min_diff = float ('-inf')

            for i in range (num_potential_bpm):
                
                # calculate the difference between the potential heart rates and the ground truth
                temp_diff = abs (potential_bpm [i] - ground_truth)

                # update the predicted heart rate
                if (min_diff == float ('-inf')):
                    min_diff = temp_diff
                    predicted_bpm = potential_bpm [i]

                if min_diff > temp_diff:
                    min_diff = temp_diff
                    predicted_bpm = potential_bpm [i]
            

            # append the error value of the predicted heart rate to the corresponding list
            bpm_error = abs (ground_truth - predicted_bpm)
            if dataset == 'accelerometer':
                accel_error.append (bpm_error)
            elif dataset == 'gyroscope_uncalibrated':
                gyro_uncalib_error.append (bpm_error)
            elif dataset == 'gyroscope':
                gyro_error.append (bpm_error)
            elif dataset == 'linear_acceleration':
                linear_accel_error.append (bpm_error)

            
            # print the results
            print ("Potential Heart Rate (bpm): %f, %f, %f" % (potential_bpm[0], potential_bpm[1], potential_bpm[2]))
            print ("Predicted Heart Rate (bpm): %f" % predicted_bpm)
            print ("Ground Truth (bpm): %f" % ground_truth)
            print ("Error: %f" % bpm_error)
        

        # Statistical Features on raw x, y and z in time domain
        X_train = pd.DataFrame()

        # mean
        X_train ['x_mean'] = pd.Series(x_moving_ave).apply(lambda x: np.mean (x))
        X_train ['y_mean'] = pd.Series(y_moving_ave).apply(lambda x: np.mean (x))
        X_train ['z_mean'] = pd.Series(z_moving_ave).apply(lambda x: np.mean (x))

        # std dev
        X_train ['x_std'] = pd.Series(x_moving_ave).apply(lambda x: np.std (x))
        X_train ['y_std'] = pd.Series(y_moving_ave).apply(lambda x: np.std (x))
        X_train ['z_std'] = pd.Series(z_moving_ave).apply(lambda x: np.std (x))


        # avg absolute diff
        X_train ['x_aad'] = pd.Series (x_moving_ave).apply (lambda x: np.mean (np.absolute (x - np.mean (x))))
        X_train ['y_aad'] = pd.Series (y_moving_ave).apply (lambda x: np.mean (np.absolute (x - np.mean (x))))
        X_train ['z_aad'] = pd.Series (z_moving_ave).apply (lambda x: np.mean (np.absolute (x - np.mean (x))))

        # min
        X_train ['x_min'] = pd.Series (x_moving_ave).apply (lambda x: np.min (x))
        X_train ['y_min'] = pd.Series (y_moving_ave).apply (lambda x: np.min (x))
        X_train ['z_min'] = pd.Series (z_moving_ave).apply (lambda x: np.min (x))

        # max
        X_train ['x_max'] = pd.Series (x_moving_ave).apply (lambda x: np.max (x))
        X_train ['y_max'] = pd.Series (y_moving_ave).apply (lambda x: np.max (x))
        X_train ['z_max'] = pd.Series (z_moving_ave).apply (lambda x: np.max (x))

        # max-min diff
        X_train ['x_maxmin_diff'] = X_train['x_max'] - X_train['x_min']
        X_train ['y_maxmin_diff'] = X_train['y_max'] - X_train['y_min']
        X_train ['z_maxmin_diff'] = X_train['z_max'] - X_train['z_min']

        # median
        X_train ['x_median'] = pd.Series(x_moving_ave).apply(lambda x: np.median (x))
        X_train ['y_median'] = pd.Series(y_moving_ave).apply(lambda x: np.median (x))
        X_train ['z_median'] = pd.Series(z_moving_ave).apply(lambda x: np.median (x))

        # median abs dev 
        X_train ['x_mad'] = pd.Series (x_moving_ave).apply (lambda x: np.median (np.absolute (x - np.median (x))))
        X_train ['y_mad'] = pd.Series (y_moving_ave).apply (lambda x: np.median (np.absolute (x - np.median (x))))
        X_train ['z_mad'] = pd.Series (z_moving_ave).apply (lambda x: np.median (np.absolute (x - np.median (x))))

        # interquartile range
        X_train ['x_IQR'] = pd.Series (x_moving_ave).apply (lambda x: np.percentile (x, 75) - np.percentile (x, 25))
        X_train ['y_IQR'] = pd.Series (y_moving_ave).apply (lambda x: np.percentile (x, 75) - np.percentile (x, 25))
        X_train ['z_IQR'] = pd.Series (z_moving_ave).apply (lambda x: np.percentile (x, 75) - np.percentile (x, 25))

        # negtive count
        X_train ['x_neg_count'] = pd.Series (x_moving_ave).apply (lambda x: np.sum (x < 0))
        X_train ['y_neg_count'] = pd.Series (y_moving_ave).apply (lambda x: np.sum (x < 0))
        X_train ['z_neg_count'] = pd.Series (z_moving_ave).apply (lambda x: np.sum (x < 0))

        # positive count
        X_train ['x_pos_count'] = pd.Series (x_moving_ave).apply (lambda x: np.sum (x > 0))
        X_train ['y_pos_count'] = pd.Series (y_moving_ave).apply (lambda x: np.sum (x > 0))
        X_train ['z_pos_count'] = pd.Series (z_moving_ave).apply (lambda x: np.sum (x > 0))

        # values above mean
        X_train ['x_above_mean'] = pd.Series (x_moving_ave).apply (lambda x: np.sum (x > np.mean (x)))
        X_train ['y_above_mean'] = pd.Series (y_moving_ave).apply (lambda x: np.sum (x > np.mean (x)))
        X_train ['z_above_mean'] = pd.Series (z_moving_ave).apply (lambda x: np.sum (x > np.mean (x)))

        # skewness
        X_train ['x_skewness'] = pd.Series (x_moving_ave).apply (lambda x: stats.skew (x))
        X_train ['y_skewness'] = pd.Series (y_moving_ave).apply (lambda x: stats.skew (x))
        X_train ['z_skewness'] = pd.Series (z_moving_ave).apply (lambda x: stats.skew (x))

        # kurtosis
        X_train ['x_kurtosis'] = pd.Series (x_moving_ave).apply (lambda x: stats.kurtosis (x))
        X_train ['y_kurtosis'] = pd.Series (y_moving_ave).apply (lambda x: stats.kurtosis (x))
        X_train ['z_kurtosis'] = pd.Series (z_moving_ave).apply (lambda x: stats.kurtosis (x))

        # energy
        X_train ['x_energy'] = pd.Series (x_moving_ave).apply (lambda x: np.sum (x**2) / 100)
        X_train ['y_energy'] = pd.Series (y_moving_ave).apply (lambda x: np.sum (x**2) / 100)
        X_train ['z_energy'] = pd.Series (z_moving_ave).apply (lambda x: np.sum (x**2 / 100))

        # avg resultant
        X_train ['avg_result_accl'] = [np.mean(i) for i in ((pd.Series(x_moving_ave)**2 + pd.Series(y_moving_ave)**2 + pd.Series(z_moving_ave)**2)**0.5)]

        # signal magnitude area
        X_train ['sma'] =    pd.Series(x_moving_ave).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(y_moving_ave).apply(lambda x: np.sum(abs(x)/100)) \
                        + pd.Series(z_moving_ave).apply(lambda x: np.sum(abs(x)/100))


        # converting the signals from time domain to frequency domain using FFT
        x_list_fft = np.abs (np.fft.fft (x_moving_ave))
        y_list_fft = np.abs (np.fft.fft (y_moving_ave))
        z_list_fft = np.abs (np.fft.fft (z_moving_ave))

        # Statistical Features on raw x, y and z in frequency domain
        # FFT mean
        X_train['x_mean_fft'] = pd.Series(x_list_fft).apply(lambda x: np.mean (x))
        X_train['y_mean_fft'] = pd.Series(y_list_fft).apply(lambda x: np.mean (x))
        X_train['z_mean_fft'] = pd.Series(z_list_fft).apply(lambda x: np.mean (x))

        # FFT std dev
        X_train['x_std_fft'] = pd.Series(x_list_fft).apply(lambda x: np.std (x))
        X_train['y_std_fft'] = pd.Series(y_list_fft).apply(lambda x: np.std (x))
        X_train['z_std_fft'] = pd.Series(z_list_fft).apply(lambda x: np.std (x))

        # FFT avg absolute diff
        X_train['x_aad_fft'] = pd.Series(x_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
        X_train['y_aad_fft'] = pd.Series(y_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
        X_train['z_aad_fft'] = pd.Series(z_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

        # FFT min
        X_train['x_min_fft'] = pd.Series(x_list_fft).apply(lambda x: np.min (x))
        X_train['y_min_fft'] = pd.Series(y_list_fft).apply(lambda x: np.min (x))
        X_train['z_min_fft'] = pd.Series(z_list_fft).apply(lambda x: np.min (x))

        # FFT max
        X_train['x_max_fft'] = pd.Series(x_list_fft).apply(lambda x: np.max (x))
        X_train['y_max_fft'] = pd.Series(y_list_fft).apply(lambda x: np.max (x))
        X_train['z_max_fft'] = pd.Series(z_list_fft).apply(lambda x: np.max (x))

        # FFT max-min diff
        X_train['x_maxmin_diff_fft'] = X_train['x_max_fft'] - X_train['x_min_fft']
        X_train['y_maxmin_diff_fft'] = X_train['y_max_fft'] - X_train['y_min_fft']
        X_train['z_maxmin_diff_fft'] = X_train['z_max_fft'] - X_train['z_min_fft']

        # FFT median
        X_train['x_median_fft'] = pd.Series(x_list_fft).apply(lambda x: np.median(x))
        X_train['y_median_fft'] = pd.Series(y_list_fft).apply(lambda x: np.median(x))
        X_train['z_median_fft'] = pd.Series(z_list_fft).apply(lambda x: np.median(x))

        # FFT median abs dev 
        X_train['x_mad_fft'] = pd.Series(x_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
        X_train['y_mad_fft'] = pd.Series(y_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
        X_train['z_mad_fft'] = pd.Series(z_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))

        # FFT Interquartile range
        X_train['x_IQR_fft'] = pd.Series(x_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        X_train['y_IQR_fft'] = pd.Series(y_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        X_train['z_IQR_fft'] = pd.Series(z_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

        # FFT values above mean
        X_train['x_above_mean_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(x > np.mean (x)))
        X_train['y_above_mean_fft'] = pd.Series(y_list_fft).apply(lambda x: np.sum(x > np.mean (x)))
        X_train['z_above_mean_fft'] = pd.Series(z_list_fft).apply(lambda x: np.sum(x > np.mean (x)))

        # FFT skewness
        X_train['x_skewness_fft'] = pd.Series(x_list_fft).apply(lambda x: stats.skew(x))
        X_train['y_skewness_fft'] = pd.Series(y_list_fft).apply(lambda x: stats.skew(x))
        X_train['z_skewness_fft'] = pd.Series(z_list_fft).apply(lambda x: stats.skew(x))

        # FFT kurtosis
        X_train['x_kurtosis_fft'] = pd.Series(x_list_fft).apply(lambda x: stats.kurtosis(x))
        X_train['y_kurtosis_fft'] = pd.Series(y_list_fft).apply(lambda x: stats.kurtosis(x))
        X_train['z_kurtosis_fft'] = pd.Series(z_list_fft).apply(lambda x: stats.kurtosis(x))

        # FFT energy
        X_train['x_energy_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(x**2)/50)
        X_train['y_energy_fft'] = pd.Series(y_list_fft).apply(lambda x: np.sum(x**2)/50)
        X_train['z_energy_fft'] = pd.Series(z_list_fft).apply(lambda x: np.sum(x**2/50))

        # FFT avg resultant
        X_train['avg_result_accl_fft'] = [np.mean(i) for i in ((pd.Series(x_list_fft)**2 + pd.Series(y_list_fft)**2 + pd.Series(z_list_fft)**2)**0.5)]

        # FFT Signal magnitude area
        X_train['sma_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(abs(x)/50)) + pd.Series(y_list_fft).apply(lambda x: np.sum(abs(x)/50)) \
                            + pd.Series(z_list_fft).apply(lambda x: np.sum(abs(x)/50))
        

        # Max Indices and Min indices 

        # index of max value in time domain
        X_train['x_argmax'] = pd.Series(x_moving_ave).apply(lambda x: np.argmax(x))
        X_train['y_argmax'] = pd.Series(y_moving_ave).apply(lambda x: np.argmax(x))
        X_train['z_argmax'] = pd.Series(z_moving_ave).apply(lambda x: np.argmax(x))

        # index of min value in time domain
        X_train['x_argmin'] = pd.Series(x_moving_ave).apply(lambda x: np.argmin(x))
        X_train['y_argmin'] = pd.Series(y_moving_ave).apply(lambda x: np.argmin(x))
        X_train['z_argmin'] = pd.Series(z_moving_ave).apply(lambda x: np.argmin(x))

        # absolute difference between above indices
        X_train['x_arg_diff'] = abs(X_train['x_argmax'] - X_train['x_argmin'])
        X_train['y_arg_diff'] = abs(X_train['y_argmax'] - X_train['y_argmin'])
        X_train['z_arg_diff'] = abs(X_train['z_argmax'] - X_train['z_argmin'])


# mean absolute error (ME) for each sensor
print ("\n------------------  Mean Absolute Error  ------------------")

# calculate mean absolute errors
ME_accel = np.mean (accel_error)
ME_gyro_uncalib = np.mean (gyro_uncalib_error)
ME_gyro = np.mean (gyro_error)
ME_linear_accel = np.mean (linear_accel_error)
ME_all = np.mean (accel_error + gyro_uncalib_error + gyro_error + linear_accel_error)

# print mean absolut errors
print ("Accelerometer: %f" % ME_accel)
print ("Gyroscope Uncalibrated: %f" % ME_gyro_uncalib)
print ("Gyroscope: %f" % ME_gyro)
print ("Linear Acceleration: %f" % ME_linear_accel)
print ("All: %f" % ME_all)


# standard deviation of the aboslute error (STD) for each sensor
print ("\n------------------  Standard Deviation  ------------------")

# calculate standard deviations
STD_accel = np.std (accel_error)
STD_gyro_uncalib = np.std (gyro_uncalib_error)
STD_gyro = np.std (gyro_error)
STD_linear_accel = np.std (linear_accel_error)
STD_all = np.std (accel_error + gyro_uncalib_error + gyro_error + linear_accel_error)

# print standard deviations
print ("Accelerometer: %f" % STD_accel)
print ("Gyroscope Uncalibrated: %f" % STD_gyro_uncalib)
print ("Gyroscope: %f" % STD_gyro)
print ("Linear Acceleration: %f" % STD_linear_accel)
print ("All: %f" % STD_all)


# root mean squared error (RMSE) for each sensor
print ("\n------------------  Root Mean Squared Error  ------------------")

# calculate root mean squared errors
RMSE_accel = np.sqrt (np.mean (np.square (accel_error)))
RMSE_gyro_uncalib = np.sqrt (np.mean (np.square (gyro_uncalib_error)))
RMSE_gyro = np.sqrt (np.mean (np.square (gyro_error)))
RMSE_linear_accel = np.sqrt (np.mean (np.square (linear_accel_error)))
RMSE_all = np.sqrt (np.mean (np.square (accel_error + gyro_uncalib_error + gyro_error + linear_accel_error)))

# print root mean squared errors
print ("Accelerometer: %f" % RMSE_accel)
print ("Gyroscope Uncalibrated: %f" % RMSE_gyro_uncalib)
print ("Gyroscope: %f" % RMSE_gyro)
print ("Linear Acceleration: %f" % RMSE_linear_accel)
print ("All: %f" % RMSE_all)


# write the error values to a csv file
error_table = pd.DataFrame ({"Sensor": ['Accelerometer', 'Gyroscope Uncalibrated', 'Gyroscope', 'Linear Acceleration', 'All'],
                              "Mean Absolute Error": [ME_accel, ME_gyro_uncalib, ME_gyro, ME_linear_accel, ME_all],
                              "Standard Deviation": [STD_accel, STD_gyro_uncalib, STD_gyro, STD_linear_accel, STD_all],
                              "Root Mean Squared Error": [RMSE_accel, RMSE_gyro_uncalib, RMSE_gyro, RMSE_linear_accel, RMSE_all]})
error_table.to_csv ('Results/Error Analysis.csv', index = False)