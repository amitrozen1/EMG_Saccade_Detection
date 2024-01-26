import pyedflib
from scipy import signal
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
import cupy as xp
from scipy.signal import butter, filtfilt
from General.HelperFunctions import isMyAnnotation
from scipy.signal import resample
from picard import picard

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], analog=False, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        y = filtfilt(b,a,data)
        return y

def window_rms_single(a, window_size=800):
    # Square the values
    a_squared = np.squeeze(a**2)

    # Apply a moving window average to the squared values
    window = np.squeeze(np.ones((1,window_size)) / float(window_size))
    a_squared_convolved = np.convolve(a_squared, window, mode='same')

    # Calculate the square root of the averaged squared values
    return np.sqrt(a_squared_convolved)


# def window_rms_single(a, window_size=800):
#     return np.sqrt(np.convolve(a**2, (np.ones(window_size)/float(window_size))))


def NotchFilterSignal(noisySignal, sampling_rate, removed_frequency=50.0, Q_Factor=30.0):
    # Design notch filter
    b_notch, a_notch = signal.iirnotch(removed_frequency, Q_Factor, sampling_rate)
    return signal.filtfilt(b_notch, a_notch, noisySignal)

def ExtractIdAndPos(s):
    splitted = s.split("_")
    return ( int(splitted[1]), splitted[2])

def extractAnnotation(s : str, timing):
    s = s.lower()
    splitted = s.split("_")
    a = Annotation()
    a.Time = timing
    if ( splitted[0] in ["smile", "angry", "blink"]):
        a.Type = splitted[0]
        a.TrialId = int(splitted[1])
        a.order = int(splitted[1])
        a.isStart = splitted[2] == "start"
        return a

    if (splitted[0][0] == "a"):
        a.Type = "Listening"
    elif ( splitted[0][0] == "r"):
        a.Type = "Reading"
    
    if(splitted[0][1] == "s"):
        a.isStart = True
    else:
        a.isStart = False
    
    a.order = int(splitted[1])
    a.TrialId = int(splitted[3])
    a.Story = splitted[5]
    return a

def GetCallibrationsTicks(chunks, calibration_state_to_analyze, num_repeats = 3):
    ticks = []
    for i in range(0,num_repeats):
        key = '%s__%d' % (calibration_state_to_analyze, i)
        chunk = chunks[key]
        ticks.append(int(chunk.Start.Time))
        ticks.append(int(chunk.End.Time))
    
    return ticks

def resample_components(components, freq, new_freq):
    if ( len(components.shape) > 1):
        return resample(components, int(len(components[0]) * new_freq / freq), axis=1)
    return resample(components, int(len(components) * new_freq / freq))

def decimate_coponent(components, freq, new_freq):
    if ( len(components.shape) > 1):
        return signal.decimate(components, int(freq/new_freq), axis=1)
    return signal.decimate(components, int(freq/new_freq))

def resample_by_taking_one_every_n(components, freq, new_freq):
    if ( len(components.shape) > 1):
        return components[:,::int(freq/new_freq)]
    return components[::int(freq/new_freq)]

def resample_by_averaging_every_n(components, freq, new_freq):
    if ( len(components.shape) > 1):
        return np.array([np.mean(components[:,i:i+int(freq/new_freq)], axis=1) for i in range(0, len(components[0]), int(freq/new_freq))])
    return np.array([np.mean(components[i:i+int(freq/new_freq)]) for i in range(0, len(components), int(freq/new_freq))])

def resample_by_averaging_every_n_convolve(components, freq, new_freq):
    N = int(freq/new_freq)
    window = (np.ones((1,N)) / float(N)).flatten()
    if ( len(components.shape) > 1):
        return np.array([np.convolve(components[i,:], window, mode='valid')[::N] for i in range(0, components.shape[0])])
    return np.convolve(components, window, mode='valid')[::N]

class EdfAnalyzer:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def readEdf(f : pyedflib.EdfReader, doButter = True, forEEG = False):
        sampling_rate = int(f.getSampleFrequency(0))
        n = 16
        # num_sets = 1
        # size_limit = int(f.getNSamples()[0] / num_sets)
        # #read EDF
        # sigbufs = np.zeros((n, size_limit))
        # for i in np.arange(n):
        #     sigbufs[i, :] = f.readSignal(i, start=0, n=size_limit)
        
        sigbufs = np.array([f.readSignal(i) for i in range(0,n)])
        #Filter before ICA
        sigbufs = NotchFilterSignal(sigbufs, sampling_rate)
        if (doButter):
            if ( not forEEG):
                sigbufs = butter_bandpass_filter(sigbufs, 20.0, 400.0, sampling_rate)
            else:
                sigbufs = butter_bandpass_filter(sigbufs, 0.5, 40.0, sampling_rate) #todo - check if this is good for EEG
        
        return sigbufs, sampling_rate
    
    @staticmethod
    def readGyroscopeData(f : pyedflib.EdfReader, doButter = True):
        gyro_indexes = [16,17,18]
        sampling_rate = int(f.getSampleFrequency(gyro_indexes[0])) 
        sigbufs = np.array([f.readSignal(i) for i in gyro_indexes])
        sigbufs = NotchFilterSignal(sigbufs, sampling_rate)

        if (doButter):
            sigbufs = butter_bandpass_filter(sigbufs, 0.1, 20.0, sampling_rate)

        return sigbufs, sampling_rate

    @staticmethod
    def getCallibrationTicks(chunks, calibration_state_to_analyze = "smile"):
        return GetCallibrationsTicks(chunks, calibration_state_to_analyze)

    @staticmethod
    def getAnnotationChunks(f : pyedflib.EdfReader, correctBy, RmsDownsampleWindow = 1):
        annotations = f.readAnnotations()
        file_sampling_rate = int(f.getSampleFrequency(0))
        startIndex = np.where(annotations[2] == "StartExperiment")[0][-1] + 1

        endIndex = len(annotations[2])
        startCorrectIdx = np.where(annotations[2] == "Smile_0_start")[0][0]
        startCorrectionTime = annotations[0][startCorrectIdx] - 60 # fix error by xTrodes. # __class__.startCorrectionTime
        
        chunks = dict()
        timing_in_seconds = 0
        index=""

        toWindowConvertionValue = int(RmsDownsampleWindow * (file_sampling_rate/1000))
        if (RmsDownsampleWindow == 1):
            toWindowConvertionValue = 1


        for i in range(startIndex, endIndex):
            annotation = str(annotations[2][i])
            
            if ( not isMyAnnotation(annotation)):
                continue
            
            #todo - no need to perform *4000 outside of this code.
            timing_in_seconds = annotations[0][i] - correctBy - startCorrectionTime # fix error by xTrodes. __class__.startCorrectionTime
            timing = timing_in_seconds*file_sampling_rate/toWindowConvertionValue
            
            extractedAnnotation = extractAnnotation(annotation, timing)
            index = "%s_%s_%d" % (extractedAnnotation.Type, extractedAnnotation.Story, extractedAnnotation.TrialId)
            if ( index in chunks):
                if (extractedAnnotation.isStart):
                    chunks[index].Start = extractedAnnotation
                else:
                    chunks[index].End = extractedAnnotation
            else:
                chunks[index] = Chunk()
                if(extractedAnnotation.isStart):
                    chunks[index].Start = extractedAnnotation
                else:
                    chunks[index].End = extractedAnnotation
        
        #get the end of the last index:
        trim_start = int(startCorrectionTime*file_sampling_rate)
        trim_end = trim_start + int((timing_in_seconds+5)*file_sampling_rate) # extra 5 seconds
        
        return chunks, trim_start, trim_end

    @staticmethod
    def RemoveRedundentData(x, trim_start, trim_end):
        return x[:, trim_start:trim_end]
    
    @staticmethod
    def Read(file_path, correctBy=0, downsample_window_in_ms=1,  doButter = True, forEEG = False):
        # Read EDF: if downsample_window_in_ms is 1, the user should downsample the data himself. annotations are handled here.
        f = pyedflib.EdfReader(file_path)
        sigbufs, freq = __class__.readEdf(f, doButter, forEEG)
        chunks, trim_start, trim_end = __class__.getAnnotationChunks(f, correctBy, downsample_window_in_ms)
        sigbufs = __class__.RemoveRedundentData(sigbufs, trim_start, trim_end)
        f.close()
        return sigbufs, chunks, freq
    
    @staticmethod
    def ReadWithGyro(file_path, correctBy=0, downsample_window_in_ms=1, doButter = True, forEEG = False):
        f = pyedflib.EdfReader(file_path)
        emgData, freq = __class__.readEdf(f, doButter, forEEG)
        gyroData, gyroFreq = __class__.readGyroscopeData(f, doButter)

        chunks, trim_start, trim_end = __class__.getAnnotationChunks(f, correctBy, downsample_window_in_ms)
        emgData = __class__.RemoveRedundentData(emgData, trim_start, trim_end)
        gyroData = __class__.RemoveRedundentData(gyroData, trim_start//4 , trim_end//4)
        
        f.close()
        return emgData, chunks, freq, gyroData, gyroFreq
    
    @staticmethod
    def ICA(matrix, n_components):
        transformer = FastICA(n_components=n_components, random_state=0 )
        x= np.transpose(transformer.fit_transform(np.transpose(matrix))) #FAST ICA
        w = transformer.components_ #np.linalg.inv(transformer.mixing_) #transformer.components_ # np.dot(unmixing_matrix, self.whitening_)
        #K, W, X = picard(matrix, n_components=n_components, ortho=True, extended=True, max_iter=200)
        return w, x #np.dot(W,K) ,X
    
    @staticmethod
    def ICA_picard(matrix, n_components):
        K, W, X = picard(matrix, n_components=n_components, ortho=True, extended=True, max_iter=200)
        return np.dot(W,K) ,X
    
    @staticmethod
    def PCA(matrix, n_components):
        transformer = PCA(n_components=n_components, random_state=0)
        x = xp.transpose(transformer.fit_transform(xp.transpose(matrix)))
        w = transformer.components_
        return w,x
    
    @staticmethod
    def window_rms(a, window_size_in_ms=200, original_freq = 4000):
        factor = original_freq//1000
        factored_window_size = window_size_in_ms*factor
        if (a.shape[0] == 1 or len(a.shape) < 2 or a.shape[1] == 1):
            return window_rms_single(a, factored_window_size)
        return np.array([window_rms_single(x,factored_window_size) for x in a])

    def resample_by_taking_one_every_n(a, window_size_in_ms, original_freq):
        return resample_by_taking_one_every_n(a, original_freq, 1000/window_size_in_ms)

    def resample_by_averaging_every_n(a, window_size_in_ms, original_freq):
        return resample_by_averaging_every_n_convolve(a, original_freq, 1000/window_size_in_ms)

    @staticmethod
    def window_rms_downsample(a, window_size_in_ms, original_freq = 4000):
        a = __class__.window_rms(a, window_size_in_ms, original_freq)
        #return resample_components(a, original_freq, 1000/window_size_in_ms)
        #return decimate_coponent(a, original_freq, 1000/window_size_in_ms)
        return resample_by_taking_one_every_n(a, original_freq, 1000/window_size_in_ms)


class Chunk:
    def __init__(self) -> None:
        self.Start = None
        self.End = None
        self.Data = None

    #todo: test these.
    def GyroStart_whenNoRmsOnGyro(self, emgRmsDownsampleWindow = 1):
        if emgRmsDownsampleWindow == 1:
            return self.Start.Time//4
        return int(self.Start.Time * emgRmsDownsampleWindow)
        
    def GyroEnd_whenNoRmsOnGyro(self, emgRmsDownsampleWindow = 1):
        if emgRmsDownsampleWindow == 1:
            return self.End.Time//4
        return int(self.End.Time * emgRmsDownsampleWindow)


class ExtendedChunk(Chunk):
    def __init__(self, chunk=None, isOther=False) -> None:
        super().__init__()
        if (chunk!=None):
            self.Start = chunk.Start
            self.End = chunk.End
            self.Data = chunk.Data
        
        self.isOther = isOther

class Annotation:
    def __init__(self) -> None:
        self.Type = ""
        self.Story = ""
        self.Reader = ""
        self.order = None
        self.TrialId = None
        self.isStart = False
        self.Time = None
