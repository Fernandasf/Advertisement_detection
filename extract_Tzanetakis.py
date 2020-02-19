import librosa
import numpy as np
from sklearn.decomposition import PCA

def extract_features(fname, m): # 0.5, 1, 5, 10 seg
        if m==10:
                audio, sr = librosa.load(fname, sr=22050)
                #print (len(audio))
                maxN = int(m * sr)
                x = audio[0:maxN]
                x = (x - np.mean(x)) / np.std(x)
                w = np.hanning(len(x))
                xw = x * w
                where_are_NaNs = np.isnan(xw)
                xw[where_are_NaNs] = 0

        else:
                audio, sr = librosa.load(fname, sr=22050) 
                #print (len(audio))
                maxN = int(m * sr)
                minN = int(sr)
                x = audio[minN:maxN+minN]
                x = (x - np.mean(x)) / np.std(x)
                w = np.hanning(len(x))
                xw = x * w
                where_are_NaNs = np.isnan(xw)
                xw[where_are_NaNs] = 0

    
        #s = np.abs(librosa.stft(xw, n_fft=2048, hop_length=1024, win_length=2048))
        mfcc = librosa.feature.mfcc(xw, sr, n_mfcc=5, dct_type=2, hop_length=1024)

        spectral_centroid = librosa.feature.spectral_centroid(xw, sr=22050, n_fft=2048, hop_length=1024)
        spectral_rolloff = librosa.feature.spectral_rolloff(xw, sr=22050, n_fft=2048, hop_length=1024)
        htzcr = librosa.feature.zero_crossing_rate(xw, frame_length=3072, hop_length=1024, center=True)
        stzcr = librosa.feature.zero_crossing_rate(xw, frame_length=1024, hop_length=1024, center=True)
        rmse = librosa.feature.rmse(xw, S=None, frame_length=2048, hop_length=1024, center=True, pad_mode='reflect')
        #flatness = librosa.feature.spectral_flatness(xw,n_fft=2048, hop_length=1024, amin=1e-10, power=2.0)
        onset_flux = librosa.onset.onset_strength(xw, sr=sr, n_fft=2048, hop_length=1024)

        onset_flux = onset_flux.reshape(1, len(onset_flux))
        
        #features = np.vstack( (spectral_centroid , spectral_rolloff , htzcr , stzcr , rmse , flatness) )
        features = np.vstack( (spectral_centroid , spectral_rolloff , htzcr , stzcr , rmse , onset_flux) )

        dfeatures = librosa.feature.delta(features)
        ddfeatures = librosa.feature.delta(dfeatures)
        all_features = np.vstack( (features, dfeatures, ddfeatures) )

        texture_len = 2
        means = np.array(\
                [np.mean(all_features[:,i*texture_len:(i+1)*texture_len], axis=1)\
                for i in range(int(all_features.shape[1] / texture_len))]\
                )
        var = np.array(\
                [np.var(all_features[:,i*texture_len:(i+1)*texture_len], axis=1)\
                for i in range(int(all_features.shape[1] / texture_len))]\
                )

        mm = np.mean(means, axis=0)
        mv = np.mean(var, axis=0)
        vm = np.var(means, axis=0)
        vv = np.var(var, axis=0)

        vector_features = np.hstack((mm, mv, vm, vv))

        ########### MFCC ################

        dmfcc = librosa.feature.delta(mfcc)
        ddmfcc = librosa.feature.delta(dmfcc)
        all_mfcc = np.vstack( (mfcc, dmfcc, ddmfcc) )

        texture_len = 2
        means_mfcc = np.array(\
                [np.mean(all_mfcc[:,i*texture_len:(i+1)*texture_len], axis=1)\
                for i in range(int(all_mfcc.shape[1] / texture_len))]\
                )
        var_mfcc = np.array(\
                [np.var(all_mfcc[:,i*texture_len:(i+1)*texture_len], axis=1)\
                for i in range(int(all_mfcc.shape[1] / texture_len))]\
                )

        mm_mfcc = np.mean(means_mfcc, axis=0)
        mv_mfcc = np.mean(var_mfcc, axis=0)
        vm_mfcc = np.var(means_mfcc, axis=0)
        vv_mfcc = np.var(var_mfcc, axis=0)

        vector_features_mfcc = np.hstack((mm_mfcc, mv_mfcc, vm_mfcc, vv_mfcc))

        #print (vector_features.shape, vector_features_mfcc.shape)

        vector_features_all = np.hstack((vector_features, vector_features_mfcc))

        #print (vector_features_all.shape)

        return vector_features_all

