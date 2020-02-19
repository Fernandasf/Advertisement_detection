import warnings
warnings.filterwarnings("ignore")

import extract_Nguyen5
import classify

import os.path
import hashlib
import os
import glob

import librosa
import numpy as np
import sklearn.svm
import sklearn.preprocessing
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.externals import joblib

########## Funções Auxiliares #########
def count_elements(seq) -> dict:
    """Tally elements from `seq`."""
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist

def extract_features_from_file(fname, datadir, m, scratchdir='scratch/'):
    """Extract features from a file
    """
    #print("Extracting features from %s\n", datadir+fname)
    data_fname = hashlib.sha224(fname.encode('utf-8')).hexdigest()

    if os.path.isfile(scratchdir+data_fname) is True:
        feats = np.loadtxt(scratchdir+data_fname)
        return feats

    vector_features = extract_Nguyen5.extract_features(datadir+fname, m)
    np.savetxt(scratchdir+data_fname, vector_features)
    return vector_features  

def extract_features_from_dataset(file_list, datadir, m):
    """Extracts features from the files specified in a list
    """
    global features 
    features = np.array([extract_features_from_file(f, datadir, m) for f in file_list])
    #print(features)
    return features

def read_fold(fold_file, datadir):
    fnames = []
    genres = []
    with open(datadir + fold_file, 'r') as f:
        for line in f.readlines():
            l = line.rstrip('\n').split('\t')
            if len(l)<2:
                continue
            fnames.append(l[0])
            genres.append(l[1])
    return fnames, genres

######## Import Dataset ########

datadir_train = 'All_dataset/MSR/'
datadir_test = 'All_dataset/MSG/'

folds_train_bases = [['Completa_5seg_train.txt'], ['NTT_5seg_train.txt'], ['TNT_5seg_train.txt']]
#folds_test_bases = ['TT_5seg_test.txt', 'NTT_5seg_test.txt', 'TNT_5seg_test.txt']
folds_test = ['Gtzan_5seg.txt']
#folds_test = ['Gtzan_5seg.txt']

col1_array = ['C']
col2_array = ['T']
col3_array = ['NT']
#col4_array = ['TNT']

f = open('Resultados_Speech/Speech_HN_MSR_MSG.txt', 'w')

for folds_train in folds_train_bases:

    print("\n ________________________________Base Treino: ", folds_train, "________________________________________________")

    M = [0.5, 1, 5]
    
    for i in M:
        ######### Apagar a pasta sempre que iniciar um novo parâmetro ##############
        files = glob.glob('scratch/*')
        print ("antes: ", len(files))
        for fi in files:
            os.remove(fi)
        files2 = glob.glob('scratch/*')
        print ('depois: ', len(files2))

        print ("\n ------------ Resultados para m = " , i, "---------------- ")
        ######## Etapa de Treinamento ########
        predictions = []
        ground_truth = []
        y_train = []
        f_train = []
        f_test = []

        # ---------------- TRAIN ------------
        for fold_number in range(len(folds_train)):
            #print("Fold number: ", fold_number)
            # Test script
            fnames_train, genres_train = read_fold(folds_train[fold_number], datadir_train) 
            features_train = extract_features_from_dataset(fnames_train,datadir_train, i)
            print ('feat_train shape:' ,features_train.shape)
            #print ('feat_train[0] shape:' ,features_train[0].shape)

            f_train = features_train
            y_train += genres_train

        # --------------- TEST ----------------   
        for fold_number2 in range(len(folds_test)):
            fnames_test, genres_test = read_fold(folds_test[fold_number2], datadir_test)
            features_test = extract_features_from_dataset(fnames_test,datadir_test, i)
            print ('feat_test:' ,features_test.shape)
            
            f_test = features_test
            ground_truth += genres_test   

        model = classify.model_fit(f_train, y_train)
        pred = classify.model_predict(model, f_test, ground_truth)

        #_______Classes ________

        joblib.dump(model, "Features_Speech/model_HN_MSR_MSG"+str(i)+".pkl")
        joblib.dump(pred, "Features_Speech/y_pred_HN_MSR_MSG"+str(i)+".pkl")
        joblib.dump(ground_truth, "Features_Speech/y_test_HN_MSR_MSG"+str(i)+".pkl")
        joblib.dump(f_train, "Features_Speech/x_train_HN_MSR_MSG"+str(i)+".pkl")
        joblib.dump(y_train, "Features_Speech/y_train_HN_MSR_MSG"+str(i)+".pkl")

        print ("\n N. Train Classes: ", count_elements(genres_train))
        print ("N. Test Classes: ", count_elements(genres_test))

        ########### Etapa de Teste ##################
        print('\nBase de teste: ', folds_train, ' - Tempo: ', i)
        print("\nClassification report for classifier %s:\n%s"
            % (model, sklearn.metrics.classification_report(ground_truth, pred)))

        F1_all = sklearn.metrics.classification_report(ground_truth, pred)

        F1 = sklearn.metrics.f1_score(ground_truth, pred, average='weighted')
        print ('F1: ', round(F1, 2))

        print("\nConfusion matrix:\n%s" % sklearn.metrics.confusion_matrix(ground_truth, pred))
        print ("\n")

        CM = sklearn.metrics.confusion_matrix(ground_truth, pred)

        #tn, fp, fn, tp = (sklearn.metrics.confusion_matrix(ground_truth, pred)).ravel()
        #print (tn, fp, fn, tp)

        ################# Armazenamento do F1-score #############
        if folds_train == ['Completa_5seg_train.txt']:
            col1_array.append(round(F1, 2))
        if folds_train == ['NTT_5seg_train.txt']:
            col2_array.append(round(F1, 2))
        if folds_train == ['TNT_5seg_train.txt']:
            col3_array.append(round(F1, 2))
        #if folds_train == ['TNT_5seg_train.txt']:
        #    col4_array.append(round(F1, 2))

        f.write('________Train set: {}'.format(folds_train))
        f.write(' -- Test set: {}'.format(folds_test))
        f.write('________\n')
        f.write('---------------------- {}'.format(i))
        f.write(' ------------------')
        f.write('\n')
        f.write('Classification Report: \n {}'.format(F1_all))
        f.write('\n')
        f.write("F1_score: {}".format(round(F1, 2)))
        f.write('\n')
        f.write('Confusion matrix:  \n {}'.format(CM))
        f.write('\n\n')

f.close()

################ Salvar em arquivo formato csv ##########################
col0_array = np.array(('dataset', '0.5seg', '1seg', '5seg'))

col1_array = np.array(col1_array)
col2_array = np.array(col2_array)
col3_array = np.array(col3_array)
#col4_array = np.array(col4_array)

np.savetxt('Resultados_Speech/Speech_HN_MSR_MSG.csv', (col0_array, col1_array, col2_array, col3_array), delimiter=',', fmt='%s')
