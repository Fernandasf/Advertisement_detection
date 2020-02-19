import warnings
warnings.filterwarnings("ignore")

import extract_Nguyen
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import itertools
from sklearn.externals import joblib

def plot_confusion_matrix(cm, target_names, time, cmap=None, normalize=False, ext=''):
    plt.figure(figsize=(10, 8))

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if ext == '.png' or ext=='NoNorm.png':
        plt.title('Confusion matrix - Handcrafted (N) - LMD+A - %i' %time)
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Resultados/CM_HN_LMDA'+str(time)+ext)
    #plt.show()

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

     vector_features = extract_Nguyen.extract_features(datadir+fname, m) #(vector_features), (spectrogram)
     np.savetxt(scratchdir+data_fname, vector_features)
     return vector_features

def extract_features_from_dataset(file_list, datadir, m):
     """Extracts features from the files specified in a list
     """
     global features 
     features = np.array([extract_features_from_file(f, datadir, m) for f in file_list])
     #print(features)
     return features

def read_fold(fold_file, datadir, path=''):
    fnames = []
    genres = []
    with open(datadir + fold_file, 'r') as f:
        for line in f.readlines():
            l = line.rstrip('\n').split(path)
            #print (l, l[0], l[1], len(l))
            if len(l)<2:
                continue
            fnames.append(l[0])
            genres.append(l[1])
    return fnames, genres

######## Import Dataset ########

datadir = 'All_dataset/'
folds_g = ['LMD.txt']

#datadir_a = '../../Propagandas_dataset/'
folds_a = ['Propagandas_10seg.txt']

for fold_number in range(len(folds_g)):
        fnames_g, genres_g = read_fold(folds_g[fold_number], datadir, path='\t')

for fold_number in range(len(folds_a)):
        fnames_a, genres_a = read_fold(folds_a[fold_number], datadir, path='\t')

fnames_a, genres_a = shuffle(fnames_a, genres_a)
fnames_a100 = fnames_a[0:130]
genres_a100 = genres_a[0:130]

fnames_all = np.concatenate((fnames_g, fnames_a100))
genres_all = np.concatenate((genres_g, genres_a100))

print('fnames_all shape: ', fnames_all.shape)
print('genres_all shape: ', genres_all.shape)

print ("\n N. Train Classes: ", count_elements(genres_all))

f_train, f_test, g_train, g_test = train_test_split(fnames_all, genres_all, train_size=0.5, stratify=genres_all)

#print(type(f_train))
#print(f_train)

def write_txt(files, labels, path=''):
    ff = open(path, 'w')
    for line in range(len(files)):
        ff.write(files[line])
        #ff.write('\t')
        #ff.write(labels[line])
        ff.write('\n')
    ff.close()
  

write_txt(f_train, g_train,  path='genres_train_LMDA.txt')
write_txt(f_test, g_test, path='genres_test_LMDA.txt')

#print(len(fnames_all), len(genres_all))
#print(len(f_train), len(g_train), len(f_test), len(g_test))

F1_array_mean = []
F1_array = []
M_array = np.array(('0.5', '1', '5', '10'))

#M = [0.5, 1, 5, 10] # 10, 5, 2, 1 seg
M = [0.5, 1, 5, 10]
#
f = open('Resultados/Genre_HN_LMDA.txt', 'w')

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
    fet_train = []
    fet_test = []

    # ---------------- TRAIN ------------
    #for fold_number in range(len(folds_all)):
    features_train = extract_features_from_dataset(f_train, datadir, i)

    fet_train = features_train
    y_train = g_train

    # --------------- TEST ----------------   
    #for fold_number2 in range(len(folds_all)):
    features_test = extract_features_from_dataset(f_test, datadir, i)
        #print ('feat_test:' ,features_test.shape)
        
    fet_test = features_test
    ground_truth = g_test   

    model = classify.model_fit(fet_train, y_train)
    pred = classify.model_predict(model, fet_test, ground_truth)

    #_______Classes ________

    print ("\n N. Train Classes: ", count_elements(g_train))
    print ("N. Test Classes: ", count_elements(g_test))

    joblib.dump(model, "Features/model_HN_LMDA"+str(i)+".pkl")
    joblib.dump(pred, "Features/y_pred_HN_LMDA"+str(i)+".pkl")
    joblib.dump(ground_truth, "Features/y_test_HN_LMDA"+str(i)+".pkl")
    joblib.dump(fet_train, "Features/x_train_HN_LMDA"+str(i)+".pkl")
    joblib.dump(y_train, "Features/y_train_HN_LMDA"+str(i)+".pkl")
    
    ########### Etapa de Teste ##################

    print("\nClassification report for classifier %s:\n%s"
        % (model, sklearn.metrics.classification_report(ground_truth, pred)))

    F1_all = sklearn.metrics.classification_report(ground_truth, pred)
    #print(F1_all)

    ########## F1-score apenas para propaganda ##############
    
    F1_class = F1_all.split('\n')[2]
    #print(F1_class)
    F1_classP = F1_class.replace(',', ' ').split()
    #print(F1_classP)
    F1_P = float(F1_classP[3])
    
    print('Tempo (seg): ', i)
    print('F1-score P: ', F1_P)
    F1_array.append(F1_P)

    F1 = sklearn.metrics.f1_score(ground_truth, pred, average='weighted')
    print ('F1-score medio: ', round(F1, 2))
    F1_array_mean.append(round(F1, 2))

    CM = sklearn.metrics.confusion_matrix(ground_truth, pred)

    print("\nConfusion matrix:\n%s" % CM)
    print ("\n")

    f.write('________________ {}'.format(i))
    f.write(' ______________________')
    f.write('\n')
    f.write('Classification Report: \n {}'.format(F1_all))
    f.write('\n')
    f.write("F1_score: {}".format(round(F1, 2)))
    f.write('\n')
    f.write('Confusion matrix:  \n {}'.format(CM))
    f.write('\n\n')

    tn = ['adver.', 'axe', 'bachata', 'bolero', 'forro', 'gaucha', 'merengue', 'pagode', 'salsa', 'sertanejo', 'tango']

    plot_confusion_matrix(CM,target_names=tn, time=i, cmap=None, normalize=True, ext='.pdf')
    plot_confusion_matrix(CM,target_names=tn, time=i, cmap=None, normalize=True, ext='.png')
    plot_confusion_matrix(CM,target_names=tn, time=i, cmap=None, normalize=False, ext='NoNorm.png')

    
F1_array = np.array(F1_array)
F1_array_mean = np.array(F1_array_mean)
M_array = np.array(M_array)

f.close()
np.savetxt('Resultados/Genre_HN_LMDA.csv', (M_array, F1_array, F1_array_mean), delimiter=',', fmt='%s')
