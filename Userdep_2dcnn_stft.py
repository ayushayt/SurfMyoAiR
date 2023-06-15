import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
from scipy import signal
import scipy.io
from scipy import signal
from scipy import interpolate
import mne
from sklearn.metrics import accuracy_score,confusion_matrix
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dropout, BatchNormalization, Input,UpSampling1D
from tensorflow.keras.layers import concatenate, Lambda, Conv2D, MaxPooling2D, GlobalAveragePooling2D,LSTM,Bidirectional
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import zscore
from scipy.signal import butter, lfilter
import time

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def process_EMG(trial,duration,winlen,interp_type,T,F):

    X = np.zeros((2600,T,F,5))
    Y = np.zeros((2600,))
    testctr = 0
    
    for folder in os.listdir('EMG_Data'):
        for rep in range (1,3):
            for num in range(65,91):
                emg_sig = np.load('EMG_Data/'+folder+'/Segmented/EMG_'+chr(num)+'_'+str(trial+1)+'_'+str(rep)+'.npy')
                
                emg_signal = np.zeros((duration,5))

                
                if len(emg_sig)<=duration:
                    for i in range(0,5):
                        x = np.linspace(0, 1,len(emg_sig))
                        y = emg_sig[:,i]
                        f = interpolate.interp1d(x, y,kind=interp_type)
                        xnew = np.linspace(0, 1,duration)
                        emg_signal[:,i] = f(xnew)

                else:
                    emg_signal = emg_sig[0:duration,:]
                
                for i in range(0,5):
                    f, t, zxx = signal.stft(emg_signal[:,i], fs=2000, nperseg=winlen,boundary=None)
                    #print(zxx.shape)
                    zxx = np.abs(zxx)
                    zxx = (zxx-np.mean(zxx))/np.std(zxx)

                
                    X[testctr,:,:,i] = np.transpose(zxx)
                    
                Y[testctr] = num-65
                #print(num)
                testctr = testctr + 1

    return X,Y
    

folderarr = []
for fo in os.listdir('EMG_Data'):
    folderarr.append(fo)
    

    
accarr = []

for fold_no in range(0,5):

    trainrefctr = 0
    testrefctr = 0

    X_train = np.zeros((10400,79,101,5))
    X_test = np.zeros((2600,79,101,5))
    Y_train = np.zeros((10400,))
    Y_test = np.zeros((2600,))

    print('Fold number : ' + str(fold_no))

    for fold_id in range(0,5):
        if fold_id==fold_no:
            [X,Y] = process_EMG(fold_id,8000,200,'cubic',79,101)
            X_test[int(testrefctr*2600):int((testrefctr+1)*2600),:,:,:] = X
            Y_test[int(testrefctr*2600):int((testrefctr+1)*2600)] = Y
            testrefctr = testrefctr + 1
            
        else:
            [X,Y] = process_EMG(fold_id,8000,200,'cubic',79,101)
            X_train[int(trainrefctr*2600):int((trainrefctr+1)*2600),:,:,:] = X
            Y_train[int(trainrefctr*2600):int((trainrefctr+1)*2600)] = Y
            trainrefctr = trainrefctr + 1
                
    print(X_train.shape)
    print(X_test.shape)
    
            
    tensorflow.keras.backend.clear_session()
    tensorflow.keras.backend.clear_session()
    i1 = Input(shape=(79,101,5))
    x1 = Conv2D(32,(3,3),padding='same',activation='relu')(i1)
    x1 = MaxPooling2D()(x1)
    x1 = Conv2D(64,(3,3),padding='same',activation='relu')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = Conv2D(128,(3,3),padding='same',activation='relu')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = Conv2D(256,(3,3),padding='same',activation='relu')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = Flatten()(x1)
    x1 = Dropout(0.5)(x1)
    output = Dense(26, activation='softmax')(x1)
    
    model = Model(inputs=i1, outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    es = EarlyStopping(monitor='val_accuracy', verbose=0, patience=10)
    model.fit(X_train, y=to_categorical(Y_train),validation_split=0.2,epochs=500, batch_size=256,verbose=0,callbacks=[es])
    
    pred = model.predict(X_test)
    Y_pred = np.argmax(pred,axis=1)
    print(accuracy_score(Y_pred,Y_test)*100)
    '''
    if fold_no==0:
        np.save('results_pred/Userdep/Y_test_fold1_stft.npy',Y_test)
        np.save('results_pred/Userdep/Y_pred_fold1_stft.npy',Y_pred)
    elif fold_no==1:
        np.save('results_pred/Userdep/Y_test_fold2_stft.npy',Y_test)
        np.save('results_pred/Userdep/Y_pred_fold2_stft.npy',Y_pred)
    elif fold_no==2:
        np.save('results_pred/Userdep/Y_test_fold3_stft.npy',Y_test)
        np.save('results_pred/Userdep/Y_pred_fold3_stft.npy',Y_pred)
    elif fold_no==3:
        np.save('results_pred/Userdep/Y_test_fold4_stft.npy',Y_test)
        np.save('results_pred/Userdep/Y_pred_fold4_stft.npy',Y_pred)
    elif fold_no==4:
        np.save('results_pred/Userdep/Y_test_fold5_stft.npy',Y_test)
        np.save('results_pred/Userdep/Y_pred_fold5_stft.npy',Y_pred)
    '''
    accarr.append(accuracy_score(Y_pred,Y_test)*100)
            
print(np.mean(accarr)) 


