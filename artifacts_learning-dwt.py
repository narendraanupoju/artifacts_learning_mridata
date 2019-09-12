# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:36:17 2019

@author: narendra
"""
import os
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from skimage import measure
import pywt

dirName_under = r"file_path_under";
dirName_fully = r"file_path_fully";

def FileRead(file_path):
    nii = nib.load(file_path)
    data = nii.get_data()
    return data

def Nifti3Dto2D(Nifti3D):
    Nifti3DWOChannel = Nifti3D#[:,:,:,0] #Considering there is only one chnnel info
    Nifti2D = Nifti3DWOChannel.reshape(np.shape(Nifti3DWOChannel)[0], np.shape(Nifti3DWOChannel)[1] * np.shape(Nifti3DWOChannel)[2])
    return Nifti2D

def Nifti2Dto1D(Nifti2D):
    Nifti1D = Nifti2D.reshape(np.shape(Nifti2D)[0] * np.shape(Nifti2D)[1])
    return Nifti1D

def Nifti1Dto2D(Nifti1D, height):
    Nifti2D = Nifti1D.reshape(height,int(np.shape(Nifti1D)[0]/height))
    return Nifti2D

def Nifti2Dto3D(Nifti2D):
    Nifti3DWOChannel = Nifti2D.reshape(np.shape(Nifti2D)[0],np.shape(Nifti2D)[0],np.shape(Nifti2D)[1]//np.shape(Nifti2D)[0])
    return Nifti3DWOChannel

def FileSave(data, file_path):
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, file_path)
    
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def getListOfFiles(dirName): 
    listOfFile = os.listdir(dirName)
    allFiles = []
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

def main_under():
    listOfFiles_under = getListOfFiles(dirName_under)
    cA_u = []
    cD_u = []
    listOfFiles_under = []
    for (dirpath, dirnames, filenames) in os.walk(dirName_under):
        listOfFiles_under += [os.path.join(dirpath, file) for file in filenames]
    for elem_under in listOfFiles_under:
        red_files_under = FileRead(elem_under)
        twod_under =normalize( Nifti3Dto2D(red_files_under))
        cA_1 , cD_1 =pywt.dwt(twod_under, 'haar') 
        cA_u.append(Nifti2Dto1D(cA_1))
        cD_u.append(Nifti2Dto1D(cD_1))
    cA_u = np.asarray(cA_u, dtype=np.float64, order='C')
    cD_u = np.asarray(cD_u, dtype=np.float64, order='C')
    return cA_u, cD_u
cA_u, cD_u = main_under()

def main_fully():
    listOfFiles_fully = getListOfFiles(dirName_fully)
    cA_f = []
    cD_f = []
    listOfFiles_fully = []
    for (dirpath, dirnames, filenames) in os.walk(dirName_fully):
        listOfFiles_fully += [os.path.join(dirpath, file) for file in filenames]
    for elem_fully in listOfFiles_fully:
        red_files_fully = FileRead(elem_fully)
        twod_fully =normalize( Nifti3Dto2D(red_files_fully))
        cA_2 , cD_2 =pywt.dwt(twod_fully, 'haar') 
        cA_u.append(Nifti2Dto1D(cA_2))
        cD_u.append(Nifti2Dto1D(cD_2))
    cA_f = np.asarray(cA_f, dtype=np.float64, order='C')
    cD_f = np.asarray(cD_f, dtype=np.float64, order='C')
    return cA_f, cD_f
cA_f, cD_f = main_under()

cA_a = cA_u - cA_f
cD_a = cD_u - cD_f

print("Shape of the cA components of artifacts",cA_a.shape)
cA_u_train, cA_u_test, cA_a_train, cA_a_test = train_test_split(cA_u, cA_a, test_size=0.2, random_state=0)
print('cA compomnents split completed')
#training knn_regressor_1 with cA components
regressor_1 = KNeighborsRegressor(n_neighbors=12)
regressor_1.fit(cA_u_train , cA_a_train)
print('nearest neighbour fit completed')

#training knn_regressor_2 with cD components
print("Shape of the cD components of artifacts",cD_a.shape)
cD_u_train, cD_u_test, cD_a_train, cD_a_test = train_test_split(cD_u, cD_a, test_size=0.2, random_state=0)
print('cD compomnents split completed')
regressor_2 = KNeighborsRegressor(n_neighbors=12)
regressor_2.fit(cD_u_train , cD_a_train)
print('nearest neighbour fit completed')
#predicting cA
cA_a_predicted = regressor_1.predict(cA_u_test)
print("Type of cA predicted data is : ",type(cA_a_predicted))

cD_a_predicted = regressor_2.predict(cD_u_test)
print("Type of cD predicted data is : ",type(cA_a_predicted))

#comaring ssim with predicted data
cA_a_pred_1 = (cA_a_predicted[1,:], 256)
cD_a_pred_1 = (cD_a_predicted[1,:], 256)
artifact_pred = pywt.idwt(cA_a, cD_a, 'haar')

cA_u = (cA_u_test[1,:], 256)
cD_u = (cD_u_test[1,:], 256)
under_samp = pywt.idwt(cA_u, cD_u , 'haar')

fully_pred = normalize(under_samp - artifact_pred)
dir_fully_test = r"path_for 65th_fully file"
fulsam_test = Nifti2Dto1D(Nifti3Dto2D(FileRead(dir_fully_test)))
print("ssim calculated = " ,measure.compare_ssim(fully_pred, fulsam_test))

#Calculating root mean square error
def rmse(data_predicted, artifacts_test):
    differences = (data_predicted) - (artifacts_test)
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    print("RMSE is : ", rmse_val)
    return rmse_val
rmse(cA_a_predicted, cA_a_test)
rmse(cD_a_predicted, cD_a_test)