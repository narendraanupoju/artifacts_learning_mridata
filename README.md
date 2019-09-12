# artifacts_learning_mridata

for artifacts_learning-dwt.py-----
artifacts_learning-dwt.py program converts the 3d mri data in .nii format to numpy array and artifacts are calculated by substracting 
nosy images form original data , the Approximation and detailed coefficients are generated using discreete wavelet transformation using 
haar wavelet and these coefficients further passed to two knn regressors to train the algorithm and predicts the coefficients and inverse
dwt is applied to generate the predicted result
other data processing steps include normalization
Reuired libraries are nibabel, mumpy, scikit-learn, pywt
