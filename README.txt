README

########################################################################################################################################################
DATASETS

Dataset from Assignment 1 (Dataset 1)

Source: 
Name: Covertype Data Set
Link: https://archive.ics.uci.edu/ml/datasets/Covertype
Number of instances: 581012 (Only small subset of random instances are used to perform the experiments)

Dataset 2

Source: UCI Machine learning repository
Name: Sensorless Drive Diagnosis Data set
Link: https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis
Number of instances: 58509

########################################################################################################################################################

Instructions to run the code:

1. Install python
2. On terminal 'python filename.py'

Note: Code contains lines with reads the dataset so the dataset should be in the same location where the code is present.
      Few files will generate and save data and plotting files will take data and plot the graphs.

########################################################################################################################################################

Folder contains:

README.txt

#Analysis
sagarwal311-analysis.pdf (Analysis)

#datasets
covtype.data (Covertype data)
Sensorless_drive_diagnosis.csv (Sensor data)


#Code 

Exp1
####
cluster_covtype.py   (Experiment 1 clusters for Covtype dataset)
cluster_func.py  (General functions to create clusters)
cluster_sensor.py (Experiment 1 clusters for Sensor dataset)

Exp2
####
dim_reduce_cluster_pca_sensor.py (PCA sensor)
dim_reduce_cluster_fa_sensor.py (FA sensor)            
dim_reduce_cluster_rp_sensor.py (RP sensor)
dim_reduce_cluster_ica_sensor.py (ICA sensor)
cluster_func.py (General cluster functions required for this experiment as well)
dim_reduce_cluster_rp_covtype.py  (RP covtype)                             
dim_reduce_cluster_fa_covtype.py   (FA covtype)
dim_reduce_cluster_ica_covtype.py  (ICA covtype)
dim_reduce_cluster_pca_covtype.py (PCA covtype)

#For plotting Exp2
###################
plot_dimensional_cluster.py (Above exp 2 files generates data, this takes the data and plots)

Exp3
####
re_neural_network_covtype.py


