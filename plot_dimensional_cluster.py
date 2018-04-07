import csv
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from os import path


##Reconstruction Error

# reconstruction_error_fa_covtype = pd.read_csv("./data/reconstruction_error_fa_covtype.csv", header= None)
# reconstruction_error_ica_covtype = pd.read_csv("./data/reconstruction_error_ica_covtype.csv", header = None)
# reconstruction_error_pca_covtype = pd.read_csv("./data/reconstruction_error_pca_covtype.csv", header = None)
# reconstruction_error_rp_covtype = pd.read_csv("./data/reconstruction_error_rp_covtype.csv", header = None)


# reconstruction_error_fa_sensor = pd.read_csv("./data/reconstruction_error_fa_sensor.csv", header= None)
# reconstruction_error_ica_sensor = pd.read_csv("./data/reconstruction_error_ica_sensor.csv", header = None)
# reconstruction_error_pca_sensor = pd.read_csv("./data/reconstruction_error_pca_sensor.csv", header = None)
# reconstruction_error_rp_sensor = pd.read_csv("./data/reconstruction_error_rp_sensor.csv", header = None)

# new_rec_error_rp_covtype = [x / (1e2) for x in list(reconstruction_error_rp_covtype.loc[:,1].values)]


# fig1, ax1 = plt.subplots()
# ax1.plot(reconstruction_error_pca_covtype.loc[:,0].values, reconstruction_error_pca_covtype.loc[:,1].values, linewidth = 2)
# ax1.plot(reconstruction_error_ica_covtype.loc[:,0].values, reconstruction_error_ica_covtype.loc[:,1].values, linewidth = 2)
# ax1.plot(reconstruction_error_rp_covtype.loc[:,0].values, new_rec_error_rp_covtype, linewidth = 2)
# ax1.plot(reconstruction_error_fa_covtype.loc[:,0].values, reconstruction_error_fa_covtype.loc[:,1].values, linewidth = 2)

# plt.legend(['PCA', 'ICA', 'RP error / 1e2', 'FA' ])
# plt.xlabel('Number of components')
# plt.ylabel('Reconstruction Error')
# plt.title('Reconstruction Error for Forest Cover Type dataset')

# new_rec_error_rp_sensor = [x / (10) for x in list(reconstruction_error_rp_sensor.loc[:,1].values)]
# fig2, ax2 = plt.subplots()
# ax2.plot(reconstruction_error_pca_sensor.loc[:,0].values, reconstruction_error_pca_sensor.loc[:,1].values, linewidth = 2)
# ax2.plot(reconstruction_error_ica_sensor.loc[:,0].values, reconstruction_error_ica_sensor.loc[:,1].values, linewidth = 2)
# ax2.plot(reconstruction_error_rp_sensor.loc[:,0].values, new_rec_error_rp_sensor, linewidth = 2)
# ax2.plot(reconstruction_error_fa_sensor.loc[:,0].values, reconstruction_error_fa_sensor.loc[:,1].values, linewidth = 2)

# plt.legend(['PCA', 'ICA', 'RP Error /10', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('Reconstruction Error')
# plt.title('Reconstruction Error for Sensorless drive diagnosis dataset')
# plt.show()

##Performance evaluation scores
#Forest Cover Type dataset

data_em_pca_covtype = pd.read_csv("./data/data_em_pca_covtype.csv", header = None)
data_em_ica_covtype = pd.read_csv("./data/data_em_ica_covtype.csv", header = None)
data_em_rp_covtype = pd.read_csv("./data/data_em_rp_covtype.csv", header = None)
data_em_fa_covtype = pd.read_csv("./data/data_em_fa_covtype.csv", header = None)

data_km_pca_covtype = pd.read_csv("./data/data_km_pca_covtype.csv", header = None)
data_km_ica_covtype = pd.read_csv("./data/data_km_ica_covtype.csv", header = None)
data_km_rp_covtype = pd.read_csv("./data/data_km_rp_covtype.csv", header = None)
data_km_fa_covtype = pd.read_csv("./data/data_km_fa_covtype.csv", header = None)

# fig3,ax3 = plt.subplots()
# ax3.plot(data_em_pca_covtype.loc[:,0].values, data_em_pca_covtype.loc[:,1].values, linewidth = 2)
# ax3.plot(data_em_ica_covtype.loc[:,0].values, data_em_ica_covtype.loc[:,1].values, linewidth = 2)
# ax3.plot(data_em_rp_covtype.loc[:,0].values, data_em_rp_covtype.loc[:,1].values, linewidth = 2)
# ax3.plot(data_em_fa_covtype.loc[:,0].values, data_em_fa_covtype.loc[:,1].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('AIC')
# plt.title('AIC curve for expected maximization for Forest Cover Type dataset')

# fig4,ax4 = plt.subplots()
# ax4.plot(data_em_pca_covtype.loc[:,0].values, data_em_pca_covtype.loc[:,2].values, linewidth = 2)
# ax4.plot(data_em_ica_covtype.loc[:,0].values, data_em_ica_covtype.loc[:,2].values, linewidth = 2)
# ax4.plot(data_em_rp_covtype.loc[:,0].values, data_em_rp_covtype.loc[:,2].values, linewidth = 2)
# ax4.plot(data_em_fa_covtype.loc[:,0].values, data_em_fa_covtype.loc[:,2].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('BIC')
# plt.title('BIC curve for expected maximization for Forest Cover Type dataset')


# fig5,ax5 = plt.subplots()
# ax5.plot(data_em_pca_covtype.loc[:,0].values, data_em_pca_covtype.loc[:,3].values, linewidth = 2)
# ax5.plot(data_em_ica_covtype.loc[:,0].values, data_em_ica_covtype.loc[:,3].values, linewidth = 2)
# ax5.plot(data_em_rp_covtype.loc[:,0].values, data_em_rp_covtype.loc[:,3].values, linewidth = 2)
# ax5.plot(data_em_fa_covtype.loc[:,0].values, data_em_fa_covtype.loc[:,3].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('Homogenity Score')
# plt.title('Homogenity Score for expected maximization for Forest Cover Type dataset')


# fig6,ax6 = plt.subplots()
# ax6.plot(data_em_pca_covtype.loc[:,0].values, data_em_pca_covtype.loc[:,4].values, linewidth = 2)
# ax6.plot(data_em_ica_covtype.loc[:,0].values, data_em_ica_covtype.loc[:,4].values, linewidth = 2)
# ax6.plot(data_em_rp_covtype.loc[:,0].values, data_em_rp_covtype.loc[:,4].values, linewidth = 2)
# ax6.plot(data_em_fa_covtype.loc[:,0].values, data_em_fa_covtype.loc[:,4].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('Completeness Score')
# plt.title('Completeness Score for expected maximization for Forest Cover Type dataset')


# fig7,ax7 = plt.subplots()
# ax7.plot(data_em_pca_covtype.loc[:,0].values, data_em_pca_covtype.loc[:,5].values, linewidth = 2)
# ax7.plot(data_em_ica_covtype.loc[:,0].values, data_em_ica_covtype.loc[:,5].values, linewidth = 2)
# ax7.plot(data_em_rp_covtype.loc[:,0].values, data_em_rp_covtype.loc[:,5].values, linewidth = 2)
# ax7.plot(data_em_fa_covtype.loc[:,0].values, data_em_fa_covtype.loc[:,5].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('Silhoutette Score')
# plt.title('Silhoutette Score for expected maximization for Forest Cover Type dataset')

# fig8,ax8 = plt.subplots()
# ax8.plot(data_em_pca_covtype.loc[:,0].values, data_em_pca_covtype.loc[:,6].values, linewidth = 2)
# ax8.plot(data_em_ica_covtype.loc[:,0].values, data_em_ica_covtype.loc[:,6].values, linewidth = 2)
# ax8.plot(data_em_rp_covtype.loc[:,0].values, data_em_rp_covtype.loc[:,6].values, linewidth = 2)
# ax8.plot(data_em_fa_covtype.loc[:,0].values, data_em_fa_covtype.loc[:,6].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('Average Log likelihood')
# plt.title('Per sample average log likelihood for EM for Forest Cover Type dataset')

# fig9,ax9 = plt.subplots()
# ax9.plot(data_km_pca_covtype.loc[:,0].values, data_km_pca_covtype.loc[:,1].values, linewidth = 2)
# ax9.plot(data_km_ica_covtype.loc[:,0].values, data_km_ica_covtype.loc[:,1].values, linewidth = 2)
# ax9.plot(data_km_rp_covtype.loc[:,0].values, data_km_rp_covtype.loc[:,1].values, linewidth = 2)
# ax9.plot(data_km_fa_covtype.loc[:,0].values, data_km_fa_covtype.loc[:,1].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('Homogenity Score')
# plt.title('Homogenity Score for kmeans for Forest Cover Type dataset')



# fig10,ax10 = plt.subplots()
# ax10.plot(data_km_pca_covtype.loc[:,0].values, data_km_pca_covtype.loc[:,2].values, linewidth = 2)
# ax10.plot(data_km_ica_covtype.loc[:,0].values, data_km_ica_covtype.loc[:,2].values, linewidth = 2)
# ax10.plot(data_km_rp_covtype.loc[:,0].values, data_km_rp_covtype.loc[:,2].values, linewidth = 2)
# ax10.plot(data_km_fa_covtype.loc[:,0].values, data_km_fa_covtype.loc[:,2].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('Silhoutette Score')
# plt.title('Silhoutette Score for kmeans for Forest Cover Type dataset')

# pca_var = [x/(1e9) for x in data_km_pca_covtype.loc[:,3].values]

# fa_var = [x/(1e4) for x in data_km_fa_covtype.loc[:,3].values]

# rp_var = [x/(1e9) for x in data_km_rp_covtype.loc[:,3].values]
# print(pca_var)
# print(fa_var)
# print(data_km_ica_covtype.loc[:,3].values)
# print(rp_var)
# fig11,ax11 = plt.subplots()
# ax11.plot(data_km_pca_covtype.loc[:,0].values, pca_var, linewidth = 2)
# ax11.plot(data_km_ica_covtype.loc[:,0].values, data_km_ica_covtype.loc[:,3].values, linewidth = 2)
# ax11.plot(data_km_rp_covtype.loc[:,0].values, rp_var, linewidth = 2)
# ax11.plot(data_km_fa_covtype.loc[:,0].values, fa_var, linewidth = 2)
# plt.legend(['PCA var / 1e9', 'ICA', 'RP var /1e9', 'FA noise var /1e4'])
# plt.xlabel('Number of components')
# plt.ylabel('Variance')
# plt.title('Variance explained by each cluster for kmeans for Forest Cover Type dataset')

# plt.show()

##Performance evaluation scores
#sensor dataset

data_em_pca_sensor = pd.read_csv("./data/data_em_pca_sensor.csv", header = None)
data_em_ica_sensor = pd.read_csv("./data/data_em_ica_sensor.csv", header = None)
data_em_rp_sensor = pd.read_csv("./data/data_em_rp_sensor.csv", header = None)
data_em_fa_sensor = pd.read_csv("./data/data_em_fa_sensor.csv", header = None)

data_km_pca_sensor = pd.read_csv("./data/data_km_pca_sensor.csv", header = None)
data_km_ica_sensor = pd.read_csv("./data/data_km_ica_sensor.csv", header = None)
data_km_rp_sensor = pd.read_csv("./data/data_km_rp_sensor.csv", header = None)
data_km_fa_sensor = pd.read_csv("./data/data_km_fa_sensor.csv", header = None)

# fig12,ax12 = plt.subplots()
# ax12.plot(data_em_pca_sensor.loc[:,0].values, data_em_pca_sensor.loc[:,1].values, linewidth = 2)
# ax12.plot(data_em_ica_sensor.loc[:,0].values, data_em_ica_sensor.loc[:,1].values, linewidth = 2)
# ax12.plot(data_em_rp_sensor.loc[:,0].values, data_em_rp_sensor.loc[:,1].values, linewidth = 2)
# ax12.plot(data_em_fa_sensor.loc[:,0].values, data_em_fa_sensor.loc[:,1].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('AIC')
# plt.title('AIC curve for expected maximization for Sensorless drive diagnosis dataset')

# fig13,ax13 = plt.subplots()
# ax13.plot(data_em_pca_sensor.loc[:,0].values, data_em_pca_sensor.loc[:,2].values, linewidth = 2)
# ax13.plot(data_em_ica_sensor.loc[:,0].values, data_em_ica_sensor.loc[:,2].values, linewidth = 2)
# ax13.plot(data_em_rp_sensor.loc[:,0].values, data_em_rp_sensor.loc[:,2].values, linewidth = 2)
# ax13.plot(data_em_fa_sensor.loc[:,0].values, data_em_fa_sensor.loc[:,2].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('BIC')
# plt.title('BIC curve for expected maximization for Sensorless drive diagnosis dataset')


fig14,ax14 = plt.subplots()
ax14.plot(data_em_pca_sensor.loc[:,0].values, data_em_pca_sensor.loc[:,3].values, linewidth = 2)
ax14.plot(data_em_ica_sensor.loc[:,0].values, data_em_ica_sensor.loc[:,3].values, linewidth = 2)
ax14.plot(data_em_rp_sensor.loc[:,0].values, data_em_rp_sensor.loc[:,3].values, linewidth = 2)
ax14.plot(data_em_fa_sensor.loc[:,0].values, data_em_fa_sensor.loc[:,3].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Homogenity Score')
plt.title('Homogenity Score for expected maximization for Sensorless drive diagnosis dataset')


fig15,ax15 = plt.subplots()
ax15.plot(data_em_pca_sensor.loc[:,0].values, data_em_pca_sensor.loc[:,4].values, linewidth = 2)
ax15.plot(data_em_ica_sensor.loc[:,0].values, data_em_ica_sensor.loc[:,4].values, linewidth = 2)
ax15.plot(data_em_rp_sensor.loc[:,0].values, data_em_rp_sensor.loc[:,4].values, linewidth = 2)
ax15.plot(data_em_fa_sensor.loc[:,0].values, data_em_fa_sensor.loc[:,4].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Completeness Score')
plt.title('Completeness Score for expected maximization for Sensorless drive diagnosis dataset')


fig16,ax16 = plt.subplots()
ax16.plot(data_em_pca_sensor.loc[:,0].values, data_em_pca_sensor.loc[:,5].values, linewidth = 2)
ax16.plot(data_em_ica_sensor.loc[:,0].values, data_em_ica_sensor.loc[:,5].values, linewidth = 2)
ax16.plot(data_em_rp_sensor.loc[:,0].values, data_em_rp_sensor.loc[:,5].values, linewidth = 2)
ax16.plot(data_em_fa_sensor.loc[:,0].values, data_em_fa_sensor.loc[:,5].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Silhoutette Score')
plt.title('Silhoutette Score for expected maximization for Sensorless drive diagnosis dataset')

fig17,ax17 = plt.subplots()
ax17.plot(data_em_pca_sensor.loc[:,0].values, data_em_pca_sensor.loc[:,6].values, linewidth = 2)
ax17.plot(data_em_ica_sensor.loc[:,0].values, data_em_ica_sensor.loc[:,6].values, linewidth = 2)
ax17.plot(data_em_rp_sensor.loc[:,0].values, data_em_rp_sensor.loc[:,6].values, linewidth = 2)
ax17.plot(data_em_fa_sensor.loc[:,0].values, data_em_fa_sensor.loc[:,6].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Average Log likelihood')
plt.title('Per sample avg log likelihood for EM for Sensorless drive data')


fig18,ax18 = plt.subplots()
ax18.plot(data_km_pca_sensor.loc[:,0].values, data_km_pca_sensor.loc[:,1].values, linewidth = 2)
ax18.plot(data_km_ica_sensor.loc[:,0].values, data_km_ica_sensor.loc[:,1].values, linewidth = 2)
ax18.plot(data_km_rp_sensor.loc[:,0].values, data_km_rp_sensor.loc[:,1].values, linewidth = 2)
ax18.plot(data_km_fa_sensor.loc[:,0].values, data_km_fa_sensor.loc[:,1].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Homogenity Score')
plt.title('Homogenity Score for kmeans for Sensorless drive dataset')



fig19,ax19 = plt.subplots()
ax19.plot(data_km_pca_sensor.loc[:,0].values, data_km_pca_sensor.loc[:,2].values, linewidth = 2)
ax19.plot(data_km_ica_sensor.loc[:,0].values, data_km_ica_sensor.loc[:,2].values, linewidth = 2)
ax19.plot(data_km_rp_sensor.loc[:,0].values, data_km_rp_sensor.loc[:,2].values, linewidth = 2)
ax19.plot(data_km_fa_sensor.loc[:,0].values, data_km_fa_sensor.loc[:,2].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Silhoutette Score')
plt.title('Silhoutette Score for kmeans for Sensorless drive dataset')

# fig20,ax20 = plt.subplots()
# ax20.plot(data_km_pca_sensor.loc[:,0].values, data_km_pca_sensor.loc[:,3].values, linewidth = 2)
# ax20.plot(data_km_ica_sensor.loc[:,0].values, data_km_ica_sensor.loc[:,3].values, linewidth = 2)
# ax20.plot(data_km_rp_sensor.loc[:,0].values, data_km_rp_sensor.loc[:,3].values, linewidth = 2)
# ax20.plot(data_km_fa_sensor.loc[:,0].values, data_km_fa_sensor.loc[:,3].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('Variance explained')
# plt.title('Variance explained by each cluster for kmeans for Sensorless drive dataset')


plt.show()