#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:15:21 2019

@author: lucky
"""
import pandas as pd
import numpy as np
from scipy.stats import bernoulli

def calc_ols_estimator(X,Y):
    ## x and y should be pandas dataframes
    beta_hat = np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ Y)
    Y_predict = X @ beta_hat
    residuals = Y - Y_predict
    
def calc_t_stat(sigma, X,beta_hat ,element_of_interest):
    #xx_1 = np.linalg.inv(X.transpose() @ X)
    var_beta =  np.linalg.inv(X.transpose() @ X)    @ X.transpose() @ sigma @ X @ np.linalg.inv(X.transpose() @ X)
    return beta_hat.iloc[element_of_interest] / np.sqrt(var_beta.iloc[element_of_interest][element_of_interest])
    
def wild_bootstrap_two_point(resid_under_null, y, x_restricted , X, beta_hat_restric):
    ber_samp = bernoulli.rvs(size=len(resid_under_null), p=0.5)
    ber_samp[ber_samp == 0] = -1
    resid_v = resid_under_null * ber_samp
    y_hat = x_restricted @ beta_hat_restric + resid_v
    (beta_hat_b, residuals_b) = calc_ols_estimator(X, y_hat)
    t_stat_b =   calc_t_stat( pd.DataFrame(np.diag(residuals_b ** 2) )
                                  ,X
                                  ,beta_hat_b
                                  ,1
                             )
    return(t_stat_b)
    
    
def wild_bootstrap_gaus(resid_under_null, y, x_restricted , X, beta_hat_restric):
    nor_samp = np.random.normal(size=len(residuals))
    resid_v = resid_under_null * nor_samp
    y_hat = x_restricted @ beta_hat_restric + resid_v
    (beta_hat_b, residuals_b) = calc_ols_estimator(X, y_hat)
    t_stat_b =   calc_t_stat( pd.DataFrame(np.diag(residuals_b ** 2) )
                                  ,X
                                  ,beta_hat_b
                                  ,1
                             )
    return(t_stat_b)

    

    
def residual_bootstrap(resid_under_null, y, x_restricted , X, beta_hat_restric ):
    ## used np.random.choice for bootstrap sample
    sample_resid = np.random.choice(resid_under_null, size=len(resid_under_null), replace=True)
    ##make diff method to calc y_hat
    y_hat = x_restricted @ beta_hat_restric + sample_resid
    
    (beta_hat_b, residuals_b) = calc_ols_estimator(X, y_hat)
    ## replace calc_t_stat with calc-stat which takes assignment number as input
    t_stat_b =   calc_t_stat( pd.DataFrame(np.diag(residuals_b ** 2) )
                                  ,X
                                  ,beta_hat_b
                                  ,1
                             )
    return(t_stat_b)
    
    
     
    print('hi')

def pairs_bootstrap():
    print('hi')
    ## found some code that says to re arrange the indices of data and resample from that
    ## https://github.com/wblakecannon/DataCamp/blob/master/07-statistical-thinking-in-python-(part-2)/2-bootstrap-confidence%3Dintervals/a-function-to-do-pairs-bootstrap.py
    
def approx_resid(y_a_r, x_a_r):
    beta_hat_ar = np.linalg.inv(x_a_r.transpose() @ x_a_r) @ (x_a_r.transpose() @ y_a_r)
    app_resid = y_a_r - x_a_r @ beta_hat_ar
    return(app_resid, beta_hat_ar)
    
    
    

X=pd.read_csv('/home/lucky/cmiec_2/data/Regressors.txt', header=None)
Y=pd.read_csv('/home/lucky/cmiec_2/data/Observables.txt', header=None)


#for i in range(0,len(Y.columns)):
#    calc_ols_estimator(X, Y.iloc[:,i])
B=99
sign_level=0.05
left_t_stat = 1 + sign_level/2 *(B+1 )
right_t_stat = (1 -sign_level/2) * (B+1)

#calc ols residuals and beta_hat
beta_hat = np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ Y)
Y_predict = X @ beta_hat
residuals = Y - Y_predict
    

## test if outcome for singele regression gave thge same values
count=[]
for i in range(1, len(Y.columns)):
    beta_hat_z = np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ Y.iloc[:,i])
    Y_predict_beta_hat_1 = X @ beta_hat_z
    residuals_1 = Y.iloc[:,i] - Y_predict_beta_hat_1
    count.append( sum(round(residuals.iloc[:,i],4) ==round(residuals_1,4)) )

t_stat_b=[]
for i in range(1, len(residuals.columns)):
    t_stat_b.append( calc_t_stat( pd.DataFrame(np.diag(residuals.iloc[:,i ] ** 2) )
                                  ,X
                                  ,beta_hat.iloc[:,i]
                                  ,1
                                )
                    )

t_stat_lower = sum(t_stat_b <= np.array(-1.96))
t_stat_higher = sum(t_stat_b >= np.array(1.96))
total_per_reject = (t_stat_higher + t_stat_higher) / len(residuals.columns)
print('Percentage rejected for 1B: ' + str(total_per_reject * 100) + '%')

## for estimating under null beta_1=0 

for i in range(0, len(Y.columns)):
    (resid_under_null, beta_hat_restric) = approx_resid(Y.iloc[:,i] ,X.iloc[:,[0,2]] )
    t_stat_resid_boot=[]
    t_stat_wild_two_boot=[]
    t_stat_wild_gaus_boot=[]
    for j in range(0,B):
        t_stat_resid_boot.append( residual_bootstrap(resid_under_null,
                                                     y,
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
        t_stat_wild_two_boot.append( wild_bootstrap_two_point(resid_under_null,
                                                     y,
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
        t_stat_wild_gaus_boot.append( wild_bootstrap_gaus(resid_under_null,
                                                     y,
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
    


t_stat=[]
for i in range(0,len(Y.columns)):
    1==1
d1 = pd.DataFrame([[1, 2]] * 3, columns=list('AB'))
d2 = pd.DataFrame(np.arange(1, 7).reshape(2, 3).T, columns=list('CD'))
kp = pd.DataFrame(np.kron(d1, d2), columns=pd.MultiIndex.from_product([d1, d2]))

    
z= Y.iloc[:,0]
beta_hat_z = np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ z)

beta_hat_1 = np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ Y.iloc[:,1])
Y_predict_beta_hat_1 = X @ beta_hat_1
residuals_1 = Y.iloc[:,1] - Y_predict_beta_hat_1
