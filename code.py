#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:15:21 2019

@author: lucky
"""
import pandas as pd
import numpy as np
from scipy.stats import bernoulli
from statsmodels.tsa.ar_model import AR


def calc_ols_estimator(X,Y):
    ## x and y should be pandas dataframes
    beta_hat = np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ Y)
    Y_predict = X @ beta_hat
    residuals = Y - Y_predict
    return(beta_hat, residuals)

    
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
    
def residual_bootstrap_2(resid_under_null, y, x_restricted , X, beta_hat_restric ):
    ## used np.random.choice for bootstrap sample
    sample_resid = np.random.choice(resid_under_null, size=len(resid_under_null), replace=True) - np.mean(resid_under_null)
    ##make diff method to calc y_hat
    y_hat =np.zeros(len(sample_resid))
    y_hat[0] = beta_hat_restric * x_restricted[0] + sample_resid[0]
    for i in range(1, len(sample_resid)):
        y_hat[i] = beta_hat_restric * y_hat[i - 1] + sample_resid[i]
        
    (beta_hat_b, residuals_b) = calc_ols_estimator(X, y_hat)
    ## replace calc_t_stat with calc-stat which takes assignment number as input
    t_stat_b =   calc_t_stat( pd.DataFrame(np.diag(residuals_b ** 2) )
                                  ,X
                                  ,pd.DataFrame(beta_hat_b)
                                  ,0
                             )
    return(t_stat_b)
     
    
def wild_bootstrap_two_point_2(resid_under_null, y, x_restricted , X, beta_hat_restric):
    ber_samp = bernoulli.rvs(size=len(resid_under_null), p=0.5)
    ber_samp[ber_samp == 0] = -1
    resid_v = resid_under_null * ber_samp
    y_hat =np.zeros(len(resid_v))
    y_hat[0] = beta_hat_restric * x_restricted[0] + resid_v[0]
    for i in range(1, len(resid_v)):
        y_hat[i] = beta_hat_restric * y_hat[i - 1] + resid_v[i]
        
    (beta_hat_b, residuals_b) = calc_ols_estimator(X, y_hat)
    t_stat_b =   calc_t_stat( pd.DataFrame(np.diag(residuals_b ** 2) )
                                  ,X
                                  ,pd.DataFrame(beta_hat_b)
                                  ,0
                             )
    return(t_stat_b)

def wild_bootstrap_gaus_2(resid_under_null, y, x_restricted , X, beta_hat_restric):
    nor_samp = np.random.normal(size=len(resid_under_null))
    resid_v = resid_under_null * nor_samp
    y_hat =np.zeros(len(resid_v))
    y_hat[0] = beta_hat_restric * x_restricted[0] + resid_v[0]
    for i in range(1, len(resid_v)):
        y_hat[i] = beta_hat_restric * y_hat[i - 1] + resid_v[i]
    (beta_hat_b, residuals_b) = calc_ols_estimator(X, y_hat)
    t_stat_b =   calc_t_stat( pd.DataFrame(np.diag(residuals_b ** 2) )
                                  ,X
                                  ,pd.DataFrame(beta_hat_b)
                                  ,0
                             )
    return(t_stat_b)
    
def block_bootstrap(resid_under_null, y, x_restricted, X, beta_hat_restric):
    B = pd.DataFrame(np.zeros((20, len(resid_under_null) - 20+ 1)))
    for i in range(0, len(B.columns)):
        B.iloc[:,i] = resid_under_null[i:i+20].reset_index(drop=True)
    index_shuf = np.random.choice(np.arange(0,len(B), 1), size=3 , replace=True)
    resid_bs = pd.concat([B.iloc[:,index_shuf[0]], B.iloc[:,index_shuf[1]], B.iloc[:,index_shuf[2]]], axis=0, ignore_index=True)
    y_hat =np.zeros(len(resid_under_null))
    y_hat[0] = beta_hat_restric * x_restricted[0] + resid_bs[0]
    for i in range(1, len(resid_under_null)):
        y_hat[i] = beta_hat_restric * y_hat[i - 1] + resid_bs[i]    
    (beta_hat_b, residuals_b) = calc_ols_estimator(X, y_hat)
    t_stat_b =   calc_t_stat( pd.DataFrame(np.diag(residuals_b ** 2) )
                                  ,X
                                  ,pd.DataFrame(beta_hat_b)
                                  ,0
                             )
    return(t_stat_b)

def sieve_bootstrap(resid_under_null, y, x_restricted, X, beta_hat_restric):
    ar_mod = AR(resid_under_null)
    ## fit can also be done with other lag
    fit = ar_mod.fit(5, trend='nc')
    heta_t = fit.resid
    ## check if need to recenter and if its needed for all simulations
    heta_t_star = pd.concat( [pd.Series(resid_under_null[0:len(fit.params) ] )  ,
                         pd.Series(np.random.choice(heta_t, size=len(heta_t), replace=True) ) ] ,
                        axis=0  , ignore_index=True)
    ## calculate e_t_star frm heta_t_star
    e_t_star = np.zeros(len(heta_t_star))
    e_t_star[0:len(fit.params)] = heta_t_star[0:len(fit.params)]
    for i in range(len(fit.params) , len(heta_t_star)):
        e_t_star[i] = np.dot(fit.params , e_t_star[i-len(fit.params) :i]) + heta_t_star[i]
    ## calcualt y_hat from e_t_star 
    y_hat =np.zeros(len(resid_under_null))
    y_hat[0] = beta_hat_restric * x_restricted[0] + e_t_star[0]
    for i in range(1, len(resid_under_null)):
        y_hat[i] = beta_hat_restric * y_hat[i - 1] + e_t_star[i]        
    (beta_hat_b, residuals_b) = calc_ols_estimator(X, y_hat)
    t_stat_b =   calc_t_stat( pd.DataFrame(np.diag(residuals_b ** 2) )
                                  ,X
                                  ,pd.DataFrame(beta_hat_b)
                                  ,0
                             )
    return(t_stat_b)

       
    
    
    
def pairs_bootstrap(y,X):
    print('hi')
    ## found some code that says to re arrange the indices of data and resample from that
    ## https://github.com/wblakecannon/DataCamp/blob/master/07-statistical-thinking-in-python-(part-2)/2-bootstrap-confidence%3Dintervals/a-function-to-do-pairs-bootstrap.py
    index_shuf = np.random.choice(np.arange(0,len(y), 1), size=len(y) , replace=True)
    y=y[index_shuf].reset_index(drop=True)
    x=X.iloc[index_shuf].reset_index(drop=True)
    
    (beta_hat_b, residuals_b) = calc_ols_estimator(x, y)    
    t_stat_b =   calc_t_stat( pd.DataFrame(np.diag(residuals_b ** 2) )
                                  ,X
                                  ,beta_hat_b
                                  ,1
                             )
    return(t_stat_b)  
    

def approx_resid(y_a_r, x_a_r):
    #    (resid_under_null, beta_hat_restric) = approx_resid(Y_b,X_b.iloc[:,1] )
    beta_hat_ar = np.linalg.inv(x_a_r.transpose() @ x_a_r) @ (x_a_r.transpose() @ y_a_r)
    app_resid = y_a_r - x_a_r @ beta_hat_ar
    return(app_resid, beta_hat_ar)

def transform_ts_data(y):   
    diff = y.diff()
    delta_y_t = diff[2:].reset_index(drop=True)
    y_t_1= y[2:].reset_index(drop=True)
    delta_y_t_1 = diff[1:len(diff)-1].reset_index(drop=True)
    x=pd.concat([y_t_1, delta_y_t_1], axis=1).reset_index(drop=True) #s1 and s2 are series
    return(x, delta_y_t)

def ols_estimator_wo_intercept(y_a_r,x_a_r):
    nom = np.dot(y_a_r, x_a_r)
    denom = np.dot(x_a_r, x_a_r)
    phi = nom/denom
    resid = y_a_r - phi * x_a_r
    return(resid, phi)
    
    
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
total_per_reject = (t_stat_higher + t_stat_lower) / len(residuals.columns)
print('Percentage rejected for 1B: ' + str(total_per_reject * 100) + '%')

## for estimating under null beta_1=0 
reject_resid_boot=[]
reject_t_stat_wild_two=[]
reject_t_stat_wil_gaus=[]
reject_t_stat_two_pairs=[]


for i in range(0, len(Y.columns)):
    (resid_under_null, beta_hat_restric) = approx_resid(Y.iloc[:,i] ,X.iloc[:,[0,2]] )
    t_stat_resid_boot=[]
    t_stat_wild_two_boot=[]
    t_stat_wild_gaus_boot=[]
    t_stat_two_pairs=[]
    for j in range(0,B):
        t_stat_resid_boot.append( residual_bootstrap(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
        t_stat_wild_two_boot.append( wild_bootstrap_two_point(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
        t_stat_wild_gaus_boot.append( wild_bootstrap_gaus(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
        t_stat_two_pairs.append( pairs_bootstrap( Y.iloc[:,i],
                                                     X
                                                 )
                               )
        
        
    t_stat_resid_boot.append(float(t_stat_b[i]))
    t_stat_resid_boot.sort()
    t_stat_wild_two_boot.append(float(t_stat_b[i]))
    t_stat_wild_two_boot.sort()
    t_stat_wild_gaus_boot.append(float(t_stat_b[i]))
    t_stat_wild_gaus_boot.sort()    
    t_stat_two_pairs.append(float(t_stat_b[i]))
    t_stat_two_pairs.sort()
    reject_resid_boot.append(abs(t_stat_b[i]) <= t_stat_resid_boot[95]  )
    reject_t_stat_wild_two.append( abs(t_stat_b[i]) <= t_stat_wild_two_boot[95] )  
    reject_t_stat_wil_gaus.append(abs(t_stat_b[i]) <= t_stat_wild_gaus_boot[95]  ) 
    reject_t_stat_two_pairs.append(abs(t_stat_b[i]) <= t_stat_two_pairs[95] )


Y_het = pd.read_csv('/home/lucky/cmiec_2/data/Timeseries_het.txt')

t_stat_het=[]
for i in range(1, len(Y_het.columns)):
    (X,Y) = transform_ts_data(Y_het.iloc[:,i])
    (beta_hat_het, residuals_het) = calc_ols_estimator(X,Y)
    t_stat_het.append( calc_t_stat( pd.DataFrame(np.diag(residuals_het ** 2) )
                                  ,X
                                  ,beta_hat
                                  ,0
                                )
                    )

t_stat_lower_het = sum(t_stat_het <= np.array(-1.95))
#t_stat_higher = sum(t_stat_b >= np.array(1.96))
total_per_reject_het = (t_stat_lower_het) / len(Y_het.columns)
print('Percentage rejected for 1B: ' + str(total_per_reject_het * 100) + '%')

reject_resid_het=[]
reject_wild_two_het=[]
reject_wild_gaus_het=[]
reject_block_het=[]
reject_siev_het=[]
for i in range(0, len(Y_het.columns)):
    (X_b,Y_b) = transform_ts_data(Y_het.iloc[:,i])
    (resid_under_null, beta_hat_restric) = ols_estimator_wo_intercept(Y_b,X_b.iloc[:,1] )
    t_stat_resid_boot_het=[]
    t_stat_wild_two_boot_het=[]
    t_stat_wild_gaus_boot_het=[]
    t_stat_block_het=[]
    t_stat_sieve_het=[]    
    for j in range(0,B):
        t_stat_resid_boot_het.append( residual_bootstrap(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
        t_stat_wild_two_boot_het.append( wild_bootstrap_two_point(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
        t_stat_wild_gaus_boot_het.append( wild_bootstrap_gaus(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )        
        t_stat_block_het.append(block_bootstrap(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )   
        t_stat_sieve_het.append(sieve_bootstrap(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
    t_stat_resid_boot_het.append(float(t_stat_het[i]))
    t_stat_resid_boot_het.sort()
    t_stat_wild_two_boot_het.append(float(t_stat_het[i]))
    t_stat_wild_two_boot_het.sort()
    t_stat_wild_gaus_boot_het.append(float(t_stat_het[i]))
    t_stat_wild_gaus_boot_het.sort()  
    t_stat_block_het.append(float(t_stat_het[i]))
    t_stat_block_het.sort()
    t_stat_sieve_het.append(float(t_stat_het[i]))
    t_stat_sieve_het.sort()

        ##check if condition is correct
    reject_resid_het.append(abs(t_stat_het[i]) <= t_stat_resid_boot_het[5]  )
    reject_wild_two_het.append( abs(t_stat_het[i]) <= t_stat_wild_two_boot_het[5] )  
    reject_wild_gaus_het.append(abs(t_stat_het[i]) <= t_stat_wild_gaus_boot_het[5]  ) 
    reject_block_het.append(abs(t_stat_het[i]) <= t_stat_block_het[5]  )
    reject_siev_het.append( abs(t_stat_het[i]) <= t_stat_sieve_het[5] )  


Y_dep = pd.read_csv('/home/lucky/cmiec_2/data/Timeseries_dep.txt')

t_stat_dep=[]
for i in range(1, len(Y_dep.columns)):
    (X,Y) = transform_ts_data(Y_dep.iloc[:,i])
    (beta_hat_dep, residuals_dep) = calc_ols_estimator(X,Y)
    t_stat_dep.append( calc_t_stat( pd.DataFrame(np.diag(residuals_dep ** 2) )
                                  ,X
                                  ,beta_hat
                                  ,0
                                )
                    )

t_stat_lower_dep = sum(t_stat_dep <= np.array(-1.95))
#t_stat_higher = sum(t_stat_b >= np.array(1.96))
total_per_reject_dep = (t_stat_lower_dep) / len(Y_dep.columns)
print('Percentage rejected for 1B: ' + str(total_per_reject_dep * 100) + '%')

reject_resid_dep=[]
reject_wild_two_dep=[]
reject_wild_gaus_dep=[]
reject_block_dep=[]
reject_siev_dep=[]
for i in range(0, len(Y_dep.columns)):
    (X_b,Y_b) = transform_ts_data(Y_dep.iloc[:,i])
    (resid_under_null, beta_hat_restric) = ols_estimator_wo_intercept(Y_b,X_b.iloc[:,1] )
    t_stat_resid_boot_dep=[]
    t_stat_wild_two_boot_dep=[]
    t_stat_wild_gaus_boot_dep=[]
    t_stat_block_dep=[]
    t_stat_sieve_dep=[]    
    for j in range(0,B):
        t_stat_resid_boot_dep.append( residual_bootstrap(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
        t_stat_wild_two_boot_dep.append( wild_bootstrap_two_point(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
        t_stat_wild_gaus_boot_dep.append( wild_bootstrap_gaus(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )        
        t_stat_block_dep.append(block_bootstrap(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )   
        t_stat_sieve_dep.append(sieve_bootstrap(resid_under_null,
                                                     Y.iloc[:,i],
                                                     X.iloc[:,[0,2]],
                                                     X,
                                                     beta_hat_restric
                                                     )
                               )
    t_stat_resid_boot_dep.append(float(t_stat_dep[i]))
    t_stat_resid_boot_dep.sort()
    t_stat_wild_two_boot_dep.append(float(t_stat_dep[i]))
    t_stat_wild_two_boot_dep.sort()
    t_stat_wild_gaus_boot_dep.append(float(t_stat_dep[i]))
    t_stat_wild_gaus_boot_dep.sort()  
    t_stat_block_dep.append(float(t_stat_dep[i]))
    t_stat_block_dep.sort()
    t_stat_sieve_dep.append(float(t_stat_dep[i]))
    t_stat_sieve_dep.sort()

    ##check if condition is correct
    reject_resid_dep.append(abs(t_stat_dep[i]) <= t_stat_resid_boot_dep[5]  )
    reject_wild_two_dep.append( abs(t_stat_dep[i]) <= t_stat_wild_two_boot_dep[5] )  
    reject_wild_gaus_dep.append(abs(t_stat_dep[i]) <= t_stat_wild_gaus_boot_dep[5]  ) 
    reject_block_dep.append(abs(t_stat_dep[i]) <= t_stat_block_dep[5]  )
    reject_siev_dep.append( abs(t_stat_dep[i]) <= t_stat_sieve_dep[5] )  


