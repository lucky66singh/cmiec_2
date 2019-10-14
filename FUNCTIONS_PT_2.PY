#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def calc_ols_estimator(X,Y):
    ## x and y should be pandas dataframes
    beta_hat = np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ Y)
    Y_predict = X @ beta_hat
    residuals = Y - Y_predict
    return(beta_hat, residuals)


def calc_t_stat(sigma, X,beta_hat ,element_of_interest):
    #xx_1 = np.linalg.inv(X.transpose() @ X)
    var_beta =  np.linalg.inv(X.transpose() @ X)    @ X.transpose() @ sigma @ X @ np.linalg.inv(X.transpose() @ X)
    return float(beta_hat.iloc[element_of_interest] / np.sqrt(var_beta.iloc[element_of_interest][element_of_interest]))

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
