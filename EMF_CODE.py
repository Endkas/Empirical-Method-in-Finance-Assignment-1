
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:26:20 2021

@author: Endrit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
import statsmodels.api as sm
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import norm
import random



#Downloading the data



filename = "Data_assignment.xlsx"

xls = pd.ExcelFile(filename)

S_P_500 = pd.read_excel(xls, 'Data_export')

S_P_500_2 = S_P_500.iloc[:, 0:452]

Index = S_P_500.iloc[:, 452]

Index_formated = np.array(Index)

Prices = np.array(S_P_500_2)

num_lines = np.size(Prices, 0)

Returns = np.divide(Prices[1:(num_lines), :], Prices[0:(num_lines - 1)]) - 1

Index_Returns = np.divide(Index_formated[1:(num_lines)], Index_formated[0:(num_lines - 1)]) - 1

Expectation_Index = np.mean(Index_Returns)

labels = list(S_P_500_2)


plt.figure(dpi=1000)
plt.plot(Prices)
plt.xlabel('Time')
plt.ylabel('Price (in $)')
plt.title('S&P 500 Stock Prices')

plt.figure(dpi=600)
plt.plot(Index_formated, 'r')
plt.xlabel('Time')
plt.ylabel('Price (in $)')
plt.title('S&P 500 Index')

plt.figure(dpi=600)
plt.plot(Returns)
plt.xlabel('Time')
plt.ylabel('Returns (in %)')
plt.title('S&P 500 Stock Price Returns')

plt.figure(dpi=600)
plt.plot(Index_Returns, 'r')
plt.xlabel('Time')
plt.ylabel('Returns (in %)')
plt.title('S&P 500 Index Return')



#Computation of Rolling Volatilities



MU = np.mean(Returns, 0)
MU = np.array(MU)


def Rolling_Vol(y):
    Rol_Var_y = (1/20) * ((Returns[y - 0, :] - MU)**2 + (Returns[y - 1, :] - MU)**2 
                          + (Returns[y - 2, :] - MU)**2 + (Returns[y - 3, :] - MU)**2 
                          + (Returns[y - 4, :] - MU)**2 + (Returns[y - 5, :] - MU)**2 
                          + (Returns[y - 6, :] - MU)**2 + (Returns[y - 7, :] - MU)**2 
                          + (Returns[y - 8, :] - MU)**2 + (Returns[y - 9, :] - MU)**2 
                          + (Returns[y - 10, :] - MU)**2 + (Returns[y - 11, :] - MU)**2 
                          + (Returns[y - 12, :] - MU)**2 + (Returns[y - 13, :] - MU)**2 
                          + (Returns[y - 14, :] - MU)**2 + (Returns[y - 15, :] - MU)**2 
                          + (Returns[y - 16, :] - MU)**2 + (Returns[y - 17, :] - MU)**2 
                          + (Returns[y - 18, :] - MU)**2 + (Returns[y - 19, :] - MU)**2 
                          + (Returns[y - 20, :] - MU)**2)
    return Rol_Var_y


ZZ_Top = Rolling_Vol(20)

m = []

for y in range(20, num_lines - 1):
    LED = Rolling_Vol(y)
    print(LED)
    m.append(LED)

M = np.array(m)

M_std_dev = np.power(M, 0.5)



#Computation of Risk Adjusted Returns



Risk_adjusted_Returns = np.divide(Returns[20:, :], M_std_dev)
plt.plot(Risk_adjusted_Returns)
plt.title('S&P 500 Stock Risk Adjusted Returns')

MU_Risk = np.mean(Risk_adjusted_Returns, 0)



#Computation of Volatility



plt.figure(dpi=600)
plt.bar(labels, MU)
plt.xticks('')

Squared_Returns = np.power(Returns, 2)
Variance_Returns = np.mean(Squared_Returns, 0) - np.power(np.mean(Returns, 0), 2)

Volatility = np.power(Variance_Returns, 0.5)
Annualized_Volatility = Volatility * np.power(52, 0.5)

plt.figure(dpi=600)
plt.bar(labels, Annualized_Volatility)
plt.xticks(size = 0)
plt.xticks('')


Squared_Risk_Returns = np.power(Risk_adjusted_Returns, 2)
Variance_Risk_Returns = np.mean(Squared_Risk_Returns, 0) - np.power(np.mean(Risk_adjusted_Returns, 0), 2)
Volatility_Risk_Returns = np.power(Variance_Risk_Returns, 0.5)



#Computation of Skewness



Skewness = skew(Returns, 0) 

plt.figure(dpi = 600)
plt.bar(labels, Skewness, color = 'red')
plt.xlabel('Stocks')
plt.xticks(rotation = 90, size = 0)
plt.xticks('')
plt.title('Simple Returns Skewness')

Skewness_Lower = Skewness[Skewness < 0]
len(Skewness_Lower)


Skewness_Risk = skew(Risk_adjusted_Returns, 0)

Skewness_Risk_Lower = Skewness_Risk[Skewness_Risk < 0]
len(Skewness_Risk_Lower)

plt.figure(dpi = 600)
plt.bar(labels, Skewness_Risk, color = 'red')
plt.xlabel('Stocks')
plt.xticks(rotation = 90, size = 0)
plt.xticks('')
plt.ylim([-0.5, 1])
plt.title('Risk Adjusted Returns Skewness')



#Computation of the Kurtosis



Kurtosis = kurtosis(Returns, 0, fisher = False)

plt.figure(dpi = 600)
plt.bar(labels, Kurtosis, color = 'red')
plt.xlabel('Stocks')
plt.xticks(rotation = 90, size = 0)
plt.xticks('')
plt.title('Simple Returns Kurtosis')

Kurtosis_Higher = Kurtosis[Kurtosis > 3]
len(Kurtosis_Higher)


Kurtosis_Risk = kurtosis(Risk_adjusted_Returns, 0, fisher=False)

Kurtosis_Risk_Higher = Kurtosis_Risk[Kurtosis_Risk > 3]
len(Kurtosis_Risk_Higher)

plt.figure(dpi = 600)
plt.bar(labels, Kurtosis_Risk, color = 'red')
plt.xlabel('Stocks')
plt.xticks(rotation = 90, size = 0)
plt.xticks('')
plt.ylim([0, 20])
plt.title('Risk Adjusted Returns Kurtosis')



#Computation of the Jarque Bera Test



test_JB = np.size(Returns, 0) * (np.power(Skewness, 2)/6 + np.power(Kurtosis - 3, 2)/24)

plt.figure(dpi = 600)
plt.plot(labels, test_JB, 'bo', labels, test_JB * 0 + 5.991, 'r--', markersize = 2)
plt.xlabel('Stocks')
plt.xticks(rotation = 90, size  = 0)
plt.legend(["JB Test", "5% Threshold"])
plt.xticks('')
plt.ylim([-10000, 250000])
plt.title('Jarque Bera Test for Simple Returns')

Rejected = test_JB[test_JB > 5.991]
len(Rejected)



test_JB_Risk = np.size(Risk_adjusted_Returns, 0) * (np.power(Skewness_Risk, 2)/6 + np.power(Kurtosis_Risk - 3, 2)/24)

plt.figure(dpi = 600)
plt.plot(labels, test_JB_Risk, 'bo', labels, test_JB_Risk * 0 + 5.991, 'r--', markersize = 2)
plt.xlabel('Stocks')
plt.legend(["JB Test", "5% Threshold"])
plt.xticks('')
plt.ylim([-60, 600])
plt.title('Jarque Bera Test for Risk Adjusted Returns')

Rejected_Risk = test_JB_Risk[test_JB_Risk > 5.991]
len(Rejected_Risk)



#Compute the Kolmogorov and Smirnov Test



from scipy import stats
stats.kstest(Returns[:, 0], 'norm')


test = [] ;

for i in range(0, 452) :

    temp = stats.kstest(Returns[:, i], 'norm');
    temp = temp.statistic
    test += [temp] ;

plt.figure(dpi=600)
plt.plot(labels, test, 'bo', color = 'red', markersize = 1)
plt.xlabel('Stocks')
plt.xticks('')
plt.title('Kolmogorov & Smirnov test for Simple Returns')

test_array = np.array(test)

Rejected_KS = test_array[test_array > (1.36 / np.sqrt(1990))]
len(Rejected_KS)

test2 = [] ;

for i in range(0, 452) :

    temp2 = stats.kstest(Risk_adjusted_Returns[:, i], 'norm');
    temp2 = temp2.statistic
    test2 += [temp2] ;

plt.figure(dpi=600)
plt.plot(labels, test2, 'bo', color = 'red', markersize = 1)
plt.xlabel('Stocks')
plt.xticks('')
plt.title('Kolmogorov & Smirnov test for Risk Adjusted Returns')

test_2_array = np.array(test2)

Rejected_KS_Risk = test_2_array[test_2_array > (1.36 / np.sqrt(1970))]
len(Rejected_KS_Risk)




#Graphs of the Distance between the Empirical CDF and the Theoretical CDF



ecdfs = np.arange(1990, dtype=float)/1990
GS = []
x = random.randint(0,452)
KS = Returns[:,x]
maxi = 0
index = 0
SOR = np.sort(KS)
for t in range(len(SOR)) :
    KS[t] = norm.cdf(SOR[t], MU[x],(Variance_Returns[x]**0.5))
    GS.append(abs(norm.cdf(SOR[t], MU[x],(Variance_Returns[x]**0.5)) - (t/len(SOR))))
    if GS[t] > maxi :
        index = t
        maxi = GS[t]
    else :
        continue
plt.figure(dpi=600)
plt.plot(ecdfs,KS,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs,ecdfs,color='k',lw=0.8,linestyle='dashed',label="Standard Normal CDF")
plt.scatter(index/1990,ecdfs[index],s=7,color='r')
plt.scatter(index/1990,KS[index],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index/1990], ecdfs[index], KS[index], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF for Simple Returns')


ecdfs = np.arange(1970, dtype=float)/1970
GS = []
x = random.randint(0,452)
KS = Risk_adjusted_Returns[:,x]
maxi = 0
index = 0
SOR = np.sort(KS)
for t in range(len(SOR)) :
    KS[t] = norm.cdf(SOR[t], MU_Risk[x],(Variance_Risk_Returns[x]**0.5))
    GS.append(abs(norm.cdf(SOR[t], MU_Risk[x],(Variance_Risk_Returns[x]**0.5)) - (t/len(SOR))))
    if GS[t] > maxi :
        index = t
        maxi = GS[t]
    else :
        continue
plt.figure(dpi=600)
plt.plot(ecdfs,KS,color='r',lw=1,label="Empirical CDF")
plt.plot(ecdfs,ecdfs,color='k',lw=0.8,linestyle='dashed',label="Standard Normal CDF")
plt.scatter(index/1970,ecdfs[index],s=7,color='r')
plt.scatter(index/1970,KS[index],s=7,color='r')
plt.xlabel("t / T")
plt.ylabel("CDF")
plt.ylim([0, 1]); plt.grid(True)
plt.vlines([index/1970], ecdfs[index], KS[index], color='b', lw=2.5,label="KS test stat")
plt.legend()
plt.title('Empirical VS Theoretical CDF for Risk Adjusted Returns')



#Compute the Ljung Box Test



def LjungBoxQ(x,lag):
    T=len(x);
    Formatedx=pd.Series(x);
    output=[];
    critical_values=[];
    for i_lag in range(1,lag+1):
        temp=0;
        for i in range(1,i_lag+1):
            temp+=1/(T-i+1)*np.power(Formatedx.autocorr(i),2)
        output+=[temp*T*(T+2)];
        critical_values+=[sci.chi2.isf(.05,i_lag)];
    return pd.DataFrame([output, critical_values])

result_Jung = [];
for i in range(0,452):
    temp = np.transpose(LjungBoxQ(Returns[:,i],10));
    if i == 451:
        result_Jung += [temp.iloc[:,0]];
        result_Jung += [temp.iloc[:,1]];
    else:
        result_Jung+=[temp.iloc[:,0]];
titles=list(S_P_500_2);
titles.append('Critical Value');
result_Jung=pd.DataFrame(np.transpose(result_Jung), list(range(0,10)),titles)


proportion_Ljung=[]
for i in range (10):
    Ljung_correlated=[]            
    for j in range (452):
        if result_Jung.iloc[i,j]>result_Jung.iloc[i,452]:
            Ljung_correlated.append(result_Jung.iloc[i,j])
    proportion_Ljung.append(np.size(Ljung_correlated)/452)




result_Jung_Risk = [];
for i in range(0,452):
    temp_Risk = np.transpose(LjungBoxQ(Risk_adjusted_Returns[:,i],10));
    if i == 451:
        result_Jung_Risk += [temp_Risk.iloc[:,0]];
        result_Jung_Risk += [temp_Risk.iloc[:,1]];
    else:
        result_Jung_Risk += [temp_Risk.iloc[:,0]];
titles=list(S_P_500_2);
titles.append('Critical Value');
result_Jung_Risk = pd.DataFrame(np.transpose(result_Jung_Risk), list(range(0,10)),titles)


proportion_Ljung_Risk = []
for i in range (10):
    Ljung_correlated_Risk = []            
    for j in range (452):
        if result_Jung_Risk.iloc[i,j] > result_Jung_Risk.iloc[i,452]:
            Ljung_correlated_Risk.append(result_Jung_Risk.iloc[i,j])
    proportion_Ljung_Risk.append(np.size(Ljung_correlated_Risk)/452)




Absolute_Returns = abs(Returns)
plt.plot(Absolute_Returns)

Result_Jung_Abs = [];
for i in range(0,452):
    temp_Abs = np.transpose(LjungBoxQ(Absolute_Returns[:,i],10));
    if i == 451:
        Result_Jung_Abs += [temp_Abs.iloc[:,0]];
        Result_Jung_Abs += [temp_Abs.iloc[:,1]];
    else:
        Result_Jung_Abs += [temp_Abs.iloc[:,0]];
titles=list(S_P_500_2);
titles.append('Critical Value');
Result_Jung_Abs = pd.DataFrame(np.transpose(Result_Jung_Abs), list(range(0,10)),titles)


proportion_Ljung_Abs = []
for i in range (10):
    Ljung_correlated_Abs = []            
    for j in range (452):
        if Result_Jung_Abs.iloc[i,j] > Result_Jung_Abs.iloc[i,452]:
            Ljung_correlated_Abs.append(Result_Jung_Abs.iloc[i,j])
    proportion_Ljung_Abs.append(np.size(Ljung_correlated_Abs)/452)




Absolute_Risk_Returns = abs(Risk_adjusted_Returns)
plt.plot(Absolute_Risk_Returns)

Result_Jung_Risk_Abs = [];
for i in range(0,452):
    temp_Risk_Abs = np.transpose(LjungBoxQ(Absolute_Risk_Returns[:,i],10));
    if i == 451:
        Result_Jung_Risk_Abs += [temp_Risk_Abs.iloc[:,0]];
        Result_Jung_Risk_Abs += [temp_Risk_Abs.iloc[:,1]];
    else:
        Result_Jung_Risk_Abs += [temp_Risk_Abs.iloc[:,0]];
titles=list(S_P_500_2);
titles.append('Critical Value');
Result_Jung_Risk_Abs = pd.DataFrame(np.transpose(Result_Jung_Risk_Abs), list(range(0,10)),titles)


proportion_Ljung_Risk_Abs = []
for i in range (10):
    Ljung_correlated_Risk_Abs = []            
    for j in range (452):
        if Result_Jung_Risk_Abs.iloc[i,j] > Result_Jung_Risk_Abs.iloc[i,452]:
            Ljung_correlated_Risk_Abs.append(Result_Jung_Risk_Abs.iloc[i,j])
    proportion_Ljung_Risk_Abs.append(np.size(Ljung_correlated_Risk_Abs)/452)


plt.figure(dpi=600)
plt.plot(proportion_Ljung)
plt.plot(proportion_Ljung_Risk, color = 'red')
plt.plot(proportion_Ljung_Abs, color = 'green')
plt.plot(proportion_Ljung_Risk_Abs, color = 'orange')
plt.legend(['Simple Returns', 'Risk Adjusted Returns', 'Abs Simple Returns', 'Abs Risk Adjusted Returns'])
plt.title('Evolution of the % of rejected test')



#Regression between the Market Portfolio and the Stock Returns



X1 = sm.add_constant(Index_Returns)

def OLS(y) :
    Model_CAPM = sm.OLS(y, X1)
    results = Model_CAPM.fit()
    return results.params


print(OLS(Returns[:, 2]))

Parameters = np.apply_along_axis(OLS, 0, Returns)

Beta = Parameters[1, :]
Beta_Higher = Beta[Beta > 1]
len(Beta_Higher)

Model_CAPM_2 = sm.OLS(Returns[:, 2], X1)
p = Model_CAPM_2.fit().params

print(Model_CAPM_2.fit().summary())

plt.figure(dpi=1000)
plt.scatter(X1[:, 1], Returns[:, 2], s = 1)
plt.plot(X1, p[0] + p[1] * X1, color = 'red')
plt.axis([-0.05, 0.05, -0.05, 0.08])
plt.xlabel('Market Returns')
plt.ylabel('Stock 3 Returns')
plt.title('Regression between the third asset and the Market')
plt.legend(["Regression"])

















