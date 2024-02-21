import pandas as pd
import wquantiles
import numpy as np
state = pd.read_csv('c:\\data_all/state.csv')
print(state['Murder.Rate'].mean())
print(np.average(state['Murder.Rate'], weights = state['Population']))

# 중위 절대 편차 (MAD)
from statsmodels import robust
print(robust.scale.mad(state['Population']))

import matplotlib.pyplot as plt

# ax = (state['Population']/1_000_000).plot.box(figsize = (2,4))
# plt.show()

ax = (state['Population']/1_000_000).plot.hist(figsize=(4,4))
ax.set_xlabel('Population(millions)')
# plt.show()

# ax = (state['Population']).plot.hist(density = True, xlim=[0,12], bins=range(1,12), figsize = (4,4))
# state['Murder.Rate'].plot.density(ax=ax)
# plt.show()

#도수 분포표
#인구수를 10개의 구간으로 분할해서 도수 분포표(value_counts) 출력
binnedPopulation = pd.cut(state['Population'], bins=10)
print(binnedPopulation.value_counts())


'''
# 다변량 탐색
univ = pd.read_csv('c:\\data_all/descriptive.csv')
univ['성별'] = '남자'
idx = 0
for val in univ['gender']:
    if val == 2:
        univ['성별'][idx] = '여자'
    idx = idx + 1
print(univ['성별'].value_counts())    

univ['학력'] = '응답없음'
idx = 0
for val in univ['level']:
    if val == 1.0:
        univ['학력'][idx] = '고졸'
    if val == 2.0:
        univ['학력'][idx] = '대졸'
    if val == 3.0:
        univ['학력'][idx] = '대학원졸'
    idx = idx + 1
univ.drop('level', axis = 1, inplace=True)
# print(univ.head())

univ['합격여부'] = '응답없음'
idx = 0
for val in univ['pass']:
  if val == 1.0:
    univ['합격여부'][idx] = '합격'
  if val == 2.:
    univ['합격여부'][idx] = '불합격'
  idx = idx + 1
univ.drop("pass", axis=1, inplace=True)
# print(univ.head())

# 교차 분할표
univ_tbl = univ[(univ['학력']=='고졸')|(univ['학력']=='대졸')|(univ['학력']=='대학원졸')]
univ_tbl = univ[(univ['합격여부']=='합격')|(univ['합격여부']=='불합격')]
print(pd.crosstab(univ_tbl['학력'], univ_tbl['합격여부']))

# cov
cov_dt = pd.read_csv('c:\\data_all/cov.csv')
x = cov_dt['x']
y = cov_dt['y']

mu_x = np.mean(x)
mu_y = np.mean(y)
print(mu_x, mu_y)

cov = sum((x-mu_x) * (y-mu_y))/(len(cov_dt)-1)
print('공분산: ', cov)

# 분산공분산 행렬 np.cov
covv = np.cov(x, y, ddof = (len(cov_dt)-(len(cov_dt)-1)))
print(covv)
'''
