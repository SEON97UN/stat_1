#자료구조 패키지
import numpy as np
import pandas as pd
#통계량 계산이나 기본적인 데이터 분석에 사용되는 패키지
import scipy as sp
#시각화 패키지
import matplotlib.pyplot as plt
#시각화에서 한글을 사용하기 위한 설정
import platform
from matplotlib import font_manager, rc

if platform.system() == 'Darwin':
  rc('font', family='AppleGothic')
  #윈도우의 경우
elif platform.system() == 'Windows':
  font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
  rc('font', family=font_name)
#시각화에서 음수를 표현하기 위한 설정
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
# Jupyter Notebook의 출력을 소수점 이하 3자리로 제한
# %precision 3
# precision은 소수점은 과학적 표기법으로 변환할 자릿수를 설정
# 아래와 같이 하면 소수점 셋째 자리 밑으로는 과학적 표기법으로 표시
pd.options.display.precision = 3  
# 경고 발생 시 무시
import warnings
warnings.filterwarnings('ignore')

# 피어슨 상관 계수 - 일반적인 상관계수
# cov / (sd_x * sd_y)
# [-1, 1] 
auto_data = pd.read_csv('c:\\data_all/auto-mpg.csv', header=None)
auto_data.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']
# print(auto_data.info())
# print(auto_data.head())
import seaborn as sns
# sns.pairplot(auto_data)  #DF에 존재하는 모든 숫자 col의 산점도 출력
# plt.show()
# print(auto_data[['mpg','cylinders', 'displacement', 'weight']].corr())
# DataFrame 산점도
# auto_data.plot(kind='scatter', x = 'weight', y = 'mpg', c = 'navy', s=10)
# plt.show()
# regplot()
# sns.regplot(x='weight', y='mpg', data=auto_data)
# plt.show()
# jointplot()
# sns.jointplot(x='weight', y='mpg', kind='reg', data=auto_data)
# plt.show()


# sp500_sym = pd.read_csv('c:\\data_all/sp500_sectors.csv')
# sp500_px = pd.read_csv('c:\\data_all/sp500_data.csv.gz', index_col=0)
# print(sp500_sym.head())
# print(sp500_px.head())

# anscombe data
# 데이터 분포를 피어슨 상관계수만으로 판단 X
import statsmodels.api as sm
data = sm.datasets.get_rdataset('anscombe')
df = data.data
# print(df)  # 8개의 col
# Pearson Corr
# print(df['x1'].corr(df['y1']))
# print(df['x2'].corr(df['y2']))
# print(df['x3'].corr(df['y3']))
# print(df['x4'].corr(df['y4']))
# plt.subplot(221) #2: 행의 수, 2: 열의 수, 1: 2X2의 첫 번째
# sns.regplot(x='x1', y='y1', data=df)
# plt.subplot(222) 
# sns.regplot(x='x2', y='y2', data=df)
# plt.subplot(223) 
# sns.regplot(x='x3', y='y3', data=df)
# plt.subplot(224) 
# sns.regplot(x='x4', y='y4', data=df)
# plt.show()

# Spearman Corr
s1 = pd.Series([1,3,5,7,9])
s2 = pd.Series([1,9,25,49,81])
# sns.regplot(x=s1, y=s2)
# plt.show()

# print('Pearson: ', s1.corr(s2))
# print('Spearman: ', s1.corr(s2, method='spearman'))

# print('1의 Spearman Corr: ', sp.stats.spearmanr(df['x1'], df['y1']))
# print('2의 Spearman Corr: ', sp.stats.spearmanr(df['x2'], df['y2']))
# print('3의 Spearman Corr: ', sp.stats.spearmanr(df['x3'], df['y3']))
# print('4의 Spearman Corr: ', sp.stats.spearmanr(df['x4'], df['y4']))

# Kendall
# print('Kendall: ', s1.corr(s2, method='kendall'))
# print('1의 Kendall Corr: ', sp.stats.kendalltau(df['x1'], df['y1']))
# print('2의 Kendall Corr: ', sp.stats.kendalltau(df['x2'], df['y2']))
# print('3의 Kendall Corr: ', sp.stats.kendalltau(df['x3'], df['y3']))
# print('4의 Kendall Corr: ', sp.stats.kendalltau(df['x4'], df['y4']))

# 산점도
tips = sns.load_dataset('tips')
# sns.jointplot(x='total_bill', y='tip', data=tips, kind='scatter')

# 육각형 차트
# sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
# plt.show()

# 등고선 차트
# kc_tax = pd.read_csv('c:\\data_all/kc_tax.csv.gz')
# print(kc_tax.shape)
# # 필터링
# kc_tax0 = kc_tax.loc[(kc_tax.TaxAssessedValue < 750000)&
#                      (kc_tax.SqFtTotLiving > 100)&
#                      (kc_tax.SqFtTotLiving < 3500), :]
# print(kc_tax0.shape)
# sns.kdeplot(data=kc_tax0, x = 'SqFtTotLiving', y = 'TaxAssessedValue')
# plt.show()


# pivot_table
# grade 와 status 교차 분할표 작성
lc_loans = pd.read_csv('c:\\data_all/lc_loans.csv')
# print(lc_loans.info())
result = lc_loans.pivot_table(index='grade', columns='status', aggfunc=lambda x:len(x), margins=True) # lambdax:len(x) -> 데이터 개수 셈
# print(result)
# print Ratio
# grade의 합계 제외한 부분
df = result.copy().loc["A":"G", :]
df.loc[:, "Charged Off":"Late"] = df.loc[:, "Charged Off":"Late"].div(df['All'], axis=0)
df['All'] = df['All'] / sum(df['All'])
# print(df)


# boxplot
airline_stat = pd.read_csv('c:\\data_all/airline_stats.csv')
# airline_stat.info()

# airline_stat.boxplot(by = 'airline', column='pct_carrier_delay')
# plt.show()

# violinplot
# sns.violinplot(data=airline_stat, x='airline', y='pct_carrier_delay')
# plt.show()

# sample data
np.random.seed(42)
x = np.random.normal(size=1000)
# print('표본평균: ', np.mean(x))
# print('표본분산: ', np.var(x))
# print('1차 적률(moment): ', sp.stats.moment(x, 1))
# print('2차 적률(moment): ', sp.stats.moment(x, 2))

# # skewness
# print('표본 왜도: ', sp.stats.skew(x))
# print('3차 적률(moment): ', sp.stats.moment(x, 3))

# kurtosis
# print('3 - 표본 첨도: ', 3 - sp.stats.kurtosis(x))
# print('4차 적률(moment): ', sp.stats.moment(x, 4))

x_m1 = sp.stats.moment(x, 1)
x_m2 = sp.stats.moment(x, 2)
x_var = np.var(x)
# print(x_m1, x_m2, x_var)

xx = np.linspace(-8, 8, 100)
# 정규 분포 객체의 pdf
# 평균 1, 표준편차 2 
rv = sp.stats.norm(loc = 1, scale = 2) 
#확률 밀도 함수
pdf = rv.pdf(xx)
# plt.plot(xx, pdf)
# # plt.show()
# 누적분포함수
cdf = rv.cdf(xx)
# plt.plot(xx, cdf)
# plt.show()


# # Bernoulli dis
# # 1이 나올 확률 = .6
# rv = sp.stats.bernoulli(.6)
# print(rv)

# # 나올 수 있는 경우
# xx = [0, 1]
# # pmf
# plt.bar(xx, rv.pmf(xx))
# # x 축 수정
# plt.xticks([0, 1], ['x=0', 'x=1'])
# plt.show()

# simulation
# sample data
# x = rv.rvs(1000, random_state = 42) #random_state: seed 고정
# sns.countplot(x=x)
# plt.show()

# 실제로는 이론과 비교하는 것이 아니고 주장과 비교
# 여러차례 시뮬레이션을 한 결과와 비교를 해서 타당성 검증

# 시뮬레이션 결과의 비율
# y = np.bincount(x, minlength=2) / float(len(x))
# print(y)
# # 이론과 시뮬레이션 결과를 하나의 데이터프레임으로 작성
# df = pd.DataFrame({'이론':rv.pmf(xx), '시뮬':y})
# df.index = [0, 1]
# print(df)
# # 시뮬레이션 결과
# result = sp.stats.describe(x)
# print(result)

# 이항 분포 
# rv = sp.stats.binom(10, 0.5) 
# xx = np.arange(11)
# plt.bar(xx, rv.pmf(xx), align='center')
# # plt.show()

# x = rv.rvs(1000) #1000번 시뮬레이션
# sns.countplot(x=x)
# # plt.show()
# # print(np.sum(x[x>8])/len(x))

# # 앞 뒷면이 나오는 확률이 동일한 동전을 100번 던졌을 때 60회 이상 나올 확률
# # (1) pmf: 1 - (0 ~ 59 까지 나올 확률 sum)
# # (2) cdf: 0번부터 나올 확률의 누적 확률
# p = sp.stats.binom.cdf(n=100, p=.5, k=59)
# # print(1-p)
# # (3) sf: 1-cdf
# sf = sp.stats.binom.sf(n=100, p=.5, k=59)
# # print(sf)

# # 20 ~ 60번 나올 확률
# p1 = sp.stats.binom.cdf(n=100, p=.5, k=19)
# p2 = sp.stats.binom.cdf(n=100, p=.5, k=60)
# # print(p2 - p1)

# p = sp.stats.binom.ppf(n=100, p=.5, q=.8)
# # print(p)
# p1 = sp.stats.binom.cdf(n=100, p=.5, k=p-1)
# p2 = sp.stats.binom.cdf(n=100, p=.5, k=p)
# print(p1, p2)


# 카테고리별 확률 - 주사위의 확률
mu = [.1, .1, .1, .1, .1, .5]

# 카테고리 분포 인스턴스 생성
rv = sp.stats.multinomial(1, mu)

# 데이터 생성
xx = np.arange(1,7)
# print(xx)

# 원 핫 인코딩 - pandas의 자료형
xx_one = pd.get_dummies(xx)
# print(xx_one) 

# 확률 질량 함수 출력
# 통계, scikit-learn 을 이요하는 전처리 & 머신러닝에서는
# numpy의 ndarray 가 기본 자료형
plt.bar(xx, rv.pmf(xx_one.values))
# plt.show()

# 시뮬레이션 - 여섯번째 열이 1인 경우가 많음
X = rv.rvs(100)
# print(X)


# 다항 분포 - 카테고리 분포 시행 횟수 여러 번
rv_multi = sp.stats.multinomial(100, [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]) # 공평한 주사위 100번 시행
X = rv_multi.rvs(1000) # 1000번 시뮬레이션 중 앞의 20개 출력
print(X[:20])

# 시뮬레이션 결과 시각화
df = pd.DataFrame(X).stack().reset_index()
df.columns = ['시도', '클래스', '데이터개수']
print(df)

# sns.swarmplot(x='클래스', y='데이터개수', data=df)
sns.violinplot(x='클래스', y='데이터개수', data=df)
plt.show()
