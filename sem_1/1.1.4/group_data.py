import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import norm

#доля случаев, когда отклонение числа отсчётов n от среднего значения не превышает (по модулю) одного, двух и трёх стандартных отклонений
def sigma_1_2_3(sec, table):
    w = dict.fromkeys(["tau", "1sigma", "2sigma", "3sigma"])
    w["tau"] = table["tau"]

    for k in range(1, 4):
        n = 0

        for  i in range(len(sec)):
            if (abs(sec[i] - table["av_n"]) <= k * table["sigma_n"]):
                n+=1

        n /= len(sec)

        w[str(k) + "sigma"] = n

    return w

def puasson(x):
    average = sum(x) / len(x)

    y = [average**x[i]/math.factorial(x[i]) * math.exp(-1 * average) for i in range(len(x))]    

    return y


def graph(sec, clr, nbins, label):
    plt.hist(sec, bins=nbins, density=True, alpha=0.6, color=clr, label=label)
    plt.legend(prop={'size': 10, })

    N = len(sec)
    tau = 4000 / N

    av_n = sum(sec) / N

    sigma_n = 0
    for i in range(N):
        sigma_n += (sec[i] - av_n) ** 2

    sigma_n = math.sqrt(sigma_n / N)

    sigma_av_n = sigma_n / math.sqrt(N)

    jot = av_n / (tau)

    sigma_jot = 0

    for i in range(N):
        sigma_jot += (sec[i] / (tau) - jot) ** 2

    sigma_jot = math.sqrt(sigma_jot) / N

    legend = {
        "tau": tau,
        "N": N,
        "av_n": av_n,
        "sigma_n": sigma_n,
        "sigma_av_n": sigma_av_n,
        "jot": jot,
        "sigma_jot": sigma_jot
    }

    return legend


def frequency(sec):
    sec_set = list(set(sec))
    unique_values = len(sec_set)

    freq = dict().fromkeys(sec_set)

    for i in range(unique_values):
        freq[sec_set[i]] = sec.count(sec_set[i]) / unique_values

    return freq


raw_data = pd.read_excel('data.xlsx')

sec_1 = raw_data.iloc[:, 0].tolist()

hist_1 = frequency(sec_1)

# ------------------ 10 SECONDS --------------------------
sec_10 = [sum(sec_1[i: i + 10]) for i in range(0, len(sec_1) - 10 + 1, 10)]

data_10 = {
    "10 сек": sec_10
}

df = pd.DataFrame(data_10)

df.to_excel("data10.xlsx", index=False)

hist_10 = frequency(sec_10)

# ------------------ 20 SECONDS --------------------------
sec_20 = [sum(sec_1[i: i + 20]) for i in range(0, len(sec_1) - 20 + 1, 20)]

data_20 = {
    "20 сек": sec_20
}

df = pd.DataFrame(data_20)

df.to_excel("data20.xlsx", index=False)

hist_20 = frequency(sec_20)

# ------------------ 40 SECONDS --------------------------
sec_40 = [sum(sec_1[i: i + 40]) for i in range(0, len(sec_1) - 40 + 1, 40)]

data_40 = {
    "40 сек": sec_40
}

df = pd.DataFrame(data_40)

df.to_excel("data40.xlsx", index=False)

hist_40 = frequency(sec_40)






# ------------------ GRAPHICS --------------------------


table_10 = graph(sec_10, 'b', len(set(sec_10)), "10 сек")
distr_10 = np.arange(0, max(sec_10) + 5, 1)
plt.plot(sec_10, puasson(sec_10), color='b', label='10 сек')

table_20 = graph(sec_20, 'g', len(set(sec_20)), "20 сек")
distr_20 = np.arange(min(sec_20), max(sec_20))
plt.plot(sec_20, puasson(sec_20), color='g', label='20 сек')

table_40 = graph(sec_40, 'y', len(set(sec_40)), "40 сек")
distr_40 = np.arange(0, max(sec_40) + 5, 1)
plt.plot(sec_40, puasson(sec_40), color='y', label='40 сек')

w_10 = sigma_1_2_3(sec_10, table_10)
w_20 = sigma_1_2_3(sec_20, table_20)
w_40 = sigma_1_2_3(sec_40, table_40)

W = dict.fromkeys(w_10.keys())

for key in w_10.keys():
    W[key] = [w_10[key], w_20[key], w_40[key]]

df = pd.DataFrame(W)
df.to_excel("sigma_1_2_3.xlsx", index=False)


plt.legend()

table = dict.fromkeys(table_10.keys())

for key in table.keys():
    table[key] = [table_10[key], table_20[key], table_40[key]]

df = pd.DataFrame(table)
df.to_excel("sigmas.xlsx", index=False)

plt.xlabel("n - частиц за отрезок времени")
plt.ylabel("Wn - частота")

plt.grid(linewidth=0.2)
plt.show()
