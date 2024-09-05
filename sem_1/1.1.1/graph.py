import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def sigma_k(x, y, k, n):
    return math.sqrt((avsq(y) / avsq(x) - k ** 2) / (n - 1))

def sigma_sist():
    return

def avsq(arr):
    res = 0
    for i in arr:
        res += i ** 2
    res /= len(arr)

    return res


def find_k(x, y):
    k = 0
    for i in range(len(x)):
        k += x[i] * y[i]

    k /= len(x)
    k /= avsq(x)

    return k


vah_20 = pd.read_excel('vah_20.xlsx', index_col=0)
vah_30 = pd.read_excel('vah_30.xlsx', index_col=0)
vah_50 = pd.read_excel('vah_50.xlsx', index_col=0)

U_v_20 = vah_20.iloc[:, 2].tolist()
I_a_20 = vah_20.iloc[:, 3].tolist()
xerr_20 = vah_20.iloc[:, 6].tolist()
yerr_20 = vah_20.iloc[:, 5].tolist()
dr_20 = vah_20.iloc[: , 9].tolist()[0]

U_v_30 = vah_30.iloc[:, 2].tolist()
I_a_30 = vah_30.iloc[:, 3].tolist()
xerr_30 = vah_30.iloc[:, 6].tolist()
yerr_30 = vah_30.iloc[:, 5].tolist()
dr_30 = vah_30.iloc[: , 9].tolist()[0]

U_v_50 = vah_50.iloc[:, 2].tolist()
I_a_50 = vah_50.iloc[:, 3].tolist()
xerr_50 = vah_50.iloc[:, 6].tolist()
yerr_50 = vah_50.iloc[:, 5].tolist()
dr_50 = vah_50.iloc[: , 9].tolist()[0]

# ---------------------- L = 20 -------------------------
k_20 = find_k(I_a_20, U_v_20)

sigma_k_20 = sigma_k(I_a_20, U_v_20, k_20, len(U_v_20))

polynomial_20 = np.poly1d([k_20, 0])
x_approx_20 = np.linspace(min(I_a_20), max(I_a_20) + 5, 100)
y_approx_20 = polynomial_20(x_approx_20)

plt.errorbar(I_a_20, U_v_20, xerr=xerr_20, yerr=yerr_20, fmt='s', capsize=3, markersize=2)
plt.plot(x_approx_20, y_approx_20, label='l = 20 см')

# ---------------------- L = 30 -------------------------
k_30 = find_k(I_a_30, U_v_30)

sigma_k_30 = sigma_k(I_a_30, U_v_30, k_30, len(U_v_30))

polynomial_30 = np.poly1d([k_30, 0])
x_approx_30 = np.linspace(min(I_a_30), max(I_a_30) + 5, 100)
y_approx_30 = polynomial_30(x_approx_30)

plt.errorbar(I_a_30, U_v_30, xerr=xerr_30, yerr=yerr_30, fmt='o', capsize=3, markersize=2)
plt.plot(x_approx_30, y_approx_30, label='l = 30 см')

# ---------------------- L = 50 -------------------------
k_50 = find_k(I_a_50, U_v_50)

sigma_k_50 = sigma_k(I_a_50, U_v_50, k_50, len(U_v_50))

polynomial_50 = np.poly1d([k_50, 0])
x_approx_50 = np.linspace(min(I_a_50), max(I_a_50) + 5, 100)
y_approx_50 = polynomial_50(x_approx_50)

plt.errorbar(I_a_50, U_v_50, xerr=xerr_50, yerr=yerr_50, fmt='^', capsize=3, markersize=2)
plt.plot(x_approx_50, y_approx_50, label='l = 50 см')

# ---------------------- EXCEL -------------------------
data = {'l, см': [50, 30, 20],
        'R1, ом': [round(k_50, 5), round(k_30, 5), round(k_20, 5)],
        'sigma_случ, ом': [round(sigma_k_50, 5), round(sigma_k_30, 5), round(sigma_k_20, 5)],
        'sigma_сист, ом': [round(dr_50, 5), round(dr_30, 5), round(dr_20, 5)],
        'sigma_полн, ом': [round(math.sqrt(dr_50**2 + sigma_k_50), 5),
                           round(math.sqrt(dr_30**2 + sigma_k_30), 5),
                           round(math.sqrt(dr_20**2 + sigma_k_20), 5)]}

df = pd.DataFrame(data)

df.to_excel('r1.xlsx', index=False)

# ---------------------- SHOW -------------------------
plt.xlabel("Ia, мВ")
plt.ylabel("Uv, мВ")

plt.xlim(0, max(I_a_20) + 50)
plt.ylim(0, max(U_v_20) + 50)

plt.title("Uv(Ia)")
plt.grid(True)
plt.legend()
plt.show()
