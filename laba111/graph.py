import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

vah_20 = pd.read_excel('vah_20.xlsx', index_col=0)
vah_30 = pd.read_excel('vah_30.xlsx', index_col=0)
vah_50 = pd.read_excel('vah_50.xlsx', index_col=0)

U_v_20 = vah_20.iloc[:,2].tolist()
I_a_20 = vah_20.iloc[:,3].tolist()
xerr_20 = vah_20.iloc[:,5].tolist()
yerr_20 = vah_20.iloc[:,6].tolist()

U_v_30 = vah_30.iloc[:,2].tolist()
I_a_30 = vah_30.iloc[:,3].tolist()
xerr_30 = vah_30.iloc[:,5].tolist()
yerr_30 = vah_30.iloc[:,6].tolist()

U_v_50 = vah_50.iloc[:,2].tolist()
I_a_50 = vah_50.iloc[:,3].tolist()
xerr_50 = vah_50.iloc[:,5].tolist()
yerr_50 = vah_50.iloc[:,6].tolist()

# l = 20
coefficients_20 = np.polyfit(U_v_20, I_a_20, 1)

print("l = 20: tg = ", coefficients_20[0], " R1 = ", 1/coefficients_20[0])

polynomial_20 = np.poly1d(coefficients_20)
x_approx_20 = np.linspace(min(U_v_20), max(U_v_20), 100)
y_approx_20 = polynomial_20(x_approx_20)

# l = 30
coefficients_30 = np.polyfit(U_v_30, I_a_30, 1)

print("l = 30: tg = ", coefficients_30[0], " R1 = ", 1/coefficients_30[0])

polynomial_30 = np.poly1d(coefficients_30)
x_approx_30 = np.linspace(min(U_v_30), max(U_v_30), 100)
y_approx_30 = polynomial_30(x_approx_30)

# l = 50
coefficients_50 = np.polyfit(U_v_50, I_a_50, 1)

print("l = 50: tg = ", coefficients_50[0], " R1 = ", 1/coefficients_50[0])

polynomial_50 = np.poly1d(coefficients_50)
x_approx_50 = np.linspace(min(U_v_50), max(U_v_50), 100)
y_approx_50 = polynomial_50(x_approx_50)



plt.errorbar(U_v_20, I_a_20, xerr=xerr_20, yerr=yerr_20, fmt='s', capsize= 3, markersize = 2)
plt.plot(x_approx_20, y_approx_20, label='l = 20 см')

plt.errorbar(U_v_30, I_a_30, xerr=xerr_30, yerr=yerr_30, fmt='o', capsize= 3, markersize = 2)
plt.plot(x_approx_30, y_approx_30, label='l = 30 см')

plt.errorbar(U_v_50, I_a_50, xerr=xerr_50, yerr=yerr_50, fmt='^', capsize= 3, markersize = 2)
plt.plot(x_approx_50, y_approx_50, label='l = 50 см')

plt.xlabel("Uv, мВ")
plt.ylabel("Ia, мВ")

plt.xlim(0, max(U_v_20) + 50)
plt.ylim(0, max(I_a_20) + 50)

plt.grid(True)

plt.legend()
plt.show()


