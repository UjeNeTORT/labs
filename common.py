import numpy as np
import matplotlib.pyplot as plt

def LeastSquares(x_arr, y_arr): # returns [k, b, sigma_k, sigma b]
    v = x_arr
    u = y_arr

    N = len(v)

    mu = np.mean(u) # средее
    mv = np.mean(v)
    mv2 = np.mean([v_i**2 for v_i in v]) # средний квадрат
    mu2 = np.mean([u_i**2 for u_i in u])
    muv = np.mean ([u[i] * v[i] for i in range(len(u))]) # среднее от произведения
    k = (muv - mu * mv) / (mv2 - mv**2)
    b = mu - k * mv

    sigma_k = np.sqrt(1/(N-2) * ( (mu2 - mu**2)/(mv2 - mv**2) - k**2 ) )
    sigma_b = sigma_k * np.sqrt(mv2)

    return (k, b, sigma_k, sigma_b)

def linear_graph(x, y, xerr, yerr, xlabel, ylabel, title):
  x_ = np.array(x)
  y_ = np.array(y)
  xerr_ = np.array(xerr)
  yerr_ = np.array(yerr)

  plt.figure(figsize=(8, 5))
  plt.errorbar(x=x_, y=y_, xerr=xerr_, yerr=yerr_, fmt='.', capsize=8,
    label=r'Экспериментальные точки')

  # Linear fit (МНК)
  coeffs = np.polyfit(x_, y_, 1)
  coeffs =  LeastSquares(x_, y_)
  linear_fit = np.poly1d(coeffs[:2])

  # Generate smooth values for the fitted line
  x_fit = np.linspace(min(x), max(x), 100)
  y_fit = linear_fit(x_fit)
  plt.plot(x_fit, y_fit, 'r', label=f'МНК: $\\varepsilon (v) = {coeffs[0]:.4f} \\cdot v + {coeffs[1]:.4f}$')

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)

  # grid
  plt.minorticks_on()
  plt.grid(True, linestyle='--', alpha=0.7, which='major')
  plt.grid(True, linestyle=':',  alpha=0.7, which='minor')
  plt.legend()
  plt.show()

  return coeffs

def plot(x, y, xerr, yerr, xlabel, ylabel, title):
  x_ = np.array(x)
  y_ = np.array(y)
  xerr_ = np.array(xerr)
  yerr_ = np.array(yerr)

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.errorbar(x=x_, y=y_, xerr=xerr_, yerr=yerr_, fmt='.', capsize=3,
    label=r'Экспериментальные точки')

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)

  # grid
  ax.minorticks_on()
  ax.grid(True, linestyle='--', alpha=0.7, which='major')
  ax.grid(True, linestyle=':',  alpha=0.7, which='minor')
  ax.legend()

  return fig, ax

def GetCalcReport (res_name, res_measure, val, sigma, n_round):
    print ("$" + res_name + " = (" + str(round(val, n_round)) + "~\\pm " + str(round(sigma, n_round)) + ") ~\\text{" + res_measure + "} ~(\\varepsilon ~\\approx " + str(round(sigma / val * 100, 2)) + " \\%) " + "$")

def latex_measurement(var_name, value, value_err, units, precision=3):
    import math

    # Determine exponent (power of 10)
    if value == 0:
        exponent = 0
    else:
        exponent = int(math.floor(math.log10(abs(value))))

    # Normalize values
    value_norm = value / (10 ** exponent)
    value_err_norm = value_err / (10 ** exponent)

    # Build format string dynamically based on precision
    fmt = f"{{:.{precision}f}}"
    value_str = fmt.format(value_norm)
    value_err_str = fmt.format(value_err_norm)

    # Relative error in %
    relative_error = (value_err / abs(value)) * 100 if value != 0 else 0

    # Build LaTeX string
    if exponent != 0:
        units_str = f"\\times 10^{{{exponent}}} ~\\text{{{units}}}"
    else:
        units_str = f"\\text{{{units}}}"

    latex_str = (
        f"${var_name} = ({value_str} \\pm {value_err_str})\\ "
        f"{units_str} \\ (\\varepsilon \\approx {fmt.format(relative_error)}\\ \\%)$"
    )

    print(latex_str)
    return latex_str

if __name__ == '__main__':
   print("the file is a library, dont execute")
