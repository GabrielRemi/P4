import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData

def chisquare(f: callable, x: np.ndarray, y: np.ndarray, yerr: np.ndarray, dof) -> float:

    x = np.array(x)
    y = np.array(y)
    yerr = np.array(y)

    chi_square = np.sum((y - f(x))**2 / yerr**2)
    return chi_square / (len(x) - dof)

# Bestimmung der Gitterkonstanten

# Wellenlängen der Quecksilberdampflampe in nm
lambda_hg = np.array([404.656, 407.783, 410.805, 433.922, 434.749, 435.833, 491.607, 546.074, 576.960, 579.066, 623.440, 671.643, 690.752])
# Winkel des Gitters in rad
w_g_hg_gemessen = np.array([52, 52, 52, 55, 55, 55, 60, 65, 69, 69, 72, 72, 74])
# Zusätzliche Winkelabweichung in mm
abweichung_d = np.array([0.0, 2.0, 4.0, -2.7, -2.1, -1.5, 0.0, 1.8, -1.6, 0.0, -1.0, 3.0, 0.0])

# Umrechnung der Winkelabweichung in Grad
abweichung_umrechnung = (abweichung_d / 300) * (180 / np.pi)

# tatsächlicher Winkel des Gitters in Grad
w_g_hg = w_g_hg_gemessen + abweichung_umrechnung
w_g_hg_err = 0.5

w_b = 140.0
w_b_err = 0.5

# in rad
w_g_hg_rad = w_g_hg * (np.pi / 180)
w_g_hg_rad_err = w_g_hg_err * (np.pi / 180)

w_b_rad = w_b * (np.pi / 180)
w_b_rad_err = w_b_err * (np.pi / 180)

# Ein- und Ausfallswinkel
alpha_hg = w_g_hg
alpha_hg_err = w_g_hg_err
beta_hg = w_b + w_g_hg - 180
beta_hg_err = np.sqrt((w_g_hg_err)**2 + (w_b_err)**2)
print('Ein- und Ausfallswinkel:', alpha_hg, alpha_hg_err, beta_hg, beta_hg_err)

# in rad
alpha_hg_rad = alpha_hg * (np.pi / 180)
alpha_hg_rad_err = alpha_hg_err * (np.pi / 180)
beta_hg_rad = beta_hg * (np.pi / 180)
beta_hg_rad_err = beta_hg_err * (np.pi / 180)

# Gitterkonstante separat berechnen in nm
g = lambda_hg / (np.sin(alpha_hg_rad) + np.sin(beta_hg_rad))
g_err = (g / (np.sin(alpha_hg_rad) + np.sin(beta_hg_rad))) * np.sqrt((np.cos(alpha_hg_rad) * alpha_hg_rad_err)**2 + (np.cos(beta_hg_rad) * beta_hg_rad_err)**2)

print('Gitterkonstante:', g, g_err)

# Gitterkonstante aus Regressionsverfahren
x_data_fit = lambda_hg[:10] * 10**(-9)
y_data_fit = np.sin(alpha_hg_rad[:10]) + np.sin(beta_hg_rad[:10])
y_data_fit_err = np.sqrt((np.cos(alpha_hg_rad[:10]) * alpha_hg_rad_err)**2 + (np.cos(beta_hg_rad[:10]) * beta_hg_rad_err)**2)
print('y-Werte:', y_data_fit, y_data_fit_err)
def gitterkonstante_fit(lam, a):
    return a * lam
plt.errorbar(x_data_fit, y_data_fit, y_data_fit_err, fmt = 'o', markersize = 3, capsize = 3, label = 'Datenpunkte')
popt1, pcov1 = curve_fit(gitterkonstante_fit, x_data_fit, y_data_fit)
plt.plot(x_data_fit, gitterkonstante_fit(x_data_fit, *popt1), label = 'Anpassungsparameter: $a = (2455100 \pm 2800)$ 1/m')
plt.xlabel('$\lambda$ / nm')
labels = [400, 425, 450, 475, 500, 525, 550, 575]
range = [400 * 10**(-9), 425 * 10**(-9), 450 * 10**(-9), 475 * 10**(-9), 500 * 10**(-9), 525 * 10**(-9), 550 * 10**(-9), 575 * 10**(-9)]
plt.xticks(range, labels)
plt.ylabel('$\sin(\\alpha) + \sin(\\beta)$ / dimensionslos')
plt.legend()
plt.grid()
plt.savefig(f"../figs/gitterkonstante_fit.png", dpi = 400)

# Chi2
y_gitter_predicted = gitterkonstante_fit(x_data_fit, popt1)
chi_squared_gitter = np.sum((y_data_fit - y_gitter_predicted)**2)
print('chi2 Gitter:', chi_squared_gitter)

# Gitterkonstante aus Fit
gitterkonstante = 1/popt1
gitterkonstante_err = (pcov1[0, 0]**0.5)/(popt1**2)
print('Gitterkonstante aus Fit:', gitterkonstante, gitterkonstante_err)

# Bestimmung der Balmerlinien

h_alpha_wg = 78 # Grad
h_alpha_isotopie = 0.2 # mm
h_alpha_isotopie_err = 0.1 # mm
h_beta_wg = 59 # Grad
h_beta_isotopie = 0.1 # mm
h_beta_isotopie_err = 0.1 # mm
h_gamma_wg = 54 # Grad
h_gamma_isotopie = 1.6 # mm
h_gamma_isotopie_err = 0.1 # mm 
wg_err = 0.5

# Umrechnung der Abweichung
h_alpha_isotopie_grad = (h_alpha_isotopie / 300) * (180 / np.pi)
h_alpha_isotopie_grad_err = (h_alpha_isotopie_err / 300) * (180 / np.pi)
h_beta_isotopie_grad = (h_beta_isotopie / 300) * (180 / np.pi)
h_beta_isotopie_grad_err = (h_beta_isotopie_err / 300) * (180 / np.pi)
h_gamma_isotopie_grad = (h_gamma_isotopie / 300) * (180 / np.pi)
h_gamma_isotopie_grad_err = (h_gamma_isotopie_err / 300) * (180 / np.pi)

# in rad
h_alpha_wg_rad = h_alpha_wg * (np.pi / 180)
h_beta_wg_rad = h_beta_wg * (np.pi / 180)
h_gamma_wg_rad = h_gamma_wg * (np.pi / 180)
wg_err_rad = wg_err * (np.pi / 180)

# Ein- und Ausfallswinkel
h_alpha_ein = h_alpha_wg
h_alpha_aus = w_b + h_alpha_wg - 180
h_beta_ein = h_beta_wg
h_beta_aus = w_b + h_beta_wg - 180
h_gamma_ein = h_gamma_wg
h_gamma_aus = w_b + h_gamma_wg - 180

h_alpha_ein_err = wg_err
h_alpha_aus_err = np.sqrt((wg_err)**2 + (w_b_err)**2)
h_beta_ein_err = wg_err
h_beta_aus_err = np.sqrt((wg_err)**2 + (w_b_err)**2)
h_gamma_ein_err = wg_err
h_gamma_aus_err = np.sqrt((wg_err)**2 + (w_b_err)**2)

# in rad
h_alpha_ein_rad = h_alpha_ein * (np.pi / 180)
h_alpha_aus_rad = h_alpha_aus * (np.pi / 180)
h_beta_ein_rad = h_beta_ein * (np.pi / 180)
h_beta_aus_rad = h_beta_aus * (np.pi / 180)
h_gamma_ein_rad = h_gamma_ein * (np.pi / 180)
h_gamma_aus_rad = h_gamma_aus * (np.pi / 180)

h_alpha_ein_rad_err = h_alpha_ein_err * (np.pi / 180)
h_alpha_aus_rad_err = h_alpha_aus_err * (np.pi / 180)
h_beta_ein_rad_err = h_beta_ein_err * (np.pi / 180)
h_beta_aus_rad_err = h_beta_aus_err * (np.pi / 180)
h_gamma_ein_rad_err = h_gamma_ein_err * (np.pi / 180)
h_gamma_aus_rad_err = h_gamma_aus_err * (np.pi / 180)

lambda_h_alpha = 1/popt1 * (np.sin(h_alpha_ein_rad) + np.sin(h_alpha_aus_rad))
lambda_h_alpha_err = np.sqrt(((np.sin(h_alpha_ein_rad) + np.sin(h_alpha_aus_rad)) * gitterkonstante_err)**2 + (1/popt1 * np.cos(h_alpha_ein_rad) * h_alpha_ein_rad_err)**2 + (1/popt1 * np.cos(h_alpha_aus_rad) * h_alpha_aus_rad_err)**2)
lambda_h_beta = 1/popt1 * (np.sin(h_beta_ein_rad) + np.sin(h_beta_aus_rad))
lambda_h_beta_err = np.sqrt(((np.sin(h_beta_ein_rad) + np.sin(h_beta_aus_rad)) * gitterkonstante_err)**2 + (1/popt1 * np.cos(h_beta_ein_rad) * h_beta_ein_rad_err)**2 + (1/popt1 * np.cos(h_beta_aus_rad) * h_beta_aus_rad_err)**2)
lambda_h_gamma = 1/popt1 * (np.sin(h_gamma_ein_rad) + np.sin(h_gamma_aus_rad))
lambda_h_gamma_err = np.sqrt(((np.sin(h_gamma_ein_rad) + np.sin(h_gamma_aus_rad)) * gitterkonstante_err)**2 + (1/popt1 * np.cos(h_gamma_ein_rad) * h_gamma_ein_rad_err)**2 + (1/popt1 * np.cos(h_gamma_aus_rad) * h_gamma_aus_rad_err)**2)
print('Ein- und Ausfallswinkel:', h_alpha_ein, h_alpha_aus, h_beta_ein, h_beta_aus, h_gamma_ein, h_gamma_aus)
print('Wellenlängen:', lambda_h_alpha, lambda_h_alpha_err, lambda_h_beta, lambda_h_beta_err, lambda_h_gamma, lambda_h_gamma_err)

# Isotopieaufspaltung
delta_beta_h_alpha = h_alpha_isotopie_grad * (np.pi / 180) 
delta_beta_h_alpha_err = h_alpha_isotopie_grad_err * (np.pi / 180)
delta_beta_h_beta = h_beta_isotopie_grad * (np.pi / 180) 
delta_beta_h_beta_err = h_beta_isotopie_grad_err * (np.pi / 180)
delta_beta_h_gamma = h_gamma_isotopie_grad * (np.pi / 180) 
delta_beta_h_gamma_err = h_gamma_isotopie_grad_err * (np.pi / 180)

delta_lambda_h_alpha = gitterkonstante * np.cos(h_alpha_aus_rad) * delta_beta_h_alpha
delta_lambda_h_alpha_err = np.sqrt((np.cos(h_alpha_aus_rad) * delta_beta_h_alpha * gitterkonstante_err)**2 + (gitterkonstante * np.sin(h_alpha_aus_rad) * delta_beta_h_alpha * h_alpha_aus_rad_err)**2 + (gitterkonstante * np.cos(h_alpha_aus_err) * delta_beta_h_alpha_err)**2)
delta_lambda_h_beta = gitterkonstante * np.cos(h_beta_aus_rad) * delta_beta_h_beta
delta_lambda_h_beta_err = np.sqrt((np.cos(h_beta_aus_rad) * delta_beta_h_beta * gitterkonstante_err)**2 + (gitterkonstante * np.sin(h_beta_aus_rad) * delta_beta_h_beta * h_beta_aus_rad_err)**2 + (gitterkonstante * np.cos(h_beta_aus_err) * delta_beta_h_beta_err)**2)
delta_lambda_h_gamma = gitterkonstante * np.cos(h_gamma_aus_rad) * delta_beta_h_gamma
delta_lambda_h_gamma_err = np.sqrt((np.cos(h_gamma_aus_rad) * delta_beta_h_gamma * gitterkonstante_err)**2 + (gitterkonstante * np.sin(h_gamma_aus_rad) * delta_beta_h_gamma * h_gamma_aus_rad_err)**2 + (gitterkonstante * np.cos(h_gamma_aus_err) * delta_beta_h_gamma_err)**2)
print(delta_lambda_h_alpha, delta_lambda_h_alpha_err, delta_lambda_h_beta, delta_lambda_h_beta_err, delta_lambda_h_gamma, delta_lambda_h_gamma_err)
print('Winkelaufspaltung:', h_alpha_isotopie_grad, h_alpha_isotopie_grad_err, h_beta_isotopie_grad, h_beta_isotopie_grad_err)
print('Isotopieaufspaltung:', delta_lambda_h_alpha, delta_lambda_h_alpha_err, delta_lambda_h_beta, delta_lambda_h_beta_err, delta_lambda_h_gamma, delta_lambda_h_gamma_err)

# CCD-Kamera

# rot
data_rot = np.loadtxt('../Daten/Rot.txt', skiprows = 1)
x_data_rot = data_rot[:,0] + 0.080
y_data_rot = data_rot[:,1]
x_data_rot_err = 0.001
y_data_rot_err = 0.1
ind_rot = np.array([i > -.0191 and i < .0401 for i in x_data_rot])

# türkis
data_türkis = np.loadtxt('../Daten/Turkis.txt', skiprows = 1)
x_data_türkis = data_türkis[:,0] - 0.075
y_data_türkis = data_türkis[:,1]
x_data_türkis_err = 0.001
y_data_türkis_err = 0.1
ind_türkis = np.array([i > -.02 and i < .03 for i in x_data_türkis])

# lila
data_lila = np.loadtxt('../Daten/Lila.txt', skiprows = 1)
x_data_lila = data_lila[:,0] - 0.473
y_data_lila = data_lila[:,1]
x_data_lila_err = 0.001
y_data_lila_err = 0.1
ind_lila = np.array([i > -.05 and i < .05 for i in x_data_lila])

#testfit
def f(a, x):
    return a[0] * np.exp(-((x - a[1])**2)/(2*a[2]**2)) + a[3] * np.exp(-((x - a[4])**2)/(2*a[5]**2)) + a[6]

exp = Model(f)
mydata_rot = RealData(x_data_rot[ind_rot], y_data_rot[ind_rot], sx = x_data_rot_err, sy = y_data_rot_err)
myodr_rot = ODR(mydata_rot, exp, beta0 = [80,0.0,0.01,10,0.03,0.01,6])
myoutput_rot = myodr_rot.run()
myoutput_rot.pprint()

mydata_türkis = RealData(x_data_türkis[ind_türkis], y_data_türkis[ind_türkis], sx = x_data_türkis_err, sy = y_data_türkis_err)
myodr_türkis = ODR(mydata_türkis, exp, beta0 = [11,0.0,0.005,1,0.02,0.004,5.5])
myoutput_türkis = myodr_türkis.run()
myoutput_türkis.pprint()

mydata_lila = RealData(x_data_lila[ind_lila], y_data_lila[ind_lila], sx = x_data_lila_err, sy = y_data_lila_err)
myodr_lila = ODR(mydata_lila, exp, beta0 = [6,0.0,0.005,1,-0.4,0.01,6])
myoutput_lila = myodr_lila.run()
myoutput_lila.pprint()

chi_square = chisquare(lambda x: f(myoutput_rot.beta, x), x_data_rot, y_data_rot,
                        np.array([y_data_rot_err]*len(x_data_rot)), 7)
print(f"chi squared: {chi_square}")

plt.subplots()
plt.errorbar(x_data_rot, y_data_rot, xerr = x_data_rot_err, yerr = y_data_rot_err, linestyle = 'none', label = 'Datenpunkte')
x1 = np.linspace(-.02, .042, 1000)
plt.plot(x1, f(myoutput_rot.beta, x1), label = 'Gauß-Anpassung')
plt.xlim([-0.02,0.042])
plt.legend()
plt.title('rot')
plt.xlabel('Winkel $\\beta$ / °')
plt.ylabel('Intensität $I$ / %')
plt.grid()
plt.savefig(f"../figs/rot_fit.png", dpi = 400)

plt.subplots()
plt.errorbar(x_data_türkis, y_data_türkis, xerr = x_data_türkis_err, yerr = y_data_türkis_err, linestyle = 'none', label = 'Datenpunkte')
x2 = np.linspace(-.02, .03, 1000)
plt.plot(x2, f(myoutput_türkis.beta, x2), label = 'Gauß-Anpassung')
plt.xlim([-0.02,0.03])
plt.legend()
plt.title('türkis')
plt.xlabel('Winkel $\\beta$ / °')
plt.ylabel('Intensität $I$ / %')
plt.grid()
plt.savefig(f"../figs/türkis_fit.png", dpi = 400)

plt.subplots()
plt.errorbar(x_data_lila, y_data_lila, xerr = x_data_lila_err, yerr = y_data_lila_err, linestyle = 'none', label = 'Datenpunkte')
x3 = np.linspace(-.05, .05, 1000)
plt.plot(x3, f(myoutput_lila.beta, x3), label = 'Gauß-Anpassung')
plt.xlim([-0.05,0.05])
plt.legend()
plt.title('violett')
plt.xlabel('Winkel $\\beta$ / °')
plt.ylabel('Intensität $I$ / %')
plt.grid()
plt.savefig(f"../figs/lila_fit.png", dpi = 400)

# Bestimmung der Rydberg-Konstanten
x_werte = np.array([(1/4 - 1/9), (1/4 - 1/16), (1/4 - 1/25)])
y_werte = np.array([1540832, 2074688, 2336448])
y_werte_err = np.array([11879, 25826, 32753])

def rydbergkonstante_fit(x, R):
    return R * x
plt.subplots()
plt.errorbar(x_werte, y_werte, y_werte_err, fmt = 'o', markersize = 3, capsize = 3, label = 'Datenpunkte')
popt2, pcov2 = curve_fit(rydbergkonstante_fit, x_werte, y_werte)
plt.plot(x_werte, rydbergkonstante_fit(x_werte, *popt2), label = 'Anpassungsparameter: $R_{\infty} = (11100000 \pm 20000)$ 1/m')
plt.xlabel('$(1/4 - 1/n^2)$ / dimensionslos')
plt.ylabel('$\\frac{1}{\lambda}$ / 1/m')
plt.legend()
plt.grid()
plt.savefig(f"../figs/rydbergkonstante_fit.png", dpi = 400)

# Chi2
y_ryd_predicted = rydbergkonstante_fit(x_werte, popt2)
chi_squared_ryd = np.sum((y_werte - y_ryd_predicted)**2)
print('chi2 Ryd:', chi_squared_ryd)

# Rydberg
rydberg_konstante = popt2
rydberg_konstante_err = pcov2[0, 0]**0.5
print('Rydberg-Konstante:', rydberg_konstante, rydberg_konstante_err)