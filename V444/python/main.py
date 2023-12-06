import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData
from monke.plots import legend

plt_params = {
    "font.size": 8.5,
    "lines.markersize": 3,
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "figure.figsize": [7, 5.5]
}
mpl.rcParams.update(plt_params)
plt.style.use("science")

# Versuchsteil 1

# Darstellung der Rohdaten - eigene Messung
glas_own_spol_data = np.loadtxt('../part1_own_data/glas2_do_0grad_avg50.txt', skiprows = 2)
glas_own_spol_angle = glas_own_spol_data[:,0] # Einfallswinkel in Grad
glas_own_spol_intensity = glas_own_spol_data[:,1] # Intensität in willkürlichen Einheiten

glas_own_ppol_data = np.loadtxt('../part1_own_data/glas2_do_45grad_avg50.txt', skiprows = 2)
glas_own_ppol_angle = glas_own_ppol_data[:,0] # Einfallswinkel in Grad
glas_own_ppol_intensity = glas_own_ppol_data[:,1] # Intensität in willkürlichen Einheiten

plt.subplots()
plt.errorbar(glas_own_spol_angle, glas_own_spol_intensity, xerr = 0.01, yerr = 0.05, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(glas_own_ppol_angle, glas_own_ppol_intensity, xerr = 0.01, yerr = 0.05, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation', color = 'red')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Intensität $I_{\mathrm{ref}}$ / w.E.')
plt.savefig(f"../protokoll/figs/referenz_own.pdf", dpi = 400)

# Erster Goldfilm
gold1_own_spol_data = np.loadtxt('../part1_own_data/au1_0grad_spol_avg50.txt', skiprows = 2)
gold1_own_spol_angle = gold1_own_spol_data[:,0] # Einfallswinkel in Grad
gold1_own_spol_intensity = gold1_own_spol_data[:,1] # Intensität in willkürlichen Einheiten

gold1_own_ppol_data = np.loadtxt('../part1_own_data/au1_45grad_ppol_avg50.txt', skiprows = 2)
gold1_own_ppol_angle = gold1_own_ppol_data[:,0] # Einfallswinkel in Grad
gold1_own_ppol_intensity = gold1_own_ppol_data[:,1] # Intensität in willkürlichen Einheiten

plt.subplots()
plt.errorbar(gold1_own_spol_angle, gold1_own_spol_intensity, xerr = 0.01, yerr = 0.05, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(gold1_own_ppol_angle, gold1_own_ppol_intensity, xerr = 0.01, yerr = 0.05, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation', color = 'red')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Intensität $I_{\mathrm{Au}}$ / w.E.')
plt.savefig(f"../protokoll/figs/au1_own.pdf", dpi = 400)

# Zweiter Goldfilm
gold2_own_spol_data = np.loadtxt('../part1_own_data/au2_0grad_spol_avg50_messung2.txt', skiprows = 2)
gold2_own_spol_angle = gold2_own_spol_data[:,0] # Einfallswinkel in Grad
gold2_own_spol_intensity = gold2_own_spol_data[:,1] # Intensität in willkürlichen Einheiten

gold2_own_ppol_data = np.loadtxt('../part1_own_data/au1_45grad_ppol_avg50_messung2.txt', skiprows = 2)
gold2_own_ppol_angle = gold2_own_ppol_data[:,0] # Einfallswinkel in Grad
gold2_own_ppol_intensity = gold2_own_ppol_data[:,1] # Intensität in willkürlichen Einheiten

plt.subplots()
plt.errorbar(gold2_own_spol_angle, gold2_own_spol_intensity, xerr = 0.01, yerr = 0.05, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(gold2_own_ppol_angle, gold2_own_ppol_intensity, xerr = 0.01, yerr = 0.05, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation', color = 'red')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Intensität $I_{\mathrm{Au}}$ / w.E.')
plt.savefig(f"../protokoll/figs/au2_own.pdf", dpi = 400)

# Dritter Goldfilm
gold3_own_spol_data = np.loadtxt('../part1_own_data/au3_0grad_spol_avg50.txt', skiprows = 2)
gold3_own_spol_angle = gold3_own_spol_data[:,0] # Einfallswinkel in Grad
gold3_own_spol_intensity = gold3_own_spol_data[:,1] # Intensität in willkürlichen Einheiten

gold3_own_ppol_data = np.loadtxt('../part1_own_data/au3_45grad_ppol_avg50.txt', skiprows = 2)
gold3_own_ppol_angle = gold3_own_ppol_data[:,0] # Einfallswinkel in Grad
gold3_own_ppol_intensity = gold3_own_ppol_data[:,1] # Intensität in willkürlichen Einheiten

plt.subplots()
plt.errorbar(gold3_own_spol_angle, gold3_own_spol_intensity, xerr = 0.01, yerr = 0.05, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(gold3_own_ppol_angle, gold3_own_ppol_intensity, xerr = 0.01, yerr = 0.05, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation', color = 'red')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Intensität $I_{\mathrm{Au}}$ / w.E.')
plt.savefig(f"../protokoll/figs/au3_own.pdf", dpi = 400)

# Vierter Goldfilm
gold4_own_spol_data = np.loadtxt('../part1_own_data/au4_0grad_spol_avg50.txt', skiprows = 2)
gold4_own_spol_angle = gold4_own_spol_data[:,0] # Einfallswinkel in Grad
gold4_own_spol_intensity = gold4_own_spol_data[:,1] # Intensität in willkürlichen Einheiten

gold4_own_ppol_data = np.loadtxt('../part1_own_data/au4_45grad_ppol_avg50.txt', skiprows = 2)
gold4_own_ppol_angle = gold4_own_ppol_data[:,0] # Einfallswinkel in Grad
gold4_own_ppol_intensity = gold4_own_ppol_data[:,1] # Intensität in willkürlichen Einheiten

plt.subplots()
plt.errorbar(gold4_own_spol_angle, gold4_own_spol_intensity, xerr = 0.01, yerr = 0.05, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(gold4_own_ppol_angle, gold4_own_ppol_intensity, xerr = 0.01, yerr = 0.05, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation', color = 'red')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Intensität $I_{\mathrm{Au}}$ / w.E.')
plt.savefig(f"../protokoll/figs/au4_own.pdf", dpi = 400)

# Darstellung der Rohdaten des Tutors
glas_spol_data = np.loadtxt('../part1_tutor/Referenz_Data_s_polarisation.txt', skiprows = 1)
glas_spol_angle = glas_spol_data[:,0] # Einfallswinkel in Grad
glas_spol_intensity = glas_spol_data[:,1] # Intensität in willkürlichen Einheiten

glas_ppol_data = np.loadtxt('../part1_tutor/Referenz_Data_p_polarisation.txt', skiprows = 1)
glas_ppol_angle = glas_ppol_data[:,0] # Einfallswinkel in Grad
glas_ppol_intensity = glas_ppol_data[:,1] # Intensität in willkürlichen Einheiten

plt.subplots()
plt.errorbar(glas_spol_angle, glas_spol_intensity, xerr = 0.01, yerr = 0.025, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(glas_ppol_angle, glas_ppol_intensity, xerr = 0.01, yerr = 0.025, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation', color = 'red')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Intensität $I_{\mathrm{ref}}$ / w.E.')
plt.savefig(f"../protokoll/figs/referenz.pdf", dpi = 400)

# Erster Goldfilm
gold1_spol_data = np.loadtxt('../part1_tutor/Probe_1_Data_s_polarisation.txt', skiprows = 1)
gold1_spol_angle = gold1_spol_data[:,0] # Einfallswinkel in Grad
gold1_spol_intensity = gold1_spol_data[:,1] # Intensität in willkürlichen Einheiten

gold1_ppol_data = np.loadtxt('../part1_tutor/Probe_1_Data_p_polarisation.txt', skiprows = 1)
gold1_ppol_angle = gold1_ppol_data[:,0] # Einfallswinkel in Grad
gold1_ppol_intensity = gold1_ppol_data[:,1] # Intensität in willkürlichen Einheiten

plt.subplots()
plt.errorbar(gold1_spol_angle, gold1_spol_intensity, xerr = 0.01, yerr = 0.025, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(gold1_ppol_angle, gold1_ppol_intensity, xerr = 0.01, yerr = 0.025, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation', color = 'red')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Intensität $I_{\mathrm{Au}}$ / w.E.')
plt.savefig(f"../protokoll/figs/au1.pdf", dpi = 400)

# Zweiter Goldfilm
gold2_spol_data = np.loadtxt('../part1_tutor/Probe_2_Data_s_polarisation.txt', skiprows = 1)
gold2_spol_angle = gold2_spol_data[:,0] # Einfallswinkel in Grad
gold2_spol_intensity = gold2_spol_data[:,1] # Intensität in willkürlichen Einheiten

gold2_ppol_data = np.loadtxt('../part1_tutor/Probe_2_Data_p_polarisation.txt', skiprows = 1)
gold2_ppol_angle = gold2_ppol_data[:,0] # Einfallswinkel in Grad
gold2_ppol_intensity = gold2_ppol_data[:,1] # Intensität in willkürlichen Einheiten

plt.subplots()
plt.errorbar(gold2_spol_angle, gold2_spol_intensity, xerr = 0.01, yerr = 0.025, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(gold2_ppol_angle, gold2_ppol_intensity, xerr = 0.01, yerr = 0.025, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation', color = 'red')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Intensität $I_{\mathrm{Au}}$ / w.E.')
plt.savefig(f"../protokoll/figs/au2.pdf", dpi = 400)

# Dritter Goldfilm
gold3_spol_data = np.loadtxt('../part1_tutor/Probe_3_Data_s_polarisation.txt', skiprows = 1)
gold3_spol_angle = gold3_spol_data[:,0] # Einfallswinkel in Grad
gold3_spol_intensity = gold3_spol_data[:,1] # Intensität in willkürlichen Einheiten

gold3_ppol_data = np.loadtxt('../part1_tutor/Probe_3_Data_p_polarisation.txt', skiprows = 1)
gold3_ppol_angle = gold3_ppol_data[:,0] # Einfallswinkel in Grad
gold3_ppol_intensity = gold3_ppol_data[:,1] # Intensität in willkürlichen Einheiten

plt.subplots()
plt.errorbar(gold3_spol_angle, gold3_spol_intensity, xerr = 0.01, yerr = 0.025, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(gold3_ppol_angle, gold3_ppol_intensity, xerr = 0.01, yerr = 0.025, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation', color = 'red')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Intensität $I_{\mathrm{Au}}$ / w.E.')
plt.savefig(f"../protokoll/figs/au3.pdf", dpi = 400)

# Vierter Goldfilm
gold4_spol_data = np.loadtxt('../part1_tutor/Probe_4_Data_s_polarisation.txt', skiprows = 1)
gold4_spol_angle = gold4_spol_data[:,0] # Einfallswinkel in Grad
gold4_spol_intensity = gold4_spol_data[:,1] # Intensität in willkürlichen Einheiten

gold4_ppol_data = np.loadtxt('../part1_tutor/Probe_4_Data_p_polarisation.txt', skiprows = 1)
gold4_ppol_angle = gold4_ppol_data[:,0] # Einfallswinkel in Grad
gold4_ppol_intensity = gold4_ppol_data[:,1] # Intensität in willkürlichen Einheiten

plt.subplots()
plt.errorbar(gold4_spol_angle, gold4_spol_intensity, xerr = 0.01, yerr = 0.025, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(gold4_ppol_angle, gold4_ppol_intensity, xerr = 0.01, yerr = 0.025, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation', color = 'red')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Intensität $I_{\mathrm{Au}}$ / w.E.')
plt.savefig(f"../protokoll/figs/au4.pdf", dpi = 400)

# Fehler Referenzmessung
intensity_glas_spol_1_err = np.max(glas_spol_intensity[100:200])
intensity_glas_spol_2_err = np.min(glas_spol_intensity[100:200])
intensity_glas_ppol_1_err = np.max(glas_ppol_intensity[100:200])
intensity_glas_ppol_2_err = np.min(glas_ppol_intensity[100:200])
print(len(glas_spol_angle))
print(intensity_glas_spol_1_err - intensity_glas_spol_2_err, intensity_glas_ppol_1_err - intensity_glas_ppol_2_err)

# Fehler Gold 1
intensity_gold1_spol_1_err = np.max(gold1_spol_intensity)
intensity_gold1_spol_2_err = np.min(gold1_spol_intensity)
print(intensity_gold1_spol_1_err - intensity_gold1_spol_2_err)

# Fehler Gold 2
intensity_gold2_spol_1_err = np.max(gold2_spol_intensity)
intensity_gold2_spol_2_err = np.min(gold2_spol_intensity)
print(intensity_gold2_spol_1_err - intensity_gold2_spol_2_err)

# Fehler Gold 3
intensity_gold3_spol_1_err = np.max(gold3_spol_intensity)
intensity_gold3_spol_2_err = np.min(gold3_spol_intensity)
print(intensity_gold3_spol_1_err - intensity_gold3_spol_2_err)

# Fehler Gold 4
intensity_gold4_spol_1_err = np.max(gold4_spol_intensity)
intensity_gold4_spol_2_err = np.min(gold4_spol_intensity)
print(intensity_gold4_spol_1_err - intensity_gold4_spol_2_err)

# Gebe Maximalwerte der Referenzmessung aus
print('Maximum s-pol:', np.max(glas_spol_intensity))
print('Maximum p-pol:', np.max(glas_ppol_intensity))

# Anpassung zur Bestimmung des Brechungsindex des Prismas

# s-Polarisation
reflexionsgrad_spol = glas_spol_intensity[1:73]/np.max(glas_spol_intensity)
reflexionsgrad_spol_err = np.sqrt((0.025/np.max(glas_spol_intensity))**2 + ((glas_spol_intensity[1:73]/np.max(glas_spol_intensity)**2)*0.025)**2)

def r_te(a, x):
    return ((a*np.cos(np.deg2rad(x)) - 1.000292*np.sqrt(1-((a/(1.000292))*np.sin(np.deg2rad(x)))**2))/(a*np.cos(np.deg2rad(x)) + 1.000292*np.sqrt(1-((a/(1.000292))*np.sin(np.deg2rad(x)))**2)))**2

func1 = Model(r_te)
mydata_glas_spol = RealData(glas_spol_angle[1:73], reflexionsgrad_spol, sx = np.repeat(0.01, 74), sy = reflexionsgrad_spol_err)
myodr_glas_spol = ODR(mydata_glas_spol, func1, beta0 = [1.5])
myoutput_glas_spol = myodr_glas_spol.run()
myoutput_glas_spol.pprint()

plt.subplots()
plt.errorbar(glas_spol_angle[1:73], reflexionsgrad_spol, xerr = 0.01, yerr = reflexionsgrad_spol_err, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.plot(glas_spol_angle[1:73], r_te(myoutput_glas_spol.beta, glas_spol_angle[1:73]), label = 'Anpassung mit $\\chi_{\\mathrm{res}}^2 = 0,28$')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Reflexionsgrad $R_{\mathrm{TE}}$')
plt.savefig(f"../protokoll/figs/glas_spol_fit.pdf", dpi = 400)

# p-Polarisation
reflexionsgrad_ppol = glas_ppol_intensity[1:73]/np.max(glas_ppol_intensity)
reflexionsgrad_ppol_err = np.sqrt((0.025/np.max(glas_ppol_intensity))**2 + ((glas_ppol_intensity[1:73]/np.max(glas_ppol_intensity)**2)*0.025)**2)

def r_tm(a, x):
    return ((a*np.sqrt(1-((a/(1.000292))*np.sin(np.deg2rad(x)))**2) - 1.000292*np.cos(np.deg2rad(x)))/(a*np.sqrt(1-((a/(1.000292))*np.sin(np.deg2rad(x)))**2) + 1.000292*np.cos(np.deg2rad(x))))**2

func2 = Model(r_tm)
mydata_glas_ppol = RealData(glas_ppol_angle[1:73], reflexionsgrad_ppol, sx = np.repeat(0.01, 75), sy = reflexionsgrad_ppol_err)
myodr_glas_ppol = ODR(mydata_glas_ppol, func2, beta0 = [1.5])
myoutput_glas_ppol = myodr_glas_ppol.run()
myoutput_glas_ppol.pprint()

plt.subplots()
plt.errorbar(glas_ppol_angle[1:73], reflexionsgrad_ppol, xerr = 0.01, yerr = reflexionsgrad_ppol_err, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation')
plt.plot(glas_ppol_angle[1:73], r_tm(myoutput_glas_ppol.beta, glas_ppol_angle[1:73]), label = 'Anpassung mit $\\chi_{\\mathrm{res}}^2 = 0,70$')
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Reflexionsgrad $R_{\mathrm{TM}}$')
plt.savefig(f"../protokoll/figs/glas_ppol_fit.pdf", dpi = 400)

# Anpassung Gold

# definiere Fresnel-Koeffizienten

def rfren(eps1, eps2, k1z, k2z, pol):
    if pol == 'TE':
        r = (k1z - k2z)/(k1z + k2z)
    else:
        r = (eps2*k1z - eps1*k2z)/(eps2*k1z + eps1*k2z)
    return r

def tfren(eps1, eps2, k1z, k2z, pol):
    if pol == 'TE':
        t = (2*k1z)/(k1z + k2z)
    else:
        t = (2*eps2*k1z)/(eps2*k1z + eps1*k2z) * np.sqrt(eps1/eps2)
    return t

# definiere Anpassfunktion

def R_slab(a, alpha):
    wavelength = 785e-9
    alphainrad = np.deg2rad(alpha + a[1])

    epsair = 1.00059
    epsdiel = 2.25
    ndiel = np.sqrt(epsdiel)

    # dielektrische Funktion von Gold
    epsr = -22.854
    epsi = 1.441
    epsm = epsr + 1j*epsi

    k0 = 2*np.pi/wavelength
    kx = k0 * ndiel * np.sin(alphainrad)
    kzdiel = np.lib.scimath.sqrt(k0*k0*epsdiel-kx*kx)
    kzm = np.lib.scimath.sqrt(k0*k0*epsm-kx*kx)
    kzair = np.lib.scimath.sqrt(k0*k0*epsair-kx*kx)

    r12 = rfren(eps1 = epsdiel, eps2 = epsm, k1z = kzdiel, k2z = kzm, pol = 'TM')
    r23 = rfren(eps1 = epsm, eps2 = epsair, k1z = kzm, k2z = kzair, pol = 'TM')
    r21 = -r12
    t12 = tfren(eps1 = epsdiel, eps2 = epsm, k1z = kzdiel, k2z = kzm, pol = 'TM')
    t21 = tfren(eps1 = epsm, eps2 = epsdiel, k1z = kzm, k2z = kzdiel, pol = 'TM')

    r = r12 + (t12*r23*t21*np.exp(2j*kzm*a[0]))/(1 - r21*r23*np.exp(2j*kzm*a[0]))

    return np.abs(r)*np.abs(r)

fit_func = Model(R_slab)

# Goldfilm 1
mydata_gold1_ppol = RealData(gold1_ppol_angle[73:200], gold1_ppol_intensity[73:200]/glas_ppol_intensity[73:200], sx = np.repeat(0.01, 128), sy = np.sqrt((0.025/glas_ppol_intensity[73:200])**2 + ((gold1_ppol_intensity[73:200]/glas_ppol_intensity[73:200]**2)*0.025)**2))
myodr_gold1_ppol = ODR(mydata_gold1_ppol, fit_func, beta0 = [30e-9, 1.0])
myoutput_gold1_ppol = myodr_gold1_ppol.run()
myoutput_gold1_ppol.pprint()

plt.subplots()
plt.errorbar(gold1_ppol_angle[73:200], gold1_ppol_intensity[73:200]/glas_ppol_intensity[73:200], xerr = 0.01, yerr = np.sqrt((0.025/glas_ppol_intensity[73:200])**2 + ((gold1_ppol_intensity[73:200]/glas_ppol_intensity[73:200]**2)*0.025)**2), linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation')
plt.errorbar(gold1_spol_angle[73:200], gold1_spol_intensity[73:200]/glas_spol_intensity[73:200], xerr = 0.01, yerr = np.sqrt((0.025/glas_spol_intensity[73:200])**2 + ((gold1_spol_intensity[73:200]/glas_spol_intensity[73:200]**2)*0.025)**2), linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.plot(gold1_ppol_angle[73:200], R_slab(myoutput_gold1_ppol.beta, gold1_ppol_angle[73:200]), label = 'Anpassung mit $\\chi_{\\mathrm{res}}^2 = 0,52$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13))
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Reflexionsgrad $R(\\theta)$')
plt.savefig(f"../protokoll/figs/gold1_ppol_fit.pdf", dpi = 400)

# Goldfilm 2
mydata_gold2_ppol = RealData(gold2_ppol_angle[73:200], gold2_ppol_intensity[73:200]/glas_ppol_intensity[73:200], sx = np.repeat(0.01, 128), sy = np.sqrt((0.025/glas_ppol_intensity[73:200])**2 + ((gold2_ppol_intensity[73:200]/glas_ppol_intensity[73:200]**2)*0.025)**2))
myodr_gold2_ppol = ODR(mydata_gold2_ppol, fit_func, beta0 = [30e-9, 1.0])
myoutput_gold2_ppol = myodr_gold2_ppol.run()
myoutput_gold2_ppol.pprint()

plt.subplots()
plt.errorbar(gold2_ppol_angle[73:200], gold2_ppol_intensity[73:200]/glas_ppol_intensity[73:200], xerr = 0.01, yerr = np.sqrt((0.025/glas_ppol_intensity[73:200])**2 + ((gold2_ppol_intensity[73:200]/glas_ppol_intensity[73:200]**2)*0.025)**2), linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation')
plt.errorbar(gold2_spol_angle[73:200], gold2_spol_intensity[73:200]/glas_spol_intensity[73:200], xerr = 0.01, yerr = np.sqrt((0.025/glas_spol_intensity[73:200])**2 + ((gold2_spol_intensity[73:200]/glas_spol_intensity[73:200]**2)*0.025)**2), linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.plot(gold2_ppol_angle[73:200], R_slab(myoutput_gold2_ppol.beta, gold2_ppol_angle[73:200]), label = 'Anpassung mit $\\chi_{\\mathrm{res}}^2 = 0,35$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13))
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Reflexionsgrad $R(\\theta)$')
plt.savefig(f"../protokoll/figs/gold2_ppol_fit.pdf", dpi = 400)

# Goldfilm 3
mydata_gold3_ppol = RealData(gold3_ppol_angle[73:200], gold3_ppol_intensity[73:200]/glas_ppol_intensity[73:200], sx = np.repeat(0.01, 128), sy = np.sqrt((0.025/glas_ppol_intensity[73:200])**2 + ((gold3_ppol_intensity[73:200]/glas_ppol_intensity[73:200]**2)*0.025)**2))
myodr_gold3_ppol = ODR(mydata_gold3_ppol, fit_func, beta0 = [30e-9, 1.0])
myoutput_gold3_ppol = myodr_gold3_ppol.run()
myoutput_gold3_ppol.pprint()

plt.subplots()
plt.errorbar(gold3_ppol_angle[73:200], gold3_ppol_intensity[73:200]/glas_ppol_intensity[73:200], xerr = 0.01, yerr = np.sqrt((0.025/glas_ppol_intensity[73:200])**2 + ((gold3_ppol_intensity[73:200]/glas_ppol_intensity[73:200]**2)*0.025)**2), linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(gold3_spol_angle[73:200], gold3_spol_intensity[73:200]/glas_spol_intensity[73:200], xerr = 0.01, yerr = np.sqrt((0.025/glas_spol_intensity[73:200])**2 + ((gold3_spol_intensity[73:200]/glas_spol_intensity[73:200]**2)*0.025)**2), linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation')
plt.plot(gold3_ppol_angle[73:200], R_slab(myoutput_gold3_ppol.beta, gold3_ppol_angle[73:200]), label = 'Anpassung mit $\\chi_{\\mathrm{res}}^2 = 0,34$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13))
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Reflexionsgrad $R(\\theta)$')
plt.savefig(f"../protokoll/figs/gold3_ppol_fit.pdf", dpi = 400)

# Goldfilm 4
mydata_gold4_ppol = RealData(gold4_ppol_angle[73:200], gold4_ppol_intensity[73:200]/glas_ppol_intensity[73:200], sx = np.repeat(0.01, 128), sy = np.sqrt((0.025/glas_ppol_intensity[73:200])**2 + ((gold4_ppol_intensity[73:200]/glas_ppol_intensity[73:200]**2)*0.025)**2))
myodr_gold4_ppol = ODR(mydata_gold4_ppol, fit_func, beta0 = [30e-9, 1.0])
myoutput_gold4_ppol = myodr_gold4_ppol.run()
myoutput_gold4_ppol.pprint()

plt.subplots()
plt.errorbar(gold4_ppol_angle[73:200], gold4_ppol_intensity[73:200]/glas_ppol_intensity[73:200], xerr = 0.01, yerr = np.sqrt((0.025/glas_ppol_intensity[73:200])**2 + ((gold4_ppol_intensity[73:200]/glas_ppol_intensity[73:200]**2)*0.025)**2), linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TE-Polarisation')
plt.errorbar(gold4_spol_angle[73:200], gold4_spol_intensity[73:200]/glas_spol_intensity[73:200], xerr = 0.01, yerr = np.sqrt((0.025/glas_spol_intensity[73:200])**2 + ((gold4_spol_intensity[73:200]/glas_spol_intensity[73:200]**2)*0.025)**2), linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = 'TM-Polarisation')
plt.plot(gold4_ppol_angle[73:200], R_slab(myoutput_gold4_ppol.beta, gold4_ppol_angle[73:200]), label = 'Anpassung mit $\\chi_{\\mathrm{res}}^2 = 0,37$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13))
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Reflexionsgrad $R(\\theta)$')
plt.savefig(f"../protokoll/figs/gold4_ppol_fit.pdf", dpi = 400)

# Versuchsteil 2

# Darstellung der Rohdaten - eigene Messung

# TE-Polarisation
silber_spol1_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_41_90.txt', skiprows = 4)
silber_spol1_angle = silber_spol1_data[:,0] # Einfallswinkel in Grad
silber_spol1_intensity = silber_spol1_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol2_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_41_96.txt', skiprows = 4)
silber_spol2_angle = silber_spol2_data[:,0] # Einfallswinkel in Grad
silber_spol2_intensity = silber_spol2_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol3_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_3.txt', skiprows = 4)
silber_spol3_angle = silber_spol3_data[:,0] # Einfallswinkel in Grad
silber_spol3_intensity = silber_spol3_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol4_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_9.txt', skiprows = 4)
silber_spol4_angle = silber_spol4_data[:,0] # Einfallswinkel in Grad
silber_spol4_intensity = silber_spol4_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol5_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_14.txt', skiprows = 4)
silber_spol5_angle = silber_spol5_data[:,0] # Einfallswinkel in Grad
silber_spol5_intensity = silber_spol5_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol6_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_21.txt', skiprows = 4)
silber_spol6_angle = silber_spol6_data[:,0] # Einfallswinkel in Grad
silber_spol6_intensity = silber_spol6_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol7_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_27.txt', skiprows = 4)
silber_spol7_angle = silber_spol7_data[:,0] # Einfallswinkel in Grad
silber_spol7_intensity = silber_spol7_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol8_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_32.txt', skiprows = 4)
silber_spol8_angle = silber_spol8_data[:,0] # Einfallswinkel in Grad
silber_spol8_intensity = silber_spol8_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol9_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_39.txt', skiprows = 4)
silber_spol9_angle = silber_spol9_data[:,0] # Einfallswinkel in Grad
silber_spol9_intensity = silber_spol9_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol10_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_45.txt', skiprows = 4)
silber_spol10_angle = silber_spol10_data[:,0] # Einfallswinkel in Grad
silber_spol10_intensity = silber_spol10_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol11_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_50.txt', skiprows = 4)
silber_spol11_angle = silber_spol11_data[:,0] # Einfallswinkel in Grad
silber_spol11_intensity = silber_spol11_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol12_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_57.txt', skiprows = 4)
silber_spol12_angle = silber_spol12_data[:,0] # Einfallswinkel in Grad
silber_spol12_intensity = silber_spol12_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol13_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_63.txt', skiprows = 4)
silber_spol13_angle = silber_spol13_data[:,0] # Einfallswinkel in Grad
silber_spol13_intensity = silber_spol13_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol14_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_68.txt', skiprows = 4)
silber_spol14_angle = silber_spol14_data[:,0] # Einfallswinkel in Grad
silber_spol14_intensity = silber_spol14_data[:,1] # Intensität in willkürlichen Einheiten

silber_spol15_data = np.loadtxt('../part2/Ag_0grad_spol/spectrum_42_75.txt', skiprows = 4)
silber_spol15_angle = silber_spol15_data[:,0] # Einfallswinkel in Grad
silber_spol15_intensity = silber_spol15_data[:,1] # Intensität in willkürlichen Einheiten

# TM-Polarisation
silber_ppol1_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_41_90.txt', skiprows = 4)
silber_ppol1_angle = silber_ppol1_data[:,0] # Einfallswinkel in Grad
silber_ppol1_intensity = silber_ppol1_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol2_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_41_96.txt', skiprows = 4)
silber_ppol2_angle = silber_ppol2_data[:,0] # Einfallswinkel in Grad
silber_ppol2_intensity = silber_ppol2_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol3_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_3.txt', skiprows = 4)
silber_ppol3_angle = silber_ppol3_data[:,0] # Einfallswinkel in Grad
silber_ppol3_intensity = silber_ppol3_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol4_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_9.txt', skiprows = 4)
silber_ppol4_angle = silber_ppol4_data[:,0] # Einfallswinkel in Grad
silber_ppol4_intensity = silber_ppol4_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol5_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_14.txt', skiprows = 4)
silber_ppol5_angle = silber_ppol5_data[:,0] # Einfallswinkel in Grad
silber_ppol5_intensity = silber_ppol5_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol6_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_21.txt', skiprows = 4)
silber_ppol6_angle = silber_ppol6_data[:,0] # Einfallswinkel in Grad
silber_ppol6_intensity = silber_ppol6_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol7_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_27.txt', skiprows = 4)
silber_ppol7_angle = silber_ppol7_data[:,0] # Einfallswinkel in Grad
silber_ppol7_intensity = silber_ppol7_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol8_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_32.txt', skiprows = 4)
silber_ppol8_angle = silber_ppol8_data[:,0] # Einfallswinkel in Grad
silber_ppol8_intensity = silber_ppol8_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol9_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_39.txt', skiprows = 4)
silber_ppol9_angle = silber_ppol9_data[:,0] # Einfallswinkel in Grad
silber_ppol9_intensity = silber_ppol9_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol10_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_45.txt', skiprows = 4)
silber_ppol10_angle = silber_ppol10_data[:,0] # Einfallswinkel in Grad
silber_ppol10_intensity = silber_ppol10_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol11_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_50.txt', skiprows = 4)
silber_ppol11_angle = silber_ppol11_data[:,0] # Einfallswinkel in Grad
silber_ppol11_intensity = silber_ppol11_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol12_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_57.txt', skiprows = 4)
silber_ppol12_angle = silber_ppol12_data[:,0] # Einfallswinkel in Grad
silber_ppol12_intensity = silber_ppol12_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol13_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_63.txt', skiprows = 4)
silber_ppol13_angle = silber_ppol13_data[:,0] # Einfallswinkel in Grad
silber_ppol13_intensity = silber_ppol13_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol14_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_68.txt', skiprows = 4)
silber_ppol14_angle = silber_ppol14_data[:,0] # Einfallswinkel in Grad
silber_ppol14_intensity = silber_ppol14_data[:,1] # Intensität in willkürlichen Einheiten

silber_ppol15_data = np.loadtxt('../part2/Ag_45grad_ppol/spectrum_42_75.txt', skiprows = 4)
silber_ppol15_angle = silber_ppol15_data[:,0] # Einfallswinkel in Grad
silber_ppol15_intensity = silber_ppol15_data[:,1] # Intensität in willkürlichen Einheiten

# Plot Rohdaten TE-Polarisation
plt.subplots()
plt.errorbar(silber_spol1_angle, silber_spol1_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '41,90°')
plt.errorbar(silber_spol2_angle, silber_spol2_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '41,96°')
plt.errorbar(silber_spol3_angle, silber_spol3_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,03°')
plt.errorbar(silber_spol4_angle, silber_spol4_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,09°')
plt.errorbar(silber_spol5_angle, silber_spol5_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,14°')
plt.errorbar(silber_spol6_angle, silber_spol6_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,21°')
plt.errorbar(silber_spol7_angle, silber_spol7_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,27°')
plt.errorbar(silber_spol8_angle, silber_spol8_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,32°')
plt.errorbar(silber_spol9_angle, silber_spol9_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,39°')
plt.errorbar(silber_spol10_angle, silber_spol10_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,45°')
plt.errorbar(silber_spol11_angle, silber_spol11_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,50°')
plt.errorbar(silber_spol12_angle, silber_spol12_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,57°')
plt.errorbar(silber_spol13_angle, silber_spol13_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,63°')
plt.errorbar(silber_spol14_angle, silber_spol14_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,68°')
plt.errorbar(silber_spol15_angle, silber_spol15_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,75°')
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('Wellenlänge $\\lambda$ / nm')
plt.ylabel('Intensität $I_{\mathrm{Ag}}$ / w.E.')
plt.savefig(f"../protokoll/figs/silber_te.pdf", dpi = 400)

# Plot Rohdaten TM-Polarisation
plt.subplots()
plt.errorbar(silber_ppol1_angle, silber_ppol1_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '41,90°')
plt.errorbar(silber_ppol2_angle, silber_ppol2_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '41,96°')
plt.errorbar(silber_ppol3_angle, silber_ppol3_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,03°')
plt.errorbar(silber_ppol4_angle, silber_ppol4_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,09°')
plt.errorbar(silber_ppol5_angle, silber_ppol5_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,14°')
plt.errorbar(silber_ppol6_angle, silber_ppol6_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,21°')
plt.errorbar(silber_ppol7_angle, silber_ppol7_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,27°')
plt.errorbar(silber_ppol8_angle, silber_ppol8_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,32°')
plt.errorbar(silber_ppol9_angle, silber_ppol9_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,39°')
plt.errorbar(silber_ppol10_angle, silber_ppol10_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,45°')
plt.errorbar(silber_ppol11_angle, silber_ppol11_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,50°')
plt.errorbar(silber_ppol12_angle, silber_ppol12_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,57°')
plt.errorbar(silber_ppol13_angle, silber_ppol13_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,63°')
plt.errorbar(silber_ppol14_angle, silber_ppol14_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,68°')
plt.errorbar(silber_ppol15_angle, silber_ppol15_intensity, xerr = 0.2, linestyle = 'None', marker = '.', linewidth = 0.5, markersize = 0.5, label = '42,75°')
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('Wellenlänge $\\lambda$ / nm')
plt.ylabel('Intensität $I_{\mathrm{Ag}}$ / w.E.')
plt.savefig(f"../protokoll/figs/silber_tm.pdf", dpi = 400)

# Dispersionsrelation

n_prisma = 1.5
theta = np.array([42.75, 42.68, 42.63, 42.57, 42.50, 42.45, 42.39, 42.32, 42.27, 42.21, 42.14, 42.09, 42.03, 41.96, 41.90])
lambda_opp = np.array([562, 568, 577, 584, 590, 596, 604, 613, 623, 634, 644, 654, 669, 685, 703])
lambda_opp_err = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4])
c_0 = 299792458 * 10**(9) # in nm/s

omega_opp = (2*np.pi*c_0)/lambda_opp
k_senk = (2*np.pi*n_prisma)/(lambda_opp) * np.sin(np.deg2rad(theta))

omega_opp_err = (2*np.pi*c_0)/(lambda_opp**2) * lambda_opp_err
k_senk_err = np.sqrt(((2*np.pi*n_prisma)/(lambda_opp**2) * np.sin(np.deg2rad(theta)) * lambda_opp_err)**2 + ((2*np.pi*n_prisma)/(lambda_opp) * np.cos(np.deg2rad(theta)) * np.deg2rad(0.06))**2)

plt.subplots()
plt.errorbar(k_senk, omega_opp, xerr = k_senk_err, yerr = omega_opp_err, marker = '.', linewidth = 0.5, markersize = 0.5, label = 'OPP-Dispersionsrelation')
plt.plot(k_senk, c_0 * k_senk, label = 'Lichtlinie')
plt.legend()
plt.xlabel('$k_{||}$ / $\mathrm{nm}^{-1}$')
plt.ylabel('$\\omega_{\\mathrm{OPP}}(k_{||})$ / $\mathrm{s}^{-1}$')
plt.savefig(f"../protokoll/figs/dispersionsrelation_opp.pdf", dpi = 400)

print(omega_opp, k_senk)
print(omega_opp_err, k_senk_err)