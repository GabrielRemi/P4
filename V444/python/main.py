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
plt.legend()
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
plt.legend()
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
plt.legend()
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
plt.legend()
plt.xlabel('Einfallswinkel $\\theta$ / °')
plt.ylabel('Reflexionsgrad $R(\\theta)$')
plt.savefig(f"../protokoll/figs/gold4_ppol_fit.pdf", dpi = 400)