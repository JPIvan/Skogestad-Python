from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

from utils import feedback, tf, marginsclosedloop, ControllerTuning, maxpeak
from utilsplot import step_response_plot, bodeclosedloop

from plotformat import *

s = tf([1, 0], 1)
G = 3*(-2*s + 1)/((10*s + 1)*(5*s + 1))

_, _, Ku, Pu = ControllerTuning(G, method='ZN')
Kc = Ku/2.2
Taui = Pu/1.2
print('Kc: {:.4f}'.format(Kc))
print('Taui: {:.4f}'.format(Taui))
print('Ku: {:.4f}'.format(Ku))
print('Pu: {:.4f}'.format(Pu))

K = Kc*(1 + 1/(Taui*s))
L = G * K
T = feedback(L, 1)
S = feedback(1, L)
u = S * K

plt.figure('Figure 2.8')
step_response_plot(T, u, 80, 0)
plt.show()

bpf = bode_plot_format(
    fig_title="Example 2.4",
    ax_title_mag="",
#    xlabel_mag="$\omega_{180}$",
    ylabel_mag="Magnitude",
    ax_title_phase="",
    xlabel_phase="Frequency $[rad/s]$",
    ylabel_phase="Phase",
    grid=True,
    legend=True,
    subplotadj_hspace=0.25
)

G.bodeclosedloop_plot(G, K=K, w=(-2, 1), n=100, printmargins=True, bodeplotformat=bpf)
"""
GM, PM, wc, wb, wbt, valid = marginsclosedloop(L)
Mt = maxpeak(T)
Ms = maxpeak(S)
print('GM: ', np.round(GM, 2))
print('PM: ', np.round(PM, 1), "deg or", np.round(PM / 180 * np.pi, 2), "rad")
print('wb: ', np.round(wb, 2))
print('wc: ', np.round(wc, 2))
print('wbt:', np.round(wbt, 2))
print('Ms: ', np.round(Ms, 2))
print('Mt: ', np.round(Mt, 2))

if valid:
    print("Frequency range wb < wc < wbt is valid")
else:
    print("Frequency range wb < wc < wbt is not valid")
"""

#TODO fix commented part