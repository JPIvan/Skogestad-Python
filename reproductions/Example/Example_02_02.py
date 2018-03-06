import matplotlib.pyplot as plot
from numpy import linspace

from utils import feedback, tf
from plotformat import *

from scipy.signal import step

# Define system

s = tf([1, 0], [1])
G = 3*(-2*s + 1)/((10*s + 1)*(5*s + 1))

# Prepare data for figure plotting and formatting

gains = (0.5, 1.5, 2.5, 3)
plotcolours = [[1-c, 0.25, c] for c in linspace(1, 0, 4)]
labels = ["K = "+str(K) for K in gains]
linestyles = ('-', '-', '-', '-.')

t_in = linspace(0, 50, 400)

# Plot

plot_setfontsizes()
fig = plot.figure(figsize=(16, 9))
ax = fig.add_subplot(111)

for K, c, l, ls in zip(gains, plotcolours, labels, linestyles):
    t_out, y_out = feedback(K*G, 1).step(0, t_in)
    ax.plot(t_out, y_out, label=l, linestyle=ls, color=c)

plotformat = plot_format(
    fig=fig,
    fig_title="Example 2.2, Figure 2.6",
    ax_title="Effect of Proportional Gain Kc on Closed Loop Response",
    xlabel="Response",
    ylabel="Time (s)",
    xlim=(0, 50),
    ylim=(-0.5, 2.5),
    grid=True,
    legend=True
)
plot_doformatting(ax, plotformat)

plot.show()

print("For system: ", G)
print("Poles: ", G.poles())
print("Zeros: ", G.zeros())
