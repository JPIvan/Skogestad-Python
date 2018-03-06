from numpy import logspace
from sympy import solve, symbols, im, Poly, And

from utils import tf


# Define system

s = tf([1, 0], [1])
G = 3*(-2*s + 1)/((10*s + 1)*(5*s + 1))

# Method 1

print("Using Method 1:")

s, Kc = symbols('s, Kc')
e = G.denominator + Kc*G.numerator

if e.expand().coeff(s**2) > 0:
    s2_coeff = 1  # Positive coefficient for s**2
else:
    s2_coeff = -1  # Negative coefficient for s**2

p = Poly(e, s)

bounds = []
for coeff in p.all_coeffs()[1:]:
    bounds.append(solve(coeff > 0))

bounds = And(*bounds).as_set()

unstable_freq = solve(e.subs('Kc', bounds.end), s)

print("Stable range for Kc: ", bounds)
print("Frequency at Kc = {:5f}: {:5f}".format(
    float(bounds.end),
    complex(unstable_freq[0])
))

# Method 2

print("\nUsing Method 2:")

_, _, _, w_180 = G.bode_plot(w=logspace(-2, 1, 1000), printmargins=True)
print("\n|L(jw_180)| = {:5f}Kc".format(G.mod(w_180)))
print("Therefore stable for Kc < 2.5")
