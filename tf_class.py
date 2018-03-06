from copy import copy

from numpy import array, asmatrix, dot, matrix

from scipy.signal import TransferFunction
from scipy.signal.filter_design import normalize
from scipy.signal import step as __scipy_signal_step__
from scipy.signal import bode as __scipy_signal_bode__

from sympy import exp, N, poly, symbols, sympify, Abs, pi
from sympy.solvers import solve

from IPython.display import display, Latex, Math
from sympy import init_printing
from sympy.printing import latex

import matplotlib.pyplot as plot

from utilsplot import plot_setfontsizes, plot_doformatting

init_printing()

def iterablepoly_to_sympypoly(iterable_poly, coeff_order='high-to-low'):

    s = symbols('s')

    poly_order = len(iterable_poly)
    sympy_poly = 0

    if coeff_order == 'high-to-low':
        coeff_order = 1
    elif coeff_order == 'low-to-high':
        coeff_order = 0
    else:
        raise("coeff_order must be 'low-to-high' or 'high-to-low'")

    for power, coeff in enumerate(iterable_poly):
        sympy_poly += coeff*s**(poly_order-(1+power)*coeff_order)

    return sympy_poly

def sympypoly_to_iterablepoly(sympy_poly, coeff_order='high-to-low'):
    sympy_numberobjects = (
        type(sympify(0)),
        type(sympify(1)),
        type(sympify(2)),
        type(sympify(0.0))
    )
    
    if type(sympy_poly) in sympy_numberobjects:
        iterable_poly = [sympy_poly] 

        # Creating this list like this is necessary because sympy does
        # not define generators to create iterables for these objects
    else:
        iterable_poly = poly(sympy_poly).all_coeffs()
        if coeff_order == 'low-to-high':
            iterable_poly.reverse()

    iterable_poly = [float(coeff) for coeff in iterable_poly]

    # This casting is necessary because the TransferFunction base class
    # cannot handle sympy.Float objects

    return iterable_poly

def __margins__(G):
    """
    Should only be used in functions that plot bode plots, tf class
    implements this.

    Calculates the gain and phase margins, together with the gain and phase
    crossover frequency for a plant model

    Parameters
    ----------
    G : tf
        plant model

    Returns
    -------
    GM : array containing a real number
        gain margin
    PM : array containing a real number
        phase margin
    wc : array containing a real number
        gain crossover frequency where |G(jwc)| = 1
    w_180 : array containing a real number
        phase crossover frequency where angle[G(jw_180] = -180 deg
    """

    Gw = freq(G)

    def mod(x):
        """to give the function to calculate |G(jw)| = 1"""
        return numpy.abs(Gw(x[0])) - 1

    # how to calculate the freqeuncy at which |G(jw)| = 1
    wc = optimize.fsolve(mod, 0.1)

    def arg(w):
        """function to calculate the phase angle at -180 deg"""
        return numpy.angle(G.w(w[0])) + pi

    # where the freqeuncy is calculated where arg G(jw) = -180 deg
    w_180 = optimize.fsolve(arg, wc)

    PM = numpy.angle(G.w(wc), deg=True) + 180
    GM = 1/G.mod(w_180)

    return GM, PM, wc, w_180

class tf(TransferFunction):
    """
        Creates a Transfer Function object with methos inherited from the
        scipy.signal library. Methods to allow arithmetic operations have
        been added.

        Addition
        Multiplication
        Division
        Potentiation

        Dead Time is supported.
        Addition of tfs with different deadtimes is not supported.

        The base TransferFunction class uses iterable representations of the
        numerator and denominator polynomial.

        In order to resolve isses with root finding this class uses sympy
        polynomials to represent the numerator and denominator which have
        been tested and shown to have better root finding algorithms than
        those of the numpy.poly1d and numpy.polynomial modules.

        Several methods have been added to allow compatibility with functions
        written before 2018 Semester 1. These are indicated as such and
        should be phased out.

        https://github.com/scipy/scipy/tree/master/scipy/signal

        Selected Inherited Attributes:
            num (numpy.ndarray): Numerator polynomial representation for base
                class.
            den (numpy.ndarray): Denominator polynomial representation for
                base class.
            dt (float): Sampling time

        Attributes:
            numerator (iterable): Numerator polynomial
            denominator (iterable): Denominator polynomial
            deadtime (int): #TODO
            zerogain (bool): #TODO
            name (str): #TODO
            u (?): #TODO
            y (?): #TODO

        Methods:
            simplify(): #TODO

        #TODO:
            Determine from num and den if a tf is zero gain.
            Add non-integer potentiation.
            Verify that all arithmetic operations return *new* objects and
                not references to old ones which have been modified.
    """

    def __init__(self, numerator, denominator=[1], deadtime=0, name='',
                 u='', y='', prec=3):

        super().__init__(numerator, denominator)
        self.simplify()

        self.numerator = iterablepoly_to_sympypoly(self.num)
        self.denominator = iterablepoly_to_sympypoly(self.den)
        self.deadtime = deadtime
        self.zerogain = False
        self.name = name
        self.u = u
        self.y = y

        self.iter_index = 0

    def __call__(self, s):
        """
            This allows the transfer function to be evaluated at
            particular values of s. Should support substitution with sympy
            expressions but has not been extensively tested.

            Effectively, this makes a tf object behave just like a function
            of s.

            TODO: Handle numpy arrays
        """
        try:
            return self.numerator.subs('s', s)/self.denominator.subs('s', s) \
            * N(exp(-s*self.deadtime))
        except:
            print(s)
            print(type(s))

    def __repr__(self):
        """
            Return representation of the system's transfer function,
            overrides TransferFunction's __repr__ function to include
            displaying the sympy numerator and denominator polynomials.
        """

        return ('{0}(\n\t{1},\n\t{2},\n\tdt: {3},\n\tnumerator: {4},'
                + '\n\tdenominator: {5}\n\tdeadtime: {6}\n)').format(
            self.__class__.__name__,
            repr(self.num),
            repr(self.den),
            repr(self.dt),
            repr(self.numerator),
            repr(self.denominator),
            repr(self.deadtime)
        )

    def __add__(self, other):
        """
            Adds two tf objects.

            Adding objects with different deadtimes is not supported.
        """

        if not isinstance(other, tf):
            other = tf(other)

        if not self.deadtime == other.deadtime:
            if not self.zerogain or other.zerogain:
                raise("Adding tfs with different deadtimes not supported.")

        ntf = self.numerator/self.denominator \
            + other.numerator/other.denominator

        ntf_sympy_num = ntf.as_numer_denom()[0].expand()
        ntf_sympy_den = ntf.as_numer_denom()[1].expand()

        ntf_num = sympypoly_to_iterablepoly(ntf_sympy_num)
        ntf_den = sympypoly_to_iterablepoly(ntf_sympy_den)

        ntf_num = [float(coeff) for coeff in ntf_num]
        ntf_den = [float(coeff) for coeff in ntf_den]

        return tf(ntf_num, ntf_den, deadtime=self.deadtime)

    def __radd__(self, other):
        """
            Right-handed addition, calls __add__ since tfs are commutative.
        """

        return self + other

    def __iter__(self):
        """
            Iteration defined to allow unpacking of numerator and denominator
            polynomial for calls to scipy.signal functions. For example:

            scipy.signal.bode(G)

            Which requires unpacking ( *G ) to be possible.
        """

        return self

    def __next__(self):
        if self.iter_index == 0:
            self.iter_index += 1
            return self.num
        elif self.iter_index == 1:
            self.iter_index += 1
            return self.den
        else:
            self.iter_index = 0
            raise StopIteration

    def __mul__(self, other):
        """
            Multiplies two tf objects.
        """

        if isinstance(other, matrix):
            return dot(other, self)
        else:
            other = tf(other)

        ntf = self.numerator*other.numerator \
            / self.denominator/other.denominator

        ntf_sympy_num = ntf.as_numer_denom()[0].expand()
        ntf_sympy_den = ntf.as_numer_denom()[1].expand()

        ntf_num = sympypoly_to_iterablepoly(ntf_sympy_num)
        ntf_den = sympypoly_to_iterablepoly(ntf_sympy_den)

        return tf(ntf_num, ntf_den, deadtime=self.deadtime+other.deadtime)

    def __rmul__(self, other):
        """
            Right-handed multiplication, calls __mul__ since tfs are
            commutative.
        """

        return self * other

    def __neg__(self):
        return tf(-self.num, self.den, self.deadtime)

    def __pow__(self, other):
        r = copy(self)

        if type(other) == int:
            if other < 0:
                r = r.inverse()**-other

            for k in range(other-1):
                r = r * self
        elif type(other) == float:
            raise("Non-integer exponents not implemented.")
        elif type(other) == complex:
            raise("Non-integer exponents not implemented.")
        else:
            raise("Exponent must be int, float or complex.")

        return tf(r.num, r.den, r.deadtime)

    def __sub__(self, other):
        """
            Subtracts two tf objects, calls __add__ and __mul__.
        """

        return self + (-other)

    def __rsub__(self, other):
        """
            Right-handed subtraction, calls __sub__ since tfs are
            commutative.
        """

        return other + (-self)

    def __truediv__(self, other):
        """
            Divides a tf object by another tf object or an object that can
            be used to create a tf object. Calls inverse() and __mul__.
        """

        if not isinstance(other, tf):
            other = tf(other)
        return self * other.inverse()

    def __rtruediv__(self, other):
        """
            Divides a tf object  or an object that can be used to create a tf
            object by a tf object. Calls inverse() and __mul__.
        """

        if not isinstance(other, tf):
            other = tf(other)
        return other/self

    def display(self):
        """
            Prints LaTeX representation of the transfer function.
        """

        s = symbols('s')

        display(
            Math(
                latex(self.numerator/self.denominator*exp(-self.deadtime*s))
            )
        )

    def inverse(self):
        return tf(self.den, self.num, -self.deadtime)

    def poles(self):
        return [N(root) for root in solve(self.denominator)]

    def zeros(self):
        return [N(root) for root in solve(self.numerator)]

    def simplify(self, dec=None):
        """
            TODO: dec parameter is legacy, remove dependencies.

            Most operations call simpify on the result, so it shouldn't be
            necessary to call this outside of class methods.
        """

        self.num, self.den = normalize(
            self.num, self.den
        )

    def w(self, w):
        """
        Returns the tf at a given frequency.

        Casting to a complex is performed to avoid type issues in the caller

        Parameters
        ----------
        w : frequency

        Returns
        -------
        G(jw) : tf response
        """
        return complex(self(1j*w))

    def mod(self, w):
        """
        Returns the magnitude of the tf at a given frequency.

        sympy.abs is used for compatibility with the current implementation
        of the tf class
        Casting to a float is performed to avoid type issues in the caller

        Parameters
        ----------
        w : frequency

        Returns
        -------
        |G(jw)| : magnitude of tf response
        """
        return float(Abs(self.w(w)))
        
    def step(system, X0=None, T=None, N=None):
        """
            Defined for compatibility with function that called the old
            tf class.

            New code should use scipy.signal.step
        """

        return __scipy_signal_step__(system, X0, T, N)
    
    def bode_plot(self, w=None, n=100):
        """
            Plot a bode plot of the transfer function using scipy.signal.bode()

            TODO: add gain and phase margin displays
        """

        w, mag, phase = __scipy_signal_bode__(self, w, n)
        GM, PM, wc, w_180 = __margins__(self)

        plot_setfontsizes()
        fig = plot.figure(figsize=(16, 9))
        ax = fig.add_subplot(2, 1, 1)
        
        ax.semilogx(w, mag)    # Bode magnitude plot
        
        plot_doformatting(
            ax,
            fig=fig,
            fig_title="Example 2.3, Figure 2.7",
            ax_title="",
            xlabel="$\omega_{180}$",
            ylabel="Magnitude",
            xlim=(min(w), max(w)),
            ylim=(min(mag), max(mag)),
            grid=True,
            legend=True,
            spadj_hspc=0.25
        )
        
        ax = fig.add_subplot(2, 1, 2)

        plot_doformatting(
            ax,
            fig=fig,
            fig_title="Example 2.3, Figure 2.7",
            ax_title="",
            xlabel="Frequency $[rad/s]$",
            ylabel="Phase",
            xlim=(min(w), max(w)),
            ylim=(min(phase), max(phase)),
            grid=True,
            legend=True
        )

        ax.semilogx(w, phase)  # Bode phase plot
        
        plot.show()
        print("Gain Margin: ", GM)
        print("Phase Margin: ", PM)
