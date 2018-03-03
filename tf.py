from copy import copy

from numpy import array, asmatrix, dot, matrix

from scipy.signal import TransferFunction
from scipy.signal.filter_design import normalize

from sympy import exp, N, poly, symbols
from sympy.solvers import solve

from IPython.display import display, Latex, Math
from sympy import init_printing
from sympy.printing import latex

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
        """

        return self.numerator.subs('s', s)/self.denominator.subs('s', s) \
            * N(exp(-s*self.deadtime))

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

        if not self.deadtime == other.deadtime:
            if not self.zerogain or other.zerogain:
                raise("Adding tfs with different deadtimes not supported.")

        ntf = self.numerator/self.denominator \
            + other.numerator/other.denominator

        ntf_sympy_num = ntf.as_numer_denom()[0].expand()
        ntf_sympy_den = ntf.as_numer_denom()[1].expand()

        ntf_num, ntf_den = \
            poly(ntf_sympy_num).all_coeffs(), \
            poly(ntf_sympy_den).all_coeffs()

        # This casting is necessary because the TransferFunction base class
        # cannot handle sympy.Float objects

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

        ntf_num, ntf_den = \
            poly(ntf_sympy_num).all_coeffs(), \
            poly(ntf_sympy_den).all_coeffs()

        # This casting is necessary because the TransferFunction base class
        # cannot handle sympy.Float objects

        ntf_num = [float(coeff) for coeff in ntf_num]
        ntf_den = [float(coeff) for coeff in ntf_den]

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