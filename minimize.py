from math import *
from numpy import dot, isinf, isnan, any, sqrt, isreal, real, nan, inf
from CG_MNIST import CG_MNIST

def minimize(X, f0, df0, length, dim, data, red=1.0, verbose=True):
    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 10
    SIG = 0.1
    RHO = SIG/2

    SMALL = 10.**-16

    i = 0                                         # zero the run length counter
    ls_failed = 0                          # no previous line search has failed
    # f0 = f(X, *args)                          # get function value and gradient
    # df0 = grad(X, *args)  
    fX = [f0]
    i = i + (length<0)                                         # count epochs?!
    s = -df0; d0 = -dot(s,s.T)    # initial search direction (steepest) and slope
    x3 = red/(1.0-d0)                             # initial step is red/(|s|+1)

    while i < abs(length):                                 # while not finished
        i = i + (length>0)                                 # count iterations?!

        X0 = X; F0 = f0; dF0 = df0              # make a copy of current values
        if length>0:
            M = MAX
        else: 
            M = min(MAX, -length-i)
        ME = 1
        while ME!=0:                      # keep extrapolating as long as necessary
            x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
            success = 0
            while (not success) and (M > 0):
                try:
                    M = M - 1; i = i + (length<0)              # count epochs?!
                    # f3 = f(X+x3*s, *args)
                    # df3 = grad(X+x3*s, *args)
                    f3, df3 = CG_MNIST(X+x3*s, dim, data) 
                    if isnan(f3) or isinf(f3) or any(isnan(df3)+isinf(df3)):
                        print("error")
                    success = 1
                except:                    # catch any error which occured in f
                    x3 = (x2+x3)/2                       # bisect and try again
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3   # keep best values
            d3 = dot(df3,s.T)                                         # new slope
            if (d3 > SIG*d0)[0][0] or (f3 > f0+x3*RHO*d0)[0][0] or M == 0:  
                ME = 0
            x1 = x2; f1 = f2; d1 = d2                 # move point 2 to point 1
            x2 = x3; f2 = f3; d2 = d3                 # move point 3 to point 2
            A = 6*(f1-f2)+3*(d2+d1)*(x2-x1)          # make cubic extrapolation
            B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
            Z = B+sqrt(complex(B*B-A*d1*(x2-x1)))
            if Z != 0.0:
                x3 = x1-d1*(x2-x1)**2/Z              # num. error possible, ok!
            else: 
                x3 = inf
            if (not isreal(x3)) or isnan(x3) or isinf(x3) or (x3 < 0): 
                                                       # num prob | wrong sign?
                x3 = x2*EXT                        # extrapolate maximum amount
            elif x3 > x2*EXT:           # new point beyond extrapolation limit?
                x3 = x2*EXT                        # extrapolate maximum amount
            elif x3 < x2+INT*(x2-x1):  # new point too close to previous point?
                x3 = x2+INT*(x2-x1)
            x3 = real(x3)

        while (abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0:  
                                                           # keep interpolating
            if (d3 > 0) or (f3 > f0+x3*RHO*d0):            # choose subinterval
                x4 = x3; f4 = f3; d4 = d3             # move point 3 to point 4
            else:
                x2 = x3; f2 = f3; d2 = d3             # move point 3 to point 2
            if f4 > f0:           
                x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2))
                                                      # quadratic interpolation
            else:
                A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)           # cubic interpolation
                B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
                if A != 0:
                    x3=x2+(sqrt(B*B-A*d2*(x4-x2)**2)-B)/A
                                                     # num. error possible, ok!
                else:
                    x3 = inf
            if isnan(x3) or isinf(x3):
                x3 = (x2+x4)/2      # if we had a numerical problem then bisect
            x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))  
                                                       # don't accept too close
            f3, df3 = CG_MNIST(X+x3*s, dim, data)
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3              # keep best values
            M = M - 1; i = i + (length<0)                      # count epochs?!
            d3 = dot(df3,s.T)                                         # new slope

        if abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:  # if line search succeeded
            X = X+x3*s; f0 = f3; fX.append(f0)               # update variables
            if verbose: print('%s %6i;  Value %4.6e\r' % (S, i, f0))
            s = (dot(df3,df3)-dot(df0,df3))/dot(df0,df0)*s - df3
                                                  # Polack-Ribiere CG direction
            df0 = df3                                        # swap derivatives
            d3 = d0; d0 = dot(df0,s)
            if d0 > 0:                             # new slope must be negative
                s = -df0; d0 = -dot(s,s)     # otherwise use steepest direction
            x3 = x3 * min(RATIO, d3/(d0-SMALL))     # slope ratio but max RATIO
            ls_failed = 0                       # this line search did not fail
        else:
            X = X0; f0 = F0; df0 = dF0              # restore best point so far
            if ls_failed or (i>abs(length)):# line search failed twice in a row
                break                    # or we ran out of time, so we give up
            s = -df0; d0 = -dot(s,s.T)                             # try steepest
            x3 = 1/(1-d0)                     
            ls_failed = 1                             # this line search failed
    if verbose: print("\n")
    return X, fX, i