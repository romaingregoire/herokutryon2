# Generated with SMOP  0.41
from libsmop import *
# example.m

    # Compute the value of the integrand at 2*pi/3.
    x=dot(2,pi) / 3
# example.m:2
    y=myIntegrand(x)
# example.m:3
    # Compute the area under the curve from 0 to pi.
    xmin=0
# example.m:6
    xmax=copy(pi)
# example.m:7
    f=myIntegrand
# example.m:8
    a=integral(f,xmin,xmax)
# example.m:9
    
@function
def myIntegrand(x=None,*args,**kwargs):
    varargin = myIntegrand.varargin
    nargin = myIntegrand.nargin

    y=sin(x) ** 3
# example.m:12
    return y
    
if __name__ == '__main__':
    pass
    