import sympy

""" linear function """
x = sympy.symbols('x')
f = (2*x) + 1
# sympy.plot(x)

""" multivariable plane """
x,y = sympy.symbols("x y")
f = (2*x) + (9*y)
# sympy.plotting.plot3d(f)

""" summation sympy """
i, n = sympy.symbols("i n")
# Obj to: 
# sum the iteration of each element '1' by '1' from i to n and multiply each element by 2
summation = sympy.Sum(2*i, (i,1,n)) 

# makes n=5 in the previous sumation object
up_to_5 = summation.subs(n,5) 
print(up_to_5.doit())

""" exponents """
x = sympy.symbols("x")
expr = (x**2) / (x**5)
print(expr)

""" logarithms """
import math
x = math.log(8, 2) # x,b -> n = ? = ans
print(x)

""" eulers number (compound interest) """
# used so much because on derivation its base does not change
p = 100
r = 0.20
t = 2.0
n = 525600 # 12 mths, 365 days or 525600 minutes, .... 149.something convergence
a = p * ( (1 + (r/n)) ** (n*t) )
print(a)
# if we make periods smaller -> a= p*e^(rt)
a_e = p * math.exp(r*t)
print(a_e)
x = math.log(8) # naturallog of 8

""" limits """
n = sympy.symbols("n")
f = (1 + (1/n)) ** n
res = sympy.limit(f, n, sympy.oo)
print(res)
print(res.evalf())

""" derivatives """
x = sympy.symbols("x")
f = x**2
dx_f = sympy.diff(f)
print(dx_f)
print(dx_f.subs(x,2)) # evaluate derivative at 2

""" partal derivatives """
x,y = sympy.symbols("x y")
f = 2*x**3 + 3*y**3
dx_f = sympy.diff(f, x)
dy_f = sympy.diff(f, x)
print(dx_f, dy_f)
# sympy.plotting.plot3d(f)

""" chain rule (neural networks) """
# compose layers, untangle derivatives from each 
# important to train NN with proper weights & biases
# rather than untangle each node derivative we can multiply them instead
# dz/dx = dz/dy * dy/dx
x,y = sympy.symbols("x y")
Y = x**2 + 1
dYdX = sympy.diff(Y)

Z = y**3 -2
dZdY = sympy.diff(Z)
chain_rule = (dYdX * dZdY).subs(y,Y)
print(chain_rule)

""" Integrals """
x = sympy.symbols("x")
f = x**2 + 1
area = sympy.integrate(f, (x, 0, 1))
print(area)