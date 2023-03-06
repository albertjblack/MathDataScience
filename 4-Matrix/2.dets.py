""" 
Simple Shears & Rotations do not affect determinant
- Scaling will change the determinant value
- Rotating will change the determinant sign
-> Determinant on a flipped space (i for j) is negative
"""

from numpy.linalg import det
from numpy import array

i_hat = array([3, 0])
j_hat = array([0, 2])

basis = array([i_hat, j_hat]).transpose()
determinant = det(basis)  # 6 because it takes an area of 1 to an area of 6

""" ! DETERMINANT TELLS YOU IS WHETHER THE TRANSFORMATION IS LINEARLY DEPENDANT ! """
# 0: space squeeshed into lesser dimension -> !! linearly dependant !!
# 1:
