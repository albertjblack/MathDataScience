from numpy import array

i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])
transform1 = array([i_hat1, j_hat1]).transpose()

i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()


print(transform1)
print()

print(transform2)
print()

combined = transform2 @ transform1
dotted = transform2.dot(transform1)

print(combined)  # matmul is better and @ than dot
print()

print(dotted)
print()
