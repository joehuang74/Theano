import matplotlib.pyplot as plt
import numpy as np

y = 0.5
a = np.arange(0.00001,1,0.01)

xent = -y*np.log(a)-(1-y)*np.log(1-a) # Cross-entropy function
quad = (y - a) * (y -a) # Quadratic function

plt.subplot(221)
plt.plot(a, xent)
plt.title("Cross-entropy with y = 0.5")

plt.subplot(222)
plt.plot(a, quad)
plt.title("Quadratic with y = 0.5")

plt.show()