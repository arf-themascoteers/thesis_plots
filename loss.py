import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
a_x = 1 / (1 + np.exp(-x))
b_x = 4200 * a_x
c_x = (x**2) + np.random.normal(0, 0.1, len(x))

fig, ax1 = plt.subplots()

ax1.plot(x, c_x, 'b-', label='Loss c(x)')
ax1.set_xlabel('x')
ax1.set_ylabel('Loss c(x)', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(x, a_x, 'r--', label='a(x) = Sigmoid(x)')
ax2.plot(x, b_x, 'g-.', label='b(x) = 4200 * a(x)')
ax2.set_ylabel('a(x) / b(x)', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.show()
