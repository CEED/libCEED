import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv("force.csv")

df.plot(x = 'Time', y = 'ForceX', legend = False)
plt.xlabel('Time')
plt.ylabel('Drag Coefficient, $C_D$')

df.plot(x = 'Time', y = 'ForceY', legend = False)
plt.xlabel('Time')
plt.ylabel('Lift Coefficient, $C_L$')


plt.show()

