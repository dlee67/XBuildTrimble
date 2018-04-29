import matplotlib.pyplot as plt
import Analyzer

plotValues = Analyzer.prepare_data_for_plotting()

for index in range(0, len(plotValues)):
    plt.scatter(plotValues[index][0], plotValues[index][1])
plt.xlabel("Testing")
plt.ylabel("Testing")
plt.show()
