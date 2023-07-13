import json
import numpy as np

with open('C:\Codes\master_thesis\momentum_history.json', 'r') as file:
    data = json.load(file)

data = np.array(data)
a = np.mean(data, axis=1)
np.savetxt('C:\Codes\master_thesis\mean_momentum_all.txt', a)