import numpy as np

my_data = np.genfromtxt('Data/LetterA/CordsLetterA.csv', delimiter=',')
my_data2 = np.genfromtxt('Data/LetterA/lDataLetterA.csv', delimiter=',')
print(my_data2.astype(int))