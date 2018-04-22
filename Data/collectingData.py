import numpy as np
from mouse_tracker.MDTPlotHandler import MDTPlotHandler

lData = []
for i in range(0,50):
    cords = MDTPlotHandler().run()
    lData.append(np.size(cords,1))
    if i == 0:
        cordList = cords
    else:
        cordList = np.concatenate((cordList,cords),1)
    print(i)
print(lData)
print(np.shape(cordList))

np.savetxt("LDataLetterC.csv", lData, delimiter=",")
np.savetxt("CordsLetterC.csv", cordList, delimiter=",")
