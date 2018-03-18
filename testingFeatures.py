print(1)
#import matplotlib.pyplot as plt
#from MDTPlotHandler import MDTPlotHandler
#from feature_extractor import FeatureExtractor

print(1)
a = MDTPlotHandler().run()
b = FeatureExtractor(a)
c = b.dist2cent()
print(c)
plt.plot(c)
plt.show()
