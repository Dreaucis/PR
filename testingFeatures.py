import matplotlib.pyplot as plt
from mouse_tracker.MDTPlotHandler import MDTPlotHandler
from feature_extracting.feature_extractor import FeatureExtractor

a = MDTPlotHandler().run()
b = FeatureExtractor(a)
c = b.dist2cent()
plt.plot(c)
plt.show()
