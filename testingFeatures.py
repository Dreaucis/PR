import matplotlib.pyplot as plt
from mouse_tracker.MDTPlotHandler import MDTPlotHandler
from feature_extracting.feature_extractor import FeatureExtractor

for i in range(0,100):
    a = MDTPlotHandler().run()
    b = FeatureExtractor(a)
    c = b.dist2cent()
    c = b.angleOfMotion()

    plt.plot(c.tolist()[0],'-')
    plt.show()
