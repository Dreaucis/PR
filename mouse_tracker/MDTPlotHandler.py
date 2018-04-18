import numpy as np
import matplotlib.pyplot as plt
from mouse_tracker.track_mouse import MouseDragTracker

class MDTPlotHandler:
    def run(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rect, = ax.plot([], [],'-')
        mdt = MouseDragTracker(rect)
        mdt.connect()
        plt.show()
        draw_crds = np.matrix(mdt.get_crds())
        return draw_crds
