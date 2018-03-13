import numpy as np
import matplotlib.pyplot as plt

class MouseDragTracker:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.isPressed = False

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.line.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidend = self.line.figure.canvas.mpl_connect('key_press_event', self.on_key)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.line.axes: return
        self.isPressed = True

    def on_key(self,event):
        print('you pressed', event.key, event.xdata, event.ydata)
        self.disconnect()


    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if not self.isPressed: return
        if event.inaxes != self.line.axes: return

        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.isPressed = False
        self.line.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)

    def get_crds(self):
        return (self.xs,self.ys)

class MDTPlotHandler:
    def run(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rect, = ax.plot([], [],'*')
        mdt = MouseDragTracker(rect)
        mdt.connect()
        plt.show()
        draw_crds = np.matrix(mdt.get_crds())
        return draw_crds


a = MDTPlotHandler().run()
