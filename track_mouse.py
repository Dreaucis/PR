from pynput import mouse

class IsPressed:
    def __init__(self):
        self.boolean = False
    def setFalse(self):
        self.boolean = False
    def setTrue(self):
        self.boolean = True
    def get(self):
        return self.boolean

isPressed = IsPressed()
myList = list()
def on_move(x, y):
    if isPressed.get():
        myList.append((x, y))

def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    if pressed:
        isPressed.setTrue()
    else:
        isPressed.setFalse()

def on_scroll(x, y, dx, dy):
    return False

# Collect events until released
with mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll) as listener:
    listener.join()

print(myList)