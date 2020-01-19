import time

from deeploader.util.opencv import *

WIN_NAME = 'label-tool'
WIN_SIZE = (1000, 800)


def get_screen_size():
    import tkinter
    win = tkinter.Tk()
    w = win.winfo_screenwidth()
    h = win.winfo_screenheight()
    win.quit()
    return w, h


WIN_SIZE = (get_screen_size())


def size_fit_center(src, dst_shape):
    rx = float(src.shape[1]) / dst_shape[1]
    ry = float(src.shape[0]) / dst_shape[0]
    # try fit x
    dx = dst_shape[1]
    dy = src.shape[0] / rx
    if dy > dst_shape[0]:
        dx = src.shape[1] / ry
        dy = dst_shape[0]
    dx = int(dx)
    dy = int(dy)
    ox = (dst_shape[1] - dx) // 2
    oy = (dst_shape[0] - dy) // 2
    return oy, ox, dy, dx


def is_in_bound(x, y, bound):
    if bound[0] <= x and x < bound[2] and bound[1] <= y and y < bound[3]:
        return True
    return False


def bound_size(bound):
    return bound[2] - bound[1], bound[3] - bound[1]


def is_in_rect(x, y, rect):
    bound = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
    return is_in_bound(bound)


def bound_union(a, b):
    c = [0, 0, 0, 0]
    c[0] = min(a[0], b[0])
    c[1] = min(a[1], b[1])
    c[2] = max(a[2], b[2])
    c[3] = max(a[3], b[3])
    return c


# Button states
NORMAL = 'normal'
HOVER = 'hover'
SELECT = 'select'


class EventListener(object):
    def __init__(self, *args, **kargs):
        pass

    def on_check(self, checked):
        pass

    def on_click(self, event, x, y):
        pass

    def on_hover(self, event, x, y):
        pass

    def on_key(self, key):
        pass


class UiRect(object):
    def __init__(self, bound=None, tag='', checked=False, *args, **kargs):
        self.bound = bound
        self.tag = tag
        self.state = NORMAL
        self.checked = checked
        self.checked_stamp = -1

    def draw(self, canvas):
        pass

    def has_hit(self, x, y):
        if self.bound is None:
            return False
        return is_in_bound(x, y, self.bound)

    def handle_mouse_event(self, event, x, y):
        return False

    def handle_key_event(self, key):
        return False


class UiButton(UiRect):
    def __init__(self, bound=None, tag='', checked=False,
                 text='',
                 text_color={NORMAL: COLOR_WHITE}, bk_color={NORMAL: COLOR_BLACK}):
        UiRect.__init__(self, bound, tag, checked)
        self.text = text
        self.text_color = text_color
        self.bk_color = bk_color
        self.listener = None

    def set_text_color(self, text_color):
        self.text_color = text_color

    def set_bk_color(self, bk_color):
        self.bk_color = bk_color

    def set_listener(self, listener):
        self.listener = listener

    @staticmethod
    def _get_color(state, text_color):
        text_color = text_color[text_color.keys()[0]]
        if state in text_color:
            text_color = text_color[state]
        return text_color

    def draw(self, canvas):
        if self.bound is None:
            return
        state = self.state
        if self.checked:
            state = SELECT
        text_color = self._get_color(state, self.text_color)
        bk_color = self._get_color(state, self.bk_color)
        # draw bk
        cvBox(canvas, self.bound, bk_color, -1)
        # draw text
        if not self.text:
            return
        w, h = bound_size(self.bound)
        cv2.putText(self.canvas, self.text, (self.bound[0] + 5, (self.bound[1] + h) // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 1)

    def handle_mouse_event(self, event, x, y):
        hit = self.has_hit(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if hit:
                self.checked = bool(1 - self.checked)
                if self.checked:
                    self.checked_stamp = time.time()

                if self.listener:
                    self.listener.on_click(event, x, y)
                    self.listener.on_check(self.checked)
        elif event == cv2.EVENT_MOUSEMOVE:
            if hit:
                self.state = HOVER
                if self.listener:
                    self.listener.on_hover(event, x, y)
            else:
                self.state = NORMAL
        return hit

    def handle_key_event(self, key):
        if self.listener:
            self.listener.on_key(key)
            return True
        return False


class UiGroup(UiRect):
    def __init__(self, bound=None, tag='', checked=False, single=False):
        UiRect.__init__(self, bound, tag, checked)
        self.single = single
        self.children = []

    def add(self, child):
        self.children.append(child)
        # update bound
        if self.bound is None:
            self.bound = child.bound
        elif child.bound:
            self.bound = bound_union(self.bound, child.bound)

    def draw(self, canvas):
        for child in self.children:
            child.draw(canvas)

    def handle_mouse_event(self, event, x, y):
        for child in self.children:
            handled = child.handle_mouse_event(event, x, y)
            if handled:
                break
        # single check group
        if self.single:
            last_child = None
            last_stamp = -1
            # cancel all
            for child in self.children:
                if child.checked:
                    child.checked = False
                    if child.checked_stamp > last_stamp:
                        last_child = child
                        last_stamp = child.checked_stamp

            if last_child:
                last_child.checked = True

    def handle_key_event(self, key):
        for child in self.children:
            child.handle_key_event(key)
