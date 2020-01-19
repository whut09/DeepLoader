import os

from deeploader.dataset.dataset_dir import FileDirReader
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
    return (oy, ox, dy, dx)


# label pos 1, neg 0
class DataAdapter(object):
    def __init__(self, dataset, label_path, *args):
        self.dataset = dataset
        self.label_path = label_path
        self.cursor = -1
        self.cur_item = None
        self.label_map = {}
        # H,W,C
        self.canvas = np.zeros((WIN_SIZE[1], WIN_SIZE[0], 3), dtype=np.uint8)
        if os.path.exists(label_path):
            self.load(label_path)

    def save(self):
        with open(self.label_path, 'w') as f:
            keys = self.label_map.keys()
            for x in keys:
                f.write('%s\t%d\n' % (x, self.label_map[x]))

    def load(self, label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                path, label = line.split('\t')
                label = int(label)
                self.label_map[path] = label

    def onLabelDone(self, idx, label):
        print('idx:%d label:%d' % (idx, label))
        item = self.dataset.getData(idx)
        self.label_map[item['path']] = label
        self.save()

    def goTodo(self):
        while True:
            if self.cur_item and self.cur_item['label'] < 0:
                break
            self.next()

    def currentDone(self, label):
        self.onLabelDone(self.cursor, label)

    def drawCurrentImage(self):
        if self.cur_item is None:
            return -1
        #
        cvZero(self.canvas)
        # draw image
        img = self.cur_item['img']
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        buttom_bar_h = 40
        div_img = (WIN_SIZE[1] - buttom_bar_h, WIN_SIZE[0])
        img_rect = size_fit_center(img, div_img)
        _img = cv2.resize(img, (img_rect[3], img_rect[2]))
        cvCopy(_img, self.canvas, img_rect)
        # cvCopy(_img, self.canvas, (0, img_rect[1], img_rect[2], img_rect[3]))
        # cvRectangleR(self.canvas, img_rect)
        # progress
        pgr = (self.cursor + 1) * 100.0 / self.size()
        label = self.cur_item['label']
        lstr = 'NUL'
        if label == 1:
            lstr = 'POS'
        elif label == 0:
            lstr = 'NEG'
        txt = '%s %5.2f%%: %d/%d' % (lstr, pgr, self.cursor + 1, self.size())

        cv2.putText(self.canvas, txt, (1, WIN_SIZE[1] - buttom_bar_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        # draw hint
        img_h = WIN_SIZE[1] - buttom_bar_h
        img_w = WIN_SIZE[0] // 2 - 30
        # cv2.putText(self.canvas, 'POS', (img_w,  int(img_h*0.5/3)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        # cv2.putText(self.canvas, 'NONE', (img_w, int(img_h*1.5/3)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        # cv2.putText(self.canvas, 'NEG', (img_w, int(img_h*2.5/3)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        cv2.putText(self.canvas, lstr, (img_w, int(img_h * 0.5 / 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

        cv2.imshow(WIN_NAME, self.canvas)
        return cv2.waitKey(20)

    def size(self):
        return self.dataset.size()

    def query(self, idx):
        item = self.dataset.getData(idx)
        key = item['path']
        if key in self.label_map:
            item['label'] = self.label_map[key]
        else:
            item['label'] = -1
        self.cur_item = item
        return item

    def next(self):
        self.cursor += 1
        if self.cursor >= self.size():
            self.cursor = self.size() - 1
        return self.query(self.cursor)

    def prev(self):
        self.cursor -= 1
        self.cursor = max(0, self.cursor)
        # return data
        return self.query(self.cursor)


def mouse_label_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEWHEEL:
        adapter = param
        # label = 0
        # if flags > 0:
        #     label = 1
        # adapter.currentDone(label)
        # item = adapter.next()
        # adapter.drawCurrentImage()
        if flags > 0:
            adapter.prev()
            adapter.drawCurrentImage()
        elif flags < 0:
            adapter.next()
            adapter.drawCurrentImage()
    elif event == cv2.EVENT_LBUTTONDOWN:
        adapter = param
        adapter.currentDone(1)
        item = adapter.next()
        adapter.drawCurrentImage()
    elif event == cv2.EVENT_RBUTTONDOWN:
        adapter = param
        adapter.currentDone(0)
        item = adapter.next()
        adapter.drawCurrentImage()


def run_label(dataset, label_path, default_label=1):
    # cv2.namedWindow(WIN_NAME)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    da = DataAdapter(dataset, label_path)
    cv2.setMouseCallback(WIN_NAME, mouse_label_callback, da)
    da.goTodo()

    while True:
        key = da.drawCurrentImage()
        if key == 27:
            break
        elif key == 106:  # j
            da.prev()
            da.drawCurrentImage()
        elif key == 108:  # l
            da.next()
            da.drawCurrentImage()
        elif key == 71:  # G
            da.goTodo()
            da.drawCurrentImage()

    cv2.destroyAllWindows()

