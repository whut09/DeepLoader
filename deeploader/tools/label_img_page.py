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
#WIN_SIZE = (1536, 864)


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
    return [oy, ox, dy, dx]

def cvSetBound(dst, rect, val):
    dst[rect[1]:rect[3], rect[0]:rect[2]] = val
    return dst

# label pos 1, neg 0
class DataPageAdapter(object):
    def __init__(self, dataset, label_path, default_label=-1, *args):
        self.dataset = dataset
        self.label_path = label_path
        self.cursor = -1
        self.cur_item = None
        self.label_map = {}
        self.default_label = default_label
        # H,W,C
        self.canvas = np.zeros((WIN_SIZE[1], WIN_SIZE[0], 3), dtype=np.uint8)
        if os.path.exists(label_path):
            self.load(label_path)
        # div layouts
        win_w, win_h = WIN_SIZE[0], WIN_SIZE[1]
        buttom_bar_h = 40
        y0 = win_h - buttom_bar_h
        x0 = int(win_w / 3)
        x0 = max(x0, win_w - y0)
        self.div_detail = (0, 0, x0, y0)
        x1 = x0 + 5
        self.div_grid = (x1, 0, win_w, y0)
        self.div_bar = (0, y0, win_w, win_h)
        # page size
        self.grid_x = 3
        self.grid_img_x = (self.div_grid[2] - self.div_grid[0]) // self.grid_x
        self.grid_y = (self.div_grid[2] - self.div_grid[0]) // self.grid_img_x
        self.page_size = self.grid_x * self.grid_y
        self.page_num = (self.dataset.size() + self.page_size - 1) // self.page_size
        self.page_idx = -1
        # cache
        self.page_items = []
        # detail
        self.detail_idx = -1
        self.detail_center = (0, 0)
        self.detail_img = None

    def save(self):
        with open(self.label_path, 'w') as f:
            keys = self.label_map.keys()
            for x in keys:
                f.write('%s\t%d\n' % (x, self.label_map[x]))

    def getLabelCount(self, target):
        keys = self.label_map.keys()
        count = 0
        for x in keys:
            if self.label_map[x] == target:
                count += 1
        return count

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

    def toggleItem(self, idx):
        label = self.page_items[idx]['label']
        if label == 1:
            label = 0
        else:
            label = 1
        self.label_map[self.page_items[idx]['path']] = label
        self.page_items[idx]['label'] = label
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
            item['label'] = self.default_label
            self.label_map[key] = self.default_label
        self.cur_item = item
        self.cursor = idx
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

    def gotoPage(self, page_idx):
        if self.page_items:
            self.save()
        self.detail_idx = -1
        # load page data
        page_items = []
        page_start_idx = page_idx * self.page_size
        page_end_idx = page_start_idx + self.page_size
        page_end_idx = min(page_end_idx, self.dataset.size())
        for idx in range(page_start_idx, page_end_idx):
            item = self.query(idx)
            page_items.append(item)
        self.page_items = page_items
        self.page_idx = page_idx
        return self.page_items

    def nextPage(self):
        self.page_idx += 1
        if self.page_idx >= self.page_num:
            self.page_idx = self.page_num - 1
        return self.gotoPage(self.page_idx)

    def prevPage(self):
        self.page_idx -= 1
        self.page_idx = max(0, self.page_idx)
        return self.gotoPage(self.page_idx)

    def _drawStatusBar(self):
        cvSetBound(self.canvas, self.div_bar, (64,64,64))
        pgr = (self.cursor + 1) * 100.0 / self.size()
        label = self.cur_item['label']
        npos = self.getLabelCount(1)
        nneg = self.getLabelCount(0)
        txt = 'PN:%d/%d %5.2f%%: %d/%d' % (npos, nneg, pgr, self.cursor + 1, self.size())
        y = int(self.div_bar[1] + (self.div_bar[3] - self.div_bar[1]) // 2)
        cv2.putText(self.canvas, txt, (1, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    def _drawDetail(self):
        if self.detail_idx < 0:
            return
        # img ref
        img_ref = self.detail_center

        img = self.detail_img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # if image is smaller than div, center it
        src_img = self.page_items[self.detail_idx]['img']
        div_w = self.div_detail[2] - self.div_detail[0]
        div_h = self.div_detail[3] - self.div_detail[1]
        div_img = (div_h, div_w)
        fit_rect = size_fit_center(src_img, div_img)
        ratio = float(fit_rect[3]) / src_img.shape[1]
        if ratio > 1:
            img_ref = (img.shape[1] // 2, img.shape[0] // 2)

        div_cx = (self.div_detail[0] + self.div_detail[2]) // 2
        div_cy = (self.div_detail[1] + self.div_detail[3]) // 2
        # map image to div
        img_x0 = div_cx - img_ref[0]
        img_y0 = div_cy - img_ref[1]
        img_x1 = img_x0 + img.shape[1]
        img_y1 = img_y0 + img.shape[0]
        # div bound
        div_x0 = self.div_detail[0]
        div_y0 = self.div_detail[1]
        div_x1 = self.div_detail[2]
        div_y1 = self.div_detail[3]

        # clip image
        clip_x0, clip_y0 = 0, 0
        clip_x1, clip_y1 = img.shape[1], img.shape[0]
        # clip x
        if img_x0 < div_x0:
            img_x0 = div_x0
        if img_x0 > div_x1:
            img_x0 = div_x1

        if img_x1 > div_x1:
            img_x1 = div_x1
        if img_x1 < div_x0:
            img_x1 = div_x0

        # clip y
        if img_y0 < div_y0:
            img_y0 = div_y0
        if img_y0 > div_y1:
            img_y0 = div_y1

        if img_y1 > div_y1:
            img_y1 = div_y1
        if img_y1 < div_y0:
            img_y1 = div_y0

        # map to image
        clip_x0 = img_x0 - (div_cx - img_ref[0])
        clip_x1 = img_x1 - (div_cx - img_ref[0])
        clip_y0 = img_y0 - (div_cy - img_ref[1])
        clip_y1 = img_y1 - (div_cy - img_ref[1])

        # crop image
        self.canvas[img_y0:img_y1, img_x0:img_x1, :] = img[clip_y0:clip_y1, clip_x0:clip_x1, :]


    def drawCurrentPage(self):
        if not self.page_items:
            return -1
        # clear canvas
        cvZero(self.canvas)
        # draw detail
        self._drawDetail()
        # div_grid
        for idx, item in enumerate(self.page_items):
            img = self.page_items[idx]['img']
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            grid_x = idx % self.grid_x
            grid_y = idx // self.grid_x

            div_img = (self.grid_img_x, self.grid_img_x)
            img_rect = size_fit_center(img, div_img)
            _img = cv2.resize(img, (img_rect[3], img_rect[2]))
            img_ox = (self.div_grid[0] + grid_x * div_img[0])
            img_oy = (self.div_grid[1] + grid_y * div_img[1])

            img_rect[1] += img_ox
            img_rect[0] += img_oy
            cvCopy(_img, self.canvas, img_rect)

            # record bound
            bound = (img_ox, img_oy, img_ox + div_img[0], img_oy + div_img[1])
            self.page_items[idx]['bound'] = bound
            self.page_items[idx]['img_rect'] = img_rect

            # indicator
            label = self.page_items[idx]['label']
            id_color = (128, 128, 128)
            if label == 1:
                id_color = (0, 255, 255)
            elif label == 0:
                id_color = (0, 255, 0)
            cv2.circle(self.canvas, (bound[2] - 20, bound[3] - 20), 10, id_color, -1)

        # draw grid
        for idx, item in enumerate(self.page_items):
            rect = self.page_items[idx]['bound']
            cvRectangleR(self.canvas, (rect[0], rect[1], self.grid_img_x, self.grid_img_x), (255,255,255), 2)


        self._drawStatusBar()
        cv2.imshow(WIN_NAME, self.canvas)
        return cv2.waitKey(20)

    def hitTest(self, x, y):
        for idx, item in enumerate(self.page_items):
            rect = self.page_items[idx]['bound']
            if x > rect[0] and x < rect[2] and y > rect[1] and y < rect[3]:
                return idx
        return -1

    def onClick(self, x, y):
        idx = self.hitTest(x, y)
        if idx < 0:
            return
        page_start_idx = self.page_idx * self.page_size
        self.toggleItem(idx)
        return self.drawCurrentPage()

    def onToggleLabel(self):
        for idx, item in enumerate(self.page_items):
            self.toggleItem(idx)
        # toggle default label
        if self.default_label == 1:
            self.default_label = 0
        else:
            self.default_label = 1

    def onMove(self, x, y):
        idx = self.hitTest(x, y)
        if idx < 0:
            self.detail_idx = -1
            return self.drawCurrentPage()

        img_rect = self.page_items[idx]['img_rect']
        src_img = self.page_items[idx]['img']

        if self.detail_idx < 0 or (self.page_items[idx]['path'] != self.page_items[self.detail_idx]['path']):
            # re-generate resized source image
            div_w = self.div_detail[2] - self.div_detail[0]
            div_h = self.div_detail[3] - self.div_detail[1]
            win_ratio = float(div_w) / WIN_SIZE[0]
            div_img = (div_h, div_w)
            fit_rect = size_fit_center(src_img, div_img)
            ratio = float(fit_rect[3]) / src_img.shape[1]
            if ratio > 1:
                _img = cv2.resize(src_img, (fit_rect[3], fit_rect[2]))
            elif ratio > win_ratio:
                _img = src_img
            else:
                _img = cv2.resize(src_img, (int(fit_rect[3]/win_ratio), int(fit_rect[2]/win_ratio)))
            self.detail_img = _img

        img = self.detail_img
        # figure out mouse position in img
        ratio = float(img.shape[1]) / img_rect[3]
        ox = x - img_rect[1]
        oy = y - img_rect[0]
        src_x = int(ox * ratio)
        src_y = int(oy * ratio)
        self.detail_idx = idx
        self.detail_center = (src_x, src_y)


def mouse_page_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEWHEEL:
        adapter = param
        if flags > 0:
            adapter.prevPage()
            adapter.drawCurrentPage()
        elif flags < 0:
            adapter.nextPage()
            adapter.drawCurrentPage()
    elif event == cv2.EVENT_LBUTTONDOWN:
        adapter = param
        adapter.onClick(x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        adapter = param
        adapter.onMove(x, y)


def run_label(dataset, label_path, default_label=1):
    # cv2.namedWindow(WIN_NAME)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    da = DataPageAdapter(dataset, label_path, default_label=default_label)
    cv2.setMouseCallback(WIN_NAME, mouse_page_callback, da)
    da.nextPage()

    while True:
        key = da.drawCurrentPage()
        if key == 27:
            break
        elif key == 106:  # j
            da.prevPage()
            da.drawCurrentPage()
        elif key == 108:  # l
            da.prevPage()
            da.drawCurrentPage()
        elif key == 114:  # r
            da.onToggleLabel()

    cv2.destroyAllWindows()



