from deeploader.tools.simple_ui import *


class LabelInfo(object):
    def __init__(self, attrs):
        self.attrs = []  # [{"name":xxx, 'vals': [vals], 'single':False}]
        self.bboxes = []  # [{'label':xxx, 'box':[4]}]
        for attr in attrs:
            self.attrs.append({'name': attr['name'], 'vals': set()})

    def find_group(self, group):
        for attr in self.attrs:
            if attr['name'] == group:
                return attr
        return None

    @staticmethod
    def has_option(attr, option):
        return option in attr['vals']

    def set_attr(self, group, option, enable):
        attr = self.find_group(group)
        if attr is None:
            return False
        if enable:
            if not option in attr['vals']:
                attr['vals'].add(option)
        else:
            if option in attr['vals']:
                attr['vals'].remove(option)

    def toggle_attr(self, group, option):
        attr = self.find_group(group)
        if attr is None:
            return False
        if not option in attr['vals']:
            attr['vals'].add(option)
        else:
            attr['vals'].remove(option)

    def get_attr(self, group, option):
        attr = self.find_group(group)
        if attr is None:
            return False
        return option in attr['vals']


class OptionListener(EventListener):
    def __init__(self, info, group, opt):
        self.info = info
        self.group = group
        self.opt = opt

    def on_check(self, checked):
        self.info.set_attr(self.group, self.opt, checked)

    def on_click(self, event, x, y):
        self.info.toggle_attr(self.group, self.opt)


class LabelListener(EventListener):
    def __init__(self, obj, label):
        self.obj = obj
        self.label = label

    def on_click(self, event, x, y):
        self.obj.set_current_label(self.label)


# label pos 1, neg 0
class DataAdapter(object):
    def __init__(self, dataset, label_dir, class_names=[], attrs=[]):
        self.dataset = dataset
        self.label_dir = label_dir
        self.cursor = -1
        self.cur_item = None
        # H,W,C
        self.canvas = np.zeros((WIN_SIZE[1], WIN_SIZE[0], 3), dtype=np.uint8)
        # Label information
        self.attrs = attrs
        self.class_names = class_names
        self.info = LabelInfo(attrs)
        self.current_label = ''
        # Layout
        H, W = WIN_SIZE[1], WIN_SIZE[0]
        top_h = int(H * 0.07)
        right_w = int(W * 0.9)
        top_w = right_w
        buttom_bar_h = 40
        x1 = right_w
        y1 = top_h
        y2 = H - buttom_bar_h
        top_div = [0, 0, x1, y1]
        right_div = [x1 + 1, y1 + 1, W, y2]
        image_div = [0, y1, x1, y2]
        bar_div = [0, y2 + 1, W, H]

        self.top_div = top_div
        self.right_div = right_div
        self.bar_div = bar_div
        self.image_div = image_div
        # top attr buttons
        groups = len(self.attrs)
        options = 0
        for group in range(groups):
            options += len(self.attrs[group])
        self.top_btns = []
        self.top_group = UiGroup(tag='header')
        if options > 0:
            btn_width = top_w // max(options, 3)
            btn_idx = 0
            for group in range(groups):
                group_name = self.attrs[group]['name']
                ui_group = UiGroup(single=self.attrs[group]['single'])
                for opt in self.attrs[group]['vals']:
                    x = top_div[0] + btn_width * btn_idx
                    btn = UiButton(bound=[x, top_div[1], x + btn_width, top_div[3]], tag=opt)
                    l = OptionListener(self.info, group_name, opt)
                    btn.set_listener(l)
                    self.top_btns.append(btn)
                    ui_group.add(btn)
                    btn_idx += 1
                self.top_group.add(ui_group)
        # right buttons
        num_classes = len(self.class_names)
        self.right_btns = []
        self.right_group = UiGroup(tag='right', single=True)
        if num_classes > 0:
            right_btn_h = max(50, (y2 - y1) / num_classes)
            btn_idx = 0
            for name in self.class_names:
                y = right_div[1] + right_btn_h * btn_idx
                btn = UiButton(bound=[right_div[0], y, right_div[2], y + right_btn_h], tag=name)
                l = LabelListener(self, name)
                btn.set_listener(l)
                self.right_btns.append(btn)
                self.right_group.add(btn)
                btn_idx += 1

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

    def set_current_label(self, label):
        self.current_label = label

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
        # prev/next image
        adapter = param
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
