class SeqAugmenter(object):
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, *args, **kwargs):
        img = args[0]
        if self.seq:
            out = self.seq.augment_image(img)
            return out.copy()
        return img


class VizAugmenter(object):
    """ Show image for debugging

    """

    def __init__(self, title='img', delay=-1):
        self.title = title
        self.delay = delay

    def __call__(self, *args, **kwargs):
        import cv2
        img = args[0]
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow(self.title, _img)
        cv2.waitKey(self.delay)
        return img


class Composer(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
