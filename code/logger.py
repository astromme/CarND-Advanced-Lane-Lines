import cv2

class Logger:
    def __init__(self, enabled, prefix):
        self.stage = 0
        self.enabled = enabled
        self.prefix = prefix

    def _filename(self, name, format='jpg'):
        return '{prefix}-stage{stage}-{name}.{format}'.format(prefix=self.prefix, stage=self.stage, name=name, format=format)

    def image(self, name, image):
        if not self.enabled:
            return

        cv2.imwrite(self._filename(name), image)
        self.stage += 1

    def plot(self, name, plot):
        if not self.enabled:
            return

        plot.savefig(self._filename(name, format='png'))
        self.stage += 1
