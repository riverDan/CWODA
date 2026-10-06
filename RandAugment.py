import random
from PIL import Image, ImageEnhance, ImageOps


class RandAugment:
    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=31, fill=(128, 128, 128)):
        self.num_ops = max(0, int(num_ops))
        self.magnitude = max(0, int(magnitude))
        self.num_magnitude_bins = max(2, int(num_magnitude_bins))
        self.fill = fill
        self.ops = [
            self._auto_contrast,
            self._brightness,
            self._color,
            self._contrast,
            self._equalize,
            self._posterize,
            self._rotate,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._solarize,
            self._translate_x,
            self._translate_y,
        ]

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError('RandAugment expects a PIL.Image input')

        augmented = img
        for op in random.choices(self.ops, k=self.num_ops):
            augmented = op(augmented)
        return augmented

    def _level(self):
        return min(self.magnitude, self.num_magnitude_bins - 1) / float(self.num_magnitude_bins - 1)

    def _signed_level(self, scale):
        value = scale * self._level()
        return value if random.random() < 0.5 else -value

    def _affine(self, img, matrix):
        try:
            return img.transform(img.size, Image.AFFINE, matrix, resample=Image.BILINEAR, fillcolor=self.fill)
        except TypeError:
            return img.transform(img.size, Image.AFFINE, matrix, resample=Image.BILINEAR)

    def _rotate(self, img):
        degrees = self._signed_level(30.0)
        try:
            return img.rotate(degrees, resample=Image.BILINEAR, fillcolor=self.fill)
        except TypeError:
            return img.rotate(degrees, resample=Image.BILINEAR)

    def _translate_x(self, img):
        offset = int(round(self._signed_level(0.45) * img.size[0]))
        return self._affine(img, (1, 0, offset, 0, 1, 0))

    def _translate_y(self, img):
        offset = int(round(self._signed_level(0.45) * img.size[1]))
        return self._affine(img, (1, 0, 0, 0, 1, offset))

    def _shear_x(self, img):
        shear = self._signed_level(0.3)
        return self._affine(img, (1, shear, 0, 0, 1, 0))

    def _shear_y(self, img):
        shear = self._signed_level(0.3)
        return self._affine(img, (1, 0, 0, shear, 1, 0))

    def _brightness(self, img):
        factor = 1.0 + self._signed_level(0.9)
        return ImageEnhance.Brightness(img).enhance(max(0.1, factor))

    def _color(self, img):
        factor = 1.0 + self._signed_level(0.9)
        return ImageEnhance.Color(img).enhance(max(0.1, factor))

    def _contrast(self, img):
        factor = 1.0 + self._signed_level(0.9)
        return ImageEnhance.Contrast(img).enhance(max(0.1, factor))

    def _sharpness(self, img):
        factor = 1.0 + self._signed_level(0.9)
        return ImageEnhance.Sharpness(img).enhance(max(0.1, factor))

    def _posterize(self, img):
        bits = max(1, 8 - int(round(4 * self._level())))
        return ImageOps.posterize(img, bits)

    def _solarize(self, img):
        threshold = int(round(256 * (1.0 - self._level())))
        threshold = min(255, max(0, threshold))
        return ImageOps.solarize(img, threshold)

    def _equalize(self, img):
        return ImageOps.equalize(img)

    def _auto_contrast(self, img):
        return ImageOps.autocontrast(img)
