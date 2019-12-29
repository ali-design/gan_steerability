from utils.transforms import *
import PIL.Image
from resources import deeplab


# overwrites ShiftXTransform in util.transforms
class ShiftXTransform(ShiftXTransform):
    def __init__(self):
        super().__init__()
        self.alpha_max = 50 # maximum 50px shift

    def test_alphas(self):
        # return np.array([-200, -150, -100, -50, 0, 50, 100, 150, 200])
        return np.linspace(-100, 100, 9)

    def vis_alphas(self, num_panels):
        # alpha range for visualization
        return np.linspace(-100, 100, num_panels)


# overwrites ShiftYTransform in util.transforms
class ShiftYTransform(ShiftYTransform):
    def __init__(self):
        super().__init__()
        self.alpha_max = 50 # maximum 50px shift

    def test_alphas(self):
        # return np.array([-200, -150, -100, -50, 0, 50, 100, 150, 200])
        return np.linspace(-100, 100, 9)

    def vis_alphas(self, num_panels):
        # alpha range for visualization
        return np.linspace(-100, 100, num_panels)


# overwrites ZoomTransform in util.transforms
class ZoomTransform(ZoomTransform):

    def test_alphas(self):
        return np.power(2., np.linspace(-2, 2, 9))

    def vis_alphas(self, num_panels):
        return np.power(2., np.linspace(-2, 2, num_panels))

# overwrites Rotate2DTransform in util.transforms
class Rotate2DTransform(Rotate2DTransform):

    def test_alphas(self):
        return np.linspace(-45, 45, num_panels)

    def vis_alphas(self, num_panels):
        return np.linspace(-45, 45, num_panels)

# overwrites Rotate3DTransform in util.transforms
class Rotate3DTransform(Rotate3DTransform):

    def test_alphas(self):
        return np.linspace(-90, 90, num_panels)

    def vis_alphas(self, num_panels):
        return np.linspace(-90, 90, num_panels)
