from utils.transforms import *
import PIL.Image
from resources import deeplab

# overwrites ColorTransform in util.transforms
class ColorTransform(ColorTransform):

    def __init__(self, channel=None):
        super().__init__(channel)

    def get_target_np(self, outputs_zs, alpha):
        if not hasattr(self, 'MODEL'):
            self.MODEL = deeplab.load_model(self.sess)
        if not np.any(alpha): # alpha is all zeros
            return outputs_zs, np.ones(outputs_zs.shape)

        # if no pascal id just do normal Color get target
        if self.dataset['pascal_id'] is None:
            return super().get_target_np(outputs_zs, alpha)

        assert(outputs_zs.shape[0] == alpha.shape[0])

        # uncomment to use standard get_target for color
        return super().get_target_np(outputs_zs, alpha)

        # uncomment to add a segmentation map to color
        # ## segmap - only change the segmented pixels, but enforce loss elsewhere too 
        # target_fn = np.copy(outputs_zs)

        # target_scaled = self.sess.run(self.uint8_im, {self.float_im: target_fn})
        # mask_out = np.ones(outputs_zs.shape)

        # N, H, W, C = outputs_zs.shape
        # for b in range(outputs_zs.shape[0]):
        #     _, seg_map = self.MODEL.run(PIL.Image.fromarray(target_scaled[b]))
        #     # resize seg_map back to original size
        #     seg_map_resize = np.asarray(PIL.Image.fromarray(np.uint8(seg_map)).resize((H, W), PIL.Image.ANTIALIAS))
        #     inds = np.where(seg_map_resize == self.dataset['pascal_id'])
        #     for i in range(self.num_channels):
        #         target_fn[b,inds[0],inds[1],i] = target_fn[b,inds[0],inds[1],i]+alpha[b,i]
        #     # mask_out[b,inds[0], inds[1], :] = 1 # use the full mask
        # return target_fn, mask_out


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
        # datasets respond differently to each transformation
        if self.dataset_name == 'cats':
            return np.linspace(-90, 90, num_panels)
        else:
            return np.linspace(-45, 45, num_panels)

# overwrites Rotate3DTransform in util.transforms
class Rotate3DTransform(Rotate3DTransform):

    def test_alphas(self):
        return np.linspace(-90, 90, num_panels)

    def vis_alphas(self, num_panels):
        # datasets respond differently to each transformation
        if self.dataset_name == 'cats':
            return np.linspace(-360, 360, num_panels)
        else:
            return np.linspace(-90, 90, num_panels)
