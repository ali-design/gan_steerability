import importlib
import numpy as np

def get_transform_graphs(model):

    print("Getting transform graphs for {} model...".format(model))

    transform_base_name = 'graphs.' + model + '.transform_base'
    base = importlib.import_module(transform_base_name)

    transform_op_name = 'graphs.' + model + '.transform_op'
    op = importlib.import_module(transform_op_name)

    class ColorGraph(base.PixelTransform,op.ColorTransform):
        def __init__(self, lr=0.001, walk_type='NNz', loss='l2', eps=0.1,
                     N_f=5, channel=None, **kwargs):
            # NN walk only changes a single op (luminance)
            nsliders = 3 if walk_type == 'linear' else 1
            base.PixelTransform.__init__(self, lr, walk_type, nsliders, loss, eps, N_f, **kwargs)
            op.ColorTransform.__init__(self, channel)

        def vis_image_batch(self, graph_inputs, filename, batch_start,
                            wgt=False, wmask=False, num_panels=7,
                            max_alpha=None, min_alpha=None):
            if max_alpha is not None and min_alpha is not None:
                alphas = np.linspace(min_alpha, max_alpha, num_panels)
            else:
                alphas = self.vis_alphas(num_panels)
            filename_base = filename

            zs_batch = graph_inputs[self.z]

            for channel, color in enumerate('RGBWD'):
                # NN walk only handles a single op
                if self.walk_type.startswith('NN') and self.channel is None and color in 'RGB' \
                   or (self.walk_type.startswith('NN') and self.channel is not
                       None and self.channel != channel):
                    continue

                alphas_to_graph = []
                alphas_to_target = []
                for a in alphas:
                    if color == 'W': # whiten
                        channel = None
                    if color == 'D': # darken
                        channel = None
                        a = -a
                    a_graph = self.scale_test_alpha_for_graph(
                        a, zs_batch, channel, contrast=True)
                    alphas_to_graph.append(a_graph)
                    a_target = np.ones((zs_batch.shape[0], 3)) * a
                    alphas_to_target.append(a_target)
                filename = filename_base + '_{}'.format(color)
                self.vis_image_batch_alphas(graph_inputs, filename,
                                            alphas_to_graph, alphas_to_target,
                                            batch_start, wgt=False, wmask=False)

        def get_distribution_statistic(self, img, channel=None):
            num_pixels = 100
            if channel is None:
                stat = np.mean(img[np.random.choice(self.img_size, num_pixels),
                                   np.random.choice(self.img_size, num_pixels), :], axis=-1)
            else:
                stat = img[np.random.choice(self.img_size, num_pixels),
                           np.random.choice(self.img_size, num_pixels), channel]
            return stat


    class ColorLabGraph(base.PixelTransform, op.ColorLabTransform):

        def __init__(self, lr=0.001, walk_type='NNz', loss='l2', eps=0.1,
                     N_f=5, channel=None, **kwargs):
            # NN walk only changes a single op (luminance)
            nsliders = 3 if walk_type == 'linear' else 1
            base.PixelTransform.__init__(self, lr, walk_type, nsliders, loss, eps, N_f, **kwargs)
            op.ColorLabTransform.__init__(self, channel)

        def vis_image_batch(self, graph_inputs, filename, batch_start,
                            wgt=False, wmask=False, num_panels=7,
                            max_alpha=None, min_alpha=None):
            if max_alpha is not None and min_alpha is not None:
                alphas = np.linspace(min_alpha, max_alpha, num_panels)
            else:
                alphas = self.vis_alphas(num_panels)

            zs_batch = graph_inputs[self.z]

            filename_base = filename
            for channel, color in enumerate('Lab'):
                # NN walk only handles a single op
                if self.walk_type.startswith('NN') and self.channel is None and color in 'ab' \
                        or (self.walk_type.startswith('NN') and self.channel is not None
                            and self.channel != channel):
                    continue
                alphas_to_graph = []
                alphas_to_target = []
                for a in alphas:
                    a_graph = self.scale_test_alpha_for_graph(
                        a, zs_batch, channel)
                    alphas_to_graph.append(a_graph)
                    a_target = np.zeros((zs_batch.shape[0], 3))
                    a_target[:, 0] = a
                    alphas_to_target.append(a_target)
                filename = filename_base + '_{}'.format(color)
                self.vis_image_batch_alphas(graph_inputs, filename,
                                            alphas_to_graph, alphas_to_target,
                                            batch_start, wgt=False, wmask=False)

        def get_distribution_statistic(self, img, channel=None):
            if channel is None:
                channel = 0 # luminance
            num_pixels = 100
            stat = img[np.random.choice(self.img_size, num_pixels),
                       np.random.choice(self.img_size, num_pixels), channel]
            return stat

    class ZoomGraph(base.BboxTransform,op.ZoomTransform):
        def __init__(self, lr=0.001, walk_type='NNz', loss='l2', eps=1.41, N_f=4, **kwargs):
            nsliders = 1
            base.BboxTransform.__init__(self, lr, walk_type, nsliders, loss, eps, N_f, **kwargs)
            op.ZoomTransform.__init__(self)

        def vis_image_batch(self, graph_inputs, filename,
                            batch_start, wgt=False, wmask=False, num_panels=7,
                            max_alpha=None, min_alpha=None):
            if max_alpha is not None and min_alpha is not None:
                alphas = np.exp(np.linspace(np.log(min_alpha), np.log(max_alpha),
                                            num_panels))
                print(alphas)
            else:
                alphas = self.vis_alphas(num_panels)

            zs_batch = graph_inputs[self.z]

            filename_base = filename

            alphas_to_graph = []
            alphas_to_target = []
            for a in alphas:
                slider = self.scale_test_alpha_for_graph(a, zs_batch)
                alphas_to_graph.append(slider)
                alphas_to_target.append(a)
            self.vis_image_batch_alphas(graph_inputs, filename,
                                        alphas_to_graph, alphas_to_target,
                                        batch_start, wgt=False, wmask=False)

        def get_distribution_statistic(self, img, *args):
            box = self.detector.detect(img, *args)
            if box:
                (bbox_x, bbox_y, right, bottom) = box
                bbox_width = int(right) - int(bbox_x)
                bbox_height = int(bottom) - int(bbox_y)
                return [bbox_width*bbox_height] # want to return as a list
            return []

    class ShiftXGraph(base.BboxTransform, op.ShiftXTransform):
        def __init__(self, lr=0.0001, walk_type='NNz', loss='l2', eps=5, N_f=10, **kwargs):
            nsliders = 1
            eps = int(eps)
            base.BboxTransform.__init__(self, lr, walk_type, nsliders, loss, eps, N_f, **kwargs)
            op.ShiftXTransform.__init__(self)


        def vis_image_batch(self, graph_inputs,  filename,
                            batch_start, wgt=False, wmask=False, num_panels=7,
                            max_alpha=None, min_alpha=None):
            if max_alpha is not None and min_alpha is not None:
                alphas = np.linspace(min_alpha, max_alpha, num_panels)
            else:
                alphas = self.vis_alphas(num_panels)
            zs_batch = graph_inputs[self.z]
            alphas_to_graph = []
            alphas_to_target = []
            for a in alphas:
                slider = self.scale_test_alpha_for_graph(a, zs_batch)
                alphas_to_graph.append(slider)
                alphas_to_target.append(a)
            self.vis_image_batch_alphas(graph_inputs, filename,
                                        alphas_to_graph, alphas_to_target,
                                        batch_start, wgt=False, wmask=False)

        def get_distribution_statistic(self, img, *args):
            box = self.detector.detect(img, *args)
            if box:
                (bbox_x, bbox_y, right, bottom) = box
                center_x = (int(right) + int(bbox_x))/2
                return [center_x] # want to return as a list
            return []

    class ShiftYGraph(base.BboxTransform, op.ShiftYTransform):
        def __init__(self, lr=0.0001, walk_type='NNz', loss='l2', eps=5, N_f=10, **kwargs):
            nsliders = 1
            eps = int(eps)
            base.BboxTransform.__init__(self, lr, walk_type, nsliders, loss, eps, N_f, **kwargs)
            op.ShiftYTransform.__init__(self)

        def vis_image_batch(self, graph_inputs,  filename,
                            batch_start, wgt=False, wmask=False, num_panels=7,
                            max_alpha=None, min_alpha=None):
            if max_alpha is not None and min_alpha is not None:
                alphas = np.linspace(min_alpha, max_alpha, num_panels)
            else:
                alphas = self.vis_alphas(num_panels)
            zs_batch = graph_inputs[self.z]
            alphas_to_graph = []
            alphas_to_target = []
            for a in alphas:
                slider = self.scale_test_alpha_for_graph(a, zs_batch)
                alphas_to_graph.append(slider)
                alphas_to_target.append(a)
            self.vis_image_batch_alphas(graph_inputs, filename,
                                        alphas_to_graph, alphas_to_target,
                                        batch_start, wgt=False, wmask=False)

        def get_distribution_statistic(self, img, *args):
            box = self.detector.detect(img, *args)
            if box:
                (bbox_x, bbox_y, right, bottom) = box
                center_y = (int(bottom) + int(bbox_y))/2
                return [center_y] # want to return as a list
            return []

    class Rotate2DGraph(base.TransformGraph,op.Rotate2DTransform):
        def __init__(self, lr=0.0002, walk_type='NNz', loss='l2', eps=10, N_f=5, **kwargs):
            nsliders = 1
            base.TransformGraph.__init__(self, lr, walk_type, nsliders, loss, eps, N_f, **kwargs)
            op.Rotate2DTransform.__init__(self)

        def vis_image_batch(self, graph_inputs, filename,
                            batch_start, wgt=False, wmask=False, num_panels=7,
                            max_alpha=None, min_alpha=None):
            if max_alpha is not None and min_alpha is not None:
                alphas = np.linspace(min_alpha, max_alpha, num_panels)
            else:
                alphas = self.vis_alphas(num_panels)
            zs_batch = graph_inputs[self.z]

            alphas_to_graph = []
            alphas_to_target = []
            for a in alphas:
                slider = self.scale_test_alpha_for_graph(a, zs_batch)
                alphas_to_graph.append(slider)
                alphas_to_target.append(a)
            self.vis_image_batch_alphas(graph_inputs, filename,
                                        alphas_to_graph, alphas_to_target,
                                        batch_start, wgt=False, wmask=False)

    class Rotate3DGraph(base.TransformGraph,op.Rotate3DTransform):
        def __init__(self, lr=0.0002, walk_type='NNz', loss='l2', eps=10, N_f=5, **kwargs):
            nsliders = 1
            base.TransformGraph.__init__(self, lr, walk_type, nsliders, loss, eps, N_f, **kwargs)
            op.Rotate3DTransform.__init__(self)

        def vis_image_batch(self, graph_inputs, filename,
                            batch_start, wgt=False, wmask=False, num_panels=7,
                            max_alpha=None, min_alpha=None):
            if max_alpha is not None and min_alpha is not None:
                alphas = np.linspace(min_alpha, max_alpha, num_panels)
            else:
                alphas = self.vis_alphas(num_panels)
            zs_batch = graph_inputs[self.z]

            alphas_to_graph = []
            alphas_to_target = []
            for a in alphas:
                slider = self.scale_test_alpha_for_graph(a, zs_batch)
                alphas_to_graph.append(slider)
                alphas_to_target.append(a)
            self.vis_image_batch_alphas(graph_inputs, filename,
                                        alphas_to_graph, alphas_to_target,
                                        batch_start, wgt=False, wmask=False)


    graphs = [ColorGraph, ColorLabGraph, ZoomGraph, ShiftXGraph, ShiftYGraph,
              Rotate2DGraph, Rotate3DGraph]

    return graphs
