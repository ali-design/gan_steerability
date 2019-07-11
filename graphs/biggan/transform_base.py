import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from . import constants
from .graph_util import *
from resources import tf_lpips_pkg as lpips_tf
from resources import get_coco_imagenet_categories
from utils import image, dataset

module_path = constants.MODULE_PATH


class TransformGraph():
    def __init__(self, lr, walk_type, nsliders, loss_type, eps, N_f):

        assert(loss_type in ['l2', 'lpips']), 'unimplemented loss'

        # module inputs
        module = self.get_biggan_module(module_path)
        inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
                  for k, v in module.get_input_info_dict().items()}
        input_z = inputs['z']
        input_y = inputs['y']
        input_trunc = inputs['truncation']
        dim_z = self.dim_z = input_z.shape.as_list()[1]
        vocab_size = self.vocab_size = input_y.shape.as_list()[1]

        # input placeholders
        Nsliders = nsliders
        z = tf.placeholder(tf.float32, shape=(None, dim_z))
        y = tf.placeholder(tf.float32, shape=(None, vocab_size))
        truncation = tf.placeholder(tf.float32, shape=None)

        # original output
        inputs_orig = {u'y': y,
                       u'z': z,
                       u'truncation': truncation}
        outputs_orig = module(inputs_orig)

        img_size = self.img_size = outputs_orig.shape[1].value
        num_channels = self.num_channels = outputs_orig.shape[-1].value

        # output placeholders
        target = tf.placeholder(tf.float32, shape=(
            None, img_size, img_size, num_channels))
        mask = tf.placeholder(tf.float32, shape=(
            None, img_size, img_size, num_channels))

        # walk pattern
        if walk_type == 'NNz':
            # alpha is the integer number of steps to take
            alpha = tf.placeholder(tf.int32, shape=())
            T1 = tf.keras.layers.Dense(dim_z, input_shape=(None, dim_z), kernel_initializer='RandomNormal', bias_initializer='zeros', activation=tf.nn.relu)
            T2 = tf.keras.layers.Dense(dim_z, input_shape=(None, dim_z), kernel_initializer='RandomNormal', bias_initializer='zeros')
            T3 = tf.keras.layers.Dense(dim_z, input_shape=(None, dim_z), kernel_initializer='RandomNormal', bias_initializer='zeros', activation=tf.nn.relu)
            T4 = tf.keras.layers.Dense(dim_z, input_shape=(None, dim_z), kernel_initializer='RandomNormal', bias_initializer='zeros')
            # forward transformation
            out_f = []
            z_prev = z
            z_norm = tf.norm(z, axis=1, keepdims=True)
            for i in range(1, N_f + 1):
                z_step = z_prev + T2(T1(z_prev))
                z_step_norm = tf.norm(z_step, axis=1, keepdims=True)
                z_step = z_step * z_norm / z_step_norm
                out_f.append(z_step)
                z_prev = z_step

            # reverse transformation
            out_g = []
            z_prev = z
            z_norm = tf.norm(z, axis=1, keepdims=True)
            for i in range(1, N_f + 1):
                z_step = z_prev + T4(T3(z_prev))
                z_step_norm = tf.norm(z_step, axis=1, keepdims=True)
                z_step = z_step * z_norm / z_step_norm
                out_g.append(z_step)
                z_prev = z_step
            out_g.reverse() # flip the reverse transformation

            # w has shape (2*N_f + 1, batch_size, dim_z)
            # elements 0 to N_f are the reverse transformation, in reverse order
            # elements N_f + 1 to 2*N_f + 1 are the forward transformation
            # element N_f is no transformation
            w = tf.stack(out_g+[z]+out_f)
        elif walk_type == 'linear':
            # # one w per category
            # w = tf.Variable(np.random.normal(0.0, 0.1, [vocab_size, z.shape[1]]),
            #         name='walk', dtype=np.float32)
            # one w all category
            alpha = tf.placeholder(tf.float32, shape=(None, Nsliders))
            w = tf.Variable(np.random.normal(0.0, 0.1, [1, z.shape[1], Nsliders]),
                    name='walk', dtype=np.float32)
            N_f = None
            eps = None
        else:
            raise NotImplementedError('Not implemented walk type:' '{}'.format(walk_type))

        # transformed output
        z_new = z
        if walk_type == 'linear':
            for i in range(Nsliders):
                z_new = z_new+tf.expand_dims(alpha[:,i], axis=1)*w[:,:,i]
        elif walk_type == 'NNz':
            # z_new = tf.reshape(tf.linalg.matmul(alpha, tf.reshape(w, [N_f*2+1, -1])), [-1, dim_z])
            # w is already z+f(z) so we can just index into w
            z_new = w[alpha,  :, :]
            # embed()

        transformed_inputs = {u'y': y,
                              u'z': z_new,
                              u'truncation': truncation}
        transformed_output = module(transformed_inputs)

        # L_2 loss
        loss = tf.losses.compute_weighted_loss(tf.square(
            transformed_output-target), weights=mask)
        loss_lpips = tf.reduce_mean(lpips_tf.lpips(
            mask*transformed_output, mask*target, model='net-lin', net='alex'))

        # losses per sample
        loss_lpips_sample = lpips_tf.lpips(
            mask*transformed_output, mask*target, model='net-lin', net='alex')
        loss_l2_sample = tf.reduce_sum(tf.multiply(tf.square(
            transformed_output-target), mask), axis=(1,2,3)) \
                / tf.reduce_sum(mask, axis=(1,2,3))

        if loss_type == 'l2':
            train_step = tf.train.AdamOptimizer(lr).minimize(
                    loss, var_list=tf.trainable_variables(scope=None), name='AdamOpter')
        elif loss_type == 'lpips':
            train_step = tf.train.AdamOptimizer(lr).minimize(
                    loss_lpips, var_list=tf.trainable_variables(scope=None), name='AdamOpter')

        # set class vars
        self.Nsliders = Nsliders
        self.y = y
        self.z = z
        self.truncation = truncation
        self.alpha = alpha
        self.target = target
        self.mask = mask
        self.w = w
        self.z_new = z_new
        self.transformed_output = transformed_output
        self.outputs_orig = outputs_orig
        self.loss = loss
        self.loss_lpips = loss_lpips
        self.loss_l2_sample = loss_l2_sample
        self.loss_lpips_sample = loss_lpips_sample
        self.train_step = train_step
        self.walk_type = walk_type
        self.N_f = N_f # NN num_steps
        self.eps = eps # NN step_size

    def get_biggan_module(self, module_path):
        tf.reset_default_graph()
        print('Loading BigGAN module from:', module_path)
        module = hub.Module(module_path)
        return module

    def initialize_graph(self):
        initializer = tf.global_variables_initializer()
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(initializer)
        saver = tf.train.Saver(tf.trainable_variables(scope=None))
        self.sess = sess
        self.saver = saver

    def clip_ims(self, ims):
        return np.uint8(np.clip(((ims + 1) / 2.0) * 256, 0, 255))

    def apply_alpha(self, graph_inputs, alpha_to_graph):

        zs_batch = graph_inputs[self.z]
        ys_batch = graph_inputs[self.y]
        trunc = graph_inputs[self.truncation]

        # print(alpha_to_graph)
        if self.walk_type == 'linear':
            best_inputs = {self.z: zs_batch, self.y: ys_batch,
                           self.truncation: trunc, self.alpha: alpha_to_graph}
            best_im_out = self.sess.run(self.transformed_output,
                                        best_inputs)
            return best_im_out
        elif self.walk_type == 'NNz':
            # alpha_to_graph is number of steps and direction
            direction = np.sign(alpha_to_graph)
            num_steps = np.abs(alpha_to_graph)
            # embed()
            single_step_alpha = self.N_f + direction
            # within the graph range, we can compute it directly
            if 0 <= alpha_to_graph + self.N_f <= self.N_f * 2:
                zs_out = self.sess.run(self.z_new, {
                    self.z:zs_batch, self.alpha:
                    alpha_to_graph + self.N_f})
                # # sanity check
                # zs_next = zs_batch
                # for n in range(num_steps):
                #     feed_dict = {self.z: zs_next, self.alpha: single_step_alpha}
                #     zs_next = self.sess.run(self.z_new, feed_dict=feed_dict)
                # zs_test = zs_next
                # assert(np.allclose(zs_test, zs_out))
            else:
                # print("recursive zs for {} steps".format(alpha_to_graph))
                zs_next = zs_batch
                for n in range(num_steps):
                    feed_dict = {self.z: zs_next, self.alpha: single_step_alpha}
                    zs_next = self.sess.run(self.z_new, feed_dict=feed_dict)
                zs_out = zs_next
            # already taken n steps at this point, so directly use zs_next
            # without any further modifications: using self.N_f index into w
            # alternatively, could also use self.outputs_orig
            best_inputs = {self.z: zs_out, self.y: ys_batch,
                           self.truncation: trunc, self.alpha: self.N_f}
            best_im_out = self.sess.run(self.transformed_output,
                                        best_inputs)
            return best_im_out


    def vis_image_batch_alphas(self, graph_inputs, filename,
                               alphas_to_graph, alphas_to_target,
                               batch_start, wgt=False, wmask=False):

        zs_batch = graph_inputs[self.z]
        ys_batch = graph_inputs[self.y]
        trunc = graph_inputs[self.truncation]

        filename_base = filename
        ims_target = []
        ims_transformed = []
        ims_mask = []
        for ag, at in zip(alphas_to_graph, alphas_to_target):
            input_test = {self.y: ys_batch,
                          self.z: zs_batch,
                          self.truncation: trunc}
            out_input_test = self.sess.run(self.outputs_orig, input_test)

            target_fn, mask_out = self.get_target_np(out_input_test, at)
            best_im_out = self.apply_alpha(input_test, ag)

            ims_target.append(target_fn)
            ims_transformed.append(best_im_out)
            ims_mask.append(mask_out)

        for ii in range(zs_batch.shape[0]):
            arr_gt = np.stack([x[ii, :, :, :] for x in ims_target], axis=0)
            if wmask:
                arr_transform = np.stack([x[j, :, :, :] * y[j, :, :, :] for x, y
                                          in zip(ims_zoomed, ims_mask)], axis=0)
            else:
                arr_transform = np.stack([x[ii, :, :, :] for x in
                                          ims_transformed], axis=0)
            arr_gt = self.clip_ims(arr_gt)
            arr_transform = self.clip_ims(arr_transform)
            if wgt:
                ims = np.concatenate((arr_gt,arr_transform), axis=0)
            else:
                ims = arr_transform

            filename = filename_base + '_sample{}'.format(ii+batch_start)
            if wgt:
                filename += '_wgt'
            if wmask:
                filename += '_wmask'
            image.save_im(image.imgrid(ims, cols=len(alphas_to_graph)), filename)

    def vis_image_batch(self, graph_inputs, filename,
                        batch_start, wgt=False, wmask=False, num_panels=7):
        raise NotImplementedError('Subclass should implement vis_image_batch')

class PixelTransform(TransformGraph):
    def __init__(self, *args, **kwargs):
        TransformGraph.__init__(self, *args, **kwargs)

    def get_distribution_statistic(self, img, channel=None):
        raise NotImplementedError('Subclass should implement get_distribution_statistic')

    def get_category_list(self):
        return dataset.get_wid_imagenet()

    def distribution_data_per_category(self, num_categories, num_samples,
                                       output_path, channel=None):
        raise NotImplementedError('Coming soon')


    def distribution_model_per_category(self, num_categories, num_samples,
                                        a, output_path, channel=None):
        raise NotImplementedError('Coming soon')

    def get_distributions_per_category(self, num_categories, num_samples,
                                       output_path, palpha, nalpha, channel=None):
        raise NotImplementedError('Coming soon')

    def get_distributions_all_categories(self, num_samples, output_path,
                                         channel=None):
        raise NotImplementedError('Coming soon')


class BboxTransform(TransformGraph):
    def __init__(self, *args, **kwargs):
        TransformGraph.__init__(self, *args, **kwargs)

    def get_distribution_statistic(self, img, channel=None):
        raise NotImplementedError('Subclass should implement get_distribution_statistic')

    def get_category_list(self):
        return get_coco_imagenet_categories()

    def distribution_data_per_category(self, num_categories, num_samples,
                                       output_path, channel=None):
        raise NotImplementedError('Coming soon')

    def distribution_model_per_category(self, num_categories, num_samples,
                                        a, output_path, channel=None):
        raise NotImplementedError('Coming soon')

    def get_distributions_per_category(self, num_categories, num_samples,
                                       output_path, palpha, nalpha,
                                       channel=None):
        raise NotImplementedError('Coming soon')

    def get_distributions_all_categories(self, num_samples, output_path,
                                         channel=None):
        raise NotImplementedError('Coming soon')

