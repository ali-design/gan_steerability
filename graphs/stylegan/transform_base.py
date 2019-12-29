import tensorflow as tf
import numpy as np
import os
import pickle
from . import constants
from .graph_util import *
from resources import tf_lpips_pkg as lpips_tf
from utils import image, util
import time

# this is mostly to solve pickle issues
import sys
sys.path.append('resources/stylegan')
import dnnlib
import dnnlib.tflib as tflib
import config

class TransformGraph():
    def __init__(self, lr, walk_type, nsliders, loss_type, eps, N_f,
                 stylegan_opts):

        assert(loss_type in ['l2', 'lpips']), 'unimplemented loss'
        assert(stylegan_opts.latent in ['z', 'w']), 'unknown latent space'

        self.dataset_name = stylegan_opts.dataset
        self.dataset = constants.net_info[stylegan_opts.dataset]
        self.latent = stylegan_opts.latent
        if hasattr(stylegan_opts, 'truncation_psi'):
            self.psi = stylegan_opts.truncation_psi
        else:
            self.psi = 1.0

        tflib.init_tf()
        with dnnlib.util.open_url(self.dataset['url'], cache_dir=config.cache_dir) as f:
            # can only unpickle where dnnlib is importable, so add to syspath
            _G, _D, Gs = pickle.load(f)

        # input placeholders
        Nsliders = nsliders
        dim_z = self.dim_z = Gs.input_shape[1]
        z = tf.placeholder(tf.float32, shape=(None, dim_z))

        # original output
        outputs_orig = tf.transpose(Gs.get_output_for(
            z, None, is_validation=True, randomize_noise=False), [0, 2, 3, 1])

        img_size = self.img_size = outputs_orig.shape[1].value
        num_channels = self.num_channels = outputs_orig.shape[-1].value

        # output placeholders
        target = tf.placeholder(tf.float32, shape=(
            None, img_size, img_size, num_channels))
        mask = tf.placeholder(tf.float32, shape=(
            None, img_size, img_size, num_channels))

        if self.latent == 'w':
            # shape [None, 16, 512]
            latent_space_orig = Gs.components.mapping.get_output_for(z, None, is_validation=True, randomize_noise=False)
            # flatten it: shape [None, 8092]
            latent_space = tf.reshape(latent_space_orig, [-1, latent_space_orig.shape[1] * latent_space_orig.shape[2]])
        else:
            # shape [None, 512]
            latent_space = z

        dim_latent_space = latent_space.shape[1].value


        # walk pattern
        scope = 'walk'
        if walk_type == 'NNz':
            with tf.name_scope(scope):
                # alpha is the integer number of steps to take
                alpha = tf.placeholder(tf.int32, shape=())
                avg_w = Gs.get_var('dlatent_avg')
                nreps = latent_space.shape[1] // len(avg_w)
                avg_w = tf.expand_dims(tf.concat([avg_w] * nreps, axis=0), axis=0)
                T1 = tf.keras.layers.Dense(dim_latent_space, input_shape=(None, dim_latent_space), kernel_initializer='glorot_normal', bias_initializer='zeros', activation=tf.nn.relu)
                T2 = tf.keras.layers.Dense(dim_latent_space, input_shape=(None, dim_latent_space), kernel_initializer='glorot_normal', bias_initializer='zeros')
                T3 = tf.keras.layers.Dense(dim_latent_space, input_shape=(None, dim_latent_space), kernel_initializer='glorot_normal', bias_initializer='zeros', activation=tf.nn.relu)
                T4 = tf.keras.layers.Dense(dim_latent_space, input_shape=(None, dim_latent_space), kernel_initializer='glorot_normal', bias_initializer='zeros')
                # forward transformation
                out_f = []
                latent_space_prev = latent_space
                latent_space_norm = tf.norm(latent_space, axis=1, keepdims=True)
                for i in range(1, N_f + 1):
                    latent_space_step = latent_space_prev + T2(T1(latent_space_prev))
                    # normalize only in z
                    if self.latent == 'z':
                        latent_space_step_norm = tf.norm(latent_space_step, axis=1, keepdims=True)
                        latent_space_step = latent_space_step * latent_space_norm / latent_space_step_norm
                    elif self.latent == 'w':
                        latent_space_step = avg_w + self.psi * (latent_space_step - avg_w)
                    out_f.append(latent_space_step)
                    latent_space_prev = latent_space_step

                # reverse transformation
                out_g = []
                latent_space_prev = latent_space
                latent_space_norm = tf.norm(latent_space, axis=1, keepdims=True)
                for i in range(1, N_f + 1):
                    latent_space_step = latent_space_prev + T4(T3(latent_space_prev))
                    # normalize only in z space
                    if self.latent == 'z':
                        latent_space_step_norm = tf.norm(latent_space_step, axis=1, keepdims=True)
                        latent_space_step = latent_space_step * latent_space_norm / latent_space_step_norm
                    elif self.latent == 'w':
                        latent_space_step = avg_w + self.psi * (latent_space_step - avg_w)
                    out_g.append(latent_space_step)
                    latent_space_prev = latent_space_step
                out_g.reverse() # flip the reverse transformation

                # w has shape (2*N_f + 1, batch_size, dim_latent_space)
                # elements 0 to N_f are the reverse transformation, in reverse order
                # elements N_f + 1 to 2*N_f + 1 are the forward transformation
                # element N_f is no transformation
                w = tf.stack(out_g+[latent_space]+out_f)
        elif walk_type == 'linear':
            alpha = tf.placeholder(tf.float32, shape=(None, Nsliders))
            w_shape = [1, latent_space.shape[1], Nsliders]
            w = tf.Variable(np.random.normal(0.0, 0.1, w_shape), name=scope, dtype=np.float32)
            N_f = None
            eps = None
        else:
            raise NotImplementedError('Not implemented walk type:' '{}'.format(walk_type))

        # transformed output
        latent_space_new = latent_space
        if walk_type == 'linear':
            for i in range(Nsliders):
                latent_space_new = latent_space_new+tf.expand_dims(
                    alpha[:,i], axis=1)*w[:, :,i]
        elif walk_type == 'NNz':
            # z_new = tf.reshape(tf.linalg.matmul(alpha, tf.reshape(w, [N_f*2+1, -1])), [-1, dim_z])
            # w is already z+f(z) so we can just index into w
            latent_space_new = w[alpha, :, :]
            # embed()

        if self.latent == 'w':
            latent_space_new_reshape = tf.reshape(latent_space_new, [-1, latent_space_orig.shape[1], latent_space_orig.shape[2]])
            # latent space new is now shape [None, 16, 512] again
            transformed_output = tf.transpose(Gs.components.synthesis.get_output_for(
                latent_space_new_reshape, is_validation=True, randomize_noise=False), [0, 2, 3, 1])
        else:
            transformed_output = tf.transpose(Gs.get_output_for(
                latent_space_new, None, is_validation=True, randomize_noise=False), [0, 2, 3, 1])

        # L_2 loss
        loss = tf.losses.compute_weighted_loss(tf.square(transformed_output-target), weights=mask)
        loss_lpips = tf.reduce_mean(lpips_tf.lpips(
            mask*transformed_output, mask*target, model='net-lin', net='alex'))

        # losses per sample
        loss_lpips_sample = lpips_tf.lpips(
            mask*transformed_output, mask*target, model='net-lin', net='alex')
        loss_l2_sample = tf.reduce_sum(tf.multiply(tf.square(
            transformed_output-target), mask), axis=(1,2,3)) \
                / tf.reduce_sum(mask, axis=(1,2,3))

        # set the scope to be 'walk'
        if loss_type == 'l2':
            train_step = tf.train.AdamOptimizer(lr).minimize(
                    loss, var_list=tf.trainable_variables(scope=scope), name='AdamOpter')
        elif loss_type == 'lpips':
            train_step = tf.train.AdamOptimizer(lr).minimize(
                    loss_lpips, var_list=tf.trainable_variables(scope=scope), name='AdamOpter')


            # rescale the stylegan output range ([-1, 1]) to uint8 range [0, 255]
        float_im = tf.placeholder(tf.float32, outputs_orig.shape)
        uint8_im = tflib.convert_images_to_uint8(tf.convert_to_tensor(float_im, dtype=tf.float32))

        # set class vars
        self.Nsliders = Nsliders
        self.z = z
        self.alpha = alpha
        self.target = target
        self.mask = mask
        self.w = w
        self.latent_space = latent_space
        self.latent_space_new = latent_space_new
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
        self.float_im = float_im
        self.uint8_im = uint8_im
        self.scope = scope

    def initialize_graph(self):
        sess = tf.get_default_session()
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        print([str(i.name) for i in not_initialized_vars]) # only for testing
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))
        print("trainable vars: {}".format(tf.trainable_variables(scope=self.scope)))
        saver = tf.train.Saver(tf.trainable_variables(scope=self.scope)) # todo double check
        self.sess = sess
        self.saver = saver

    def clip_ims(self, ims):
        return self.sess.run(self.uint8_im, {self.float_im: ims})

    def apply_alpha(self, graph_inputs, alpha_to_graph):

        zs_batch = graph_inputs[self.z]

        # print(alpha_to_graph)
        if self.walk_type == 'linear':
            best_inputs = {self.z: zs_batch, self.alpha: alpha_to_graph}
            best_im_out = self.sess.run(self.transformed_output, best_inputs)
            return best_im_out
        elif self.walk_type.startswith('NN'):
            # alpha_to_graph is number of steps and direction
            direction = np.sign(alpha_to_graph)
            num_steps = np.abs(alpha_to_graph)
            # embed()
            single_step_alpha = self.N_f + direction
            # within the graph range, we can compute it directly
            if 0 <= alpha_to_graph + self.N_f <= self.N_f * 2:
                latent_space_out = self.sess.run(self.latent_space_new, {
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
                latent_space_batch = self.sess.run(self.latent_space, {self.z: zs_batch})
                latent_space_next = latent_space_batch
                for n in range(num_steps):
                    feed_dict = {self.latent_space: latent_space_next, self.alpha: single_step_alpha}
                    latent_space_next = self.sess.run(self.latent_space_new, feed_dict=feed_dict)
                latent_space_out = latent_space_next
            # already taken n steps at this poin
            best_inputs = {self.latent_space_new: latent_space_out} #, self.alpha: self.N_f}
            best_im_out = self.sess.run(self.transformed_output,
                                        best_inputs)
            return best_im_out


    def vis_image_batch_alphas(self, graph_inputs, filename,
                               alphas_to_graph, alphas_to_target,
                               batch_start, wgt=False, wmask=False):

        zs_batch = graph_inputs[self.z]

        filename_base = filename
        ims_target = []
        ims_transformed = []
        ims_mask = []
        for ag, at in zip(alphas_to_graph, alphas_to_target):
            input_test = {self.z: zs_batch}
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

    def get_distribution(self, num_samples, channel=None):
        random_seed = 0
        rnd = np.random.RandomState(random_seed)
        inputs = graph_input(self, num_samples, seed=random_seed)
        batch_size = constants.BATCH_SIZE
        model_samples = []
        for a in self.test_alphas():
            distribution = []
            start = time.time()
            print("Computing attribute statistic for alpha={:0.2f}".format(a))
            for batch_num, batch_start in enumerate(range(0, num_samples,
                                                          batch_size)):
                s = slice(batch_start, min(num_samples, batch_start + batch_size))
                inputs_batch = util.batch_input(inputs, s)
                zs_batch = inputs_batch[self.z]
                a_graph = self.scale_test_alpha_for_graph(a, zs_batch, channel)
                ims = self.clip_ims(self.apply_alpha(inputs_batch, a_graph))
                for img in ims:
                    img_stat = self.get_distribution_statistic(img, channel)
                    distribution.extend(img_stat)
            end = time.time()
            print("Sampled {} images in {:0.2f} min".format(num_samples, (end-start)/60))
            model_samples.append(distribution)

        model_samples = np.array(model_samples)
        return model_samples

class BboxTransform(TransformGraph):
    def __init__(self, *args, **kwargs):
        TransformGraph.__init__(self, *args, **kwargs)

    def get_distribution_statistic(self, img, channel=None):
        raise NotImplementedError('Subclass should implement get_distribution_statistic')

    def get_distribution(self, num_samples, **kwargs):
        if 'is_face' in self.dataset:
            from utils.detectors import FaceDetector
            self.detector = FaceDetector()
        elif self.dataset['coco_id'] is not None:
            from utils.detectors import ObjectDetector
            self.detector = ObjectDetector(self.sess)
        else:
            raise NotImplementedError('Unknown detector option')

        # not used for faces: class_id=None
        class_id = self.dataset['coco_id']

        random_seed = 0
        rnd = np.random.RandomState(random_seed)

        model_samples = []
        for a in self.test_alphas():
            distribution = []
            total_count = 0
            start = time.time()
            print("Computing attribute statistic for alpha={:0.2f}".format(a))
            while len(distribution) < num_samples:
                inputs = graph_input(self, 1, seed=total_count)
                zs_batch = inputs[self.z]
                a_graph = self.scale_test_alpha_for_graph(a, zs_batch)
                ims = self.clip_ims(self.apply_alpha(inputs, a_graph))
                img = ims[0, :, :, :]
                img_stat = self.get_distribution_statistic(img, class_id)
                if len(img_stat) == 1:
                    distribution.extend(img_stat)
                total_count += 1
            end = time.time()
            print("Sampled {} images to detect {} boxes in {:0.2f} min".format(
                total_count, num_samples, (end-start)/60))
            model_samples.append(distribution)

        model_samples = np.array(model_samples)
        return model_samples
