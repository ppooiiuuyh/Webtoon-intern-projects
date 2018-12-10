import os
import time
import tensorflow as tf
from utils import *
from imresize import *
from metrics import *
import matplotlib.pyplot as plt
import pprint
import math
import numpy as np
import sys
import glob
from tqdm import tqdm
import argparse
from ops import *


class ARCNN(object):
# ==========================================================
# class initializer
# ==========================================================
    def __init__(self, sess):
        self.sess = sess
        self.model_name = "ARCNN"
        self.model_dict = {"batch_size":80,
                           "patch_size":40,
                           "num_blocks":8,
                           "c_dim":1}

        self.train_dict = {"base_lr":1e-5,
                           "min_lr": 1e-6,
                           "lr_-decay_rate":1e-1,
                           "lr_step_size":20,
                           "epoch":80,
                           "batch_size":128}

        self.model()
        #self.other_tensors()
        #self.init_model()


# ==========================================================
# build model
# ==========================================================
    def model(self):
        with tf.variable_scope(self.model_name) as scope:
            shared_inner_model_template = tf.make_template('shared_model', self.inner_model)

            self.images = tf.placeholder(tf.float32, [None, self.model_dict["patch_size"], self.model_dict["patch_size"], self.model_dict["c_dim"]],  name='images')
            self.labels = tf.placeholder(tf.float32, [None, self.model_dict["patch_size"], self.model_dict["patch_size"], self.model_dict["c_dim"]],  name='labels')
            self.output = [shared_inner_model_template(self.images,4,i) for i in range(min(len(combination(8,4)),5))]


            self.image_test = tf.placeholder(tf.float32, [1, None, None, self.model_dict["c_dim"]], name='image_test')
            self.label_test = tf.placeholder(tf.float32, [1, None, None, self.model_dict["c_dim"]], name='labels_test')
            self.output_test = [shared_inner_model_template(self.image_test,4,i) for i in range(min(len(combination(8,4)),5))]

        show_variables(self.model_name)


# ===========================================================
# inner model
# ===========================================================
    def inner_model(self, inputs,num_use,idx):
        channel = 64
        cardinality = 8

        """ input layer """
        with tf.variable_scope("input_layer") as scope:
            layer = conv(inputs, channels=channel, kernel = 3, stride=1, pad=1, pad_type='reflect', use_bias=True, sn=False, scope='conv_0')
            alpha = tf.get_variable("prelu_alpha",[1],initializer=tf.constant_initializer(0.2))
            layer = tf.nn.leaky_relu(layer,alpha=alpha)


        """ hidden layer """
        for i in range(self.model_dict["num_blocks"]):
            with tf.variable_scope("resnext_b{}".format(i)) as scope:
                layer = resNext_CWD(layer,channels=channel,num_card=cardinality,num_use=num_use,idx=idx)
                alpha = tf.get_variable("prelu_alpha", [1], initializer=tf.constant_initializer(0.2))
                layer = tf.nn.leaky_relu(layer, alpha=alpha)


        """ output layer """
        with tf.variable_scope("output_layer") as scope:
            layer = conv(layer, channels=self.model_dict["c_dim"], kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, sn=False, scope='conv_0')
            layer = tf.identity(layer + inputs, name="output")

        return layer





# ============================================================
# other tensors related with training
# ============================================================
    def other_tensors(self):
        with tf.variable_scope("trainer") as scope:
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.loss = tf.reduce_mean(tf.square(self.pred - self.labels))  # L1 is betther than L2
            self.learning_rate = tf.maximum(tf.train.exponential_decay(self.args.base_lr, self.global_step,
                                                                           len(self.train_label) // self.args.batch_size * self.args.lr_step_size,
                                                                           self.args.lr_decay_rate,
                                                                           staircase=True),
                                                self.args.min_lr)  # stair case showed better result

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

            # tensor board
            self.summary_writer = tf.summary.FileWriter("./board", self.sess.graph)
            self.loss_history = tf.summary.scalar("loss", self.loss)
            self.summary = tf.summary.merge_all()
            self.psnr_history = []
            self.ssim_history = []



# ============================================================
# init tensors
# ============================================================
    def init_model(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=0)
        if self.cpkt_load(self.args.checkpoint_dir, self.args.cpkt_itr):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    def cpkt_save(self, checkpoint_dir, step):
        model_name = "checks.model"
        model_dir = "checks"
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.type, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def cpkt_load(self, checkpoint_dir, checkpoint_itr):
        print(" [*] Reading checkpoints...")
        model_dir = "checks"
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.type, model_dir)

        if checkpoint_itr == 0:
            print("train from scratch")
            return True

        elif checkpoint_dir == -1:
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)

        else:
            ckpt = os.path.join(checkpoint_dir, "checks.model-" + str(checkpoint_itr))

        print(ckpt)
        if ckpt:
            self.saver.restore(self.sess, ckpt)
            return True
        else:
            return False





# ==========================================================
# functions
# ==========================================================
    def inference(self, input_img):
        if (np.max(input_img) > 1): input_img = (input_img / 255).astype(np.float32)

        size = input_img.shape
        if (len(input_img.shape) == 3):
            infer_image_input = input_img[:, :, 0].reshape(1, size[0], size[1], 1)
        else:
            infer_image_input = input_img.reshape(1, size[0], size[1], 1)
        sr_img = self.sess.run(self.pred_test, feed_dict={self.image_test: infer_image_input})
        # sr_img = np.expand_dims(sr_img,axis=-1)


        #input_img = imresize(input_img,self.args.scale)
        if (len(input_img.shape) == 3):
            input_img[:, :, 0] = sr_img[0, :, :, 0]
        else:
            input_img = sr_img[0]

        return input_img #return as ycbcr




# ==========================================================
# train
# ==========================================================
    def train(self):
        self.test()
        print("Training...")
        start_time = time.time()


        for ep in range(self.args.epoch):
            # =============== shuffle and prepare batch images ============================
            seed = int(time.time())
            np.random.seed(seed); np.random.shuffle(self.train_label)
            np.random.seed(seed); np.random.shuffle(self.train_input)

            #================ train rec ===================================================
            batch_idxs = len(self.train_label) // self.args.batch_size
            for idx in tqdm(range(0, batch_idxs)):
                batch_labels = np.expand_dims(np.array(self.train_label[idx * self.args.batch_size: (idx + 1) * self.args.batch_size])[:,:,:,0],-1)
                batch_inputs = np.expand_dims(np.array(self.train_input[idx * self.args.batch_size: (idx + 1) * self.args.batch_size])[:,:,:,0],-1)

                feed = {self.images: batch_inputs, self.labels:batch_labels}
                _, err, lr, summary = self.sess.run( [self.train_op, self.loss, self.learning_rate, self.summary], feed_dict=feed)
                self.summary_writer.add_summary(summary,self.global_step.eval())



            #=============== print log =====================================================
            if ep % 1 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_com: [%.8f], lr: [%.8f]" \
                      % ((ep + 1), self.global_step.eval(), time.time() - start_time, np.mean(err), lr))
                self.test()


            #================ save checkpoints ===============================================
            if ep % self.args.save_period == 0:
                self.cpkt_save(self.args.checkpoint_dir, ep + 1)


# ==========================================================
# test
# ==========================================================
    def test(self):
        print("Testing...")
        psnrs_preds = []
        ssims_preds = []

        preds = []
        labels = []
        images = []

        for idx in range(0, len(self.test_label)):
            test_label = np.array(self.test_label[idx]) #none,none,3
            test_input = np.array(self.test_input[idx])

            # === original =====
            for f in [5, 10, 20, 40, 50, 60, 80, 90, 100]:
                cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "ori" + str(f) + ".jpg"),
                            (ycbcr2rgb(test_label)*255)[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), f])
            cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "ori.PNG"), (ycbcr2rgb(test_label)*255)[..., ::-1])
            # ==================
            result = self.inference(test_input)
            cv2.imwrite(os.path.join(self.args.result_dir,str(idx)+ "rec"+str(self.args.jpgqfactor)+".bmp"), (ycbcr2rgb(result)*255)[...,::-1])

            preds.append(result)
            labels.append(test_label)


        # cal PSNRs for each images upscaled from different depths
        for i in range(len(self.test_label)):
            if len(np.array(labels[i]).shape)==3 : labels[i] = np.array(labels[i])[:,:,0]
            if len(np.array(preds[i]).shape)==3 : preds[i] = np.array(preds[i])[:,:,0]
            psnrs_preds.append(psnr(labels[i], preds[i], max=1.0, scale=self.args.scale))
            ssims_preds.append(ssim(labels[i], preds[i], max=1.0, scale=self.args.scale))

        # print evalutaion results
        print("===================================================================================")
        print("PSNR: " + str(round(np.mean(np.clip(psnrs_preds, 0, 100)), 3)) + "dB")
        print("SSIM: " + str(round(np.mean(np.clip(ssims_preds, 0, 100)), 5)))
        print("===================================================================================")

        self.psnr_history.append(str(round(np.mean(np.clip(psnrs_preds, 0, 100)), 3)))
        self.ssim_history.append(str(round(np.mean(np.clip(ssims_preds, 0, 100)), 5)))
        print()
        print(self.psnr_history)
        print(self.ssim_history)





#=========================  test case =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ common configuration """
    parser.add_argument("--gpu", type=int, default=3)  # -1 for CPU
    args = parser.parse_args()

    """ system configuration """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session(config=config) as sess:
        model = ARCNN(sess=sess)
        tf.summary.FileWriter("./board", sess.graph)
