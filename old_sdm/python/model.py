import tensorflow as tf
import numpy as np
import time,os,random
import scipy.io as sio
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
import scipy.interpolate as interpolate
import h5py
import icp
xrange = range

from utils import *
from render import *
from feature2vertex import *


timeline_use = False
tensorboard = False
vaeshare  = False

class SDM_NET():

    VAE = 'SDM-NET'
    no_opt = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.ON_2,
                            do_common_subexpression_elimination=True,
                            do_function_inlining=True,
                            do_constant_folding=True)
    gpu_opt = tf.GPUOptions(allocator_type = 'BFC', allow_growth = True, per_process_gpu_memory_fraction=0.8) #there was no per_process
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=no_opt),
                        allow_soft_placement=False, gpu_options = gpu_opt )

#-------------------------------------------------------------network framework---------------------------------------------------------------------------------------

    def __init__(self, datainfo):
        self.activate = datainfo.activate
        self.union = datainfo.jointly                                                     # training jointly or separately
        self.interval = 50
        self.modelname = datainfo.modelname

        self.part_name = datainfo.part_name
        self.part_num = len(datainfo.part_name)
        self.part_dim = 2*self.part_num+9
        self.cube_point_num = np.shape(datainfo.mesh.vertices)[0]
        self.cube_vertex_dim = 9

        self.batch_size = datainfo.batch_size
        self.pointnum = datainfo.pointnum
        self.vertex_dim = datainfo.vertex_dim
        self.hiddendim = datainfo.hidden_dim
        # self.latent_dim = datainfo.latent_dim
        self.maxdegree = datainfo.maxdegree
        self.finaldim = datainfo.finaldim
        self.layers = datainfo.layers
        self.lambda0 = datainfo.lambda0
        # self.lambda1 = datainfo.lambda1
        self.lambda2 = datainfo.lambda2
        self.lambda3 = datainfo.lambda3
        self.lambda4 = datainfo.lambda4
        # self.lambda5 = datainfo.lambda5
        self.lr = datainfo.lr
        self.decay_step = 5000
        self.decay_rate = 0.8
        self.maxepoch_deform = datainfo.epoch_deform
        self.maxepoch_structure = datainfo.epoch_structure

        self.num_vaes = datainfo.num_vaes
        self.reconm = datainfo.reconmatrix
        self.w_vdiff = datainfo.vdiff
        self.control_id = datainfo.control_idx
        self.all_vertex = datainfo.all_vertex
        self.deform_reconmatrix_holder = datainfo.deform_reconmatrix
        self.mesh = datainfo.mesh

        self.neighbourmat = datainfo.neighbour
        self.degrees = datainfo.degrees
        self.feature = datainfo.feature
        self.symmetry_feature = datainfo.symmetry_feature
        self.cot_w = datainfo.cotw1
        self.L1_ = datainfo.L1

        self.outputdir = datainfo.output_dir
        self.flog = datainfo.flog
        self.iddat_name = datainfo.iddat_name
        self.train_percent = datainfo.train_percent
        self.train_id = datainfo.train_id
        self.valid_id = datainfo.valid_id

        if not datainfo.use_ae_or_vae.find('tan') == -1:
            self.ae = True
            self.lambda3 = 0.0
            datainfo.lambda3 = 0.0
        else:
            self.ae = False

        # tf.set_random_seed(1)

        self.inputs_feature = tf.placeholder(tf.float32, [None, self.part_num, self.cube_point_num, self.cube_vertex_dim], name = 'input_mesh') #feature matrix
        self.inputs_symmetry = tf.placeholder(tf.float32, [None, self.part_num, self.part_dim], name = 'input_mesh_sym') #structure information

        inputs_feature = self.inputs_feature
        inputs_symmetry = self.inputs_symmetry

        self.nb = tf.constant(self.neighbourmat, dtype = 'int32', shape=[self.cube_point_num, self.maxdegree], name='nb_relation')
        self.degrees = tf.constant(self.degrees, dtype = 'float32', shape=[self.cube_point_num, 1], name = 'degrees')
        self.cw = tf.constant(self.cot_w, dtype = 'float32', shape=[self.cube_point_num, self.maxdegree, 1], name = 'cw')

        # ---------------------------------------------------feature2vertex
        self.f2v = Feature2Vertex(datainfo, name = 'sdm-net')

        self.initial_vae()
        variables_vae = []
        print(self.part_num)
        for i in range(self.part_num):
            if vaeshare:
                reuse = tf.AUTO_REUSE
                output_vae = self.build_vae_block(inputs_feature[:,i,:,:], self.hiddendim[0], name = 'vae_block', reuse = reuse)
                variables_vae.append(slim.get_variables(scope='vae_block'))
            else:
                output_vae = self.build_vae_block(inputs_feature[:,i,:,:], self.hiddendim[0], name = 'vae_block' + self.part_name[i]) #if no share, each partVAE has its own name
                variables_vae.append(slim.get_variables(scope='vae_block' + self.part_name[i]))
            self.post_output_vae(output_vae) #!!!!!!!
            # partvae until here

        if self.union: #default 0
            hidden_code = tf.transpose(self.encode, perm = [1, 0, 2])
            print(np.shape(hidden_code))
            structure_feature = tf.reshape(tf.concat([hidden_code, inputs_symmetry], axis = 2), [tf.shape(self.encode)[1], -1])
        else:

            hidden_code = tf.transpose(self.test_encode, perm = [1, 0, 2])
            print(np.shape(hidden_code))
            structure_feature = tf.reshape(tf.concat([hidden_code, inputs_symmetry], axis = 2), [tf.shape(self.test_encode)[1], -1]) #obtain features for SPVAE

        output_vae = self.build_vae_block_for_structure(structure_feature, self.hiddendim[1], name = 'vae_block_structure') #for SPVAE

        self.post_output_vae(output_vae) #for SPVAE
        #SPVAE untill here, so the latent vecotrs of SPVAE stores in the 7th of self.test_code

        variables_vae.append(slim.get_variables(scope="vae_block_structure"))

        train_variables_vae_all = []
        for id in range(self.part_num + 1):
            variables = variables_vae[id]
            train_variables_vae = []
            for v in variables:
                if v in tf.trainable_variables():
                    train_variables_vae.append(v)

            train_variables_vae_all.append(train_variables_vae)

        print(np.expand_dims(tf.trainable_variables(), axis = 0))
        # print(np.expand_dims(tf.global_variables(), axis = 0))
        self.get_total_loss()
        self.train_op = []
        if self.union:
            global_step_all = tf.Variable(0, trainable = False, name = 'global_step_all')
            learning_rate_deform = tf.train.exponential_decay(self.lr, global_step_all, 3000, self.decay_rate, staircase = True)
            learning_rate_deform = tf.maximum(learning_rate_deform, 0.0000001)
            optimizer_vae = tf.train.AdamOptimizer(learning_rate_deform, name='Adam_vae_block_all')
            self.total_trainop = tf.contrib.training.create_train_op(tf.reduce_sum(self.loss_vae), optimizer_vae, global_step = global_step_all,
                                                                variables_to_train=tf.trainable_variables(),
                                                                summarize_gradients=False)
        else:
            for id in range(self.part_num + 1):
                if id == self.part_num:
                    name = 'struc'
                else:
                    name = self.part_name[id]
                global_step_all = tf.Variable(0, trainable = False, name = 'global_step_'+name)
                if id == self.part_num:
                    learning_rate_deform = tf.train.exponential_decay(self.lr, global_step_all, 3000, self.decay_rate, staircase = True)
                else:
                    learning_rate_deform = tf.train.exponential_decay(self.lr, global_step_all, 1000, self.decay_rate, staircase = True)

                learning_rate_deform = tf.maximum(learning_rate_deform, 0.0000001)
                optimizer_vae = tf.train.AdamOptimizer(learning_rate_deform, name='Adam_vae_block'+name)
                trainop = tf.contrib.training.create_train_op(self.loss_vae[id], optimizer_vae, global_step = global_step_all,
                                                                variables_to_train=train_variables_vae_all[id],
                                                                summarize_gradients=False)
                self.train_op.append(trainop)

        self.checkpoint_dir_structure = os.path.join(self.outputdir, 'structure_ckt')

        if tensorboard:
            lr_all = tf.summary.scalar('all_learning_rate', learning_rate_deform)
            self.train_summary_var.append([lr_all])
            self.get_var_summary()
        else:
            self.train_summary = tf.constant(0)
            self.valid_summary = tf.constant(0)

        self.saver = tf.train.Saver(max_to_keep = 10)
        # self.print_netinfo()

    def initial_vae(self):
        self.encode = []
        self.decode = []
        self.encode_std = []
        self.test_encode = []
        self.test_decode = []
        self.test_encode_std = []
        self.embedding_inputs = []
        self.embedding_decode = []
        self.generation_loss = []
        self.distance_norm = []
        self.weights_norm = []
        self.kl_diver = []
        self.l2_loss = []
        self.test_loss = []
        self.testkl_diver = []
        self.train_summary_var = []
        self.valid_summary_var = []
        self.loss_vae = []
        self.op_vae = []
        self.valid_summary_one = []
        self.train_summary_one = []
        self.laplacian_check = []
        self.region = []

    def post_output_vae(self, output_vae):
        #encode, decode, test_encode, test_decode, embedding_inputs, embedding_decode, generation_loss, l2_loss, test_generation_loss, 
        #train_summary_var, valid_summary_var, total_loss
        if self.ae:
            self.encode.append(output_vae[0])
            self.decode.append(output_vae[1])
            self.test_encode.append(output_vae[2])
            self.test_decode.append(output_vae[3])
            self.embedding_inputs.append(output_vae[4])
            self.embedding_decode.append(output_vae[5])

            self.generation_loss.append(output_vae[6])
            self.weights_norm.append(output_vae[7])
            self.l2_loss.append(output_vae[8])

            self.test_loss.append(output_vae[9])

            self.train_summary_var.append(output_vae[10])
            self.valid_summary_var.append(output_vae[11])

            self.loss_vae.append(output_vae[12])
            if tensorboard:
                self.valid_summary_one.append(tf.summary.merge(output_vae[11]))
                self.train_summary_one.append(tf.summary.merge(output_vae[10]))
            else:
                self.valid_summary_one.append(tf.constant(0))
                self.train_summary_one.append(tf.constant(0))

        else:
            #encode, decode, encode_std, test_encode, test_decode, test_encode_std, embedding_inputs, embedding_decode,
            #generation_loss, kl_diver, l2_loss, test_generation_loss, testkl_diver, train_summary_var, valid_summary_var, total_loss
            print(len(output_vae))
            print(output_vae[16])

            self.encode.append(output_vae[0])
            self.decode.append(output_vae[1])
            self.encode_std.append(output_vae[2])

            self.test_encode.append(output_vae[3])
            self.test_decode.append(output_vae[4])
            self.test_encode_std.append(output_vae[5])

            self.embedding_inputs.append(output_vae[6])
            self.embedding_decode.append(output_vae[7])

            self.generation_loss.append(output_vae[8])
            self.weights_norm.append(output_vae[9])
            self.kl_diver.append(output_vae[10])
            self.l2_loss.append(output_vae[11])

            self.test_loss.append(output_vae[12])
            self.testkl_diver.append(output_vae[13])

            self.train_summary_var.append(output_vae[14])
            self.valid_summary_var.append(output_vae[15])

            self.loss_vae.append(output_vae[16])
            if tensorboard:
                self.valid_summary_one.append(tf.summary.merge(output_vae[15]))
                self.train_summary_one.append(tf.summary.merge(output_vae[14]))
            else:
                self.valid_summary_one.append(tf.constant(0))
                self.train_summary_one.append(tf.constant(0))

    def get_total_loss(self):

        self.total_generation_loss = tf.reduce_sum(self.generation_loss)
        self.total_weights_norm = tf.reduce_sum(self.weights_norm)
        self.total_l2_loss = tf.reduce_sum(self.l2_loss)

        self.total_test_loss = tf.reduce_sum(self.test_loss)# + self.validtotalvae_loss
        if not self.ae:
            self.total_kl_loss = tf.reduce_sum(self.kl_diver)
            self.total_testkl_loss = tf.reduce_sum(self.testkl_diver)

    def get_var_summary(self):
        t_vars = tf.trainable_variables()
        fc_vars = [tf.summary.histogram(var.name, var) for var in t_vars if 'fclayer' in var.name]

        total_loss = tf.summary.tensor_summary('total_loss', self.loss_vae)
        generation_loss = tf.summary.tensor_summary('generation_loss', self.generation_loss)
        l2_loss = tf.summary.tensor_summary('l2_loss', self.generation_loss)
        test_loss = tf.summary.tensor_summary('test_loss', self.generation_loss)

        self.train_summary_var.append([total_loss, generation_loss, l2_loss]+fc_vars)
        self.valid_summary_var.append([test_loss])

        if not self.ae:
            kl_loss = tf.summary.tensor_summary('kl_loss', self.kl_diver)
            testkl_loss = tf.summary.tensor_summary('testkl_loss', self.testkl_diver)
            self.train_summary_var.append([kl_loss])
            self.valid_summary_var.append([testkl_loss])

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # trainable_variables = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        valid_summary_var = list(np.reshape(self.valid_summary_var, [-1]))

        train_summary_var = [x for x in summaries if x not in valid_summary_var]

        self.valid_summary = tf.summary.merge(valid_summary_var)
        self.train_summary = tf.summary.merge(train_summary_var)

        self.merge_summary = tf.summary.merge_all()

        # self.valid_summary = tf.summary.merge(list(np.reshape(self.valid_summary_var, [-1])))

    def print_netinfo(self):
        # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # # trainable_variables = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        # valid_summary_var = list(np.reshape(self.valid_summary_var, [-1]))

        # train_summary_var = [x for x in summaries if x not in valid_summary_var]

        # variables_encoder = slim.get_variables(scope="vae_block0")

        # variables_encoder = slim.get_variables(scope="*/fclayer")
        # variables_encoder = slim.get_variables(scope="*/fclayer_std")
        # print(np.expand_dims(variables_encoder, axis=0))
        # t_vars = tf.trainable_variables()
        # self.d_vars = [var for var in t_vars if 'd_' in var.name]
        # a = set(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='vae_block0'))
        # reg_losses = [var for var in a if 'batch_norm' not in var.name]
        # print(np.shape(self.decode))
        # print(np.shape(self.test_decode))

        # print(np.expand_dims(valid_summary_var, axis=0))

        # print(np.expand_dims(train_summary_var, axis=0))

        # print(np.expand_dims(summaries, axis=0))

        # [print(x.name) for x in a]
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        [print(x) for x in summaries]

#-------------------------------------------------------------network framework---------------------------------------------------------------------------------------

#-------------------------------------------------------------network tools---------------------------------------------------------------------------------------

    def build_vae_block_for_structure(self, inputs, hiddendim, name = 'vae_block'): #SPVAE
        with tf.variable_scope(name) as scope:
            embedding_inputs = tf.placeholder(tf.float32, [None, hiddendim], name = 'embedding_inputs')

            object_stddev = tf.constant(np.concatenate((np.array([1, 1]).astype('float32'), 1 * np.ones(hiddendim - 2).astype('float32'))))
            self.weight_hist = []
            # train
            if not self.ae:
                encode, encode_std = self.encoder_symm(inputs, training = True)
                encode_gauss = encode + encode_std*embedding_inputs
                decode = self.decoder_symm(encode_gauss, training = True)
            else:
                encode, _ = self.encoder_symm(inputs, training = True)
                decode = self.decoder_symm(encode, training = True)

            if not self.ae:
                kl_diver = 0.5 * tf.reduce_sum(tf.square(encode) + tf.square(encode_std / object_stddev) - tf.log(1e-8 + tf.square(encode_std / object_stddev)) - 1, 1)
                kl_diver = self.lambda3*tf.reduce_mean(kl_diver)
            else:
                kl_diver = 0.0

            # test
            test_encode, test_encode_std = self.encoder_symm(inputs, training = False)
            test_decode = self.decoder_symm(test_encode, training = False)

            if not self.ae:
                testkl_diver = 0.5 * tf.reduce_sum(tf.square(test_encode) + tf.square(test_encode_std / object_stddev) - tf.log(1e-8 + tf.square(test_encode_std / object_stddev)) - 1, 1)
                testkl_diver = self.lambda3*tf.reduce_mean(testkl_diver)
            else:
                testkl_diver = 0.0

            embedding_decode = self.decoder_symm(embedding_inputs, training = False)

            generation_loss = self.lambda0 * tf.reduce_mean(tf.reduce_sum(tf.pow(inputs-decode, 2.0), [1])) * 100

            test_generation_loss = self.lambda0 * tf.reduce_mean(tf.reduce_sum(tf.pow(inputs-test_decode, 2.0), [1])) * 100

            reg_losses = set(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope = name))
            reg_losses = [var for var in reg_losses if 'batch_norm' not in var.name]
            # [print(x) for x in reg_losses]
            weight_norm = tf.constant(0.0, dtype=tf.float32)

            l2_loss = self.lambda4 * sum(reg_losses)

            total_loss = generation_loss + kl_diver + l2_loss
            if tensorboard:
                s1 = tf.summary.scalar('generation_loss', generation_loss)
                s3 = tf.summary.scalar('kl_diver', kl_diver)
                s5 = tf.summary.scalar('l2_loss', l2_loss)

                if self.ae:
                    train_summary_var = [s1, s3, s5, self.weight_hist]
                else:
                    s4 = tf.summary.scalar('kl_diver', kl_diver)
                    train_summary_var = [s1, s3, s4, s5, self.weight_hist]

                s1 = tf.summary.scalar('valid_generation_loss', test_generation_loss)

                if self.ae:
                    valid_summary_var = [s1]
                else:
                    s2 = tf.summary.scalar('valid_KL_loss', testkl_diver)
                    valid_summary_var = [s1, s2]
            else:
                train_summary_var = []
                valid_summary_var = []

        if self.ae:
            return encode, decode, test_encode, test_decode, embedding_inputs, embedding_decode, generation_loss, weight_norm, l2_loss, test_generation_loss, train_summary_var, valid_summary_var, total_loss
        else:
            return encode, decode, encode_std, test_encode, test_decode, test_encode_std, embedding_inputs, embedding_decode, generation_loss, weight_norm, kl_diver, l2_loss, test_generation_loss, testkl_diver, train_summary_var, valid_summary_var, total_loss

    def build_vae_block(self, inputs, hiddendim, name = 'vae_block', reuse = None):
        with tf.variable_scope(name, reuse = reuse) as scope:
            # if vaeshare:
                # scope.reuse_variables()
            vae_block_para_en = []
            vae_block_para_de = []
            n_weight_en = []
            e_weight_en = []
            n_weight_de = []
            e_weight_de = []

            embedding_inputs = tf.placeholder(tf.float32, [None, hiddendim], name = 'embedding_inputs')

            object_stddev = tf.constant(np.concatenate((np.array([1, 1]).astype('float32'), 1 * np.ones(hiddendim - 2).astype('float32'))))

            for i in range(0, self.layers):
                if i == self.layers - 1:
                    n_en, e_en = self.get_conv_weights(self.vertex_dim, self.finaldim, name = 'en_convw'+str(i+1))
                    n_de, e_de = self.get_conv_weights(self.vertex_dim, self.finaldim, name = 'de_convw'+str(i+1))
                else:
                    n_en, e_en = self.get_conv_weights(self.vertex_dim, self.vertex_dim, name = 'en_convw'+str(i))
                    n_de, e_de = self.get_conv_weights(self.vertex_dim, self.vertex_dim, name = 'de_convw'+str(i))

                n_weight_en.append(n_en)
                e_weight_en.append(e_en)
                n_weight_de.append(n_de)
                e_weight_de.append(e_de)
                if tensorboard:
                    tf.summary.histogram('convw_n_en'+str(i+1), n_en)
                    tf.summary.histogram('convw_e_en'+str(i+1), e_en)
                    tf.summary.histogram('convw_n_de'+str(i+1), n_de)
                    tf.summary.histogram('convw_e_de'+str(i+1), e_de)

                if i == 0:
                    l2_loss = tf.nn.l2_loss(n_en) + tf.nn.l2_loss(e_en) + tf.nn.l2_loss(n_de) + tf.nn.l2_loss(e_de)
                else:
                    l2_loss += tf.nn.l2_loss(n_en) + tf.nn.l2_loss(e_en) + tf.nn.l2_loss(n_de) + tf.nn.l2_loss(e_de)

            vae_block_para_en.append(n_weight_en)
            vae_block_para_en.append(e_weight_en)
            vae_block_para_de.append(n_weight_de)
            vae_block_para_de.append(e_weight_de)

            fcparams_en = tf.get_variable("weights_en", [self.pointnum*self.finaldim, hiddendim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            vae_block_para_en.append(fcparams_en)
            fcparams_de = tf.get_variable("weights_de", [self.pointnum*self.finaldim, hiddendim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            vae_block_para_de.append(fcparams_de)

            if not self.ae:
                fcparams_std_en = tf.get_variable("std_weights_en", [self.pointnum*self.finaldim, hiddendim], tf.float32, tf.random_normal_initializer(stddev=0.02))
                vae_block_para_en.append(fcparams_std_en)

            if self.ae:
                l2_loss += tf.nn.l2_loss(fcparams_de) + tf.nn.l2_loss(fcparams_en)
            else:
                l2_loss += tf.nn.l2_loss(fcparams_de) + tf.nn.l2_loss(fcparams_en) + tf.nn.l2_loss(fcparams_std_en)

            l2_loss = self.lambda4 * l2_loss

            if not self.ae:
                encode, l0, weights_norm = self.encoder(inputs, vae_block_para_en, train = True, name = name)
                encode_std = self.encoder_std(l0, fcparams_std_en)
                encode_gauss = encode + encode_std*embedding_inputs
                decode = self.decoder(encode_gauss, vae_block_para_de, train = True, name = name)
            else:
                encode, _ , weights_norm= self.encoder(inputs, vae_block_para_en, train = True, name = name)
                decode = self.decoder(encode, vae_block_para_de, train = True, name = name)

            weights_norm = self.lambda2*weights_norm

            if not self.ae:
                kl_diver = 0.5 * tf.reduce_sum(tf.square(encode) + tf.square(encode_std / object_stddev) - tf.log(1e-8 + tf.square(encode_std / object_stddev)) - 1, 1)
                kl_diver = self.lambda3*tf.reduce_mean(kl_diver)
            else:
                kl_diver = 0.0
            # test
            test_encode, test_l0 = self.encoder(inputs, vae_block_para_en, train = False, name = name)
            test_decode = self.decoder(test_encode, vae_block_para_de, train = False, name = name)

            if not self.ae:
                test_encode_std = self.encoder_std(test_l0, fcparams_std_en)
                testkl_diver = 0.5 * tf.reduce_sum(tf.square(test_encode) + tf.square(test_encode_std / object_stddev) - tf.log(1e-8 + tf.square(test_encode_std / object_stddev)) - 1, 1)
                testkl_diver = self.lambda3*tf.reduce_mean(testkl_diver)
            else:
                testkl_diver = 0.0

            embedding_decode = self.decoder(embedding_inputs, vae_block_para_de, train = False, name = name)
            # total loss
            generation_loss = self.lambda0 * tf.reduce_mean(tf.reduce_sum(tf.pow(inputs-decode, 2.0), [1,2]))

            test_generation_loss = self.lambda0 * tf.reduce_mean(tf.reduce_sum(tf.pow(inputs-test_decode, 2.0), [1,2]))

            total_loss = generation_loss + kl_diver + l2_loss + weights_norm

            if tensorboard:
                s1 = tf.summary.scalar('generation_loss', generation_loss)
                s2 = tf.summary.scalar('weights_norm', weights_norm)
                s3 = tf.summary.scalar('l2_loss', l2_loss)

                s1hist = tf.summary.histogram('fcparams', fcparams_en)

                if self.ae:
                    # train_summary_var = [s1, s2, s3, s5, s6, s1hist, slr]
                    train_summary_var = [s1, s2, s1hist, s3]
                else:
                    s4 = tf.summary.scalar('kl_diver', kl_diver)
                    s2hist = tf.summary.histogram('fcparams_std', fcparams_std_en)
                    # train_summary_var = [s1, s2, s3, s4, s5, s6, s1hist, s2hist, slr]
                    train_summary_var = [s1, s2, s4, s1hist, s2hist, s3]

                s1 = tf.summary.scalar('valid_generation_loss', test_generation_loss)

                if self.ae:
                    valid_summary_var = [s1]
                else:
                    s2 = tf.summary.scalar('valid_KL_loss', testkl_diver)
                    valid_summary_var = [s1, s2]

            else:
                train_summary_var = []
                valid_summary_var = []

        if self.ae:
            return encode, decode, test_encode, test_decode, embedding_inputs, embedding_decode, generation_loss, weights_norm, l2_loss, test_generation_loss, train_summary_var, valid_summary_var, total_loss
        else:
            return encode, decode, encode_std, test_encode, test_decode, test_encode_std, embedding_inputs, embedding_decode, generation_loss, weights_norm, kl_diver, l2_loss, test_generation_loss, testkl_diver, train_summary_var, valid_summary_var, total_loss

    def load(self, sess, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        saver = self.saver

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # import the inspect_checkpoint library
        from tensorflow.python.tools import inspect_checkpoint as chkp

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            # print all tensors in checkpoint file
            # chkp.print_tensors_in_checkpoint_file(os.path.join(checkpoint_dir, ckpt_name), tensor_name='', all_tensors=False, all_tensor_names=True)
            # chkp._count_total_params

            if not ckpt_name.find('best') == -1:
                counter = 0
            else:
                counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0  # model = convMESH()

    def leaky_relu(self, input_, alpha = 0.1):
        return tf.nn.leaky_relu(input_)

    def softplusplus(self,input_, alpha=0.2):
        return tf.log(1.0+tf.exp(input_*(1.0-alpha)))+alpha*input_-tf.log(2.0)

    def batch_norm_wrapper(self, inputs, name = 'batch_norm',is_training = False, decay = 0.9, epsilon = 1e-5):
        with tf.variable_scope(name) as scope:
            if is_training == True:
                scale = tf.get_variable('scale', dtype=tf.float32, trainable=True, initializer=tf.ones([inputs.get_shape()[-1]],dtype=tf.float32))
                beta = tf.get_variable('beta', dtype=tf.float32, trainable=True, initializer=tf.zeros([inputs.get_shape()[-1]],dtype=tf.float32))
                pop_mean = tf.get_variable('overallmean',  dtype=tf.float32,trainable=False, initializer=tf.zeros([inputs.get_shape()[-1]],dtype=tf.float32))
                pop_var = tf.get_variable('overallvar',  dtype=tf.float32, trainable=False, initializer=tf.ones([inputs.get_shape()[-1]],dtype=tf.float32))
            else:
                scope.reuse_variables()
                scale = tf.get_variable('scale', dtype=tf.float32, trainable=True)
                beta = tf.get_variable('beta', dtype=tf.float32, trainable=True)
                pop_mean = tf.get_variable('overallmean', dtype=tf.float32, trainable=False)
                pop_var = tf.get_variable('overallvar', dtype=tf.float32, trainable=False)

            if is_training:
                axis = list(range(len(inputs.get_shape()) - 1))
                batch_mean, batch_var = tf.nn.moments(inputs,axis)
                train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
            else:
                return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

    def convlayer(self, input_feature, input_dim, output_dim, nb_weights, edge_weights, name = 'meshconv', training = True, special_activation = False, no_activation = False, bn = True):
        with tf.variable_scope(name) as scope:

            padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, input_dim], tf.float32)

            padded_input = tf.concat([padding_feature, input_feature], 1)

            total_nb_feature = tf.gather(padded_input, self.nb, axis = 1)
            mean_nb_feature = tf.reduce_sum(total_nb_feature, axis = 2)/self.degrees

            nb_feature = tf.tensordot(mean_nb_feature, nb_weights, [[2],[0]])

            edge_bias = tf.get_variable("edge_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            edge_feature = tf.tensordot(input_feature, edge_weights, [[2],[0]]) + edge_bias

            total_feature = edge_feature + nb_feature

            if bn == False:
                fb = total_feature
            else:
                fb = self.batch_norm_wrapper(total_feature, is_training = training)

            if no_activation == True:
                fa = fb
            elif special_activation == False:
                fa = self.leaky_relu(fb)
            else:
                fa = tf.nn.tanh(fb)
                # print('tanh')

            return fa

    def linear(self, input_, input_size, output_size, name='Linear', training = True, special_activation = False, no_activation = False, bn = True, stddev=0.02, bias_start=0.0):
        with tf.variable_scope(name) as scope:

            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            matrix = tf.get_variable("weights", [input_size, output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size], tf.float32, initializer=tf.constant_initializer(bias_start))

            if training:
                if tensorboard:
                    matrixhist = tf.summary.histogram('fc_weights', matrix)
                    biashist = tf.summary.histogram('fc_bias', bias)
                    self.weight_hist.append(matrixhist)
                    self.weight_hist.append(biashist)

            output = tf.matmul(input_, matrix) + bias

            if bn == False:
                fb = output
            else:
                fb = self.batch_norm_wrapper(output, is_training = training)

            if no_activation == True:
                fa = fb
                # print('dont use activate function')
            elif special_activation == False:
                if self.activate == 'elu':
                    fa = tf.nn.elu(fb)
                elif self.activate == 'spp':
                    fa = self.softplusplus(fb)
                else:
                    fa = self.leaky_relu(fb)
            else:
                fa = tf.nn.tanh(fb)
                # print('tanh')

        return fa

    def get_conv_weights(self, input_dim, output_dim, name = 'convweight'):
        with tf.variable_scope(name) as scope:
            n = tf.get_variable("nb_weights", [input_dim, output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            e = tf.get_variable("edge_weights", [input_dim, output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))

            return n, e

    def encoder(self, input_feature, para, train = True, name = "stack"):
        with tf.variable_scope("encoder") as scope:
            if train == False:
                scope.reuse_variables()

            prev = input_feature

            for i in range(0, self.layers):
                if i == self.layers - 1:
                    if self.layers == 1:
                        conv = self.convlayer(prev, self.vertex_dim, self.finaldim, para[0][i], para[1][i], name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)
                    else:
                        conv = self.convlayer(prev, self.vertex_dim, self.finaldim, para[0][i], para[1][i], name = 'conv'+str(i+1), no_activation = True, training = train, bn = False)
                else:
                    prev = self.convlayer(prev, self.vertex_dim, self.vertex_dim, para[0][i], para[1][i], name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)

            l0 = tf.reshape(conv, [tf.shape(conv)[0], self.pointnum * self.finaldim])

            l1 = tf.matmul(l0, para[2])

            if train == True:
                weights_maximum = tf.reduce_max(tf.abs(l1), 0) - 0.95
                zeros = tf.zeros_like(weights_maximum)
                weights_norm = tf.reduce_mean(tf.maximum(weights_maximum, zeros))
                return l1, l0, weights_norm
                # return l1, l0
            else:
                return l1, l0

    def encoder_std(self, l0, para_std):
        # return tf.nn.softplus(tf.matmul(l0, para_std))
        return 2 * tf.nn.sigmoid(tf.matmul(l0, para_std))
        # return tf.sqrt(tf.nn.softsign(tf.matmul(l0, para_std))+1)

    def decoder(self, latent_tensor, para, train = True, name = "stack"):
        with tf.variable_scope("decoder") as scope:
            if train == False:
                scope.reuse_variables()

            l1 = tf.matmul(latent_tensor, tf.transpose(para[2]))

            l2 = tf.reshape(l1, [tf.shape(l1)[0], self.pointnum, self.finaldim])

            prev = l2

            for i in range(0, self.layers):
                if i == 0:
                    conv = self.convlayer(prev, self.finaldim, self.vertex_dim, tf.transpose(para[0][self.layers-1]), tf.transpose(para[1][self.layers-1]), name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)
                else:
                    conv = self.convlayer(prev, self.vertex_dim, self.vertex_dim, tf.transpose(para[0][self.layers-1-i]), tf.transpose(para[1][self.layers-1-i]), name = 'conv'+str(i+1), special_activation = True, training = train, bn = False)

                prev = conv

        return conv

    def encoder_symm(self, input_mesh, training = True, keep_prob = 1.0):
        with tf.variable_scope("encoder_symm") as scope:
            if(training == False):
                keep_prob = 1.0
                scope.reuse_variables()

            if self.union:
                bn = True
            else:
                bn = False

            h1 = self.linear(input_mesh, self.part_num*(self.hiddendim[0]+self.part_dim), 2048, name = 'fc_1', training = training, special_activation = False, bn = bn)
            h1 = tf.nn.dropout(h1, keep_prob = keep_prob)

            h2 = self.linear(h1, 2048, 512, name = 'fc_2', training = training, special_activation = False, bn = bn)
            h2 = tf.nn.dropout(h2, keep_prob = keep_prob)

            h3 = self.linear(h2, 512, 128, name = 'fc_3', training = training, special_activation = False, bn = bn)
            h3 = tf.nn.dropout(h3, keep_prob = keep_prob)

            '''
            h4, weights = linear(h3a, 1024, 512, 'h4')
            h4bn = batch_norm_wrapper(h4, name = 'h4bn',is_training = training, decay = 0.9)
            h4a = leaky_relu(h4bn)
            '''

            mean = self.linear(h3, 128, self.hiddendim[1], name = 'mean', training = training, no_activation = True, bn = False)
            stddev = self.linear(h3, 128, self.hiddendim[1], name = 'stddev', training = training, no_activation = True, bn = False)
            stddev = tf.sqrt(tf.nn.softsign(stddev)+1.0)

        return mean, stddev

    def decoder_symm(self, z, training = True, keep_prob = 1.0):
        with tf.variable_scope("decoder_symm") as scope:
            if(training == False):
                keep_prob = 1.0
                scope.reuse_variables()

            if self.union:
                bn = True
            else:
                bn = False

            h1 = self.linear(z, self.hiddendim[1], 128, name = 'fc_1', training = training, special_activation = False, bn = bn)
            h1 = tf.nn.dropout(h1, keep_prob = keep_prob)

            h2 = self.linear(h1, 128, 512, name = 'fc_2', training = training, special_activation = False, bn = bn)
            h2 = tf.nn.dropout(h2, keep_prob = keep_prob)

            h3 = self.linear(h2, 512, 2048, name = 'fc_3', training = training, special_activation = False, bn = bn)
            h3 = tf.nn.dropout(h3, keep_prob = keep_prob)

            output = self.linear(h3, 2048, self.part_num*(self.hiddendim[0]+self.part_dim), name = 'fc_4', training = training, no_activation = True, bn = False)

        return output

#------------------------------------------------------------network tools---------------------------------------------------------------------------------------


#------------------------------------------------------------training function------------------------------------------------------------------------------------

    def train_total_deform_structure(self):# trian with split the dataset to test the generalization error
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        rng = np.random.RandomState(23456)
        batch_size = self.batch_size
        inf = float('inf')
        if timeline_use: # use the timeline to analyze the efficency of the program
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            many_runs_timeline = TimeLiner()
        else:
            options = None
            run_metadata = None

        # with tf.Session(config = self.config) as sess:
        tf.global_variables_initializer().run()

        could_load, checkpoint_counter = self.load(self.sess, self.checkpoint_dir_structure)
        if tensorboard:
            summary_writer = tf.summary.FileWriter(self.outputdir+'/logs', self.sess.graph)

        # train_id, valid_id = spilt_dataset(len(self.feature), self.train_percent, self.iddat_name)
        Allepoch = self.maxepoch_deform+self.maxepoch_structure+1
        for epoch in range(checkpoint_counter, Allepoch):
            # train
            rng.shuffle(self.train_id)
            # printout(self.flog,"Train Epoch: %5d" % epoch)
            for bidx in xrange(0, len(self.train_id)//batch_size + 1):

                train_feature = [self.feature[i] for i in self.train_id[bidx*batch_size:min(len(self.train_id), bidx*batch_size+batch_size)]]
                train_symmetry_feature = [self.symmetry_feature[i] for i in self.train_id[bidx*batch_size:min(len(self.train_id), bidx*batch_size+batch_size)]]
                if len(train_feature) == 0:
                    continue

                dictbase = {self.inputs_feature: train_feature, self.inputs_symmetry: train_symmetry_feature}
                if self.ae:
                    random = np.zeros((len(train_feature),200)).astype('float32')
                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    _, cost_generation, weight_norm, cost_l2, train_summary = self.sess.run([self.total_trainop, self.total_generation_loss, self.total_weights_norm, self.total_l2_loss, self.train_summary], feed_dict = feed_dict, options=options, run_metadata=run_metadata)

                    printout(self.flog,"Epoch: {:6d} generation_loss: {:08.4f} weight_loss: {:08.4f} l2_loss: {:08.4f}".format(epoch, cost_generation, weight_norm, cost_l2),epoch)

                else:
                    dictrand = {x: gaussian(len(train_feature), np.shape(x)[1]) for x in self.embedding_inputs}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    _, cost_generation, cost_kl, weight_norm, cost_l2, train_summary = self.sess.run([self.total_trainop, self.total_generation_loss, self.total_kl_loss, self.total_weights_norm, self.total_l2_loss, self.train_summary], feed_dict = feed_dict, options=options, run_metadata=run_metadata)

                    printout(self.flog,"Epoch: {:6d} generation_loss: {:08.4f} weight_loss: {:08.4f} kl_loss: {:08.4f} l2_loss: {:08.4f}".format(epoch, cost_generation, weight_norm, cost_kl, cost_l2), epoch)

                if (epoch+1) % 5000 == 0:
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

                if cost_generation < 60 and len(self.valid_id)==0 and cost_generation < inf:
                    inf = cost_generation
                    printout(self.flog,"Save Best(cost_generation): {:08.4f}\n".format(cost_generation))

            # valid
            # printout(self.flog,"Valid Epoch: %5d" % epoch)
            rng.shuffle(self.valid_id)
            valid_loss = 0
            for bidx in xrange(0, len(self.valid_id)//batch_size + 1):

                valid_feature = [self.feature[i] for i in self.valid_id[bidx*batch_size:min(len(self.valid_id), bidx*batch_size+batch_size)]]
                valid_symmetry_feature = [self.symmetry_feature[i] for i in self.valid_id[bidx*batch_size:min(len(self.valid_id), bidx*batch_size+batch_size)]]
                if len(valid_feature) == 0:
                    continue

                dictbase = {self.inputs_feature: valid_feature, self.inputs_symmetry: valid_symmetry_feature}
                random = np.zeros((len(valid_feature),200)).astype('float32')
                if self.ae:
                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
                    feed_dict = merge_two_dicts(dictbase, dictrand)

                    cost_generation_valid, valid_summary = self.sess.run([self.total_test_loss, self.valid_summary], feed_dict = feed_dict)

                    printout(self.flog,"Epoch: {:6d} valid_gen_loss: {:08.4f}".format(epoch, cost_generation_valid), epoch)
                else:

                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    cost_generation_valid, cost_kl_valid, valid_summary = self.sess.run([self.total_test_loss, self.total_testkl_loss, self.valid_summary], feed_dict = feed_dict)

                    printout(self.flog,"Epoch: {:6d} valid_gen_loss:{:08.4f} valid_kl_loss: {:08.4f}".format(epoch, cost_generation_valid, cost_kl_valid),epoch)
                valid_loss+=cost_generation_valid*len(valid_feature)
            if len(self.valid_id)>0:
                valid_loss/=len(self.valid_id)
                if valid_loss < 50 and valid_loss < inf:
                    inf = valid_loss
                    printout(self.flog,"Save Best(cost_generation_valid): {:08.4f}\n".format(valid_loss))
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-modelbest')

            if tensorboard:
                summary_writer.add_summary(train_summary, epoch)
                if not len(valid_feature) == 0:
                    summary_writer.add_summary(valid_summary, epoch)

            if timeline_use:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                many_runs_timeline.update_timeline(chrome_trace)

            if (epoch+1) == Allepoch:
                self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

        if timeline_use:
            many_runs_timeline.save('timeline_03_merged_{}_runs.json'.format(self.maxepoch_deform-checkpoint_counter))

    def train_pre(self):
        tf.global_variables_initializer().run()
        # tf.local_variables_initializer().run()
        # self.write = tf.summary.FileWriter(logfolder + '/logs/', self.sess.graph)
        if tensorboard:
            self.summary_writer = tf.summary.FileWriter(self.outputdir+'/logs', self.sess.graph)

        if not os.path.exists(self.checkpoint_dir_structure):
            os.makedirs(self.checkpoint_dir_structure)

        could_load_struture, checkpoint_counter_struture = self.load(self.sess, self.checkpoint_dir_structure)

        if (could_load_struture and checkpoint_counter_struture <= self.maxepoch_deform):
            self.start = 'DEFORM'
            self.start_step_deform = checkpoint_counter_struture
            self.start_step_structure = 0
        elif (could_load_struture and checkpoint_counter_struture <= self.maxepoch_structure):
            self.start = 'STRUCTURE'
            self.start_step_structure = checkpoint_counter_struture
        else:
            self.start_step_deform = 0
            self.start_step_structure = 0
            self.start = 'DEFORM'
            print('we start from VAE...')

    def train_deform(self):
        printout(self.flog,"Train DEFORM Net...")
        rng = np.random.RandomState(23456)
        batch_size = self.batch_size
        inf = float('inf')
        if timeline_use: # use the timeline to analyze the efficency of the program
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            many_runs_timeline = TimeLiner()
        else:
            options = None
            run_metadata = None

        for epoch in range(self.start_step_deform, self.maxepoch_deform):
            # train
            rng.shuffle(self.train_id)

            # time1=time.time()
            for bidx in xrange(0, len(self.train_id)//batch_size + 1):

                train_feature = [self.feature[i,:,:,:] for i in self.train_id[bidx*batch_size:min(len(self.train_id), bidx*batch_size+batch_size)]]
                # train_symmetry_feature = [self.symmetry_feature[i] for i in train_id[bidx*batch_size:min(len(train_id), bidx*batch_size+batch_size)]]
                if len(train_feature) == 0:
                    continue
                # train_feature = np.unique(train_feature, axis=0)
                dictbase = {self.inputs_feature: train_feature}
                if self.ae:
                    random = np.zeros((len(train_feature),200)).astype('float32')
                    # dictrand = {self.embedding_inputs[part_id]: random[:, 0: np.shape(self.embedding_inputs[part_id])[1]]}
                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs[:-1]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    # _, cost_generation, weights_norm, cost_l2, train_summary = self.sess.run([self.train_op[part_id], self.generation_loss[part_id], self.weights_norm[part_id], self.l2_loss[part_id], self.train_summary_one[part_id]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                    _, cost_generation, weights_norm, cost_l2, train_summary = self.sess.run([self.train_op[:-1], self.generation_loss[:-1], self.weights_norm[:-1], self.l2_loss[:-1], self.train_summary_one[:-1]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)

                    printout(self.flog,"Train Epoch: {:6d} \nPart  Name: {} \ngener_loss: {} \nweight_loss: {} \nl2222_loss: {}".format(epoch, self.part_name, cost_generation, weights_norm, cost_l2),epoch)

                else:
                    # dictrand = {self.embedding_inputs[part_id]: gaussian(len(train_feature), np.shape(self.embedding_inputs[part_id])[1])}
                    dictrand = {x: gaussian(len(train_feature), np.shape(x)[1]) for x in self.embedding_inputs[:-1]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    # _, cost_generation, weights_norm, cost_kl, cost_l2, train_summary = self.sess.run([self.train_op[part_id], self.generation_loss[part_id], self.weights_norm[part_id], self.kl_diver[part_id], self.l2_loss[part_id], self.train_summary_one[part_id]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)
                    _, cost_generation, weights_norm, cost_kl, cost_l2, train_summary = self.sess.run([self.train_op[:-1], self.generation_loss[:-1], self.weights_norm[:-1], self.kl_diver[:-1], self.l2_loss[:-1], self.train_summary_one[:-1]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)

                    printout(self.flog, "Train Epoch: {:6d} \nPart  Name: {} \ngener_loss: {} \nweigh_loss: {} \nkllll_loss: {} \nl2222_loss: {}".format(epoch, self.part_name, cost_generation, weights_norm, cost_kl, cost_l2), epoch)

                if (epoch+1) % 5000 == 0:
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

                if np.max(cost_generation) < 60 and len(self.valid_id)==0 and np.max(cost_generation) < inf:
                    inf = cost_generation
                    printout(self.flog,"Save Best(cost_generation): %.8f"%(cost_generation))
            # print(time.time()-time1)
            # valid
            rng.shuffle(self.valid_id)
            # printout(self.flog, "Valid Epoch: %5d Part ID: %5d Part Name: %s" % (epoch, part_id, self.part_name[part_id]))
            valid_loss=0
            for bidx in xrange(0, len(self.valid_id)//batch_size + 1):

                valid_feature = [self.feature[i,:,:,:] for i in self.valid_id[bidx*batch_size:min(len(self.valid_id), bidx*batch_size+batch_size)]]
                # valid_symmetry_feature = [self.symmetry_feature[i] for i in valid_id[bidx*batch_size:min(len(valid_id), bidx*batch_size+batch_size)]]
                if len(valid_feature) == 0:
                    continue
                # valid_feature = np.unique(valid_feature, axis=0)
                dictbase = {self.inputs_feature: valid_feature}
                random = np.zeros((len(valid_feature),200)).astype('float32')
                if self.ae:
                    # dictrand = {self.embedding_inputs[part_id]: random[:, 0: np.shape(self.embedding_inputs[part_id])[1]]}
                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs[:-1]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)

                    cost_generation_valid, valid_summary = self.sess.run([self.test_loss[:-1], self.valid_summary_one[:-1]], feed_dict = feed_dict)

                    printout(self.flog,"Train Epoch: {:6d} \nPart  Name: {} \nvalid_gen_loss: {}".format(epoch, self.part_name[:-1], cost_generation_valid),epoch)
                else:

                    # dictrand = {self.embedding_inputs[part_id]: random[:, 0: np.shape(self.embedding_inputs[part_id])[1]]}
                    dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs[:-1]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    cost_generation_valid, cost_kl_valid, valid_summary = self.sess.run([self.test_loss[:-1], self.testkl_diver[:-1], self.valid_summary_one[:-1]], feed_dict = feed_dict)

                    printout(self.flog,"Train Epoch: {:6d} \nPart  Name: {} \nvalid_Gen_loss: {} \nvalid_kl_loss: {}".format(epoch, self.part_name, cost_generation_valid, cost_kl_valid),epoch)
                valid_loss+=np.max(cost_generation_valid)*len(valid_feature)

            if len(self.valid_id)>0:
                valid_loss/=len(self.valid_id)
                if valid_loss < 50 and valid_loss < inf:
                    inf = valid_loss
                    printout(self.flog,"Save Best(cost_generation_valid): {:08.4f}".format(valid_loss))
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-modelbest')

            if tensorboard:
                self.summary_writer.add_summary(train_summary, epoch)
                if not len(valid_feature) == 0:
                    self.summary_writer.add_summary(valid_summary, epoch)

            if timeline_use:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                many_runs_timeline.update_timeline(chrome_trace)

            if (epoch+1) == self.maxepoch_deform:
                self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

        if timeline_use:
            many_runs_timeline.save('timeline_03_merged_{}_runs.json'.format(epoch))

    def train_structure(self):
        printout(self.flog,"Train Structure Net...")
        rng = np.random.RandomState(23456)
        batch_size = self.batch_size
        inf = float('inf')
        if timeline_use: # use the timeline to analyze the efficency of the program
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            many_runs_timeline = TimeLiner()
        else:
            options = None
            run_metadata = None

        for epoch in range(self.start_step_structure, self.maxepoch_structure):
            # train
            rng.shuffle(self.train_id)
            # printout(self.flog,"Train Epoch: %5d" % epoch)
            # time1=time.time()
            for bidx in xrange(0, len(self.train_id)//batch_size + 1):

                train_feature = [self.feature[i,:,:,:] for i in self.train_id[bidx*batch_size:min(len(self.train_id), bidx*batch_size+batch_size)]]
                train_symmetry_feature = [self.symmetry_feature[i] for i in self.train_id[bidx*batch_size:min(len(self.train_id), bidx*batch_size+batch_size)]]
                if len(train_feature) == 0:
                    continue
                # train_feature = np.unique(train_feature, axis=0)
                dictbase = {self.inputs_feature: train_feature, self.inputs_symmetry: train_symmetry_feature}
                if self.ae:
                    random = np.zeros((len(train_feature),200)).astype('float32')
                    dictrand = {self.embedding_inputs[-1]: random[:, 0: np.shape(self.embedding_inputs[-1])[1]]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    _, cost_generation, cost_l2, train_summary = self.sess.run([self.train_op[-1], self.generation_loss[-1], self.l2_loss[-1], self.train_summary_one[-1]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)

                    printout(self.flog,"Epoch: {:6d} generation_loss: {:08.4f} l2_loss: {:08.4f}".format(epoch, cost_generation, cost_l2),epoch)

                else:
                    dictrand = {self.embedding_inputs[-1]: gaussian(len(train_feature), np.shape(self.embedding_inputs[-1])[1])}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    _, cost_generation, cost_kl, cost_l2, train_summary = self.sess.run([self.train_op[-1], self.generation_loss[-1], self.kl_diver[-1], self.l2_loss[-1], self.train_summary_one[-1]], feed_dict = feed_dict, options=options, run_metadata=run_metadata)

                    printout(self.flog,"Epoch: {:6d} generation_loss: {:08.4f} kl_loss: {:08.4f} l2_loss: {:08.4f}".format(epoch, cost_generation, cost_kl, cost_l2),epoch)

                if (epoch+1) % 50 == 0: #was 5000
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

                if cost_generation < 60 and len(self.valid_id)==0 and cost_generation < inf:
                    inf = cost_generation
                    printout(self.flog,"Save Best(cost_generation): {:08.4f}".format(cost_generation))
            # print(time.time()-time1)
            # valid
            rng.shuffle(self.valid_id)
            # printout(self.flog,"Valid Epoch: %5d" % epoch)
            valid_loss=0
            for bidx in xrange(0, len(self.valid_id)//batch_size + 1):

                valid_feature = [self.feature[i,:,:,:] for i in self.valid_id[bidx*batch_size:min(len(self.valid_id), bidx*batch_size+batch_size)]]
                valid_symmetry_feature = [self.symmetry_feature[i] for i in self.valid_id[bidx*batch_size:min(len(self.valid_id), bidx*batch_size+batch_size)]]
                if len(valid_feature) == 0:
                    continue
                # valid_feature = np.unique(valid_feature, axis=0)
                dictbase = {self.inputs_feature: valid_feature,self.inputs_symmetry: valid_symmetry_feature}
                random = np.zeros((len(valid_feature),200)).astype('float32')
                if self.ae:
                    dictrand = {self.embedding_inputs[-1]: random[:, 0: np.shape(self.embedding_inputs[-1])[1]]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)

                    cost_generation_valid, valid_summary = self.sess.run([self.test_loss[-1], self.valid_summary_one[-1]], feed_dict = feed_dict)

                    printout(self.flog,"valid_gen_loss: {:08.4f}".format(cost_generation_valid),epoch)
                else:

                    dictrand = {self.embedding_inputs[-1]: random[:, 0: np.shape(self.embedding_inputs[-1])[1]]}
                    feed_dict = merge_two_dicts(dictbase, dictrand)
                    cost_generation_valid, cost_kl_valid, valid_summary = self.sess.run([self.test_loss[-1], self.testkl_diver[-1], self.valid_summary_one[-1]], feed_dict = feed_dict)

                    printout(self.flog,"valid_gen_loss: {:08.4f}, valid_kl_loss: {:08.4f}".format(cost_generation_valid, cost_kl_valid),epoch)
                valid_loss+=cost_generation_valid*len(valid_feature)
            if len(self.valid_id)>0:
                valid_loss/=len(self.valid_id)
                if valid_loss < 50 and valid_loss < inf:
                    inf = valid_loss
                    printout(self.flog,"Save Best(cost_generation_valid): {:08.4f}".format(valid_loss))
                    self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-modelbest')

            if tensorboard:
                self.summary_writer.add_summary(train_summary, epoch)
                if not len(valid_feature) == 0:
                    self.summary_writer.add_summary(valid_summary, epoch)

            if timeline_use:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                many_runs_timeline.update_timeline(chrome_trace)

            if (epoch+1) == self.maxepoch_structure:
                self.saver.save(self.sess, self.checkpoint_dir_structure + '/convmesh-model', global_step = epoch+1)

        if timeline_use:
            many_runs_timeline.save('timeline_03_merged_{}_runs.json'.format(epoch))

    def train_scvae(self):

        with tf.Session(config = self.config) as self.sess:
            if self.union:
                self.train_total_deform_structure()
            else:
                self.train_pre()
                if self.start == 'DEFORM':
                    self.train_deform()
                    self.train_structure()
                elif self.start == 'STRUCTURE':
                    self.train_structure()
                else:
                    print('Training Ending!')

        print(self.outputdir)

#------------------------------------------------------------training function------------------------------------------------------------------------------------

#------------------------------------------------------------applications--------------------------------------------------------------------------------------

    def recover_mesh(self, datainfo, epoch = 0):
        with tf.Session(config = self.config) as sess:
            tf.global_variables_initializer().run()
            if epoch == 0:
                _success, epoch = self.load(sess, self.checkpoint_dir_structure)

            if not _success:
                raise Exception("error")
            path = self.checkpoint_dir_structure +'/../recon'+str(epoch)

            if not os.path.isdir(path):
                os.makedirs(path)

            random = np.zeros((len(self.feature), 200)).astype('float32')
            dictbase = {self.inputs_feature: self.feature, self.inputs_symmetry: self.symmetry_feature}
            dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
            feed_dict = merge_two_dicts(dictbase, dictrand)

            for i in range(self.part_num+1):
                recover, emb, std = sess.run([self.test_decode[i], self.test_encode[i], self.test_encode_std[i]], feed_dict = feed_dict)

                if i < self.part_num:
                    v, _ = self.f2v.get_vertex(recover, i, path + '/' + self.part_name[i], self.modelname, np.arange(len(self.modelname)),self.part_name[i],one=False)
                    rs, rlogr = recover_data(recover, datainfo.logrmin[i], datainfo.logrmax[i], datainfo.smin[i], datainfo.smax[i], self.pointnum)
                    sio.savemat(path + '/' + self.part_name[i]+'/recover.mat', {'RS':rs, 'RLOGR':rlogr, 'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})
                    printout(self.flog, "Erms: %.8f" % (self.f2v.calc_erms(v, id = i)))
                else:
                    recoversym = np.reshape(recover, (len(recover), self.part_num, self.part_dim+self.hiddendim[0]))
                    symmetryf = recover_datasym(recoversym[:,:,-self.part_dim:], datainfo.boxmin[0], datainfo.boxmax[0])
                    sio.savemat(path+'/recover_sym.mat', {'symmetry_feature':symmetryf, 'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})
                    for k in range(self.part_num):
                        recover1 = sess.run([self.embedding_decode[k]], feed_dict = {self.embedding_inputs[k]: recoversym[:,k,:self.hiddendim[0]]})[0]
                        v, _ = self.f2v.get_vertex(recover1, k, path + '/struc_' + self.part_name[k], self.modelname, np.arange(len(self.modelname)),self.part_name[k],one=False)

                        rs, rlogr = recover_data(recover1, datainfo.logrmin[k], datainfo.logrmax[k], datainfo.smin[k], datainfo.smax[k], self.pointnum)
                        sio.savemat(path + '/struc_' + self.part_name[k]+'/recover.mat', {'RS':rs, 'RLOGR':rlogr, 'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})
                        printout(self.flog, "Erms: %.8f" % (self.f2v.calc_erms(v, id = k)))

        print(path)

        return

    def recover_mesh_v2(self, datainfo, epoch = 0):
        with tf.Session(config = self.config) as sess:
            tf.global_variables_initializer().run()
            if epoch == 0:
                _success, epoch = self.load(sess, self.checkpoint_dir_structure)

            if not _success:
                raise Exception("error")
            path = self.checkpoint_dir_structure +'/../recon'+str(epoch)

            if not os.path.isdir(path):
                os.makedirs(path)

            random = np.zeros((len(self.feature), 200)).astype('float32')
            dictbase = {self.inputs_feature: self.feature, self.inputs_symmetry: self.symmetry_feature}
            dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
            feed_dict = merge_two_dicts(dictbase, dictrand)

            i = self.part_num

            recover, emb, std = sess.run([self.test_decode[i], self.test_encode[i], self.test_encode_std[i]], feed_dict = feed_dict)

            recoversym = np.reshape(recover, (len(recover), self.part_num, self.part_dim+self.hiddendim[0]))
            symmetryf = recover_datasym(recoversym[:,:,-self.part_dim:], datainfo.boxmin[0], datainfo.boxmax[0])
            sio.savemat(path+'/recover_sym.mat', {'symmetry_feature':symmetryf, 'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})

            k = 0
            recover1 = sess.run([self.embedding_decode[k]], feed_dict = {self.embedding_inputs[k]: recoversym[:,k,:self.hiddendim[0]]})[0]
            v, _ = self.f2v.get_vertex(recover1, k, path + '/struc_' + self.part_name[k], self.modelname, np.arange(len(self.modelname)),self.part_name[k],one=False)
            rs, rlogr = recover_data(recover1, datainfo.logrmin[k], datainfo.logrmax[k], datainfo.smin[k], datainfo.smax[k], self.pointnum)
            sio.savemat(path + '/struc_' + self.part_name[k]+'/recover.mat', {'RS':rs, 'RLOGR':rlogr, 'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})
            printout(self.flog, "Erms: %.8f" % (self.f2v.calc_erms(v, id = k)))

        print(path)

        return

    def obtain_embeddings_v2(self, datainfo, epoch = 0):
        with tf.Session(config = self.config) as sess:
            tf.global_variables_initializer().run()
            if epoch == 0:
                _success, epoch = self.load(sess, self.checkpoint_dir_structure)

            if not _success:
                raise Exception("error")
            path = self.checkpoint_dir_structure +'/../recon'+str(epoch)

            if not os.path.isdir(path):
                os.makedirs(path)

            random = np.zeros((len(self.feature), 200)).astype('float32')
            dictbase = {self.inputs_feature: self.feature, self.inputs_symmetry: self.symmetry_feature}
            dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
            feed_dict = merge_two_dicts(dictbase, dictrand)

            for i in range(self.part_num+1):
                recover, emb, std = sess.run([self.test_decode[i], self.test_encode[i], self.test_encode_std[i]], feed_dict = feed_dict)

                if i < self.part_num:
                    if not os.path.isdir(path + '/' + self.part_name[i]):
                        os.makedirs(path + '/' + self.part_name[i])
                    sio.savemat(path + '/' + self.part_name[i]+'/recover.mat', {'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})

                else:
                    recoversym = np.reshape(recover, (len(recover), self.part_num, self.part_dim+self.hiddendim[0]))
                    symmetryf = recover_datasym(recoversym[:,:,-self.part_dim:], datainfo.boxmin[0], datainfo.boxmax[0])

                    sio.savemat(path+'/recover_sym.mat', {'symmetry_feature':symmetryf, 'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})
                    for k in range(self.part_num):
                        if not os.path.isdir(path + '/struc_' + self.part_name[k]):
                            os.makedirs(path + '/struc_' + self.part_name[k])
                        sio.savemat(path + '/struc_' + self.part_name[k]+'/recover.mat', {'tid':self.train_id, 'vid':self.valid_id, 'emb':emb, 'std':std})

        print(path)

        return

    def obtain_embeddings(self, datainfo, epoch = 0):
        with tf.Session(config = self.config) as sess:
            tf.global_variables_initializer().run()
            if epoch == 0:
                _success, epoch = self.load(sess, self.checkpoint_dir_structure)

            if not _success:
                raise Exception("error")
            path = self.checkpoint_dir_structure +'/../recon'+str(epoch)

            if not os.path.isdir(path):
                os.makedirs(path)

            random = np.zeros((len(self.feature), 200)).astype('float32')
            dictbase = {self.inputs_feature: self.feature, self.inputs_symmetry: self.symmetry_feature}
            dictrand = {x: random[:, 0: np.shape(x)[1]] for x in self.embedding_inputs}
            feed_dict = merge_two_dicts(dictbase, dictrand)

            for i in range(self.part_num+1):
                emb = sess.run(self.test_encode[i], feed_dict = feed_dict)

                if i < self.part_num:
                    # v, _ = self.f2v.get_vertex(recover, i, path + '/' + self.part_name[i], self.modelname, np.arange(len(self.modelname)),self.part_name[i],one=False)
                    # rs, rlogr = recover_data(recover, datainfo.logrmin[i], datainfo.logrmax[i], datainfo.smin[i], datainfo.smax[i], self.pointnum)
                    if not os.path.isdir(path + '/' + self.part_name[i]):
                        os.makedirs(path + '/' + self.part_name[i])
                    sio.savemat(path + '/' + self.part_name[i]+'/recover.mat', {'tid':self.train_id, 'vid':self.valid_id, 'emb':emb})

                else:
                    # recoversym = np.reshape(recover, (len(recover), self.part_num, self.part_dim+self.hiddendim[0]))
                    # symmetryf = recover_datasym(recoversym[:,:,-self.part_dim:], datainfo.boxmin[0], datainfo.boxmax[0])
                    sio.savemat(path+'/recover_sym.mat', {'tid':self.train_id, 'vid':self.valid_id, 'emb':emb})

        print(path)

        return

    def random_gen(self, datainfo, epoch = 0):
        with tf.Session(config = self.config) as sess:
            tf.global_variables_initializer().run()
            if epoch == 0:
                _success, epoch = self.load(sess, self.checkpoint_dir_structure)

            if not _success:
                raise Exception("error")

            path = self.checkpoint_dir_structure +'/../random'+str(epoch)

            if not os.path.isdir(path):
                os.makedirs(path)

            for i in range(self.part_num+1):

                if i < self.part_num:
                    random = gaussian(200, self.hiddendim[0])
                    recover = sess.run([self.embedding_decode[i]], feed_dict = {self.embedding_inputs[i]: random})[0]
                    num_list_new = [str(x) for x in np.arange(len(recover))]
                    self.f2v.get_vertex(recover, i, path + '/' + self.part_name[i], num_list_new, num_list_new, self.part_name[i], one = True)

                    # render_parallel(path + '/recon'+restore[-4:])
                    rs, rlogr = recover_data(recover, datainfo.logrmin[i], datainfo.logrmax[i], datainfo.smin[i], datainfo.smax[i], self.pointnum)
                    sio.savemat(path + '/' + self.part_name[i]+'/random.mat', {'RS':rs, 'RLOGR':rlogr, 'tid':self.train_id, 'vid':self.valid_id, 'emb':random})
                    # printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v1)))
                else:
                    random = gaussian(200, self.hiddendim[1])
                    recover = sess.run([self.embedding_decode[i]], feed_dict = {self.embedding_inputs[i]: random})[0]
                    recoversym = np.reshape(recover, (len(recover), self.part_num, self.part_dim+self.hiddendim[0]))

                    symmetryf = recover_datasym(recoversym[:,:,-self.part_dim:], datainfo.boxmin[0], datainfo.boxmax[0])
                    sio.savemat(path+'/random_sym.mat', {'symmetry_feature':symmetryf, 'tid':self.train_id, 'vid':self.valid_id, 'emb':random})

                    for k in range(self.part_num):
                        recover1 = sess.run([self.embedding_decode[k]], feed_dict = {self.embedding_inputs[k]: recoversym[:,k,:self.hiddendim[0]]})[0]
                        num_list_new = [str(x) for x in np.arange(len(recover1))]
                        self.f2v.get_vertex(recover1, k, path + '/struc_' + self.part_name[k], num_list_new, num_list_new, self.part_name[k], one = True)

                        rs, rlogr = recover_data(recover1, datainfo.logrmin[k], datainfo.logrmax[k], datainfo.smin[k], datainfo.smax[k], self.pointnum)
                        sio.savemat(path + '/struc_' + self.part_name[k]+'/random.mat', {'RS':rs, 'RLOGR':rlogr, 'tid':self.train_id, 'vid':self.valid_id, 'emb':recoversym[:,k,self.part_dim:]})
                        # printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v1)))

        print(path)

        return

    def interpolate1(self, datainfo, inter_id, epoch = 0): # [2, 10]
        with tf.Session(config = self.config) as sess:
            tf.global_variables_initializer().run()
            if epoch == 0:
                _success, epoch = self.load(sess, self.checkpoint_dir_structure)

            if not _success:
                raise Exception("error")
            path = self.checkpoint_dir_structure +'/../interpolation'+str(epoch)
            print(np.expand_dims([i for i in datainfo.modelname], axis=0))
            for i in range(len(inter_id)):
                if isinstance(inter_id[i], str) and len(inter_id[i])>len(str(datainfo.modelnum)):
                    inter_id[i] = datainfo.modelname.index(inter_id[i])
                else:
                    inter_id[i] = int(inter_id[i])
                print('ID: {:3d} Name: {}'.format(inter_id[i], datainfo.modelname[inter_id[i]]))

            if not os.path.isdir(path):
                os.makedirs(path)

            for i in range(self.part_num+1):

                if inter_id:
                    x = np.zeros([2, self.pointnum, self.finaldim])
                    inter_feature = self.feature[inter_id]
                    symmetry_feature = self.symmetry_feature[inter_id]
                    embedding = sess.run([self.encode[i]], feed_dict = {self.inputs_feature: inter_feature, self.inputs_symmetry: symmetry_feature})[0]

                else:
                    embedding = gaussian(2, self.hiddendim[0])

                random2_intpl = interpolate.griddata(np.linspace(0, 1, len(embedding) * 1), embedding, np.linspace(0, 1, 200), method='linear')

                if i < self.part_num:
                    # random = gaussian(200, self.hiddendim[0])
                    recover = sess.run([self.embedding_decode[i]], feed_dict = {self.embedding_inputs[i]: random2_intpl})[0]
                    num_list_new = [str(x) for x in np.arange(len(recover))]
                    self.f2v.get_vertex(recover, i, path + '/' + self.part_name[i], num_list_new, num_list_new, self.part_name[i], one = True)

                    # render_parallel(path + '/recon'+restore[-4:])
                    rs, rlogr = recover_data(recover, datainfo.logrmin[i], datainfo.logrmax[i], datainfo.smin[i], datainfo.smax[i], self.pointnum)
                    sio.savemat(path + '/' + self.part_name[i]+'/inter.mat', {'RS':rs, 'RLOGR':rlogr, 'emb':random2_intpl})
                    # printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v1)))
                else:
                    # random = gaussian(200, self.hiddendim[1])
                    recover = sess.run([self.embedding_decode[i]], feed_dict = {self.embedding_inputs[i]: random2_intpl})[0]
                    recoversym = np.reshape(recover, (len(recover), self.part_num, self.part_dim+self.hiddendim[0]))

                    symmetryf = recover_datasym(recoversym[:,:,-self.part_dim:], datainfo.boxmin[0], datainfo.boxmax[0])
                    sio.savemat(path+'/inter_sym.mat', {'symmetry_feature':symmetryf, 'emb':random2_intpl})

                    for k in range(self.part_num):
                        recover1 = sess.run([self.embedding_decode[k]], feed_dict = {self.embedding_inputs[k]: recoversym[:,k,:self.hiddendim[0]]})[0]
                        num_list_new = [str(x) for x in np.arange(len(recover1))]
                        self.f2v.get_vertex(recover1, k, path + '/struc_' + self.part_name[k], num_list_new, num_list_new, self.part_name[k], one = True)

                        rs, rlogr = recover_data(recover1, datainfo.logrmin[k], datainfo.logrmax[k], datainfo.smin[k], datainfo.smax[k], self.pointnum)
                        sio.savemat(path + '/struc_' + self.part_name[k]+'/inter.mat', {'RS':rs, 'RLOGR':rlogr, 'emb':recoversym[:,k,self.part_dim:]})
                        # printout(self.flog, "Erms: %.8f" % (self.calc_erms(deforma_v1)))

        print(path)

        return

#------------------------------------------------------------applications----------------------------------------------------------------------------------------
