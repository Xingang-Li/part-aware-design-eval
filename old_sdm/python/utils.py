import scipy.io as sio
import numpy as np
import random
import pickle
import configparser, argparse, os
import base64
import openmesh as om
import json
import h5py
import scipy
import glob, pymesh

class Id:
    def __init__(self, Ia):
        self.Ia=Ia
    def show(self):
        print('A: %s\n'%(self.Ia))

class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        print ("values: {}".format(values))
        for kv in values:
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

def load_datanew(path):
    resultmin = -0.95
    resultmax = 0.95

    data = h5py.File(path,mode = 'r')
    datalist = data.keys()

    logr = np.transpose(data['FLOGRNEW'], (3, 2, 1, 0)).astype('float32')
    s = np.transpose(data['FS'], (3, 2, 1, 0)).astype('float32')
    neighbour1 = np.transpose(data['neighbour']).astype('float32')
    cotweight1 = np.transpose(data['cotweight']).astype('float32')

    L1 = []

    nb1 = neighbour1
    cotw1 = cotweight1
    pointnum1 = neighbour1.shape[0]
    maxdegree1 = neighbour1.shape[1]
    modelnum = len(logr)

    degree1 = np.zeros((neighbour1.shape[0], 1)).astype('float32')
    for i in range(neighbour1.shape[0]):
        degree1[i] = np.count_nonzero(nb1[i])

    laplacian_matrix = np.transpose(data['L']).astype('float32')
    reconmatrix = np.transpose(data['recon']).astype('float32')
    vdiff = np.transpose(data['vdiff'], (2, 1, 0)).astype('float32')
    all_vertex = np.transpose(data['vertex'], (3, 2, 1, 0)).astype('float32')

    logrmin_set = []
    logrmax_set = []
    smin_set = []
    smax_set = []
    f = np.zeros((modelnum, logr.shape[1], pointnum1, 9)).astype('float32')
    for i in range(logr.shape[1]):
        logr_part = logr[:,i,:,:]
        s_part = s[:,i,:,:]

        logrmin = logr_part.min()
        logrmin = logrmin - 1e-6
        logrmax = logr_part.max()
        logrmax = logrmax + 1e-6
        smin = s_part.min()
        smin = smin - 1e-6
        smax = s_part.max()
        smax = smax + 1e-6

        logrmin_set.append(logrmin)
        logrmax_set.append(logrmax)
        smin_set.append(smin)
        smax_set.append(smax)

        rnew = (resultmax - resultmin) * (logr_part - logrmin) / (logrmax - logrmin) + resultmin
        snew = (resultmax - resultmin) * (s_part - smin) / (smax - smin) + resultmin

        print(np.shape(rnew))
        print(np.shape(snew))
        f[:,i,:,:] = np.concatenate((rnew, snew), axis=2)


    sym_feature = np.transpose(data['symmetryf'], (2, 1, 0)).astype('float32')

    partnum = sym_feature.shape[1]
    modelnum = sym_feature.shape[0]
    vertex_dim = sym_feature.shape[2]
    bbxmin_set = []
    bbxmax_set = []

    sym_feature_tmp = sym_feature
    sym_feature_tmp[np.where(sym_feature_tmp == 0)] = -1
    sym_feature_tmp[:,:,-4:]=sym_feature[:,:,-4:]
    sym_feature_tmp[:,:,-8:-5]=sym_feature[:,:,-8:-5]
    sym_feature = sym_feature_tmp
    # print(data['symmetryf'][1])
    # print(sym_feature[:,:,1])

    sym_featuremin = sym_feature.min() - 1e-6
    sym_featuremax = sym_feature.max() + 1e-6

    bbxmin_set.append(sym_featuremin)
    bbxmax_set.append(sym_featuremax)
    # print(data['symmetryf'][1])
    # print(sym_feature[:,:,1])
    # print(sym_featuremax)
    sym_feature = (resultmax-resultmin)*(sym_feature-sym_featuremin)/(sym_featuremax - sym_featuremin) + resultmin
    print(sym_featuremin)
    print(sym_featuremax)

    modelname = []
    for column in data['modelname']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(u''.join(map(chr, data[column[row_number]][:])))
        modelname.append(row_data[0])
    # modelname = np.squeeze(modelname)
    partname = data['partlist']['name'][0]
    partlist = []
    for i in range(len(partname)):
        partlist.append(''.join(chr(v) for v in data[partname[i]]))

    refmesh_V = np.transpose(data['ref_V']).astype('float32')
    refmesh_F = np.transpose(data['ref_F']).astype('int32') - 1
    refmesh_path = path.split('_')[0] + '.obj'
    mesh = pymesh.form_mesh(refmesh_V, refmesh_F)
    pymesh.save_mesh(refmesh_path, mesh, ascii=True)

    print(all_vertex.shape, vdiff.shape)
    print(logrmin)
    print(logrmax)
    print(smin)
    print(smax)

    return f, nb1, degree1, logrmin_set, logrmax_set, smin_set, smax_set, modelnum, pointnum1, maxdegree1, L1, cotw1, laplacian_matrix, reconmatrix, vdiff, all_vertex, sym_feature, bbxmin_set, bbxmax_set, modelname, partlist, mesh

def load_data_sym(path):
    resultmax = 0.95
    resultmin = -0.95

    data = h5py.File(path,mode = 'r')
    datalist = data.keys()

    sym_feature = np.transpose(data['symmetryf']).astype('float32')
    # print(data['symmetryf'][1])
    # print(sym_feature[:,:,1])

    partnum = sym_feature.shape[1]//5
    modelnum = sym_feature.shape[0]
    vertex_dim = 5
    # print(data['symmetryf'][1])
    # print(sym_feature[:,:,1])

    sym_featuremin = sym_feature.min() - 1e-6
    sym_featuremax = sym_feature.max() + 1e-6
    # print(data['symmetryf'][1])
    # print(sym_feature[:,:,1])
    # print(sym_featuremax)
    sym_feature = (resultmax-resultmin)*(sym_feature-sym_featuremin)/(sym_featuremax - sym_featuremin) + resultmin

    print(sym_featuremin)
    print(sym_featuremax)

    return sym_feature, partnum, vertex_dim, modelnum, sym_featuremin, sym_featuremax

def recover_datasym(recover_feature, sym_featuremin, sym_featuremax):
    resultmax = 0.95
    resultmin = -0.95

    f = (sym_featuremax - sym_featuremin) * (recover_feature - resultmin) / (resultmax - resultmin) + sym_featuremin

    ftmp = f
    ftmp=np.round(ftmp)
    ftmp[np.where(ftmp==-1)]=0
    ftmp[:,:,-8:-5] = f[:,:,-8:-5]
    f = ftmp

    return f

def recover_datasymv2(recover_feature, bbx_centermin, bbx_centermax): # recover for every part
    resultmax = 0.95
    resultmin = -0.95
    modelnum = recover_feature.shape[0]
    partnum = (recover_feature.shape[1]-9)//2


    bbx_center = recover_feature[:,:3]
    symmetry_para = recover_feature[:,3:7]
    binary_part_f = recover_feature[:,7:(8+2*partnum)]
    symmetry_exist = np.expand_dims(recover_feature[:,-1], axis=1)
    bbx_center = (bbx_centermax - bbx_centermin) * (bbx_center - resultmin) / (resultmax - resultmin) + bbx_centermin

    f = np.concatenate([binary_part_f, bbx_center, symmetry_exist, symmetry_para], axis=1)

    return f

def recover_data(recover_feature, logrmin, logrmax, smin, smax, pointnum):
    # print(base)
    # recover_feature = recover_feature + base
    logr = recover_feature[:,:,0:3]
    s = recover_feature[:,:,3:9]
    base_s = np.array([1,0,0,1,0,1]).astype('float32')

    if isinstance(logrmin, np.float32) or isinstance(logrmin, np.float64):
        # print('yangjie')

        resultmax = 0.95
        resultmin = -0.95

        s = (smax - smin) * (s - resultmin) / (resultmax - resultmin) + smin
        logr = (logrmax - logrmin) * (logr - resultmin) / (resultmax - resultmin) + logrmin

        # s = s + base_s

    elif isinstance(logrmin, np.ndarray):
        # mean_logr, mean_s, std_logr, std_s
        resultmax = 0.95
        resultmin = -0.95

        r_min = smax['rmin']
        r_max = smax['rmax']
        s_min = smax['smin']
        s_max = smax['smax']

        s = (s_max - s_min) * (s - resultmin) / (resultmax - resultmin) + s_min
        logr = (r_max - r_min) * (logr - resultmin) / (resultmax - resultmin) + r_min

        logr = logr * smin + logrmin
        s = s * smax['std_s'] + logrmax

    else:
        print('error')


    return s, logr

def gaussian(batch_size, n_dim, mean=0.0, var=1.0, n_labels=10, use_label_info=False):
    if use_label_info:
        if n_dim != 2:
            raise Exception("n_dim must be 2.")

        def sample(n_labels):
            x, y = np.random.normal(mean, var, (2,))
            angle = np.angle((x - mean) + 1j * (y - mean), deg=True)

            label = (int(n_labels * angle)) // 360

            if label < 0:
                label += n_labels

            return np.array([x, y]).reshape((2,)), label

        z = np.empty((batch_size, n_dim), dtype=np.float32)
        z_id = np.empty((batch_size, 1), dtype=np.int32)
        for batch in range(batch_size):
            for zi in range(int(n_dim / 2)):
                a_sample, a_label = sample(n_labels)
                z[batch, zi * 2:zi * 2 + 2] = a_sample
                z_id[batch] = a_label
        return z, z_id
    else:
        z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
        return z

def spilt_dataset(num, percent_to_train, name="id.dat"):

    if os.path.isfile(name):
        id = pickle.load(open(name, 'rb'))
        id.show()
        Ia = id.Ia
    else:
        Ia = np.arange(num)
        Ia = random.sample(list(Ia), int(num * percent_to_train))

        id = Id(Ia)
        f = open(name, 'wb')
        pickle.dump(id, f, 0)
        f.close()
        id.show()

    Ia_C=list(set(np.arange(num)).difference(set(Ia)))

    return Ia, Ia_C

def printout(flog, data, epoch=0, interval = 50, write_to_file = True):
    if epoch % interval==0:
        print(data)
        flog.write(str((data + '\n')*write_to_file))

def argpaser2file(args, name='example.ini'):
    d = args.__dict__
    cfpar = configparser.ConfigParser()
    cfpar['default'] = {}
    for key in sorted(d.keys()):
        cfpar['default'][str(key)]=str(d[key])
        print('%s = %s'%(key,d[key]))

    with open(name, 'w') as configfile:
        cfpar.write(configfile)

def inifile2args(args, ininame='example.ini'):

    config = configparser.ConfigParser()
    config.read(ininame)
    defaults = config['default']
    result = dict(defaults)
    # print(result)
    # print('\n')
    # print(args)
    args1 = vars(args)
    # print(args1)

    args1.update({k: v for k, v in result.items() if v is not None})  # Update if v is not None

    # print(args1)
    args.__dict__.update(args1)

    # print(args)

    return args

def getFileName(path, postfix = '.ini'):
    filelist =[]
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():
        if os.path.splitext(i)[1] == postfix:
            print("[{}] {}".format(f_list.index(i),i))
            filelist.append(i)

    return filelist

def random_sample_range(_min, _max, num=25):
    rng = np.random.RandomState(12345)
    x = np.random.rand(num)
    x = x * (_max - _min) + _min
    rng.shuffle(x)

    return x

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def readmesh(objpath):
    # mesh = om.TriMesh()
    # mesh.request_halfedge_normals()
    # mesh.request_vertex_normals()
    # result =
    return om.read_trimesh(objpath)

def savemesh(mesh, objpath, newv):
    # get all points of the mesh
    point_array = mesh.points()
    # print(np.shape(point_array))

    # translate the mesh along the x-axis
    for vh in mesh.vertices():
        # print(vh.idx())
        point_array[vh.idx()] = newv[vh.idx()]
    # point_array = newv
    # point_array += np.array([1, 0, 0])

    # write and read meshes
    om.write_mesh(objpath, mesh)

def savemesh_pymesh(mesh, objpath, newv):
    # write and read meshes
    new_mesh = pymesh.form_mesh(newv, mesh.faces)
    pymesh.save_mesh(objpath, new_mesh, ascii=True)

def get_batch_data(data1, data2, batch_size):
    data_num = len(data1)
    if data_num == 0:
        return data1, data2

    import math
    if data_num < batch_size:
        if batch_size//data_num>1:
        # remainder = batchsize - data_num
            reminder = math.pow(2, math.ceil(math.log(data_num, 2))) - data_num
        else:
            reminder = batch_size - data_num
    else:
        reminder = batch_size-(data_num%batch_size)

    Ia = np.arange(data_num)
    Ia = random.sample(list(Ia), int(reminder))
    data1 = np.concatenate((data1, data1[Ia]), axis = 0)
    data2 = np.concatenate((data2, data2[Ia]), axis = 0)
    return data1, data2

def allFactor(n):
    if n == 0: return [0]
    if n == 1: return [1]
    rlist = []
    i = 1
    while i <= n:
        if n % i == 0:
            rlist.append(i)
            n = n // i
            i = 2
            continue
        i += 1
    rlist=rlist[1:]
    return rlist

def get_batch_data1(data1, data2, repeat_epoch, batch_size):
    data_num = len(data1)
    if data_num == 0:
        return data1, data2, int(0)
    big_factor=allFactor(data_num*repeat_epoch)
    small_factor=allFactor(batch_size)

    from collections import Counter
    c1 = Counter(big_factor)
    c2 = Counter(small_factor)
    diff = c2 - c1
    factors = np.prod(list(diff.elements()))
    remainder = factors - (data_num*repeat_epoch)%factors

    Ia = np.arange(data_num)
    Ia = random.sample(list(Ia), int(remainder))
    data1 = np.concatenate((data1, data1[Ia]), axis = 0)
    data2 = np.concatenate((data2, data2[Ia]), axis = 0)

    epoch = int((len(data1)*repeat_epoch)/batch_size)

    return data1, data2, epoch

class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

def traversalDir_FirstDir(path, perfix = ''):
    # os.path.getmtime()
    # os.path.getctime()
    dir_list = []
    if (os.path.exists(path)):
        files = glob.glob(path + '/' + perfix + '*' )
        for file in files:
            if (os.path.isdir(file)):
                h = os.path.split(file)
                dir_list.append(h[1])
        dir_list = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse = True)
        return dir_list




