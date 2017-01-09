import numpy as np
import pdb
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.io as spio
import sys

class Node(object):
    def __init__(self, I_0, alpha, theta_0, sm_time, dt):
        self.theta = theta_0
        self.I_0 = I_0
        self.theta_0 = theta_0
        self.alpha = alpha
        self.I = I_0
        self.dt = dt
        self.sm_time = sm_time
        self.epsilon = -0.00001
        self.step = int(np.floor(self.sm_time / self.dt))

    def theta_der(self, theta, alpha, I):
        return self.dt*( 1 - np.cos(theta) + alpha * I * (1 + np.cos(theta)))

    def model(self,  w, t_input_dirac):
        assert(len(w) == len(t_input_dirac))
        dirac_flag = True
        fire_t = []
        thetas = []
        self.theta = self.theta_0
        for t in range(self.step):
            self.I = self.I_0
            thetas.append(self.theta) 
            if t in t_input_dirac:    # The first time cross the t_input_dirac
                id_ = np.where(t_input_dirac == t)[0][0]
                if id_ != len(t_input_dirac):
                    self.I += 1/self.dt*w[id_]  # Synaptic currents as Diracs
            delta_theta = self.theta_der(self.theta, self.alpha, self.I)
            theta_new = self.theta + delta_theta
            #print theta_new
            if self.theta < np.pi < theta_new:  # cross pi
                self.theta = theta_new
                fire_t.append(t)
            else:
                self.theta = theta_new
        if len(fire_t) == 0:
            fire_t.append(self.step)
        return fire_t, thetas



class NET(object):
    def __init__(self, I_0, sm_time, dt):
        self.I_0 = I_0
        self.nlayer = 0
        self.sm_time = sm_time
        self.dt = dt
        self.layers = []
        self.alphas = []
        self.step = int( np.floor(self.sm_time / self.dt))

    def addlayer(self, cur_nodes, alpha):
        layers = np.array([], dtype=object)
        theta_0 = -np.arccos((1 + alpha * self.I_0) / (1 - alpha * self.I_0)) 
        layers = np.concatenate((layers, np.array([ Node(self.I_0, alpha, theta_0, self.sm_time, self.dt) for i in xrange(cur_nodes)])))
        self.layers.append(layers)
        self.alphas.append(alpha)
        self.nlayer += 1

    def stimulate(self, input_, w, layer):
        n_nodes = len(self.layers[layer]) 
        [pre_nodes, cur_nodes] = np.shape(w)
        assert(n_nodes == cur_nodes)
        t = np.zeros(n_nodes)
        thetas = np.zeros((n_nodes, self.step))
        for i in xrange(n_nodes):
            w_tmp = np.transpose(w)
            t_tmp, thetas[i] = self.layers[layer][i].model(w_tmp[i], input_)
            t[i] = t_tmp[0]
        return t, thetas

class SparseCoding(object):
    def __init__(self, net, samples):
        (self.nsample, self.in_) = np.shape(samples)
        self.dt = 0.1
        self.sm_time = 20
        self.tao = 0.1
        self.C = 50 
        self.lambda_ = float(sys.argv[2])#0#.0000001 #0.00001 #0.000000001
        self.mu = float(sys.argv[3]) #0.0000001
        self.wrange = [0, 0.3]
        self.step = self.sm_time / self.dt
        self.net = net
        self.trainX = samples

    def delta_t_delta_w(self, theta_before, theta_after, alpha, I_0):
        return -(1+np.cos(theta_before))*alpha / ((1-np.cos(theta_after)) + alpha* I_0*( 1+ np.cos(theta_after)))


    def forward(self, _input, w ):
        half_nlayer = self.net.nlayer / 2
        t_input_dirac =  np.floor(_input / self.dt)
        last_in = t_input_dirac
        last_thetas = [];
        for layer in xrange(self.net.nlayer):
            n_node = len(self.net.layers[layer])
            if layer < half_nlayer:
                cur_w = w[layer]
                t_all, thetas =  self.net.stimulate(last_in, cur_w, layer)

            if layer >= half_nlayer:
                w[layer] =  np.transpose( w[2* half_nlayer - layer - 1] ) 
                cur_w = w[layer]
                t_all, thetas =  self.net.stimulate(last_in, cur_w, layer)
            if layer != self.net.nlayer - 1:
                last_in = t_all
                last_thetas = thetas
        return t_all, thetas, last_thetas, last_in

    def shuffle_data(self, train_rate):
        assert(train_rate >= 0 and train_rate <=1)
        trainset = set([]);
        ntrain = int(np.floor(train_rate*self.nsample))
        while len(trainset) < ntrain:
            remain = ntrain - len(trainset)
            train_ids = [ int(np.random.randint(0,self.nsample)) for i in xrange(remain) ]
            trainset.update( train_ids)
#            valset = [i for i in xrange(self.nsample) if i not in trainset ] 
	val_data = generate_fixed_samples()
        #return self.trainX[list(trainset)], self.trainX[list(valset)]
        return self.trainX[list(trainset)], val_data[0:1024]
        
    def learn(self, ISI, oper):
        print 'Start learning....'
        PLOT_INTERVAL = 128
        assert(self.net.nlayer % 2 == 0)
        half_nlayer = self.net.nlayer/2
        n = self.in_
	if oper == 'train':
	    self.w = []
	    for i in xrange(self.net.nlayer):
		if i < half_nlayer:
		    m = len(self.net.layers[i])
		    self.w.append(np.random.rand(n,m) * (wrange[1] - wrange[0]) + wrange[0])
		    #self.w.append(np.array([[ 1.43237788,2.99588509], [ 0.81954857,3.04929579],[ 0.88254488,3.05563769]]))
		    #self.w.append(np.array([[ 1.10075154,3.34409603],[ 0.47409087,3.39723133],[ 0.537345, 3.40360319]]))
		    #self.w.append(np.array([[ 1.38964519,3.21326752],[ 0.85084747,3.24406681],[ 0.85547578,3.24433349]]))
		    n = m
		else:
		    self.w.append(np.transpose(self.w[2*half_nlayer - i - 1]))
        p_node = self.in_
        c_node = len(self.net.layers[0]) 
        delta = np.zeros((c_node, p_node))
        eps = 0.1
        sum_error = 1
        max_iter = 200
        it = 0
        train_rate = 0.5
        last_w = np.copy( self.w[0] )
        w0 = np.zeros((int(max_iter*self.nsample*train_rate), p_node* c_node))
        #w0 = np.zeros((int(max_iter), p_node* c_node))
        convergence_count = 0
        E_all = np.zeros((int(max_iter),1))
	best_w = 0
        while it < max_iter and sum_error > eps:
            E_total = np.zeros(p_node)
            train_set, val_set = self.shuffle_data(train_rate)
            [train_nsample, _] = np.shape(train_set)
            #w0[it] = self.w[0].flatten()
            count = 0
#	    self.mu = max( 0.0002, self.mu - np.floor(it)*0.00005)
            batch = 128
            w_tmp = np.zeros((c_node, p_node)) 
            for i in xrange(train_nsample):
                w0[it*train_nsample + i] = self.w[0].flatten()
                t, thetas, last_thetas, res = self.forward(train_set[i], self.w)
                # update last layer of weight
                for j in xrange(p_node):
                    for k in xrange(c_node):
                        if res[k] < self.step - 1:
                            theta_before = thetas[j][int(res[k])]
                            theta_after = thetas[j][int(res[k])+1]
                    # dts/dwi
                            delta[k][j] = self.delta_t_delta_w(theta_before, theta_after, self.net.alphas[self.net.nlayer-1], self.net.I_0) 
                        else:
                            delta[k][j] = 0
                #print delta
                phi = np.zeros(c_node)
                for k in xrange(c_node): 
                    phi[k] = phi[k]* self.tao + (1-self.tao)*(res[k] - np.mean(res))*self.dt
                    for j in xrange(p_node):
                        error_correct = self.lambda_*phi[k]
                        gradient = -2* self.mu* (t[j]*self.dt - train_set[i][j] - ISI)* max(delta[k][j], -self.C) 
#                        if error_correct > 0.5*gradient:
#                            w_tmp[k][j] = gradient
#                        else:
                        w_tmp[k][j] += gradient - error_correct
                if i != 0 and i % batch == 0:
                    last_w = np.copy(self.w[0])
                    self.w[0] += np.transpose(w_tmp)/batch
                    self.w[self.net.nlayer - 1] += w_tmp/batch
                    sum_error = sum(sum(abs(self.w[0] - last_w)))
                E= abs(t* self.dt - train_set[i]  - np.ones(self.in_)*ISI)
                E_total += E
                if i % PLOT_INTERVAL == 0:
		    print 'Layer Y fires: ', res[0:64]*self.dt
                    print 'desired output:', train_set[i][0:10] + ISI
		    print 'actual fire:', t[0:10]*self.dt
                    print 'sample #', i, 'Error ', E[0:10]
                    # plot weight changes
                    w1 = np.transpose(w0)
                    plt.figure('weight')
                    for k in xrange(p_node* c_node):
			if k % 128 == 0:
			    plt.plot(xrange( (it)*train_nsample + i ), w1[k][:(it)*train_nsample+i])
                        #plt.plot(xrange( (it+1) ), w1[i][:(it+1)])
                    plt.draw()
                    plt.show(block=False)
                    plt.pause(0.001)
                    plt.clf()
            # plot theta changes
            plt.figure('Theta')
            plt.plot(thetas[0])
            plt.draw()
            plt.show(block=False)
            plt.pause(1)
            gt, pred = self.predict(val_set[0:4])
            plot_single_patch(gt, pred)
            print 'Round #%d, step: %f, Error %f'%(it,self.mu, sum(E_total)/train_nsample/256)        
	    w1 = self.w[0].flatten()
            print 'current weight:', w1[0:10]
            E_all[it] = sum(E_total)/train_nsample/256
	    if E_all[it] < E_all[best_w]:
		best_w = it
		pickle_save('sparse_model_w_%s_%f_%f.pkl'%(oper, self.mu, self.lambda_), w0[(best_w+1)*train_nsample-1])
            plt.figure('Error_%f_%f'%(self.mu, self.lambda_))
            plt.plot(xrange( (it+1) ), E_all[:(it+1)])
            plt.draw()
            plt.show(block=False)
            plt.pause(0.001)
            plt.clf()
            if it != 0 and sum(E_all[it - 1])  < sum(E_total):
                convergence_count += 1
            else:
                convergence_count = 0
            it += 1


    def predict(self, samples):
        print 'Predicting...'
        (N, dim) = np.shape(samples)
	pred = []
        for i in xrange(N):
	    print 'Predicting image patch %d'%i
            t, deltas, last_thetas, res = self.forward(samples[i], self.w)
	    #pred.append(t*self.dt-9)
            t_fire = t *self.dt
	    pred.append(t_fire)
	    print 'Firing time: ', self.dt*t[0:5]
	    print 'origin image:', samples[i, 0:5]
	    print 'predicted image:', pred[i][0:5]
	pred = np.array(pred)
        return samples, pred
def plot_single_patch(gt, pd):
    (N, dim) = np.shape(gt)
    patch_size = np.sqrt(dim)
    print 'Ploting...'
    plt.figure('image')
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.imshow(gt[0].reshape(patch_size, patch_size), cmap='gray')
    ax2.imshow(pd[0].reshape(patch_size, patch_size),cmap='gray')
    ax3.imshow(gt[1].reshape(patch_size, patch_size), cmap='gray')
    ax4.imshow(pd[1].reshape(patch_size, patch_size),cmap='gray')
    plt.draw()
    plt.show(block=False)
    plt.pause(0.001)
    plt.clf()

def plot_pca(gt, pd):
    (N, dim) = np.shape(gt)
    image_size = int(np.sqrt(N*dim))
    patch_size = int(np.sqrt(dim))
    gt_image = np.zeros((image_size, image_size))
    pd_image = np.zeros((image_size, image_size))
    for i in xrange(int(np.sqrt(N))):
	for j in xrange(int(np.sqrt(N))):
	    id_ = int(i*np.sqrt(N)+j)
	    gt_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = gt[id_].reshape(patch_size, patch_size)
	    pd_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = pd[id_].reshape(patch_size, patch_size)
    print 'Ploting...'
    plt.figure('image')
    plt.subplot(121)
    plt.imshow(gt_image, cmap='gray')
    plt.subplot(122)
    plt.imshow(pd_image, cmap='gray')
    plt.draw()
    plt.show(block=False)
    plt.pause(0.001)
    plt.clf()

def pickle_save(fname, data):
    with open(fname, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
        print "saved to %s"%fname

def pickle_load(fname):
    with open(fname, 'rb') as _input:
        return pickle.load(_input)

def generate_random_samples(N):
    samples = loadmat('IMAGES_RAW.mat')['IMAGESr']
    patch_size = 16
    (width, height, pnum) = np.shape(samples)
    patches = []
    for i in xrange(N):
        iid = int(np.floor(np.random.rand(1)*pnum))
        corner = np.floor( np.multiply(np.random.rand(2), np.array([ width - patch_size, height - patch_size ]))) 
	corner = [int(corner[i]) for i in xrange(len(corner))]
        patch = samples[corner[0]:corner[0]+patch_size,corner[1]:corner[1]+patch_size,iid]
        # normalize path
        max_pixel = np.amax(patch)
        min_pixel = np.amin(patch)
        patch = (patch - min_pixel)*1.0/(max_pixel - min_pixel)
	patch = patch.flatten()
        patches.append(patch)
    return np.array(patches)

def generate_fixed_samples():
    samples = loadmat('IMAGES_RAW.mat')['IMAGESr']
    patch_size = 16
    (width, height, pnum) = np.shape(samples)
    patches = []
    for i in xrange(1):
	for j in xrange( width / patch_size ):
	    for k in xrange( height / patch_size):
		patch = samples[j*patch_size: (j+1)*patch_size,k*patch_size:(k+1)*patch_size,i]
		patch = patch.flatten()
		patches.append(patch)
    # normalize patch
    patch = np.array(patches)
    mean_patches = np.mean(patches)
    std_patches = np.std(patches)
    patches = (patches - mean_patches) / std_patches
    min_p = np.amin(patches)
    patches -= min_p
    return patches 


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray):
            dict[strg] = _tolist(elem)
        else:
            dict[strg] = elem
    return dict

def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem,np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list

if __name__ == '__main__':
    # construct network
    I_0 = -0.01
    alpha = 0.1
    wrange = [0,0.3]
    N = 300
    sm_time = 20
    dt = 0.1
    ISI = 9
    m = 64
    n = 256
    #samples = generate_random_samples(N)
    samples = generate_fixed_samples()
    samples = samples[:1024]
    net = NET(I_0,sm_time, dt)
    net.addlayer(m, alpha*m/n)
    net.addlayer(n, alpha)
    oper = sys.argv[1]
    w_model = 'sparse_model_w.pkl'
    sample_name = 'sparse_samples.pkl'
    if oper == 'retrain':
        w_model = sys.argv[4]
        samples = pickle_load(sample_name)
        w = pickle_load(w_model)
        w = w.reshape(256,64)
        w = [w, np.transpose(w)]
        pca = SparseCoding(net, samples)
        pca.w = w
        pca.learn(ISI, oper)
    elif oper == 'train':
        samples = generate_fixed_samples()
        print np.shape(samples)
        pickle_save(sample_name, samples)
        pca = SparseCoding(net, samples)
        pca.learn(ISI, oper)
    else:
        w_model = sys.argv[4]
        samples = pickle_load(sample_name)
        pca = SparseCoding(net, samples)
        w = pickle_load(w_model)
        w = w.reshape(256,64)
        w = [w, np.transpose(w)]
        pca.w = w
 
    test_samples = samples #generate_fixed_samples()
    gt, pred = pca.predict(test_samples[:1024])
    pickle_save('gt.pkl', gt)
    pickle_save('pred.pkl', pred)
    pdb.set_trace()
    plot_pca(gt, pred)
    pdb.set_trace()
 
