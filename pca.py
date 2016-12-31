import numpy as np
import pdb
import matplotlib.pyplot as plt


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
        return ( 1 - np.cos(theta) + alpha * I * (1 + np.cos(theta)))

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
                    self.I += w[id_]  # Synaptic currents as Diracs
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

class PCA(object):
    def __init__(self, net, samples):
        (self.nsample, self.in_) = np.shape(samples)
        self.dt = 0.05
        self.sm_time = 20
        self.tao = 0.1
        self.C = 50
        self.lambda_ = 0.001
        self.mu = 0.00001
        self.wrange = [0.5, 1.5]
        self.step = self.sm_time / self.dt
        self.net = net
        self.trainX = samples

    def delta_t_delta_w(self, theta_before, theta_after, alpha, I_0):
        return -(1+np.cos(theta_before))*alpha / ((1-np.cos(theta_after)) + alpha* I_0*( 1+ np.cos(theta_after)))


    def forward(self, _input, w ):
        half_nlayer = self.net.nlayer / 2
        t_input_dirac =  np.floor(_input / self.dt)
        last_in = t_input_dirac
        p_node = len(last_in)
        last_thetas = [];
        for layer in xrange(self.net.nlayer):
            n_node = len(self.net.layers[layer])
            if layer < half_nlayer:
                cur_w = w[layer]
                t_all, thetas =  self.net.stimulate(last_in, cur_w, layer)
                last_in = t_all

            if layer >= half_nlayer:
                w[layer] =  np.transpose( w[2* half_nlayer - layer - 1] ) 
                cur_w = w[layer]
                t_all, thetas =  self.net.stimulate(last_in, cur_w, layer)
            if layer != self.net.nlayer - 1:
                last_in = t_all
                p_node = n_node
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
            valset = [i for i in xrange(self.nsample) if i not in trainset ] 
        return self.trainX[list(trainset)], self.trainX[list(valset)]

    def learn(self, ISI):
        print 'Start learning....'
        assert(self.net.nlayer % 2 == 0)
        half_nlayer = self.net.nlayer/2
        self.w = []
        n = self.in_
        for i in xrange(self.net.nlayer):
            if i < half_nlayer:
                m = len(self.net.layers[i])
                self.w.append(np.random.rand(n,m) * (wrange[1] - wrange[0]) + wrange[0])
                n = m
            else:
                self.w.append(np.transpose(self.w[2*half_nlayer - i - 1]))
        p_node = self.in_
        c_node = len(self.net.layers[0]) 
        delta = np.zeros((c_node, p_node))
        eps = 0.000001
        sum_error = 1
        max_iter = 100
        it = 0
        train_rate = 0.8
        last_w = np.copy( self.w[0] )
        w0 = np.zeros((int(max_iter*self.nsample*train_rate), p_node* c_node))
        while it < max_iter and sum_error > eps:
            E_total = 0
            train_set, val_set = self.shuffle_data(train_rate)
            [train_nsample, _] = np.shape(train_set)
            for i in xrange(train_nsample):
                w0[it*train_nsample + i] = self.w[0].flatten()
                t, thetas, last_thetas, res = self.forward(train_set[i], self.w)
                # print t*self.dt
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
                w_tmp = np.transpose(self.w[0])
                #pdb.set_trace()
                for k in xrange(c_node): 
                    #phi[k] = phi[k]* self.tao + (1-self.tao)*(res[k] - np.mean(res))*self.dt
                    for j in xrange(p_node):
                        if delta[k][j] < 0 and delta[k][j] > - self.C:
                            w_tmp[k][j] += -2* self.mu* (t[j]*self.dt - train_set[i][j] - ISI) * delta[k][j] - self.lambda_ * phi[k]
                        else:
                            w_tmp[k][j] += 2* self.mu* (t[j]*self.dt - train_set[i][j] - ISI) * self.C - self.lambda_ * phi[k]
                pdb.set_trace()
                self.w[0] = np.transpose(w_tmp)
                self.w[self.net.nlayer - 1] = w_tmp
                E= np.sum((t* self.dt - train_set[i]  - np.ones((self.in_, 1))*ISI)**2)
                #print 'sample #%d, Error %f'%(i, E)
                E_total += E
                sum_error = sum(sum(abs(self.w[0] - last_w)))
                last_w = np.copy(self.w[0])
            gt, pred = self.predict(val_set)
            plot_pca(gt, pred)
            w1 = np.transpose(w0)
            plt.figure('weight')
            for i in xrange(p_node* c_node):
                plt.plot(xrange( (it+1)*train_nsample ), w1[i][:(it+1)*train_nsample])
            plt.draw()
            plt.show(block=False)
            plt.pause(0.001)
            print 'Round #%d, Error %f'%(it, E_total)        
            it += 1
        w0 = np.transpose(w0)
        for i in xrange(p_node* c_node):
            plt.plot(xrange(it ), w0[i][:it])

        plt.show()


    def predict(self, samples):
        print 'Predicting...'
        (N, dim) = np.shape(samples)
        samples_t = np.transpose(samples)
        gt = np.zeros((2,N))
        gt[0] = samples_t[1] - samples_t[0]
        gt[1] = samples_t[2] - samples_t[0]
        pred = np.zeros((2, N))
        for i in xrange(N):
            t, deltas, last_thetas, res = self.forward(samples[i], self.w)
            pred[0][i] = (t[1] - t[0])*self.dt
            pred[1][i] = (t[2] - t[0])*self.dt
        return gt, pred

def plot_pca(gt, pd):
    print 'Ploting...'
    plt.figure('scatter')
    plt.scatter(gt[0], gt[1], c=['b'], alpha=0.5)
    plt.hold(True)
    plt.scatter(pd[0], pd[1], c=['r'], alpha=0.5)
    plt.draw()
    plt.show(block=False)
    plt.pause(0.001)
    plt.clf()


    
def generate_samples(N):
    '''
    return N samples of 3-D in shape (N, 3)
    '''
    samples = np.random.randn(2,N)
    t0 = 3*np.ones((1,N))
    t1 = t0+ np.cos(np.pi/3)*samples[0] + 0.5* np.sin(np.pi/3)* samples[1]
    t2 = t0+ np.cos(np.pi/3)*samples[1]*0.5 +  np.sin(np.pi/3)* samples[0]
    return np.transpose(np.concatenate( (t0, t1, t2), axis = 0))

if __name__ == '__main__':
    # construct network
    I_0 = -0.01
    alpha = 0.1
    wrange = [0.5,1.5]
    N = 500
    sm_time = 20
    dt = 0.05
    ISI = 5
    samples = generate_samples(N)
    net = NET(I_0,sm_time, dt)
    net.addlayer(2, alpha*2/3)
    net.addlayer(3, alpha)

    pca = PCA(net, samples)
    pca.learn(ISI)
    gt, pred = pca.predict(samples)
    plot_pca(gt, pred)
