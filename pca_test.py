import numpy as np
import matplotlib.pyplot as plt

def get_input():
    data = []
    #with open('2D_gaussian_dist.txt') as fin:
    with open('uniform.txt') as fin:
    #with open('2D_gaussian_dist_shrink.txt') as fin:
    #with open('2D_gaussian_dist_positive.txt') as fin:
    #with open("2D_gaussian_dist_small_var_positive.txt") as fin:
        lines = fin.readlines()
    for line in lines:
        s = []
        vals = line.strip().split()
        for val in vals:
            s.append(float(val))
        data.append(s)
    return np.array(data)


def check_firing_X(X_ft, step, timestep):
    I_Y = np.zeros(3)
    time = step * timestep
    for k in range(3):
        if 0.0 <= time - X_ft[k] < timestep:
            I_Y[k] = 1.0
    return I_Y
    
def theta_der(theta, alpha, I):
    return 1.0 - np.cos(theta) + alpha * I * (1.0 + np.cos(theta))

def update_theta(thetas, I_i, w, I_0, alpha, timestep):
    I = np.dot(I_i, w) + I_0
    delta_thetas = theta_der(thetas, alpha, I)
    new_thetas = thetas + delta_thetas # * timestep

    firing = ((thetas < np.pi) & (new_thetas >= np.pi)).astype(int)
    #print 'new_thetas:', new_thetas
    #print 'firing:', np.shape(firing)
    #print firing
    return new_thetas, firing

def delta_weight(alpha, cos_theta_minus, cos_theta_plus, I_0, eta, C, ft_actual, ft_desired):
    der_ts_wi = -alpha * (1 + cos_theta_minus) / ((1 - cos_theta_plus) + alpha * I_0 * (1 + cos_theta_plus))
    delta_wi = -2 * eta * (ft_actual - ft_desired) * der_ts_wi

    #print 'cos_theta_minus/plus:', cos_theta_minus, cos_theta_plus

    if der_ts_wi <= -C or der_ts_wi >= 0:
        delta_wi = 2 * eta * (ft_actual - ft_desired) * C
        #print 'cos_theta_minus/plus:', cos_theta_minus, cos_theta_plus, 'delta_wi:', delta_wi
    return delta_wi

def plot_w_update(w_history):
    for x in range(6):
        plt.plot(w_history[x])
    plt.show()
    #print w_history

def pca():
    # Parameters
    n, m = 3, 2     # Size of neuron population X and Y
    I_0 = -0.01
    tau = 0.1
    alpha_X = 0.1
    alpha_Y = alpha_X * 1.0 * m / n
    timestep = 0.05   # ms
    num_iter_trial = 401
    ISI = 0.5   # ms
    C = 1000
    eta = 0.00001
    lamuda = 10.0 * eta
    num_iter_learning = 20000


    # Topology
    w = np.random.uniform(0.5, 1.5, (2, 3)) # Initialize the weight matrix
    #w = np.array([[0.69986636, 0.26886332, 0.04155199], [-0.06064578, 0.3710673, 0.60037819]])
    #w = np.array([[ 0.63182949, -0.08249607,  0.63205652], [ 0.08574707,  0.72654316,  0.08158783]])

    #w = np.array([[-1.50879568,  0.44550418,  1.90464732], [ 2.69228429,  0.5133216,  -1.24438609]])#x1
    #w = np.array([[-2.82295985,  0.43981246,  3.32091401], [ 5.35372586 , 0.53064058, -2.65269579]]) #x2
    #w = np.array([[-3.59960994,  0.40231348 , 4.2435657 ], [ 6.89327113,  0.53724807, -3.60413348]]) #x3

    #w = np.array([[ 1.37049837, -0.4799603,  -1.04055838], [-2.03708624,  0.67384842,  0.68324663]])# positive1
    #w = np.array([[1.34733182, -0.46383421, -1.36568038], [-4.19716343,  0.69846158,  0.70854153]])#positive2

    #w = np.array([[-0.06104861,  0.65189425,  0.64795393], [ 0.80379834, -0.03142514,  0.03048122]]) # positive_new 1
    #w = np.array([[-0.06109321,  0.65298658,  0.6339056 ], [ 0.80375245, -0.02127992,  0.01800156]])# positive_new 2
    #w = np.array([[-0.06184917,  0.64656136,  0.6323295 ], [ 0.80323337, -0.01515438,  0.02725966]]) # positive_new 3

    #w = np.array([[-0.647, 2.41, 3.294], [3.94, 1.07, 0.053]]) # paper

    #w = np.array([[-0.65153531890373018, 2.3871210136956238, 3.2949539562446173], [3.9381781637219668, 0.69381154703036074, -0.14940643314926633]])
    #w = np.array([[-0.65226165379659284, 2.3892026484443698, 3.2964837590526859], [3.9360338198366169, 0.68320719582864187, -0.20506202274993959]]) # Final

    #3/8 pi
    #w = np.array([[2.1652134673146062, 2.167390038392893, 2.0798503575640668], [1.2639322851552297, 0.058276076049662937, 0.87921623515580105]])
    
    #test
    #w = np.array([[-0.47510387138909832, 2.5219989675378485, 3.433469556605937], [3.7954872100973676, 0.51916490608819388, -0.58669136953534251]])

    # linear
    w = np.array([[1.4053885118930365, 1.2666870092287592, 0.72938860548410267], [1.5549516373945689, 1.6333115795030138, 1.9350923932911437]])


    # add timetemp

    w_history = [[] for k in range(6)]

    plot = True
    predict = False
    predict = True
    pred_X, pred_Y = [], []
    actual_X, actual_Y = [], []

    phi = np.zeros(2)

    input_data = get_input()
    print "Input size:", np.shape(input_data)

    for learn in range(num_iter_learning + 1):
        if learn == len(input_data):
            input_data = np.random.permutation(input_data)

        if learn % 1 == 0:
            log = True
        else:
            log = False
        if log:
            print 'Trial:', learn, '=============================='


        if predict and learn >= len(input_data):
            break
        if not predict and plot:
            temp = w[:].reshape(6, 1)
            for x in range(len(temp)):
                w_history[x].append(temp[x][0])
            if learn % 2500 == 0 and learn > 0:
                plot_w_update(w_history)

        X_ft = input_data[learn % len(input_data)]
        #X_ft = input_data[2]

        # Round the input_data # FIXME
        #for k in range(len(X_ft)):
        #    X_ft[k] = int(X_ft[k] / timestep) * 1.0 * timestep


        if log:
            print 'input:', X_ft,
        theta_Y = np.array([ - np.arccos((1 + alpha_Y * I_0) / (1 - alpha_Y * I_0)) for k in range(2)])
        theta_Xprime = np.array([ - np.arccos((1 + alpha_X * I_0) / (1 - alpha_X * I_0)) for k in range(3)])

        X_ft_desired = X_ft + ISI
        X_ft_actual = [timestep * (num_iter_trial-1) for k in range(3)]
        cos_theta_minus = [[None for k in range(2)] for i in range(3)]
        cos_theta_plus = [[None for k in range(2)] for i in range(3)]

        #print 'Current_weight:', w
        Y_firing_time = np.array([20.0, 20.0])

        display = False
        # Begin simulation
        for step in range(num_iter_trial):
            if display:
                print 'time:', step * timestep,
            I_i_Y = check_firing_X(X_ft, step, timestep)
            if display and np.sum(I_i_Y) > 0:
                print 'Dirac:', I_i_Y,
            theta_Y_new, I_i_Xprime = update_theta(theta_Y, I_i_Y, w.T, I_0, alpha_Y, timestep)
            theta_Y = theta_Y_new
            theta_Xprime_new, firing_Xprime = update_theta(theta_Xprime, I_i_Xprime, w, I_0, alpha_X, timestep)
            if display:
                print 'theta_Y:', theta_Y_new, 'I_i_Xprime:', I_i_Xprime,
                print 'theta_Xprime:', theta_Xprime_new, 
            # Update cos_theta_minus and _plus if Y fires
            for y in range(2):
                for x in range(3):
                    if I_i_Xprime[y] == 1 or (cos_theta_minus[x][y] == None and step == num_iter_trial - 1): # Firing, or last step
                        cos_theta_minus[x][y] = np.cos(theta_Xprime[x])
                        cos_theta_plus[x][y] = np.cos(theta_Xprime_new[x])
            theta_Xprime = theta_Xprime_new

            if display:
                if np.sum(firing_Xprime):
                    print 'firing:', firing_Xprime,
                else:
                    print 'No_output_firing',
            for k in range(3):  # Get firing time at the X' layer
                if firing_Xprime[k] == 1:
                    X_ft_actual[k] = step * timestep
            for k in range(2):  # Get firing time at the Y layer
                if I_i_Xprime[k] == 1:
                    Y_firing_time[k] = step * 1.0 * timestep

            if display:
                print ''
            if step > 10 and False:
                break

        # Update weight matrix
        # if predict, then keep the given weight matrix
        #print 'cos_theta_minus, cos_theta_plus:', cos_theta_minus, cos_theta_plus
        #print np.shape(w), np.shape(X_ft_actual),  np.shape(X_ft_desired), np.shape(cos_theta_minus), np.shape(cos_theta_plus)
        delta_w_mat = np.zeros((2, 3))
        for y in range(2):
            for x in range(3):
                if predict:
                    break
                #    w[y][x] += delta_weight(alpha_X, cos_theta_minus[x][y], cos_theta_plus[x][y], I_0, \
                #                            eta * 0.5, C, X_ft_actual[x], X_ft_desired[x])
                #    continue
                delta_w_mat[y][x] += delta_weight(alpha_X, cos_theta_minus[x][y], cos_theta_plus[x][y], I_0, eta, C, X_ft_actual[x], X_ft_desired[x])
                delta_w_mat[y][x] -= lamuda * phi[y]
        w += delta_w_mat

        phi = phi * tau + (1 - tau) * (Y_firing_time - np.mean(Y_firing_time))
        #print 'w:', w[0], w[1]
        if log:
            print 'w:', [list(w[0]), list(w[1])]
            print 'Error:', X_ft_desired - X_ft_actual, 'Desired:', np.round(X_ft_desired,5), 'Actual:',np.round(X_ft_actual,5)
            print 'Y_firing_time:', Y_firing_time
            print 'delta_w_mat:', delta_w_mat[0], delta_w_mat[1]
            if np.max(X_ft_actual) > 18:
                print '##########################'
                #temp = raw_input()
        
        #break

        # Predict
        if predict:
            #if X_ft_actual[1] - X_ft_actual[0]> 0.6:
            #    continue

            pred_X.append(X_ft_actual[1] - X_ft_actual[0] + 0.01 * np.random.randn())
            pred_Y.append(X_ft_actual[2] - X_ft_actual[0] + 0.01 * np.random.randn())
            
            actual_X.append(X_ft_desired[1] - X_ft_desired[0] + 0.01 * np.random.randn())
            actual_Y.append(X_ft_desired[2] - X_ft_desired[0] + 0.01 * np.random.randn())

    if predict:
        #print 'Congratulations! Training completion!'
        plt.scatter(pred_X, pred_Y, alpha=0.3, color='r', label='Prediction')
        plt.scatter(actual_X, actual_Y, alpha=0.3, label='Actual')
        plt.legend()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.savefig('centering.png')
        plt.show()
    


if __name__ == '__main__':
    pca()

