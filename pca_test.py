import numpy as np
import matplotlib.pyplot as plt

def get_input():
    data = []
    with open('2D_gaussian_dist.txt') as fin:
    #with open('uniform.txt') as fin:
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
    I = 25.0 * np.dot(I_i, w) + I_0     # XXX The amplitude of the input current
    delta_thetas = theta_der(thetas, alpha, I)
    new_thetas = thetas + delta_thetas * timestep   # XXX

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
    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel('Values of $w$s', fontsize=14)
    plt.savefig('weight_evolution.png')
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
    ISI = 5   # ms
    C = 1000
    eta = 0.0005
    lamuda = 1.0 * eta * 0
    num_iter_learning = 20000


    # Topology
    w = np.random.uniform(0.5, 1.5, (2, 3)) # Initialize the weight matrix

    #w = np.array([[-0.647, 2.41, 3.294], [3.94, 1.07, 0.053]]) # paper
    #w = np.array([[-0.6, 2, 3], [3, 1.0, 0.0]]) # paper w + noise

    #w = np.array([[-0.63802349174938167, 2.4523969927136875, 3.3442600199684529], [3.9371855848428123, 1.1621748973987442, 0.0099998921875246762]])  # start from paper w, Transient current dirac fucntion = 25.0

    #w = np.array( [[-0.34014524085091696, 2.4984120043146971, 3.3826309273837993], [3.9604700734489273, 1.0451106102281138, -0.27651269887877949]]  )  # start from paper w + noise

    #w = np.array( [[1.3393724648195791, 1.1078659141549303, 1.2910507614431805], [2.4882029488392532, 2.5584422674500242, 2.5292646270325059]])
    #w = np.array([[3.7500113682604646, -0.027695515392296293, -0.82365578626699421], [2.393367577016023, 2.6830377946894912, 2.7787575984740487]])
    #w = np.array([[3.7569613329593552, 0.057528364648807562, -0.69249128753537026], [1.8533979967612757, 2.8173749415724663, 3.0945438586595642]])


    # Local minimum
    w = np.array([[1.0495854584232804, 1.0278752838593483, 1.2963497108427307], [2.5814114885299038, 2.5942425259423234, 2.5684321081005832]])
    w = np.array([[3.7675472479198109, 0.02960945788428803, -0.85644914652842097], [2.3504787468350066, 2.677358590481218, 2.8518639597855548]])    #stable
    w = np.array([[3.7654484344702723, -0.061728412044310496, -0.75222654640069864], [2.3534712976610068, 2.7112629477736196, 2.7732706677102237]])

    w = np.array([[3.8760013994978411, -0.079921516855307295, -0.81076340314914874], [2.2839364371198831, 2.6603809828672071, 2.9383202738616667]])

    w = np.array([[4.1135971432729743, -0.12442774816536092, -0.5947342610954659], [1.4644912255890914, 2.8283849942492241, 3.0526581921437086]])

    w = np.array([[4.0845921487407502, 0.80071088244585054, -0.12623240092991136], [-0.49143363947168128, 2.6967656403979992, 3.2797358555207903]])

    w = np.array([[4.0823062445548794, 0.92894373957735532, -0.23957641578299593], [-0.50585379597958069, 2.6067705591068342, 3.3720953488872292]])

    w = np.array([[4.0789826198195707, 0.96472554539292421, -0.26692934134239449], [-0.48440624169023211, 2.5901997927923603, 3.3860248359864666]])

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

    # See the evolution of total MSE
    total_error = [[], [], []]
    batch_error = []
    for learn in range(num_iter_learning + 1):
        
        if learn % len(input_data) == 0:
            input_data = np.random.permutation(input_data)
            if learn > 0:
                for i in range(3):
                    temp = np.mean(batch_error, axis=1)
                    total_error[i].append(temp[i])
                batch_error = []
                #print "temp:", temp
                #print 'total_error:', total_error
                

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
            if learn % 2000 == 0 and learn > 0:
                plot_w_update(w_history)
                for i in range(3):
                    plt.plot(total_error[i], ls='--', label='MSE of $t_' + str(i) + '$')
                plt.plot(np.sum(total_error, axis=0), label='Total MSE')
                plt.xlabel('Number of Batches', fontsize=14)
                plt.ylabel('MSE', fontsize=14)
                #plt.plot(total_error[0] + total_error[1] + total_error[2], label='Total error')
                #plt.ylim(0, 1.0)
                plt.legend()
                plt.savefig('mse.png')
                plt.show()

        X_ft = input_data[learn % len(input_data)]

        # XXX Change the w
        '''
        if X_ft[1] < 3 and X_ft[2] < 3:
            eta = 0.06
        else:
            eta = 0.001
        '''

        if log:
            print 'input:', X_ft,
        theta_Y = np.array([ - np.arccos((1 + alpha_Y * I_0) / (1 - alpha_Y * I_0)) for k in range(2)])
        theta_Xprime = np.array([ - np.arccos((1 + alpha_X * I_0) / (1 - alpha_X * I_0)) for k in range(3)])

        X_ft_desired = X_ft + ISI
        X_ft_actual = [timestep * (num_iter_trial-1) for k in range(3)]
        cos_theta_minus = [[None for k in range(2)] for i in range(3)]
        cos_theta_plus = [[None for k in range(2)] for i in range(3)]

        #print 'Current_weight:', w
        Y_firing_time = np.array([num_iter_trial * timestep, num_iter_trial * timestep])

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

        temp = (X_ft_desired - X_ft_actual) * (X_ft_desired - X_ft_actual)
        batch_error.append(temp)
        
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
        print 'Total error:', total_error
        plt.scatter(pred_X, pred_Y, alpha=0.3, color='r', label='Reconstruction')
        plt.scatter(actual_X, actual_Y, alpha=0.3, label='Actual')
        plt.legend(loc='upper left')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        #plt.savefig('centering.png')
        plt.show()



if __name__ == '__main__':
    pca()

