from os import stat
import numpy as np
from scipy.stats import norm
import math

def getStationaryProb(trans_mat):
    trans_mat = trans_mat.T
    A = trans_mat - np.identity(trans_mat.shape[0])
    for i in range(state_num):
        A[state_num-1][i] = 1
    #print(A)
    B = np.zeros([state_num,1])
    B[state_num-1][0] = 1
    #print(B)

    X = np.linalg.inv(A).dot(B)
    return X

#----------------Viterbi Function---------------------
def viterbi(y, transition_m, emission_m, initial_prob):
    state_count = transition_m.shape[0]
    T = len(y)

    T1 = np.zeros((T, state_count))
    T2 = np.zeros((T - 1, state_count))

    for state in range(state_count):
        T1[0][state] = np.log(initial_prob[state] * emission_m[0][state])    
    
    for t in range(1,T):
        for state in range(state_count):    
            probability = []
            for prevstate in range(state_count):
                probability.append(T1[t-1][prevstate] + np.log(transition_m[prevstate][state]*emission_m[t][state]))
            probability = np.array(probability)        
            T1[t][state] = np.max(probability)
            T2[t-1][state] = np.argmax(probability)

    #print(T1)
    last_state = np.argmax(T1[T - 1, :])
    #print(last_state)

    X = np.zeros(T) 
    X[0] = last_state
    
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        X[backtrack_index] = T2[i, int(last_state)]
        last_state = T2[i, int(last_state)]
        backtrack_index += 1
    
    X = np.flip(X, axis=0)

    result = []
    for x in X:
        if x == 0:
            result.append("\"El Nino\"")
        else:
            result.append("\"La Nina\"")
 
    return result

#------------------------Baum-Welch Implementation-----------------------------
def forward(init_prob, transition_m, emission_m, state_count, T):
    f = np.zeros((emission_m.shape[0],emission_m.shape[1]))
    for state in range(state_count):
        f[0][state] = init_prob[state] * emission_m[0][state]
    f[0] = f[0]/f[0].sum()
    
    for t in range(1,T):
        norm_sum = 0
        for state in range(state_count):
            prob = []
            for prevstate in range(state_count):
                prob.append(f[t-1][prevstate] * transition_m[prevstate][state] * emission_m[t][state] )

            f[t][state] = sum(prob)
            norm_sum = norm_sum + sum(prob)
        f[t] = f[t]/f[t].sum()    
        # for state in range(state_count):
        #     f[t][state] = f[t][state]/norm_sum 
    
    #print(f)
    return f

def backward(transition_m, emission_m, state_count, T):
    b = np.zeros((T,state_count))
    b[T - 1] = np.ones((state_count))
    b[T-1] = b[T-1]/b[T-1].sum()
    #print(b)    
    
    for t in range(T - 2, -1, -1):
        for state in range(state_count):
            prob = []
            for prevstate in range(state_count):
                prob.append( b[t+1][prevstate] * transition_m[state][prevstate] * emission_m[t+1][prevstate] )
            b[t][state] = sum(prob)
        b[t] = b[t]/b[t].sum()

    #print(b) 
    return b

def get_pi_star_star(f, b, transition_m, emission_m, fsink, state_count, T):
    pi_star_star = np.zeros((T-1,state_count*state_count))
    
    for t in range(1,T):
        for prevstate in range(state_count):
            for state in range(state_count):
                indx = prevstate*state_count + state
                pi_star_star[t-1][indx] = (f[t-1][prevstate] * transition_m[prevstate][state] * emission_m[t][state] * b[t][state])/fsink
    
    for i in range(T-1):
        pi_star_star[i] = pi_star_star[i]/sum(pi_star_star[i])
    
    return pi_star_star


def baum_welch(y, transition_m, emission_m, initial_prob, iterations):
    state_count = transition_m.shape[0]
    T = len(y)
    
    #  IT HAS TO BE CHANGED TO ITERATIONS
    prev_means = np.zeros((1,state_num))
    prev_variances = np.zeros((1,state_num))
    for iter in range(iterations):
        f = forward(initial_prob, transition_m, emission_m, state_count, T)
        b = backward( transition_m, emission_m, state_count, T)
        


        fsink = f[T-1].sum()
        pi_star = f*b/fsink
        for t in range(T):
            pi_star[t] = pi_star[t]/sum(pi_star[t])
        
        pi_star_star = get_pi_star_star(f, b, transition_m, emission_m, fsink,state_count, T)

        transitions = pi_star_star.sum(axis=0)
        
        for prevstate in range(state_count):
            for state in range(state_count):
                transition_m[prevstate][state] = transitions[prevstate*state_count+state]
        
        for state in range(state_count):
            transition_m[state] = transition_m[state]/sum(transition_m[state])
        
        initial_prob = getStationaryProb(transition_m)
        
        observations = np.array(y).reshape((T,1))
        l_means = (pi_star*observations).sum(axis=0)/pi_star.sum(axis=0)
        
        l_variances = (pi_star * np.square(observations-l_means)).sum(axis=0)/pi_star.sum(axis=0)
        #l_variances = np.sqrt(l_variances)
        # print(l_variances)
        # print(l_means)
        # print()
        
        emission_m = getEmissionMatrix(y, l_means, l_variances)


        # learned_vars =[]
        # sums = np.zeros(state_count)
        # sumP = np.zeros(state_count)
            
        # for state in range(state_count):
        #     for t in range(T):
        #         sums[state]= sums[state] + (pi_star[t][state] * math.pow(observations[t]-l_means[state] ,2))
        #         sumP[state]= sumP[state] + pi_star[t][state]
        #     learned_vars.append(np.sqrt(sums[state]/sumP[state]))
        
        #print(pi_star.sum(axis=0))
        # print("learned_vars")
        # print(learned_vars)


        if abs((l_variances-prev_variances).sum()) + abs((l_means-prev_means).sum()) < 0.00001:
            prev_means = l_means
            prev_variances= l_variances
            # print("broken"+ str(iter))
            break
        
        prev_means = l_means
        prev_variances= l_variances
    
    # print("final print")
    # print(transition_m)
    # print(prev_variances)
    # print(prev_means)

    outputfile2 = open("parameters_learned.txt.txt", 'w')
    outputfile2.write(str(state_count)+"\n")
    for i in range(state_count):
        for j in range(state_count):
            outputfile2.write(str(transition_m[i][j])+"  ")
        outputfile2.write("\n")
    for i in range(state_count):
        outputfile2.write(str(prev_means[i])+"  ")
    outputfile2.write("\n")
    for i in range(state_count):
        outputfile2.write(str(prev_variances[i])+"  ")
    outputfile2.write("\n")
    for i in range(state_count):
        outputfile2.write(str(initial_prob[i][0])+"  ")
    
    result = viterbi(y, transition_m, emission_m, initial_prob)

    outputfile3  = open("states_Viterbi_after_learning.txt",'w')
    for r in result:
        outputfile3.write(r+"\n")

    #------------------check with output------------
    # output1  = open("Output/states_Viterbi_after_learning.txt",'r')
    # for r in result:
    #     line = output1.readline().rstrip("\n")
    #     if r!=line:
    #         print(r)
    # print("checks finished")
    
        
def getEmissionMatrix(y,m,v):
    Emission_matrix=[]
    for o in y:
        e_m = []
        for state in range(state_num):
            e_m.append(norm.pdf(o, m[state], math.sqrt(v[state])))
        Emission_matrix.append(e_m)

    Emission_matrix = np.array(Emission_matrix)

    return Emission_matrix






#-------------Input parameters---------------------------------------
f = open("Input/parameters.txt.txt", 'r')
state_num = int(f.readline())

mat = []

for i in range(state_num):
    line = f.readline()
    a = []
    for n in line.split()[0:]:
        a.append(float(n))
    mat.append(a)

transition_matrix = np.array(mat)

stationary_prob = getStationaryProb(transition_matrix)
#print(stationary_prob)

means = []
for n in f.readline().split():
    means.append(float(n))

variances = []
for n in f.readline().split():
    variances.append(float(n))

f2 = open("Input/data.txt", 'r')
lines = f2.readlines()
O = []
for line in lines:
    O.append(float(line))

# Emission_matrix=[]
# for o in O:
#     e_m = []
#     for state in range(state_num):
#         e_m.append(norm.pdf(o, means[state], math.sqrt(variances[state])))
#     Emission_matrix.append(e_m)

# Emission_matrix = np.array(Emission_matrix)

Emission_matrix = getEmissionMatrix(O,means,variances)
result  = viterbi(O, transition_matrix, Emission_matrix, stationary_prob)

outputfile1  = open("states_Viterbi_wo_learning.txt",'w')
for r in result:
    outputfile1.write(r+"\n")

#------------------check with output------------
# output1  = open("Output/states_Viterbi_wo_learning.txt",'r')
# for r in result:
#     line = output1.readline().rstrip("\n")
#     if r!=line:
#         print(r)
# print("checks finished")



baum_welch(O, transition_matrix, Emission_matrix, stationary_prob, 10)
