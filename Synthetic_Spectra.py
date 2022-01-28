import numpy as np
import random
from matplotlib import pyplot as plt 
from bsbl import * #import bsbl
import time

Lambda = 2000 # number of pixels
J = 1000 # number of basis functions

#Create Dictionary Matrix
D = np.zeros((J,Lambda))
D_sparsity = 0.99
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        if np.random.uniform() > D_sparsity:
            D[i,j] = np.random.uniform()

for j in range(D.shape[1]):
    D[:,j]=D[:,j]/np.sum(D[:,j])
D = D.transpose()    
        
#Create Blocks
BlockMin = 5
BlockMax = 10
Block_Lengths  = []
Block_Indices = []
BlockSum = 0

while BlockSum < J-BlockMax:
    BlockLen = np.random.randint(BlockMin,BlockMax)
    Block_Lengths.append(BlockLen)
    Block_Indices.append(BlockSum)
    BlockSum += BlockLen
Block_Lengths.append(J-BlockSum)
Block_Indices.append(BlockSum)
Active_Blocks = np.random.randint(0,len(Block_Lengths),np.random.randint(3,10))

x_true = np.zeros(J)
for a in Active_Blocks:
    for i in range(Block_Lengths[a]):
        if np.random.uniform()>0.5:
            x_true[i+Block_Indices[a]] = np.random.uniform()

y_true = np.matmul(D,x_true)
print(np.linalg.norm(y_true,0),len(Active_Blocks))

NOISE = np.zeros(Lambda)
SNR = 0.1
for l in range(len(NOISE)):
    NOISE[l] = np.random.normal(0,SNR)
y_data = y_true + NOISE


#======================================================================
#            Algorithm Comparison
#======================================================================
ind = (np.abs(x_true)>0).nonzero()[0]

# 1. Benchmark
t1 = time.time()
supt = ind
x_ls = np.dot(lp.pinv(D[:,supt]), y_data)
x0 = np.zeros(J)
x0[supt] = x_ls
t2 = time.time()
mse_bench = (lp.norm(x_true - x0)/lp.norm(x_true))**2
time_bench = t2 - t1


# 2. BSBL-BO
BInd = np.zeros(len(Block_Lengths))
for i in range(len(BInd)):
    BInd[i] = Block_Indices[i]
BInd = BInd.astype(int)
print("start")
t1 = time.time()
clf = bo(learn_lambda=1, learn_type=1, lambda_init=1e-3, #bsbl.bo
              epsilon=1e-5, max_iters=100, verbose=1)
x1 = clf.fit_transform(D, y_true, BInd)
t2 = time.time()
print("end")
mse_bo = (lp.norm(x_true - x1)/lp.norm(x_true))**2
time_bo = t2 - t1
# visualize
plt.figure()
plt.plot(x_true, linewidth=4)
plt.plot(x0, 'g-', linewidth=0.5)
plt.plot(x1, 'r-', linewidth=2)
#plt.plot(x2, 'y-', linewidth=2)
plt.xlabel('Samples')
plt.legend(('Original',
            'MSE (LS) = ' + str(mse_bench),
            'MSE (BO) = ' + str(mse_bo)),
            loc='best')
print("Bench",mse_bench,time_bench)
print("BSBL-BO",mse_bo,time_bo)

