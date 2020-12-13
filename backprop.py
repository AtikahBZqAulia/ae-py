def backprop(VISHID, VISBIASES, PENRECBIASES, PENRECBIASES2, HIDRECBIASES, HIDPEN, HIDPEN2, HIDGENBIASES, HIDGENBIASES2, HIDTOP, TOPRECBIASES, TOPGENBIASES):
# def backprop():
    import numpy as np
    import scipy.io as sio
    from makebatches import makebatches
    from mnistdisp import mnistdisp
    from minimize import minimize
    from CG_MNIST import CG_MNIST

    MAX_EPOCH = 200
    print('Fine-tuning deep autoencoder by minimizing cross entropy error.')
    print('60 batches of 1000 cases each.')

    BATCH_DATA_ = sio.loadmat("batchdata_py.mat", verify_compressed_data_integrity=False)
    BATCHDATA = BATCH_DATA_['batchdata']
    TEST_BATCH_DATA_ = sio.loadmat("testbatchdata_py.mat", verify_compressed_data_integrity=False)
    TESTBATCHDATA = TEST_BATCH_DATA_['testbatchdata']

    VISHID_ = sio.loadmat("vishid_mnistvh.mat", verify_compressed_data_integrity=False)
    VISHID = VISHID_['vishid_']
    
    W1 = np.append(VISHID, HIDRECBIASES.reshape(1, -1), axis = 0)
    W2 = np.append(HIDPEN, PENRECBIASES.reshape(1, -1), axis = 0)
    W3 = np.append(HIDPEN2, PENRECBIASES2.reshape(1, -1), axis = 0)
    W4 = np.append(HIDTOP, TOPRECBIASES.reshape(1, -1), axis = 0)
    W5 = np.append(HIDTOP.T, TOPGENBIASES.reshape(1, -1), axis = 0)
    W6 = np.append(HIDPEN2.T, HIDGENBIASES2.reshape(1, -1), axis = 0)
    W7 = np.append(HIDPEN.T, HIDGENBIASES.reshape(1, -1), axis = 0)
    W8 = np.append(VISHID.T, VISBIASES.reshape(1, -1), axis = 0)

    L1 = W1.shape[0]-1
    L2 = W2.shape[0]-1
    L3 = W3.shape[0]-1
    L4 = W4.shape[0]-1
    L5 = W5.shape[0]-1
    L6 = W6.shape[0]-1
    L7 = W7.shape[0]-1
    L8 = W8.shape[0]-1
    L9 = L1

    TEST_ERR=[]
    TRAIN_ERR=[]

    for epoch in range(1, MAX_EPOCH):
        ERR = 0
        NUM_CASES, NUM_DIMS, NUM_BATCHES = BATCHDATA.shape
        N = NUM_CASES
        for batch in range(1, NUM_BATCHES):
            data = BATCHDATA[:,:,batch]
            data = np.append(data, np.ones((N, 1)), axis=1)
            
            W1_PROBS = 1.0/(1 + np.exp(np.matmul(-data, W1)))
            W1_PROBS = np.append(W1_PROBS, np.ones((N, 1)), axis=1)
            W2_PROBS = 1.0/(1 + np.exp(np.matmul(-W1_PROBS, W2)))
            W2_PROBS = np.append(W2_PROBS, np.ones((N, 1)), axis=1)
            W3_PROBS = 1.0/(1 + np.exp(np.matmul(-W2_PROBS,W3)))
            W3_PROBS = np.append(W3_PROBS, np.ones((N, 1)), axis=1)
            W4_PROBS = np.matmul(W3_PROBS, W4)
            W4_PROBS = np.append(W4_PROBS, np.ones((N, 1)), axis=1)
            W5_PROBS = 1.0/(1 + np.exp(np.matmul(-W4_PROBS,W5)))
            W5_PROBS = np.append(W5_PROBS, np.ones((N, 1)), axis=1)
            W6_PROBS = 1.0/(1 + np.exp(np.matmul(-W5_PROBS, W6)))
            W6_PROBS = np.append(W6_PROBS, np.ones((N, 1)), axis=1)
            W7_PROBS = 1.0/(1 + np.exp(np.matmul(-W6_PROBS, W7)))
            W7_PROBS = np.append(W7_PROBS, np.ones((N, 1)), axis=1)
            DATAOUT = 1.0/(1 + np.exp(np.matmul(-W7_PROBS, W8)))
            ERR += 1/N*np.sum(np.sum(np.square(data[:,:-1]-DATAOUT), axis=0), axis=0)
        TRAIN_ERR = ERR/NUM_BATCHES

        print('Displaying in figure 1: Top row - real data, Bottom row -- reconstructions')
        OUTPUT = np.array([])
        for ii in range(15):
            A = np.append(data[ii, :-1].T, DATAOUT[ii, :].T)
            A = A.reshape(784, 2)
            OUTPUT = np.append(OUTPUT, A)
        # mnistdisp(OUTPUT)
        # plt.show()

        TESTNUMCASES, TESTNUMDIMS, TESTNUMBATCHES = TESTBATCHDATA.shape
        N = TESTNUMCASES
        ERR = 0
        for batch in range(1, TESTNUMBATCHES):
            data = TESTBATCHDATA[:,:,batch]
            data = np.append(data, np.ones((N, 1)), axis=1)
                
            W1_PROBS = 1.0/(1 + np.exp(np.matmul(-data, W1)))
            W1_PROBS = np.append(W1_PROBS, np.ones((N, 1)), axis=1)
            W2_PROBS = 1.0/(1 + np.exp(np.matmul(-W1_PROBS, W2)))
            W2_PROBS = np.append(W2_PROBS, np.ones((N, 1)), axis=1)
            W3_PROBS = 1.0/(1 + np.exp(np.matmul(-W2_PROBS,W3)))
            W3_PROBS = np.append(W3_PROBS, np.ones((N, 1)), axis=1)
            W4_PROBS = np.matmul(W3_PROBS, W4)
            W4_PROBS = np.append(W4_PROBS, np.ones((N, 1)), axis=1)
            W5_PROBS = 1.0/(1 + np.exp(np.matmul(-W4_PROBS,W5)))
            W5_PROBS = np.append(W5_PROBS, np.ones((N, 1)), axis=1)
            W6_PROBS = 1.0/(1 + np.exp(np.matmul(-W5_PROBS, W6)))
            W6_PROBS = np.append(W6_PROBS, np.ones((N, 1)), axis=1)
            W7_PROBS = 1.0/(1 + np.exp(np.matmul(-W6_PROBS, W7)))
            W7_PROBS = np.append(W7_PROBS, np.ones((N, 1)), axis=1)
            DATAOUT = 1.0/(1 + np.exp(np.matmul(-W7_PROBS, W8)))
            ERR += 1/N*np.sum(np.sum(np.square(data[:,:-1]-DATAOUT), axis=0), axis=0)
        TEST_ERR = ERR/NUM_BATCHES
        print('Before epoch {} Train squared error: {} Test squared error: {}'.format(epoch,TRAIN_ERR,TEST_ERR))

        TT = 0
        for batch in range(int(NUM_BATCHES/10)):
            print('epoch {} batch {}'.format(epoch,batch))
            TT+=1
            data=np.empty((100,784), int)
            for kk in range(10):
                data = np.append(data, BATCHDATA[:,:,((TT-1)*10+kk)], axis=0)
                
            MAX_ITER = 3
            VV = np.concatenate((W1.reshape(1,-1), W2.reshape(1,-1), W3.reshape(1,-1),
                    W4.reshape(1,-1), W5.reshape(1,-1), W6.reshape(1, -1), W7.reshape(1, -1),
                    W8.reshape(1, -1)), axis=1)
            DIM = np.array([L1, L2, L3, L4, L5, L6, L7, L8, L9]).reshape(1, -1).T

            f, df = CG_MNIST(VV, DIM, data)
            X, fX, i = minimize(VV,f, df, MAX_ITER, DIM, data, 1.0, True) 

            W1 = X[0][0:(L1+1)*L2].reshape(L1+1,L2)
            X3 = (L1+1)*L2
            W2 = X[0][X3:X3+(L2+1)*L3].reshape(L2+1, L3)
            X3 = X3+(L2+1)*L3
            W3 = X[0][X3:X3+(L3+1)*L4].reshape(L3+1, L4)
            X3 = X3+(L3+1)*L4
            W4 = X[0][X3:X3+(L4+1)*L5].reshape(L4+1, L5)
            X3 = X3+(L4+1)*L5
            W5 = X[0][X3:X3+(L5+1)*L6].reshape(L5+1, L6)
            X3 = X3+(L5+1)*L6
            W6 = X[0][X3:X3+(L6+1)*L7].reshape(L6+1, L7)
            X3 = X3+(L6+1)*L7
            W7 = X[0][X3:X3+(L7+1)*L8].reshape(L7+1, L8)
            X3 = X3+(L7+1)*L8
            W8 = X[0][X3:X3+(L8+1)*L9].reshape(L8+1, L9)
    return ERR


# if __name__ == "__main__":
#     backprop()