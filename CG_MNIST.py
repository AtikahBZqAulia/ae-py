def CG_MNIST(VV, Dim, X2):
    import numpy as np
    L1 = Dim[0]
    L2 = Dim[1]
    L3 = Dim[2]
    L4 = Dim[3]
    L5 = Dim[4]
    L6 = Dim[5]
    L7 = Dim[6]
    L8 = Dim[7]
    L9 = Dim[8]
    N = X2.shape[0]

    # Do decomversion
    W1 = VV[0][0:(L1[0]+1)*L2[0]].reshape(L1[0]+1,L2[0])
    X3 = (L1[0]+1)*L2[0]
    W2 = VV[0][X3:X3+(L2[0]+1)*L3[0]].reshape(L2[0]+1, L3[0])
    X3 = X3+(L2[0]+1)*L3[0]
    W3 = VV[0][X3:X3+(L3[0]+1)*L4[0]].reshape(L3[0]+1, L4[0])
    X3 = X3+(L3[0]+1)*L4[0]
    W4 = VV[0][X3:X3+(L4[0]+1)*L5[0]].reshape(L4[0]+1, L5[0])
    X3 = X3+(L4[0]+1)*L5[0]
    W5 = VV[0][X3:X3+(L5[0]+1)*L6[0]].reshape(L5[0]+1, L6[0])
    X3 = X3+(L5[0]+1)*L6[0]
    W6 = VV[0][X3:X3+(L6[0]+1)*L7[0]].reshape(L6[0]+1, L7[0])
    X3 = X3+(L6[0]+1)*L7[0]
    W7 = VV[0][X3:X3+(L7[0]+1)*L8[0]].reshape(L7[0]+1, L8[0])
    X3 = X3+(L7[0]+1)*L8[0]
    W8 = VV[0][X3:X3+(L8[0]+1)*L9[0]].reshape(L8[0]+1, L9[0])

    X2 = np.append(X2, np.ones((N, 1)), axis=1)
    W1_PROBS = 1.0/(1 + np.exp(np.matmul(-X2, W1)))
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
    X2_OUT = 1.0/(1 + np.exp(np.matmul(-W7_PROBS, W8)))

    f = -1/N*np.sum(np.sum(X2[:,:-1]*np.log(X2_OUT)+ (1-X2[:, :-1])*np.log(1-X2_OUT)))
    IO = 1/N*(X2_OUT-X2[:,:-1])
    Ix8 = IO
    DW8 = np.matmul(W7_PROBS.T,Ix8)

    Ix7 = (np.matmul(Ix8,W8.T))*W7_PROBS*(1-W7_PROBS)
    Ix7 = Ix7[:, :-1]
    DW7 = np.matmul(W6_PROBS.T,Ix7)

    Ix6 = (np.matmul(Ix7, W7.T))*W6_PROBS*(1-W6_PROBS)
    Ix6 = Ix6[:, :-1]
    DW6 = np.matmul(W5_PROBS.T,Ix6)

    Ix5 = (np.matmul(Ix6, W6.T))*W5_PROBS*(1-W5_PROBS)
    Ix5 = Ix5[:, :-1]
    DW5 = np.matmul(W4_PROBS.T,Ix5)

    Ix4 = np.matmul(Ix5, W5.T)
    Ix4 = Ix4[:, :-1]
    DW4 = np.matmul(W3_PROBS.T,Ix4)

    Ix3 = np.matmul(Ix4, W4.T)*W3_PROBS*(1-W3_PROBS)
    Ix3 = Ix3[:, :-1]
    DW3 = np.matmul(W2_PROBS.T,Ix3)

    Ix2 = np.matmul(Ix3, W3.T)*W2_PROBS*(1-W2_PROBS)
    Ix2 = Ix2[:, :-1]
    DW2 = np.matmul(W1_PROBS.T, Ix2)

    Ix1 = np.matmul(Ix2, W2.T)*W1_PROBS*(1-W1_PROBS)
    Ix1 = Ix1[:, :-1]
    DW1 = np.matmul(X2.T, Ix1)

    # df = lst.extend([DW1[:].T, DW2[:].T, DW3[:].T, DW4[:].T, DW5[:].T, DW6[:].T, DW7[:].T, DW8[:].T]).T
    df = np.concatenate((DW1.reshape(1,-1), DW2.reshape(1,-1), DW3.reshape(1,-1),
                DW4.reshape(1,-1), DW5.reshape(1,-1), DW6.reshape(1, -1), DW7.reshape(1, -1),
                DW8.reshape(1, -1)), axis=1)
    return f, df