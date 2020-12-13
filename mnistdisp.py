def mnistdisp(digits):
    import numpy as np
    import matplotlib.pyplot as plt

    COL = 28
    ROW = 28
    digits = digits.reshape(784, 30)
    dd, N = digits.shape
    img2 = np.zeros((2*28, int(np.ceil(N/2)*28)))

    for nn in range(N):
        ii = (nn+1)% 2
        if (ii==0):
            ii=2
        jj = int(np.ceil((nn+1)/2))

        img1 = digits[:,nn].reshape(ROW, COL)
        index1_a =int((ii-1)*ROW) 
        index1_b = int(ii*ROW) 
        index2_a =int((jj-1)*COL) 
        index2_b = int(jj*COL) 
        img2[index1_a:index1_b,index2_a:index2_b]= img1.T
    plt.imshow(img2, cmap='gray')
    plt.show()
    err = 0

    return err