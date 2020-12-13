def rbm(BATCH_DATA, RESTART, NUM_HID, NUM_DIMS, MAX_EPOCH, RANDOMIN, VISHID):
    import scipy.io as sio
    import numpy as np

    EPSILON_W = 0.1
    EPSILON_VB = 0.1
    EPSILON_HB = 0.1
    WEIGHT_COST = 0.0002
    INITIAL_MOMENTUM = 0.5
    FINAL_MOMENTUM = 0.9

    NUM_CASES, NUM_DIMS, NUM_BATCHES = BATCH_DATA.shape

    if RESTART==1:
        RESTART = 0
        EPOCH = 0

        # VISHID = 0.1 * np.random.randn(NUM_DIMS, NUM_HID)
        if NUM_HID==500:
            VISHID_ = sio.loadmat("vis_pre2.mat", verify_compressed_data_integrity=False)
            VISHID = VISHID_['vishid_']
            RANDOMIN_ = sio.loadmat("randomin_pre2.mat", verify_compressed_data_integrity=False)
            RANDOMIN = RANDOMIN_['randomin']
        elif NUM_HID==250:
            VISHID_ = sio.loadmat("vis_pre3.mat", verify_compressed_data_integrity=False)
            VISHID = VISHID_['vishid_']
            RANDOMIN_ = sio.loadmat("randomin_pre3.mat", verify_compressed_data_integrity=False)
            RANDOMIN = RANDOMIN_['randomin']

        HIDBIASES = np.zeros(NUM_HID)
        VISBIASES = np.zeros(NUM_DIMS)
        POSHIDPROBS = np.zeros((NUM_CASES, NUM_HID))
        NEGHIDPROBS = np.zeros((NUM_CASES, NUM_HID))
        POSPRODS = np.zeros((NUM_DIMS, NUM_HID))
        NEGPRODS = np.zeros((NUM_DIMS, NUM_HID))
        VISHIDINC = np.zeros((NUM_DIMS, NUM_HID))
        HIDBIASINC = np.zeros(NUM_HID)
        VISBIASINC = np.zeros(NUM_DIMS)
        BATCHPOSHIDPROBS = np.zeros((NUM_CASES, NUM_HID, NUM_BATCHES))

    for epoch in range(EPOCH, MAX_EPOCH):
        print('epoch {}'.format(epoch))
        ERR_SUM = 0
        for batch in range(NUM_BATCHES):
            print('epoch {} batch {}'.format(epoch,batch))
            
            data = BATCH_DATA[:,:,batch]

            POSHIDPROBS = 1.0/(1 + (np.exp(np.matmul(-data,VISHID)- np.tile(HIDBIASES, (NUM_CASES, 1)))))
            BATCHPOSHIDPROBS[:,:,batch] = POSHIDPROBS
            POSPRODS = np.matmul(data.T,POSHIDPROBS)
            POSHIDACT = np.sum(POSHIDPROBS, axis=0)
            POSVISACT = np.sum(data, axis=0)

            POSHIDSTATES = POSHIDPROBS > RANDOMIN
            POSHIDSTATES = np.where(POSHIDSTATES, 1, 0)

            NEGDATA = 1.0/(1 + np.exp(np.matmul(-POSHIDSTATES, VISHID.T) - np.tile(VISBIASES, (NUM_CASES, 1))))
            NEGHIDPROBS = 1.0/(1 + np.exp(np.matmul(-NEGDATA, VISHID) - np.tile(HIDBIASES, (NUM_CASES,1))))
            NEGPRODS = np.matmul(NEGDATA.T, NEGHIDPROBS)
            NEGHIDACT = np.sum(NEGHIDPROBS, axis=0)
            NEGVISACT = np.sum(NEGDATA, axis=0)

            ERR = np.sum(np.sum(np.square(data-NEGDATA), axis=0), axis=0)
            ERR_SUM += ERR

            if epoch>5:
                MOMENTUM = FINAL_MOMENTUM
            else:
                MOMENTUM = INITIAL_MOMENTUM

            VISHIDINC = MOMENTUM* VISHIDINC + EPSILON_W  * ((POSPRODS-NEGPRODS)/NUM_CASES - WEIGHT_COST*VISHID)
            VISBIASINC = MOMENTUM* VISBIASINC + (EPSILON_VB/NUM_CASES) *(POSVISACT-NEGVISACT)
            HIDBIASINC = MOMENTUM * HIDBIASINC + (EPSILON_HB/NUM_CASES)* (POSHIDACT-NEGHIDACT)
            VISHID += VISHIDINC
            VISBIASES += VISBIASINC
            HIDBIASES += HIDBIASINC
        print('epoch {} error {}'.format(epoch, ERR_SUM))
    return BATCHPOSHIDPROBS, VISHID, HIDBIASES, VISBIASES

