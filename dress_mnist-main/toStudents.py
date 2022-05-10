from sklearn.neighbors import KNeighborsClassifier #머신러닝 분석할때 쓰이는 라이브러
from sklearn.naive_bayes import GaussianNB #나이브베이즈
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
import cv2

trainNum = 60000
testNum = 1000

def init_data():
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')    
    return x_train, y_train, x_test, y_test

def data_ready(x_train, y_train, x_test, y_test):
    x = np.zeros((trainNum,28,28))
    y = np.zeros(trainNum)
    xx = np.zeros((testNum,28,28))
    yy = np.zeros(testNum)
    for ii in range(10):
        x[ii*(trainNum//10):(ii+1)*(trainNum//10),::] = x_train[y_train==ii,::][0:(trainNum//10)]
        y[ii*(trainNum//10):(ii+1)*(trainNum//10)] = y_train[y_train==ii][0:(trainNum//10)]
        xx[ii*(testNum//10):(ii+1)*(testNum//10),::] = x_test[y_test==ii,::][0:(testNum//10)]
        yy[ii*(testNum//10):(ii+1)*(testNum//10)] = y_test[y_test==ii][0:(testNum//10)]
    x_train2 = x
    y_train2 = y
    x_test2 = xx
    y_test2 = yy

    return x_train2, y_train2, x_test2, y_test2

def data_ready_knn(trainSet, testSet):
    trs = len(trainSet) // 10
    tes = len(testSet) // 10

    trainSetf = np.zeros((len(trainSet), 28*28))
    testSetf = np.zeros((len(testSet), 28*28))
    for i in range(10):
        for j in range(trs):
            trainSetf[(i * trs) + j] = trainSet[j+(i*trs)].flatten()

    for i in range(10):
        for j in range(tes):
            testSetf[(i * tes) + j] = testSet[j+(i*tes)].flatten()
    return trainSetf/255.0, testSetf/255.0

# 준비된 사진 보는 함수
def print_data(data, row, col, data_num):
    num = row * col
    fig = plt.figure()
    for i in range(1,num+1):
        ax = fig.add_subplot(row,col,i)
        plt.imshow(data[i+data_num])
    plt.show()

def createTmpl(trainSet,label=[]):
    tmpl = np.zeros((28,28*len(label)))
    for i in range(len(label)):
        imsi = trainSet[(trainNum//10)*i : (trainNum//10)*i+(trainNum//10)]
        tmpl[:,i*28:(i+1)*28] = np.mean(imsi, axis = 0)
    return tmpl

def tmplMatch(tmpl, testSet, label=[]):
    result = np.zeros((testNum//10, len(label))) #100 x 10
    for i in range(10): # 10
        for j in range(testNum//10): # 1000
            imsiTest = np.tile(testSet[j+i*(testNum//10)], (1,10))

            error = np.abs(tmpl - imsiTest) #6000x28x28
            errorSum = [error[:,0:28].sum(), error[:,28:56].sum(),\
                        error[:,56:84].sum(), error[:,84:112].sum(),\
                        error[:,112:140].sum(), error[:,140:168].sum(),\
                        error[:,168:196].sum(), error[:,196:224].sum(),\
                        error[:,224:252].sum(), error[:,252:280].sum(),]
            result[j,i] = np.argmin(errorSum)
    return result

def tmplMatch2(tmpl, testSet, label=[]):
    result = np.zeros((len(te_cloth)//10, len(label))) #100 x 10

    for i in range(len(label)): # 10
        for j in range(len(testSet)//10): # 1000

            imsiTest = np.tile(testSet[j+i*len(testSet)//10], (1,6))

            error = np.abs(tmpl - imsiTest) #6000x28x28
            errorSum = [error[:,0:28].sum(), error[:,28:56].sum(),\
                        error[:,56:84].sum(), error[:,84:112].sum(),\
                        error[:,112:140].sum(), error[:,140:168].sum(),\
                        error[:,168:196].sum(), error[:,196:224].sum(),\
                        error[:,224:252].sum(), error[:,252:280].sum(),]
        
            result[j,i] = np.argmin(errorSum)

    return result



def knn(trainSet, testSet, k): 
    trS1,trS2 = trainSet.shape # 6000, 784
    teS1,teS2 = testSet.shape # 1000, 784

    trS3 = int(trS1/10) # 600
    teS3 = int(teS1/10) # 100

    label = np.tile(np.arange(0,10), (teS3,1)) 
    result = np.zeros((teS3,10))

    for i in range(teS1): 
        imsi = np.sum((trainSet - testSet[i,:])**2,axis=1) 
        no = np.argsort(imsi)[0:k] 
        hist, bins = np.histogram(no//trS3, np.arange(-0.5,10.5,1))
        result[i%teS3, i//teS3] = np.argmax(hist) 
    return result

def knn2(trainSet, testSet, tr_label, k): 
    trS1,trS2 = trainSet.shape # 6000, 784
    teS1,teS2 = testSet.shape # 1000, 784

    trS3 = int(trS1/10) # 600
    teS3 = int(teS1/10) # 100

    label = np.tile(tr_label, (teS3,1)) 
    result = np.zeros((teS3,len(tr_label)))

    for i in range(len(tr_label)):
        for j in range(teS1//10):
            imsi = np.sum((trainSet - testSet[i*(testNum//10)+j])**2,axis=1) 
            no = np.argsort(imsi)[0:k] 
            hist, bins = np.histogram(no//trS3, np.arange(-0.5,len(tr_label)+0.5,1))
            result[j , i] = tr_label[np.argmax(hist)]

    return result



def knn3(trainSet, testSet, k):

    # 3D
    cov_mat = np.cov(trainSet.T) #784 x 784 
    eigen_vec, eigen_val, v = np.linalg.svd(cov_mat) #u 고유벡터 #s 고유값 

    train_z = (eigen_vec.T[:3] @ trainSet.T).T #2 x 784 @ 784 x 6000
    test_z = (eigen_vec.T[:3] @ testSet.T).T   #2 x 784 @ 784 x 1000

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(10):
        ax.scatter(train_z[i * (trainNum//10) : (i+1) * (trainNum//10) ,0],\
                    train_z[i * (trainNum//10) : (i+1) * (trainNum//10) ,1],\
                    train_z[i * (trainNum//10) : (i+1) * (trainNum//10) ,2],)
    plt.show()


def calcMat(result):
    label = np.tile(np.arange(0,10),(100,1))
    bound = np.arange(-0.5, 10.5, 1)
    cMat = np.zeros((10,10))

    for i in range(10):
        hist, bins = np.histogram(result[:,i], bound)
        cMat[i,:] = hist

    cm = pd.DataFrame(cMat, index = [i for i in range(1,11)],\
                      columns = [i for i in range(1,11)])
    
    sns.heatmap(cm, annot = True)
    plt.show()

def calcMat2(result,tr_label):
    label = np.tile(tr_label,(60,1))
    bound = np.arange(-0.5, len(tr_label)+0.5, 1)
    
    cMat = np.zeros((len(tr_label),len(tr_label)))

    for i in range(len(tr_label)):
        hist, bins = np.histogram(result[:,i], bound)
        cMat[i,:] = hist

    cm = pd.DataFrame(cMat, index = [i for i in range(1,len(tr_label)+1)],\
                      columns = [i for i in range(1,len(tr_label)+1)])

    
    sns.heatmap(cm, annot = True)
    plt.show()

def calcMeasure(result):

    label = np.tile(np.arange(0,10), (testNum//10,1))
    conf_mat = np.zeros((10,10))
    TP = []; TN = []; FN = []; FP = []
    for i in range(10):
        TP.append(((result == label) & (label == i)).sum())
        TN.append(((result != i) & (label != i)).sum())
        FP.append(((result != label) & (label == i)).sum())
        FN.append(((result == i) & (label != i)).sum())

        conf_mat[i,i] = ((result == label) & (label == i)).sum()
    

    TP = np.array(TP); TN = np.array(TN); FP = np.array(FP); FN = np.array(FN);
    acc = (TP + TN)/(TP + TN + FP + FN)
    pre = TP/(TP + FP)
    rec = TP/(TP + FN)
    f1 = 2*pre*rec/(pre+rec)
    recog_rate = TP/100
    
    return recog_rate

def calcMeasure2(result, tr_label):

    label = np.tile(tr_label, (len(result),1))
    conf_mat = np.zeros((len(tr_label),len(tr_label)))
    TP = []; TN = []; FN = []; FP = []

    for i in range(len(tr_label)):
        TP.append(((result == label) & (label == i)).sum())
        TN.append(((result != i) & (label != i)).sum())
        FP.append(((result != label) & (label == i)).sum())
        FN.append(((result == i) & (label != i)).sum())
        
        conf_mat[i,i] = ((result == label) & (label == i)).sum()
    
    print(conf_mat)
    TP = np.array(TP); TN = np.array(TN); FP = np.array(FP); FN = np.array(FN);
    acc = (TP + TN)/(TP + TN + FP + FN)
    pre = TP/(TP + FP)
    rec = TP/(TP + FN)
    f1 = 2*pre*rec/(pre+rec)
    recog_rate = TP/60

    
    
    return recog_rate

def feat1(trainSet, testSet):
    trS1 = 10; trS2 = trainNum // 10 #600
    teS1 = 10; teS2 = testNum // 10 #100

    trainSetf = np.zeros((trS1 * trS2, 5))
    testSetf = np.zeros((teS1 * teS2, 5))

    for i in range(trS1):
        for j in range(trS2):
            imsi = trainSet[j+(i*trS2)]
            imsi = np.where(imsi != 0)
            imsi2 = np.mean(imsi,1)
            imsi3 = np.cov(imsi)
            trainSetf[j + (i*trS2)] = np.array([imsi2[0], imsi2[1], imsi3[0,0],imsi3[0,1],imsi3[1,1]])

    for i in range(teS1):
        for j in range(teS2):
            imsi = testSet[j+(i*teS2)]
            imsi = np.where(imsi != 0)
            imsi2 = np.mean(imsi,1)
            imsi3 = np.cov(imsi)
            testSetf[j + (i*teS2)] = np.array([imsi2[0],imsi2[1],imsi3[0,0],imsi3[0,1],imsi3[1,1]])
    return trainSetf, testSetf

def feat2(trainSet,testSet, mask_size, dx):

    input_size = trainSet[0].shape[0]
    stride = dx
    padding = 0

    output = int((input_size - mask_size + 2*padding)/stride + 1)

    #mask1
    mask = np.ones((mask_size,mask_size))
    mask = mask * (1/(mask_size)**2)

    tr_result = np.zeros((output,output))
    te_result = np.zeros((output,output))
    
    trainSetf = np.zeros((trainNum,output*output))
    testSetf = np.zeros((testNum,output*output))
    

    for k in range(trainNum):
        for i in range(output):
            for j in range(output):
                tr_result[i,j] = np.sum(trainSet[k][dx*i:dx*i+mask_size,dx*j:dx*j+mask_size] * mask)
                trainSetf[k,:] = tr_result.flatten()

    for k in range(testNum):
        for i in range(output):
            for j in range(output):
                te_result[i,j] = np.sum(testSet[k][dx*i:dx*i+mask_size,dx*j:dx*j+mask_size] * mask)
                testSetf[k,:] = te_result.flatten()

    return trainSetf, testSetf

def pca(trainSetf, testSetf, k):
    cov_mat = np.cov(trainSetf.T) #784 x 784 
    eigen_vec, eigen_val, v = np.linalg.svd(cov_mat) #u 고유벡터 #s 고유값 

    train_z = (eigen_vec.T[:k] @ trainSetf.T).T #2 x 784 @ 784 x 6000
    test_z = (eigen_vec.T[:k] @ testSetf.T).T   #2 x 784 @ 784 x 1000

    z = eigen_vec.T[:k] @ trainSetf.T
    cov_z = np.cov(z)

    x_label = np.arange(len(cov_z))+1

    gamma_sum = np.diag(cov_z).sum()
    rate = np.diag(cov_z)/gamma_sum


##    plt.plot(x_label, rate)
##    plt.show()
        
    return train_z, test_z

def lda(trainSetf, testSetf, k):
    trS1, trS2 = trainSetf.shape; trS3 = int(trS1/10)
    covMat = np.zeros((10, trS2, trS2))
    meanV = np.zeros((10, trS2))

    for i in range(10):
        covMat[i,::] = np.cov(trainSetf[i*trS3:(i+1)*trS3,:].T)
        meanV[i,:] = np.mean(trainSetf[i*trS3:(i+1)*trS3,:],0)
  
    meanC = np.mean(meanV,0)
    Sb = (meanV-meanC).T.dot(meanV-meanC)
    Sw = covMat.sum(0)
    imsi = np.linalg.pinv(Sw).dot(Sb)

    eigen_vec, eigen_val, v = np.linalg.svd(imsi) #u 고유벡터 #s 고유값 

    train_z = (eigen_vec.T[:k] @ trainSetf.T).T #2 x 784 @ 784 x 6000
    test_z = (eigen_vec.T[:k] @ testSetf.T).T   #2 x 784 @ 784 x 1000

    return train_z, test_z

def nBayes(trainSet, testSet, case):

    trS1,trS2 = trainSet.shape # 6000, 784
    teS1,teS2 = testSet.shape # 1000, 784

    trS3 = int(trS1/10) # 600
    teS3 = int(teS1/10) # 100


    
    result = np.zeros((teS3,10))
    meanV = np.zeros((10, trS2))
    covC = np.zeros((10,trS2,trS2))
    g = np.zeros((10))



    for i in range(10):
        meanV[i,:] = np.mean(trainSet[i*trS3:(i+1)*trS3,:], axis = 0)
        covC[i,::] = np.cov(trainSet[i*trS3:(i+1)*trS3,:].T)

    if case == 3:
        covC = np.tile(np.mean(covC, axis = 0), (10,1,1))
    if case == 4:
        for i in range(10):
            covC[i,::] = np.diag(np.diag(covC[i,:]))

    for i in range(teS3):
        for j in range(10):
            print(i*10+j)
            g[j] = -0.5*(testSet[i,:] - meanV[j,:]).dot(np.linalg.pinv(covC[j,::])).dot(\
                (testSet[i,:]-meanV[j,:]).T) - 0.5*np.log(np.diag(covC[j,::]).sum())
        result[i % teS3, i//teS3] = np.argmax(g)

    return result

def sklearn_knn(x_train, y_train, x_test, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(x_train, y_train)

    pred = knn.predict(testSet)
    acc = knn.score(x_test, y_test)

    plot_confusion_matrix(knn, testSet, y_test)
    cm = confusion_matrix(y_test, pred)
    rec_rate = np.diag(cm)/100
    plt.show()    


    
    return rec_rate

def sklearn_bayes(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()

    gnb.fit(x_train, y_train)
    
    pred = gnb.predict(testSet)

    plot_confusion_matrix(gnb, testSet, y_test)
    cm = confusion_matrix(y_test, pred)
    rec_rate = np.diag(cm)/100
    plt.show()


    
    return rec_rate

def create_threshold_data(trainSet, testSet, val):
    zeros_tr = np.zeros_like(trainSet)
    zeros_te = np.zeros_like(testSet)
    

    for i in range(trainSet.shape[0]):
        img = np.uint8(trainSet[i])

        ret, thr1 = cv2.threshold(img, 8, 255, cv2.THRESH_BINARY)
        zeros_tr[i] = thr1
        
    for j in range(testSet.shape[0]):
        img2 = np.uint8(testSet[j])

        ret, thr2 = cv2.threshold(img2, 8, 255, cv2.THRESH_BINARY)
        zeros_te[j] = thr2

    return zeros_tr, zeros_te

def group_classification(train, test):
    train_list = [0 for i in range(10)]
    test_list = [0 for i in range(10)]
    
        
    for i in range(10):
        cnt_shoe, cnt_cloth = 0, 0
        for j in range(trainNum//10):        
            x, y = np.where(train[i*(trainNum//10)+j][:10,11:17]==0)
            if len(x) > 25:
                cnt_shoe += 1
            else:
                cnt_cloth += 1
        if cnt_shoe < cnt_cloth:
            train_list[i] = 1

    for i in range(10):
        cnt_shoe, cnt_cloth = 0, 0
        for j in range(testNum//10):        
            x, y = np.where(test[i*(testNum//10)+j][:10,11:17]==0)
            if len(x) > 25:
                cnt_shoe += 1
            else:
                cnt_cloth += 1
        if cnt_shoe < cnt_cloth:
            test_list[i] = 1    
            
    return train_list, test_list
        
def data_classification(train_list, test_list, x_train, y_train):
    cnt1,cnt2,cnt3,cnt4 = 0,0,0,0
    tr_clo_zeros = np.zeros((train_list.count(1)*trainNum//10,28,28))
    tr_shoe_zeros = np.zeros((train_list.count(0)*trainNum//10,28,28))

    te_clo_zeros = np.zeros((test_list.count(1)*testNum//10,28,28))
    te_shoe_zeros = np.zeros((test_list.count(0)*testNum//10,28,28))



    tr_clo_label, tr_shoe_label, te_clo_label, te_shoe_label = [],[],[],[]

    for i in range(len(train_list)):
        if train_list[i] == 1:
            tr_clo_label.append(i)
            for j in range(trainNum//10):
                tr_clo_zeros[cnt1] = x_train[i*(trainNum//10)+j]
                cnt1+=1
        elif train_list[i] == 0:
            tr_shoe_label.append(i)
            for j in range(trainNum//10):
                tr_shoe_zeros[cnt2] = x_train[i*(trainNum//10)+j]
                cnt2+=1            

    for i in range(len(test_list)):
        if test_list[i] == 1:
            te_clo_label.append(i)
            for j in range(testNum//10):
                te_clo_zeros[cnt3] = y_train[i*(testNum//10)+j]
                cnt3+=1
        elif test_list[i] == 0:
            te_shoe_label.append(i)
            for j in range(testNum//10):
                te_shoe_zeros[cnt4] = y_train[i*(testNum//10)+j]
                cnt4+=1   
   

    zeros1 = np.array([])
    zeros2 = np.array([])
    zeros3 = np.array([])
    zeros4 = np.array([])
    
    for i in range(len(tr_clo_label)):
        label1 = np.full(trainNum//10, tr_clo_label[i])
        zeros1 = np.hstack((zeros1, label1))

    for j in range(len(tr_shoe_label)):
        label2 = np.full(trainNum//10, tr_shoe_label[j])
        zeros2 = np.hstack((zeros2, label2))

    for k in range(len(te_clo_label)):
        label3 = np.full(testNum//10, te_clo_label[k])
        zeros3 = np.hstack((zeros3, label3))

    for l in range(len(te_shoe_label)):
        label4 = np.full(testNum//10, te_shoe_label[l])
        zeros4 = np.hstack((zeros4, label4))


    return tr_clo_zeros, tr_shoe_zeros, te_clo_zeros, te_shoe_zeros,\
           zeros1, zeros2, zeros3, zeros4
    
################################## main ################################

x_train, y_train, x_test, y_test = init_data()
x_train2, y_train2, x_test2, y_test2 = data_ready(x_train, y_train, x_test, y_test)
trainSet, testSet = data_ready_knn(x_train2, x_test2)
result = knn3(trainSet, testSet, 2)



##recog_rate = calcMeasure2(result, tr_clo_label)
##print(recog_rate)
##cmat = calcMat2(result, tr_clo_label)


##======================   =============================
##train_th, test_th = create_threshold_data(x_train2, x_test2, 2)
##======================   =============================
##train_list, test_list = group_classification(train_th, test_th)
##tr_cloth, tr_shoes, te_cloth, te_shoes,\
##          tr_clo_label, tr_shoe_label, te_clo_label, te_shoe_label = \
##          data_classification(train_list, test_list , x_train2, x_test2)
##
##tmpl = createTmpl(tr_cloth,tr_clo_label)
##result = tmplMatch2(tmpl, te_cloth, tr_clo_label)
##
### A section
##trainSet, testSet = data_ready_knn(tr_shoes, te_shoes)
##knn_rate = sklearn_knn(trainSet, tr_shoe_label.ravel(), testSet,\
##                       te_shoe_label.ravel(),15)
##
### B section
##trainSet, testSet = data_ready_knn(tr_cloth, te_cloth)
##knn_rate = sklearn_knn(trainSet, tr_clo_label.ravel(), testSet,\
##                       te_clo_label.ravel(), 15)
##===================================================================





