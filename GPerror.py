import numpy as np
from mlxtend.data import mnist_data
import sklearn.preprocessing as skpp
import sherpa
from DNNGP import DNNGP
import time
from scipy.special import softmax

def mixed_mnist(count):
    total = count*10
    valid = int(np.floor(count/10))
    rawset = mnist_data()
    arranged_data = np.zeros((1,len(rawset[0][0])))
    arranged_target = np.zeros((1,1))
    for i in range(10):
        arranged_data = np.append(arranged_data,rawset[0][500*i:(500*i)+count],axis=0)
        arranged_target = np.append(arranged_target,np.ones((count,),dtype=int)*i)
    for i in range(10): #validation
        arranged_data = np.append(arranged_data,rawset[0][500*(i+1)-valid:500*(i+1)],axis=0)
        arranged_target = np.append(arranged_target,np.ones((valid,),dtype=int)*i)
    arranged_data = np.delete(arranged_data,0,axis=0)
    arranged_target = np.delete(arranged_target,0,axis=0)
    return [arranged_data,arranged_target,valid*10]


def data_prep():
    data = mixed_mnist(200) #How many samples per class; x*10
    min_max_scaler = skpp.MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(data[0])
    enc = skpp.OneHotEncoder()
    enc.fit(data[1].reshape(-1,1))
    classes = len(enc.transform([[1]]).toarray()[0])
    enc.transform(data[1][0].reshape(-1,1)).toarray()[0]

    X_train = np.zeros((len(data[0])-data[2],len(data[0][0])+classes))
    X_test = np.zeros((data[2],len(data[0][0])+classes)) 
    
    for i in range(len(data[0][:-data[2]])):
        X_train[i] = np.append(data[0][i],enc.transform(data[1][i].reshape(-1,1)).toarray()[0])
    for i in range(data[2]):
        X_test[i] = np.append(data[0][-(i+1)],np.ones(classes)*(1/classes))

    X_train_norm = skpp.normalize(X_train, norm='l2')
    X_test_norm = skpp.normalize(X_test, norm='l2')

    return X_train_norm, data, X_test_norm

def GPerror(GP,data,sigb,sigw,layers):
    """Creates the corresponding GP evaluates and returns the error
    """
    GP.setHyper(sigb,sigw,layers)
    GP.train(approximate=True)
    #evaluate and calculate the error here:
    predict = GP.rawPrediction()
    predict = predict[::-1]

    error = 0.0
    test_count = len(data[1][-data[2]:])
    #predict = softmax(predict)
    for i in range(test_count):
        if (np.argmax(predict[i]) == data[1][-data[2]+i]):
            error += 0
        else:
            for j in range(len(predict[i])):
                if j == data[1][-data[2]+i]:
                    error += (1-predict[i][j])**2 #MSE
                    #error += -1*(np.log(abs(predict[i][j]))) #Log-loss
                else:
                    #error += (predict[i][j])**2
                    pass
            #Might be worth trying to softmax the output of the function
            #and/or adding Cross entropy loss as mentioned in the paper
            #MSE found
    return error / test_count


def sherpaopt():
    train, targ, test = data_prep()
    sigb, sigw, layers = 0.35204672, 2.1220488, 87
    gp = DNNGP(train,targ[1][:-targ[2]],test,sigb,sigw,layers)

    t0 = time.time()
    parameters = [sherpa.Discrete(name='layers',range=[2,100]),
                  sherpa.Continuous(name='bias',range=[0,5]),
                  sherpa.Continuous(name='weight',range=[.1,2.09])]
    bayesopt = sherpa.algorithms.PopulationBasedTraining(4)
    stop = sherpa.algorithms.MedianStoppingRule(0,1)
    study = sherpa.Study(parameters=parameters,
                         algorithm = bayesopt,
                         stopping_rule=stop,
                         lower_is_better=True,
                         disable_dashboard=True)
    
    train = study.get_suggestion()
    
    for trial in study:
        print('still going: ',trial.id)
        for iteration in range(1):
            error = GPerror(gp,targ,trial.parameters["bias"],
                            trial.parameters["weight"],
                            trial.parameters["layers"])
            study.add_observation(trial=trial,
                                  iteration=iteration,
                                  objective=error)
        study.finalize(trial)
        if(trial.id == 100): #set to around ~200
            break
    print(study.get_best_result())
    print("Time Optimizing: ", (time.time() - t0)/60, " minutes")
    
sherpaopt()
