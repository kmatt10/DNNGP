import numpy as np
import math

class DNNGP:
    """raw implementation of deep neural network gaussian process as shown in
    the Google brain paper
    """

    def __init__(self,train_data,train_target,test_data,sigb,sigw,layers):
        """target should just be y in regression case
        classification targets should be correct labels; numerical classes
        """
        self.trainNum = len(train_data)
        self.testNum = len(test_data)
        self.sigb = sigb
        self.sigw = sigw
        self.layers = layers
        X_train = train_data.copy()
        X_test = test_data.copy()
        
        self.all_data = np.append(X_train,X_test,axis=0)
        self.train_target = train_target

    def setHyper(self,sigb,sigw,layers):
        self.sigb = sigb
        self.sigw = sigw
        self.layers = layers

    def K_next_layer(self,K,a,b,approx=False):
        if approx:
            return self.sigb + ((self.sigw/(2*math.pi))*(2.142*K[a,b] + 1*math.sqrt(K[b,b]*K[a,a])))
        else:
            theta = math.acos(K[a,b]/math.sqrt(K[a,a]*K[b,b]))
            J1_0 = (math.sin(theta) + (math.pi - theta)*math.cos(theta))
            return self.sigb +((self.sigw/(2*math.pi))*math.sqrt(K[a,a]*K[b,b])*J1_0)

    def build_targ_vec(self,targets):
        targvec = np.ones((len(targets),len(np.unique(targets))))*-.1
        for i in range(len(targets)):
            targvec[i][int(targets[i])] = .9
        return targvec

    def K_next_layer_flat(self,K):
        weipi = ((self.sigw/(2*math.pi)))
        alpha = 2.142
        beta = 1
        a = self.sigb
        u = self.sigb
        b = alpha*weipi
        v = (alpha+beta)*weipi
        z = self.sigb + self.sigw
        A = a*(b**self.layers)*((1/b)**2-(1/b)**(self.layers+1))/(1-(1/b))
        C = (b**self.layers)*(a + self.sigw*K) #only changing element
        B = (b**(-2)-(1/b)**(self.layers+1))/(1-(1/b))-(1/v)*((v*(1/b))**2 - (v*(1/b))**(self.layers+1))/(1-(v*(1/b)))
        B = (u*(b**self.layers)/(1-v))*B + ((z*b**self.layers)/v)*( (v*(1/b))**2 - (v*(1/b))**(self.layers+1))/(1-(v*(1/b)))
        return A + (beta*(self.sigw/(2*math.pi))*B) + C
        

    def evaluate(self,K):
        divider = self.testNum
        KDD = np.array(K[:-divider,:-divider]) #Data-Data
        self.KTD = np.array(K[-divider:,:-divider]) #Test-Data
        gausNoise =  10**-10 * (np.eye(len(KDD[:])))
        self.invKDD = np.linalg.inv(KDD + gausNoise)
        prod1 = np.matmul(self.KTD,self.invKDD)
        targets = self.build_targ_vec(self.train_target)
        return np.matmul(prod1,targets)
        
    def train(self,approximate=False, one_shot=False):
        '''
            This should be called after instancing
        '''
        K = np.zeros((len(self.all_data),len(self.all_data)))
        if one_shot:
            for i in range(len(self.all_data)):
                for j in range(i + 1):
                    K[i,j] = np.dot(self.all_data[i],self.all_data[j])
                    K[j,i] = K[i,j]
            Ktemp = np.zeros((len(self.all_data),len(self.all_data)))
            for i in range(len(self.all_data)):
                for j in range(i + 1):
                          Ktemp[i,j] = self.K_next_layer_flat(K[i,j])
                          Ktemp[j,i] = Ktemp[i,j]
            K = Ktemp.copy()
        else:
            for i in range(len(self.all_data)):
                for j in range(i + 1):
                    K[i,j] = self.sigb + (self.sigw*np.dot(self.all_data[i],self.all_data[j]))
                    K[j,i] = K[i,j]
            for a in range(self.layers):
                Ktemp = np.zeros((len(self.all_data),len(self.all_data)))
                for i in range(len(self.all_data)):
                    for j in range(i + 1):
                        Ktemp[i,j] = self.K_next_layer(K,i,j,approximate)
                        Ktemp[j,i] = Ktemp[i,j]
                K = Ktemp.copy()

        self.Kfinal = K
        self.final_mat = self.evaluate(self.Kfinal)

    def prediction(self):
        """returns vector with class predictions based on max arg
        for classification
        """
        predict = np.zeros(self.testNum)
        for i in range(self.testNum):
            predict[i] = np.argmax(self.final_mat[i])
        return predict
    
    def rawPrediction(self):
        return self.final_mat

    def confidence(self):
        #should return confidence intervals for predictions
        # KTT - KTD(KDD)^-1np.trans(KTD)
        KTT = np.array(self.Kfinal[-divider:],self.Kfinal[-divider:])
        return KTT - np.matmul(np.matmul(self.KTD,self.invKDD),np.transpose(KTD))
    
