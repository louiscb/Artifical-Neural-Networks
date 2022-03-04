from unittest import skip
import numpy as np
import random

class RBFNetwork():
    def __init__(self,
                 n_inputs,
                 n_rbf,
                 n_outputs,
                 n_epochs=20,
                 min_val=0.05,
                 max_val=2 * np.pi,
                 rbf_var=0.1,
                 learning_rate = 0.4,
                 CL_learning_rate= 0.1,
                 rbf_positioning = 'linspace'):

        self.n_inputs = n_inputs
        self.n_rbf = n_rbf
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.rbf_var = rbf_var
        self.learning_rate = learning_rate
        self.CL_learning_rate = CL_learning_rate
        self.min_val = min_val
        self.max_val = max_val
        self.rbf_positioning = rbf_positioning 

        if self.rbf_positioning == 'linspace':
            self.rbf_centers = np.array([np.linspace(min_val, max_val, n_rbf)])
        else:
            self.rbf_centers = np.array([np.random.uniform(min_val, max_val, n_rbf)])

            
        self.w = np.array([np.random.normal(0, 1, n_rbf)])
        self.RBF = np.vectorize(self.base_func)
  

    def base_func(self, x, center):
        return np.exp(-np.linalg.norm(x - center)**2 / (2 * self.rbf_var**2))

    def phi(self, data):
        self.data = np.array([data]).T

        phi = self.RBF(self.data, self.rbf_centers)
        return phi


    def train_leastsquaresbatch(self, data, f):

        MSEs = np.zeros(self.n_epochs)
        self.data = np.array([data]).T
        phi = self.RBF(self.data, self.rbf_centers)

        for epoch in range(self.n_epochs):
            f_hat = np.dot(self.w, phi.T).flatten()
            MSEs[epoch] = self.calc_mse(f_hat, f)

            self.w = np.dot(np.dot(np.linalg.pinv(np.dot(phi.T, phi)), phi.T), f)

        return MSEs

    def test_leastsquaresbatch(self, data, f):

        self.data = np.array([data]).T
        phi = self.RBF(self.data, self.rbf_centers)

        self.w = np.dot(np.dot(np.linalg.pinv(np.dot(phi.T, phi)), phi.T), f)
        f_hat = np.dot(self.w, phi.T).flatten()

        MSE = self.calc_mse(f_hat, f)
        return MSE

    def calc_delta_w(self, pattern, target, error):
        error = target - self.predict(pattern)
        delta_w = self.learning_rate * error * self.RBF(pattern, self.rbf_centers)
        return delta_w

    def train_sequential_delta(self, data, targets, CL_iterations=0, CL_winners=1):


        if CL_iterations!=0:

            for i in range(CL_iterations):

                distances=[]

                trainvector= data.copy()
                random.shuffle(trainvector)
                
                train_v = trainvector[0]

                for position in self.rbf_centers.flatten():
                    distances.append(np.linalg.norm(position - train_v))

                index= np.argpartition(distances, 1)
                winner = np.argmin(index)
                

                self.rbf_centers[0, winner] += self.learning_rate*(train_v - self.rbf_centers[0, winner])

                if CL_winners ==2:
                    distances.pop(winner)
                    index= np.argpartition(distances, 1)
                    winner = np.argmin(index)

                    self.rbf_centers[0, winner] += self.learning_rate*(train_v - self.rbf_centers[0, winner])




        MSEs = np.zeros(self.n_epochs)
        #data = np.array([data]).T

     

        for epoch in range(self.n_epochs):
            _data = list(zip(data, targets))
            random.shuffle(_data)
            data, targets = list(zip(*_data))
            data = np.array(data)
            targets= np.array(targets)

            preds = np.zeros(len(targets))
            point_no = 0

            

            for data_point, target in zip(data, targets):



                phi_k = np.zeros(self.n_rbf)
                pred = 0

                for index in range(self.n_rbf):
                    pred = pred + np.matmul([self.w.flatten()[index]], [self.base_func(data_point, self.rbf_centers.flatten()[index])] )
                    phi_k[index] = self.base_func(data_point, self.rbf_centers.flatten()[index])
                preds[point_no] = pred
                point_no= point_no +1

                    
                MSEs[epoch] = self.calc_mse(preds, targets)

                error = target - pred
                delta_w = self.learning_rate * error * phi_k
                self.w = self.w + delta_w
                
            
        return MSEs


 

        
    
    def test_sequential_delta(self, data, targets):

        _data = list(zip(data, targets))
        random.shuffle(_data)
        data, targets = list(zip(*_data))
        data = np.array(data)
        targets= np.array(targets)

        preds = np.zeros(len(targets))
        point_no = 0


        for data_point, target in zip(data, targets):

            
            pred = 0

            for index in range(self.n_rbf):
                pred = pred + np.matmul([self.w.flatten()[index]], [self.base_func(data_point, self.rbf_centers.flatten()[index])] )
            
            preds[point_no] = pred
            point_no= point_no +1

        
        MSE = self.calc_mse(preds, targets)
        return MSE


        




                
                


        
    def calc_mse(self, preds, targets):
        return np.sum(np.power(preds - targets, 2))/len(targets)

    def predict(self, x):
        x = np.array([x]).T
        return np.dot(self.w, self.RBF(x, self.rbf_centers).T).flatten()

