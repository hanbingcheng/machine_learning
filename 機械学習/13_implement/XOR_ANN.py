import numpy as np
import matplotlib.pyplot as plt
from mpmath import lerchphi

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

'''
NEURAL NETWORKS WITH BACKPROPAGATION FOR XOR USING ONE HIDDEN LAYER
'''
class ANNBPModel:
    
    def __init__(self, layers, activation = 'sigmoid'):
        
        # set activate function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
            
        # Set weights
        self.weights = []
        
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = (2 * np.random.random((layers[i-1] + 1, layers[i] + 1)) -1) * 0.01
            self.weights.append(r)
            
        # output layer - random((2+1, 1)) : 3 x 1
        r = (2 * np.random.random( (layers[i] + 1, layers[i+1])) -1) * 0.01
        self.weights.append(r)
        print (self.weights)
            
    def fit(self, X, y, learning_rate = 0.1, epochs = 10000, momentum = 0.1):
        
        # Add column of ones to X
        X = np.c_[np.ones((X.shape[0])), X]
        lossHistory = []
        run_cnt = 0
        for e in range(epochs):
            lost = 0.0
            run_cnt += 1
            # input layer
            #i = np.random.randint(X.shape[0]);
            for i in range(X.shape[0]):
                o = [X[i]]
            
                # Hidden Layer Output
                for k in range(len(self.weights)):
                    dot_value = np.dot(o[k], self.weights[k])
                    o_k = self.activation(dot_value)
                    o.append(o_k)
                
                #output layer
                error  = y[i] - o[-1]
                lost += abs(error)
                deltas = [error * self.activation_prime(o[-1])]
                
                # from the second layer to the layer before last layer
                for l in range(len(o) - 2, 0, -1): 
                    delta = deltas[-1].dot(self.weights[l].T) * self.activation_prime(o[l])
                    deltas.append(delta)
                
                # reverse
                # [layer3(output)->layer2(hidden)]  => [layer2(hidden)->layer3(output)]
                deltas.reverse()
                
                # back propagation
                # 1. Multiply its output delta and input activation 
                #    to get the gradient of the weight.
                # 2. Subtract a ratio (percentage) of the gradient from the weight.
                 
                for j in range(len(self.weights)):
                    layer = np.atleast_2d(o[j])
                    delta = np.atleast_2d(deltas[j])
                    self.weights[j] += learning_rate * layer.T.dot(delta) + momentum * self.weights[j]
           
            
            lossHistory.append(abs(lost))
            if e % 100 == 0:
                print ('epochs:{},lost:{}'.format(e, lost))
    
            if lost < 0.01:
                print ('epochs:{},lost:{}'.format(e, lost))
                print (self.weights)
                break
                    
        fig = plt.figure()
        plt.plot(np.arange(0, run_cnt), lossHistory)
        fig.suptitle("Training Loss(XOR_ANN)")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.show()     
            
        
    def predict(self, x):
        x = np.insert(x, 0, 1)   
        out = np.array(x)
        for l in range(0, len(self.weights)):
            out = self.activation(np.dot(out, self.weights[l]))
        return out
    

if __name__ == "__main__":
    
    nn = ANNBPModel([2, 2, 1], 'sigmoid')
    X = np.array([  [0, 0], [0, 1], [1, 0], [1, 1] ] )
    y = np.array([0, 1, 1, 0])
    
    nn.fit(X, y, learning_rate = 0.01, epochs = 100000, momentum = 0.001)
    
    for input_data in X:
        print(input_data, nn.predict(input_data))
        
    
                
                
                
                
                
                
                
