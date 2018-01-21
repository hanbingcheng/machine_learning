import numpy as np

X = np.asarray([ [1, 0, 0],
                 [1, 0, 1], 
                 [1, 1, 0],
                 [1, 1,1] ])

y = np.asarray([[1], [1], [1], [0]])
lr = 0.1
p = np.float64(0.501)
lossHistory = []


def step():
    W = np.zeros((X.shape[1], 1))
    found = False
    epoch = 1
    while not found:
        loss = 0
        for i in range(X.shape[0]):
            pred = np.sum(X[i].T.dot(W))
            print ("pred:%f" %pred)
            pred = 1 if pred > p else 0
            error = y[i] - pred
            print("[INFO] epoch #{},sample #{}, error={:.7f}".format(epoch + 1, i +1, error[0]))
            loss = loss + abs(error)
            
            gradient = (X[i] * error[0]).reshape(X.shape[1], 1)
            W += lr * gradient
            print (W)
            
        if(loss == 0):
                found = True
                break;
        epoch += 1
        
def batch():
    W = np.zeros((X.shape[1], 1))
    found = False
    epoch = 1
    while not found:
        preds = X.dot(W) 
        print ("preds:")
        print (preds)
        preds[preds > p] = 1
        preds[preds <= p] = 0
    
        error = y - preds
        #print ("err:")
        #print (error)
        loss = np.sum(abs(error))
        lossHistory.append(loss)
        
        print("[INFO] epoch #{}, loss={:.7f}".format(epoch + 1, loss))
        
        if loss == 0:
            print ("W:")
            print (W)
            found = True
        
        gradient = X.T.dot(error) / X.shape[0]
        
        W += lr * gradient
        epoch += 1
        


#step()
batch()