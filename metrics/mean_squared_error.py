import numpy as np

def mean_squared_error(Y_true,Y_pred,multioutput='uniform_average'):
    mse=np.sum(np.square(np.array(Y_true)-np.array(Y_pred))/len(Y_true),axis=0)
    return np.sum(mse) if multioutput=='uniform_average' else mse

if __name__=='__main__':
    y_true = [[0.5, 1],[-1, 1],[7, -6]]
    y_pred = [[0, 2],[-1, 2],[8, -5]]
    print(mean_squared_error(y_true,y_pred,multioutput='raw_values'))
