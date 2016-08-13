import numpy as np
import pdb

from scipy import optimize

def residual_two_functions(params, x, y1, y1err, y2, y2err):
    """
    returns residuals
    between two lines, specified by parameters in variable params,
    and data y1 and y2
    """
    b   = params[0]
    alpha = params[1]
    c  = params[2]
    beta = params[3]
    asymptote = params[4]
    diff1 = (y1 - (asymptote + b  * alpha ** x))**2 / y1err
    diff2 = (y2 - (asymptote + c * beta ** x))**2 / y2err
    return np.concatenate((diff1, diff2))


def predict_learning_curve(train_set_size,error_test,error_train=None,pinit = [1.0, -1.0],funcs=1):
    """
    returns parameters to predict learning curve as a power function
    with the form
    y = a + b * x**alpha
    where x is the training set size, i.e., the independent variable
    
    You provide the function with your data: a vector of the training set sizes you used, and arrays of the error
    you found when training models with those training sets. The function then returns the fit parameters.
    
    arguments:
    train_set_size -- vector of m integers representing number of samples in training sets, should increase monotonically
    error_test -- m x n array of errors where error_train[m,n] is the error measured for replicate n of training a model
    with train_set_size[m] samples. Error is measured on the *training* set
    error_test -- same as error_train except the error should be measured on a test set separate from the training set 
    pinint -- initial guess for parameters b and alpha, default is [1.0, -1.0]
    funcs -- number of functions to fit, default is 1. Hello
    If funcs==1 and only test error is passed as an argument, a power function is fit just to the test error
    If funcs==1 and both test error and train error are passed as arguments,
    it is assumed the train error and test error can be fit with same
    exponent and scaling parameter. 
    If funcs==2, both test error and train error must be passed
    and each is fit with separate exponent and scaling parameters, but both share an extra
    parameter which is the asymptote.
      
    returns:
    a: asymptotic value of error predicted for infinite training data
    b: scaling parameter of power function
    alpha: exponent parameter of power function
    *** if funcs = 2 ***
    c: scaling parameter of power function fit to train error (b fits test error)
    beta: exponent parameter of power function fit to train error (alpha fits test error)
    """
    if funcs!=1 and funcs !=2:
        raise ArgumentError('funcs argument should equal 1 or 2')
    
    if funcs==2 and error_train is None:
        raise ArgumentError('error_train is a required argument when funcs==2')
    
    if train_set_size.shape[0] != error_test.shape[0]:
        raise ValueError(
            'Number of elements in train_set_size does not match number of columns in error_test')

    fitfunc = lambda p, x: p[0] + p[1] * x   
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    
    logx = np.log10(train_set_size)
 
    if error_train is None: # if we just have test error, fit with power function
        y = np.mean(error_test, axis = 1)
        logy = np.log10(y)
        yerr = np.std(error_test, axis = 1)
        logyerr = yerr / y
        out1 = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy, logyerr), full_output=1)
        pfinal = out1[0]
        b = 10.0**pfinal[0]
        alpha = pfinal[1]
        return b,alpha
        
    elif error_train is not None and funcs==1: # if we have train error too, then try Cortes et al. 1994 approach
        err_diff = error_test - error_train
        y = np.mean(err_diff, axis = 1)
        logy = np.log10(y)
        yerr = np.std(err_diff, axis = 1)
        logyerr = yerr / y
        out1 = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy, logyerr), full_output=1)    
        pfinal = out1[0]
        b = (10.0**pfinal[0]) / 2
        alpha = pfinal[1]
        
        err_sum = error_test + error_train
        y2 = np.mean(err_sum, axis = 1)
        logy2 = np.log10(y2)
        y2err = np.std(err_sum, axis = 1)
        logy2err = y2err / y
        #take mean of logy as best estimate of horizontal line
        estimate = np.average(logy2,weights=logy2err)
        a = (10.0**estimate) / 2
        return a,b,alpha
    
    elif error_train is not None and funcs==2:
        y1 = np.mean(error_test, axis = 1)
        y1err = np.std(error_test, axis = 1)
        logy1 = np.log10(y1)
        y2 = np.mean(error_train, axis = 1)
        y2err = np.std(error_train, axis = 1)
        logy2 = np.log10(y2)
        if len(pinit) < 3: # if default pinit from function declaration
            # change instead to default pinit in next line 
            pinit = [1.0, -1.0, 1.0, 1.0, 0.05]
        best, cov, info, message, ier = optimize.leastsq(residual_two_functions,
                                        pinit,
                                        args=(train_set_size, y1, y1err, y2, y2err),
                                        full_output=True)
        return best

