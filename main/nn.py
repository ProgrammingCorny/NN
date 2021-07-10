import numpy as np

class ConfigurationError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message) 

class DenseLayer:
    def __init__(self, size_nn, activationfuncs_nn, weights):
        def linear(x):
            return x
        def d_linear(x):
            return 1
        def ReLU(x):
            return np.maximum(0,x)
        def d_ReLu(x):
            x[x<=0] = 0
            x[x>0] = 1
            return x
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        def d_sigmoid(x):
            return x * (1-x)
        def tanh(x):
            return 2 / (1 + np.exp(-2*x)) - 1
        def d_tanh(x):
            return 1 - (x)**2
        def softmax(x):
            maxim = np.max(x)
            return np.exp(x-maxim) / np.sum(np.exp(x-maxim))
        def d_softmax(x):
            return x * (1 - x)
        self.__size_nn = size_nn
        for e in activationfuncs_nn:
            if e not in ["Linear", "TanH", "Sigmoid", "ReLu", "Softmax"]:
                raise ConfigurationError("No activationfunction is named " + e + ". Available activationfunctions: Linear, TanH, Sigmoid, ReLu, Softmax")
        self.__activationfuncs_nn = {}
        self.__derivative_activationfuncs_nn = {}
        for e in activationfuncs_nn:
            if e == "Linear":
                self.__activationfuncs_nn["Linear"] = linear
                self.__derivative_activationfuncs_nn["Linear"] = d_linear
            if e == "TanH":
                self.__activationfuncs_nn["TanH"] = tanh
                self.__derivative_activationfuncs_nn["TanH"] = d_tanh
            if e == "ReLu":
                self.__activationfuncs_nn["ReLu"] = ReLU
                self.__derivative_activationfuncs_nn["ReLu"] = d_ReLu
            if e == "Sigmoid":
                self.__activationfuncs_nn["Sigmoid"] = sigmoid
                self.__derivative_activationfuncs_nn["Sigmoid"] = d_sigmoid
            if e == "Softmax":
                self.__activationfuncs_nn["Softmax"] = softmax
                self.__derivative_activationfuncs_nn["Softmax"] = d_softmax
        if len(self.__size_nn)-1 != len(self.__activationfuncs_nn.keys()):
            raise ConfigurationError("Between two Layers there must be one activationfunction!")
        self.__weights = weights
        
    def forward(self, inputs):
        if type(inputs) == np.ndarray:
            self.__inputs = inputs
        if type(inputs) == list:
            self.__inputs = np.array(inputs)
        if type(inputs) != np.ndarray and type(inputs) != list:
            raise TypeError("The input must be a np.ndarray or a list!")
        self.__out_layers = []
        self.__out_layers.append(self.__inputs)
        for i in range(len(self.__size_nn)-1): #I don't know if this one, one line down, is so elegant: self.__activationfuncs_nn[list(self.__activationfuncs_nn.keys())[i]]
            self.__out_layers.append(self.__activationfuncs_nn[list(self.__activationfuncs_nn.keys())[i]](np.dot(self.__out_layers[i], self.__weights[i])))
        return self.__out_layers[-1]

        
    def backward(self, desired_out, e):
        self._delta = [0 for i in range(len(self.__size_nn)-1)]
        self._delta[-1] = (desired_out-self.__out_layers[-1]) * self.__derivative_activationfuncs_nn[list(self.__derivative_activationfuncs_nn.keys())[-1]](self.__out_layers[-1])
        if len(self.__size_nn)-2 > 0:
            for i in range(len(self.__size_nn)-3, -1, -1):
                self._delta[i] = np.array(self._delta[i+1]).dot(self.__weights[i+1].T) * self.__derivative_activationfuncs_nn[list(self.__derivative_activationfuncs_nn.keys())[i]](self.__out_layers[i+1])
        for i in range(len(self.__weights)-1, -1, -1):
            self.__weights[i] += e * self.__out_layers[i].T.dot(self._delta[i])
