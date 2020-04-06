#wszystkie dążą do 1 why???????
import math, random, statistics
def sigmoid(x):
    return 1/(1+math.e**(-x))
def derivsigmoid(x):
    return x*(1-x)
def mse_loss(y,y_pred):
    #obliczanie straty
    return ((y-y_pred)**2)

class Neuron:
    w0 = random.random()
    w1 = random.random()
    w2 = random.random()
    w3 = random.random()
    w4 = random.random()
    w5 = random.random()

    b0 = random.random()
    b1 = random.random()
    b2 = random.random()

    def feedforward(self, x):
        h0 = sigmoid(self.w0 * x[0] + self.w1 * x[1] + self.b0)  # H0 = f(w0*x0 + w1*x1 + b0)
        h1 = sigmoid(self.w2 * x[0] + self.w3 * x[1] + self.b1)  # H1 = f(w2*x0 + w3*x1 + b1)
        oo = sigmoid(self.w4 * h0 + self.w5 * h1 + self.b2)  # z wzoru  O0 = f(w4*H0 + w5*H1 + b2)
        # zwracamy wartość z ostatniego neuronu
        return oo

    def train(self, x, y_data, learn_rate, epochs):
        for i in range(epochs):
            y_res=[]
            for j in range(4):
                h_0S = self.w0 * x[j][0] + self.w1 * x[j][1]+self.b0
                h0 = sigmoid(h_0S)
                h1 = sigmoid(self.w2 * x[j][0] + self.w3 * x[j][1]+self.b1)
                h_1S = self.w2 * x[j][0] + self.w3 * x[j][1]+self.b1
                oo = sigmoid((h0 * self.w4 + self.w5 * h1+self.b2))
                y_pred = oo

                D_MSE = -2 * (1 - y_pred)

                dW5 = h1 * derivsigmoid(oo)
                dW4 = h0 * derivsigmoid(oo)

                self.b2 = derivsigmoid(oo)

                dh0 = self.w4 * derivsigmoid(oo)
                dh1 = self.w5 * derivsigmoid(oo)

                self.b0 = derivsigmoid(h_0S)
                self.b1 = derivsigmoid(h_1S)

                dW0 = x[j][0] * derivsigmoid(h_0S)
                dW1 = x[j][1] * derivsigmoid(h_0S)
                dW2 = x[j][0] * derivsigmoid(h_1S)
                dW3 = x[j][1] * derivsigmoid(h_1S)
                self.w0 = self.w0 - learn_rate * dW0 * dh0 * D_MSE
                self.w1 = self.w1 - learn_rate * dW1 * dh0 * D_MSE
                self.w2 = self.w2 - learn_rate * dW2 * dh1 * D_MSE
                self.w3 = self.w3 - learn_rate * dW3 * dh1 * D_MSE
                self.w4 = self.w4 - learn_rate * dW4 * D_MSE
                self.w5 = self.w5 - learn_rate * dW5 * D_MSE
                y_res.append(y_pred)
            if i % 100 == 0:
                for k in range(4):
                    print(y_pred,end=' ')
                print('\n')


data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 0, 0, 1]
learn_rate = 0.1
epchos = 1000
siec = Neuron()

siec.train(data, y_data, learn_rate, epchos)

