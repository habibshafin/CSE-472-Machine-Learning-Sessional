import numpy as np
from sklearn.metrics import f1_score,accuracy_score

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def readCifar10():
    unpickled_data = unpickle('cifar-10-batches-py/data_batch_1')
    label = unpickled_data['labels']
    
    data_x = unpickled_data['data'].reshape((len(unpickled_data['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    data_x = data_x.reshape(len(unpickled_data['data']), 3, 32, 32)
    labels = np.array(label)
    labels = labels.reshape(len(unpickled_data['data']), 1)

    return data_x, labels

def read_mnist():
    import gzip
    f = gzip.open('train-images-idx3-ubyte.gz','r')

    image_size = 28
    num_images = 10000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    data = data.reshape(num_images, 1, image_size, image_size)
    data = data/255

    f = gzip.open('train-labels-idx1-ubyte.gz','r')
    f.read(8)
    labels = np.zeros((1,num_images))
    for i in range(0,num_images):   
        buf = f.read(1)
        label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        labels[0, i] = label
    labels = labels.T
    return data, labels

def read_mnist_test():
    import gzip
    f = gzip.open('t10k-images-idx3-ubyte.gz','r')

    image_size = 28
    num_images = 200

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    data = data.reshape(num_images, 1, image_size, image_size)
    data = data/255

    f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
    f.read(8)
    labels = np.zeros((1,num_images))
    for i in range(0,num_images):   
        buf = f.read(1)
        label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        labels[0, i] = label
        
    labels = labels.T
    return data, labels

def add_pad(X, pad):
    # return np.pad(X, ((0,0), (0,0),(pad, pad), (pad, pad) ))
    return np.pad(X, ( (0,0),(pad, pad), (pad, pad) ))
    
class Conv:
    def __init__(self, ch_num, f_size, str, pad, X_shape) -> None:
        self.channel_num = ch_num
        self.filter_dim = f_size
        self.stride = str
        self.padding = pad
        self.dataX_shape = X_shape
        self.out_shape = (self.channel_num, int(1 + (self.dataX_shape[1] - f_size + 2*pad) / str), int(1 + (self.dataX_shape[2] - f_size+ 2*pad) / str))
        # self.filter = np.random.randint( 1, size=( self.channel_num, self.dataX_shape[1], self.filter_dim, self.filter_dim )) - .5
        # self.bias = np.random.randint( 1, size=( self.channel_num , self.out_shape[1], self.out_shape[2] )) - .5
        self.filter = np.random.randn( self.channel_num, self.dataX_shape[0], self.filter_dim, self.filter_dim )/9
        self.bias = np.random.randn( self.channel_num , self.out_shape[1], self.out_shape[2] )/9
        
        #print(self.filter.shape)


    def forward_single_step(self, x_slice, channel, b):
        Z = 0
        for color in range(self.dataX_shape[0]):
            s = np.multiply(x_slice[color], self.filter[channel,color])
            z = np.sum(s)
            Z = Z + z
        Z = Z + b
        
        return Z

    def forward_single_data(self, dataX):
        self.last_input = dataX
        dataX = add_pad(dataX, self.padding)
        # print(self.filter)
        # print(self.bias)
        
        out = np.zeros(self.out_shape)
        # print(dataX.shape)
        # print(self.bias.shape)
        for ch in range(self.channel_num):
            for h in range(self.out_shape[1]):
                vert_start = self.stride * h 
                vert_end = vert_start + self.filter_dim
                for w in range(self.out_shape[2]):
                    h_start = self.stride * w
                    h_end = h_start + self.filter_dim
                    out[ch, h, w] = self.forward_single_step(dataX[ : , vert_start:vert_end, h_start:h_end],ch, self.bias[ch, h,w])
        return out

    def backward_single_data(self, grad, learning_rate):
        X = self.last_input
        X = add_pad(X, self.padding)
        df = np.zeros(self.filter.shape)
        db = np.zeros(self.bias.shape)
        out_grad = np.zeros(X.shape)
        #print(out_grad.shape)

        for ch in range(self.channel_num):
            for h in range(self.out_shape[1]):
                for w in range(self.out_shape[2]):
                    vert_start = self.stride * h 
                    vert_end = vert_start + self.filter_dim
                    h_start = self.stride * w
                    h_end = h_start + self.filter_dim
                    x_slice = X[:, vert_start:vert_end,h_start:h_end]

                    df[ch,:,:] += x_slice * grad[ch, h, w]
                    db[ch,:,:] += grad[ch, h, w]

                    out_grad[:, vert_start:vert_end, h_start:h_end] += self.filter[ch, :, :] * grad[ch, h, w]

        #print(out_grad.shape)
        if self.padding!=0:
            out_grad = out_grad[:, self.padding:-self.padding, self.padding:-self.padding]
            
        self.filter = self.filter - learning_rate* df
        self.bias = self.bias - learning_rate*db

        return out_grad


class Relu:
    def __init__(self, prev_shape) -> None:
        self.out_shape = prev_shape
    def forward_single_data(self, input):
        self.last_input = input
        return np.maximum(input, 0)
    def backward_single_data(self, grad, learning_rate):
        return (self.last_input>0)*grad

class Pool:
    def __init__(self, f_dim, str , input_shape) -> None:
        self.filter_dim = f_dim
        self.stride = str
        self.out_shape = (input_shape[0], int(( input_shape[1]-self.filter_dim )/self.stride)+1, int((input_shape[2]-self.filter_dim )/self.stride)+1 )
    
    def forward_single_data(self, input):
        self.last_input = input
        out = np.zeros(self.out_shape)
        
        for i in range(input.shape[0]):
            for j in range(out.shape[1]):
                for k in range(out.shape[2]):
                    out[i, j, k] = np.max(input[i, j*self.stride:j*self.stride + self.filter_dim , k*self.stride : k*self.stride + self.filter_dim ])
        
        return out
    
    def create_mask_from_slice(self, slice):
        return slice == np.max(slice)

    def backward_single_data(self, grad, learning_rate):
        out_grad = np.zeros(self.last_input.shape)
        for ch in range(self.out_shape[0]):
            for h in range(self.out_shape[1]):
                for w in range(self.out_shape[2]):
                    vert_start = self.stride * h 
                    vert_end = vert_start + self.filter_dim
                    h_start = self.stride * w
                    h_end = h_start + self.filter_dim
                    
                    x_slice = self.last_input[ch, vert_start:vert_end,h_start:h_end]
                    mask = self.create_mask_from_slice( x_slice )

                    out_grad[ch, vert_start:vert_end,h_start:h_end] += mask * grad[ch, h, w]
        
        return out_grad


class FC:
    def __init__(self, size, prevsize) -> None:
        self.size1 = size
        self.size2 = prevsize[0]
        # self.w = np.random.randint(3, size=(self.size1, self.size2))
        # self.bias = np.random.randint(3, size=(1, 1))
        self.w = np.random.randn(self.size1, self.size2)/9
        self.bias = np.random.randn(self.size1, 1)/9
        self.out_shape = (self.size1, prevsize[1]) 

    def forward_single_data(self, input):
        self.last_in_shape = input.shape
        
        input = input.flatten()
        input = input.reshape(input.shape[0],1)
        
        out = np.dot(self.w, input)
        out = out + self.bias
        
        self.last_input = input
        return out
    
    def backward_single_data(self, gradient, learning_rate):
        dw = np.dot(gradient, self.last_input.T)
        db = gradient.reshape(self.bias.shape)

        self.w -= learning_rate * dw
        self.bias -= learning_rate * db
        
        grad = np.dot(self.w.T, gradient)
        return grad.reshape(self.last_in_shape)


    

class Softmax:
    def __init__(self, prevshape) -> None:
        self.out_shape = prevshape

    def forward_single_data(self, input):
        e_x = np.exp(input - np.max(input))
        self.output = e_x / e_x.sum()
        return self.output

    def backward_single_data(self, gradient, learning_rate):
        return np.dot((np.identity(self.output.shape[0]) - self.output.T) * self.output, gradient)

x,y = read_mnist()
# x, y = readCifar10()

c = Conv(6,5,1,2, x[0].shape)
#c = Conv(2,2,1,0, d.shape)
o = c.forward_single_data(x[0,:,:,:])

R = Relu(c.out_shape)
res = R.forward_single_data(o)
P = Pool(2,2, R.out_shape)
p_res = P.forward_single_data(res)

C2 = Conv(12, 5, 1, 0, P.out_shape )
o2 = C2.forward_single_data(p_res)
R2 = Relu(C2.out_shape)
res2 = R2.forward_single_data(o2)
P2 = Pool(2,2, R2.out_shape)
p_res2 = P2.forward_single_data(res2)

C3 = Conv(100, 5, 1, 0, P2.out_shape )
o3 = C3.forward_single_data(p_res2)
R3 = Relu(C3.out_shape)
res3 = R3.forward_single_data(o3)

fc =  FC(10, R3.out_shape)
fc_out = fc.forward_single_data(res3)

# print(fc_out)
# print(fc.out_shape)

sm = Softmax(fc.out_shape)
smout = sm.forward_single_data(fc_out)

class CNN:
    def __init__(self, ep, b, l_rate) -> None:
        self.network = []
        self.network.append(c)
        self.network.append(R)
        self.network.append(P)
        self.network.append(C2)
        self.network.append(R2)
        self.network.append(P2)
        self.network.append(C3)
        self.network.append(R3)
        self.network.append(fc)
        self.network.append(sm)
        self.learning_rate = l_rate
        self.epoch = ep
        self.batch_size = b

    def cross_entropy_loss(self, predicted, dataY):
        if predicted[int(dataY),0] == 0:
            return .5
        return - np.log(predicted[int(dataY),0])

    def cross_entropy_prime_single_data(self, predicted, actual_value):
        act_mat = np.zeros(predicted.shape)
        act_mat[int(actual_value),0] = 1
        return predicted - act_mat

    def predict(self, datax):
        out = datax
        for i in range(len(self.network)):
            out = self.network[i].forward_single_data(out)
        
        return out

    def backward(self, grad):
        for layer in reversed(self.network):
            # print()
            # print(i)
            grad = layer.backward_single_data(grad, self.learning_rate)
            # print(grad.shape)
            # i = i + 1
            
        return
    
    def batch_train(self, batch_x, batch_y):
        grad = np.zeros((10,1))
        error = 0
        for i in range(batch_x.shape[0]):
            pred = self.predict(batch_x[i])
            error = error + self.cross_entropy_loss(pred, y[i,0])

            grad = grad + self.cross_entropy_prime_single_data(pred, batch_y[i,0])
        
        grad = grad/batch_x.shape[0]

        self.backward(grad)

        return error/batch_x.shape[0]

    def test(self):
        y_true = y_test.astype(int)
        y_true = np.ndarray.tolist(y_true.T)[0]
        # print(y_true)
        y_pred = []
        for i in range(x_test.shape[0]):
            pred = self.predict(x_test[i])
            y_pred.append(np.argmax(pred))
        
        # print(y_pred)
        print(accuracy_score(y_true, y_pred))
        print(f1_score(y_true, y_pred, average='macro'))
        
        return

    def train(self):
        import time
        for i in range(self.epoch):
            start = time.time()
            print(str(i+1) + " epoch starts")
            start_index = 0
            j = 0
            while True:
                err = self.batch_train(x[start_index: min(start_index+self.batch_size, x.shape[0])], y[start_index: min(start_index+self.batch_size, x.shape[0])])
                start_index = start_index + self.batch_size
                j = j+1
                # if j%32==0:
                #     print(err)

                if start_index >= x.shape[0]:
                    # print("epoch finished")
                    break
                    
            end = time.time()
            print(end-start)

            self.test()
        
        
            
        # print("works")
        return


# print("CNN :")
x_test, y_test = read_mnist_test()

cnn = CNN(10, 32, .001)

cnn.train()