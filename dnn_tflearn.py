import tflearn
import tflearn.datasets.mnist as mnist

# data loading and pre-processing
x, y, test_x, test_y = mnist.load_data(one_hot=True)

# building the dnn
print("building the net")
inp_layer = tflearn.input_data(shape=[None,784])
dense1 = tflearn.fully_connected(inp_layer,64,activation='tanh')

dropout1 = tflearn.dropout(dense1,.8)
dense2 = tflearn.fully_connected(dropout1,64,activation='tanh')

dropout2 = tflearn.dropout(dense2,.8)
softmax = tflearn.fully_connected(dropout2,10,activation='softmax')

# regression using sgd and top 3 accuracy
sgd = tflearn.SGD(learning_rate=.1,lr_decay=.96,decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax,optimizer=sgd,metric=top_k,loss='categorical_crossentropy')

#training
print('training the model')
model = tflearn.DNN(net,tensorboard_verbose=0)
model.fit(x,y,n_epoch=20,validation_set=(test_x,test_y),show_metric=True,run_id='dense_model')


