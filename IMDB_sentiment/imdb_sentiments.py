from keras.datasets import imdb
import numpy as np
from keras import models,layers,optimizers,losses,metrics
import matplotlib.pyplot as plt


(train_data , train_labels),(test_data,test_labels) = imdb.load_data(num_words= 10000)

#A quick peek at the decoding the review
word_index = imdb.get_word_index()
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

#Encoding an integer sequence to binary matrix
def vectorize_sequences(sequences,dimension= 10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#Model Definition
model = models.Sequential()
model.add(layers.Dense(16,'relu',(10000,)))
model.add(layers.Dense(16,'relu'))
model.add(layers.Dense(1,'sigmoid'))

#Compiling the Model
model.compile(optimizer= optimizers.RMSprop(lr= 0.001),
              loss= losses.binary_crossentropy,
              metrics= [metrics.binary_accuracy])

#Preparing Validation set
x_val = x_train[:10000]
partial_x_val = x_train[10000:]
y_val = y_train[:10000]
partial_y_val = y_train[10000:]

#Train your model
history = model.fit(partial_x_val,
                    partial_y_val,
                    epochs= 20,
                    batch_size= 512,
                    validation_data=(x_val,y_val))

#Plotting the training and validation loss
history_dict = history.history
epochs = range(1,len(history_dict['loss'])+1)
plt.figure(figsize= (20,10))

plt.subplot(121)
plt.plot(epochs,history_dict['loss'],'bo',label= 'Training Loss')
plt.plot(epochs,history_dict['val_loss'],'b',label= 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(epochs,history_dict['binary_accuracy'],'bo',label= 'Training Accuracy')
plt.plot(epochs,history_dict['val_binary_accuracy'],'b',label= 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#Retrain a model from scratch toll 4 epochs
new_model = models.Sequential()
new_model.add(layers.Dense(16,'relu',(10000,)))
new_model.add(layers.Dense(16,'relu'))
new_model.add(layers.Dense(1,'sigmoid'))

new_model.compile(optimizer= optimizers.RMSprop(lr= 0.001),
              loss= losses.binary_crossentropy,
              metrics= [metrics.binary_accuracy])

new_model.fit(x_train,
              y_train,
              epochs= 4,
              batch_size= 512)

results = new_model.evaluate(x_test,y_test)
print(new_model.metrics_names)
print("Test Accuracy is " + str(results["binary_accuracy"] * 100 + "%"))