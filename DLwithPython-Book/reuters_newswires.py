from keras.datasets import  reuters
import numpy as np
from keras import models,layers
import matplotlib.pyplot as plt

(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)

# Decoding newswires back to text
word_index = reuters.get_word_index()
reverse_word_index = dict([value,key] for (key,value) in word_index.items())
decoded_newswire = ' '.join([reverse_word_index.get(i-3 , '?') for i in train_data[0]])
print(decoded_newswire)

# Encoding the data and labels
def vectorize_sequences(sequences, dimension= 10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1-
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# Model Definition
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

# Compile the model
model.compile(optimizer = 'rmsprop',
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'])

# Validation approach
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs= 20,
                    batch_size= 512,
                    validation_data= (x_val, y_val))

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
plt.plot(epochs,history_dict['accuracy'],'bo',label= 'Training Accuracy')
plt.plot(epochs,history_dict['val_accuracy'],'b',label= 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Retrain the model from scratch
new_model = models.Sequential()
new_model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
new_model.add(layers.Dense(64,activation='relu'))
new_model.add(layers.Dense(46,activation='softmax'))

new_model.compile(optimizer = 'rmsprop',
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'])

new_model.fit(partial_x_train,
          partial_y_train,
          epochs= 9,
          batch_size= 512,
          validation_data= (x_val, y_val))

results = new_model.evaluate(x_test,one_hot_test_labels)
print("Testing accuracy is " + str(round((results[1] * 100),2)) + "%")



