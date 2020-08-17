from keras.datasets import boston_housing
from keras import models,layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_target) , (test_data,test_target) = boston_housing.load_data()

# Feature Normalization
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
test_data /= std

test_data -= mean
test_data /= std

# Build the model
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation= 'relu',input_shape= (train_data.shape[1],)))
    model.add(layers.Dense(64,'relu'))
    model.add(layers.Dense(1))
    model.compile('rmsprop','mse',metrics=['mae'])
    return model

# Using Kfold validation
k = 5
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories= []

for i in range(k):
    print("Processing fold #",i)
    val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
    val_targets = train_target[i * num_val_samples : (i+1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                        train_data[(i+1) * num_val_samples:]],
                                        axis=0)
    partial_train_target = np.concatenate([train_target[:i * num_val_samples],
                                        train_target[(i+1) * num_val_samples:]],
                                        axis=0)

    model = build_model()
    history = model.fit(partial_train_data,
                        partial_train_target,
                        validation_data=(val_data,val_targets),
                        epochs= num_epochs,
                        batch_size= 1,
                        verbose= 0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)

avg_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# Plot the Validation Scores
plt.plot(range(1,len(avg_mae_history)+1),avg_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Build a model from Scratch with new epoch

new_model = build_model()
new_model.fit(train_data,train_target,
          epochs=150, batch_size=16, verbose=0)
test_mse_score,test_mae_score = new_model.evaluate(test_data,test_target)

print("Testing MAE is $"+str(round((test_mae_score* 1000), 2)))

