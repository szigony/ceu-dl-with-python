from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam
import numpy as np

def network_build(input_dim, output_dim, dense_layers=1, dense_units=[128], dropout_layers=True, dropout_ratio=[0.3], 
                  activation='relu'):
    '''This function builds a dynamic dense network based on the provided inputs.'''
    
    # Error handling
    if dense_layers != len(dense_units):
        raise ValueError('The length of the dense_units array must be equal to the number of dense layers.')
    if dropout_layers:
        if len(dropout_ratio) == 0:
            raise ValueError('If you want to include dropout layers, you must provide an array of dropout ratios.')
        if len(dropout_ratio) != 0 and len(dense_units) != len(dropout_ratio):
            raise ValueError('The number of dropout layers must be the same as the number of dense layers.')
        
    # Model building
    model = Sequential()
    for i in range(0, dense_layers):
        model.add(Dense(units=dense_units[i], input_dim=input_dim, activation=activation))
        if dropout_layers:
            model.add(Dropout(rate=dropout_ratio[i]))
    model.add(Dense(units=output_dim, activation='softmax'))
      
    return model
	
def compile_fit_evaluate_network(model, X_train, y_train, X_test, y_test, epochs=100, validation_split=0.2, batch_size=64, 
                                 loss_function='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'], 
                                 num_of_repeats=10):
    '''This function compiles the model, fits it on the train set and evaluates its accuracy against the test set. It also 
    repeats the experiment a selected number of times and takes the average of their accuracy as the final metric, 
    also known as grand average.'''
    
    # Error handling
    if type(loss_function) != str:
        raise ValueError('You must only provide one loss function.')
    
    # Network building
    accuracies = []
    for i in range(0, num_of_repeats):
        # Compile the model
        model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

        # Fit the model
        model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, batch_size=batch_size, verbose=False)

        # Evaluate the model on the test set
        model_loss, model_accuracy = model.evaluate(X_test, y_test, verbose=False)
        accuracies.append(model_accuracy)

    grand_mean = np.mean(accuracies)
    print(f'The overall accuracy of the model on the test set is {grand_mean:.2%}')
        
    return model, grand_mean