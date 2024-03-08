import sherpa
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.optimizers import Adam
import tempfile
import os
import shutil

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

parameters = [sherpa.Continuous('learning_rate', [1e-4, 1e-2], 'log'),
              sherpa.Discrete('num_units', [32, 128]),
              sherpa.Choice('activation', ['relu', 'tanh', 'sigmoid'])]

algorithm = alg = sherpa.algorithms.SuccessiveHalving(r=1, R=9, eta=3, s=0, max_finished_configs=1)

study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     disable_dashboard=True,
                     lower_is_better=False)

# study = sherpa.Study(parameters=parameters,
#                      algorithm=algorithm,
#                      disable_dashboard=True,
#                      lower_is_better=False,
#                      dashboard_port=8995)

model_dir = tempfile.mkdtemp()
for trial in study:
    # Getting number of training epochs
    initial_epoch = {1: 0, 3: 1, 9: 4}[trial.parameters['resource']]
    epochs = trial.parameters['resource'] + initial_epoch

    print("-" * 100)
    print(f"Trial:\t{trial.id}\nEpochs:\t{initial_epoch} to {epochs}\nParameters:{trial.parameters}\n")

    if trial.parameters['load_from'] == "":
        print(f"Creating new model for trial {trial.id}...\n")

        # Get hyperparameters
        lr = trial.parameters['learning_rate']
        num_units = trial.parameters['num_units']
        act = trial.parameters['activation']

        # Create model
        model = Sequential([Flatten(input_shape=(28, 28)),
                            Dense(num_units, activation=act),
                            Dense(10, activation='softmax')])
        optimizer = Adam(lr=lr)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
    else:
        print(f"Loading model from: ", os.path.join(model_dir, trial.parameters['load_from']), "...\n")

        # Loading model
        model = load_model(os.path.join(model_dir, trial.parameters['load_from']))

    # Train model
    for i in range(initial_epoch, epochs):
        model.fit(x_train, y_train, initial_epoch=i, epochs=i + 1)
        loss, accuracy = model.evaluate(x_test, y_test)

        print("Validation accuracy: ", accuracy)
        study.add_observation(trial=trial, iteration=i,
                              objective=accuracy,
                              context={'loss': loss})

    study.finalize(trial=trial)
    print(f"Saving model at: ", os.path.join(model_dir, trial.parameters['save_to']))
    model.save(os.path.join(model_dir, trial.parameters['save_to']))

    study.save(model_dir)

# print (sherpa.Study.load_dashboard("."))

print (study.get_best_result())
print(os.path.join(model_dir, study.get_best_result()['save_to']))

shutil.rmtree(model_dir)
