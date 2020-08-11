from ann import ANN
import plotting_service
import sys
import init_utils


class Runner:

    def __init__(self):
        pass

    def run(self, initialization="zeros"):

        # load image dataset: blue/red dots in circles
        train_X, train_Y, test_X, test_Y = init_utils.load_dataset()

        ann = ANN()
        learning_rate = 0.01
        parameters, costs = ann.fit(train_X, train_Y, learning_rate=learning_rate, initialization=initialization)

        print ("On the train set:")
        predictions_train = ann.predict(train_X, train_Y, parameters)
        print ("On the test set:")
        predictions_test = ann.predict(test_X, test_Y, parameters)

        plotting_service.plot_loss_per_iteration_for_learning_rate(costs, learning_rate)

        plotting_service.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)


if __name__ == "__main__":

    initialization = sys.argv[1]
    Runner().run(initialization=initialization)
