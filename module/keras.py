from keras.models import Sequential
from keras.layers import *
from .utils import *
from .player import AI


class Keras:
    def __init__(self, time_step=10, batch_size=32, epoch=10):
        model = Sequential()
        model.add(Embedding(9, 9, input_length=1))
        model.add(GRU(5, input_shape=(None, 1)))
        model.add(Dense(3))
        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        self.model = model
        self.data_dict = Dictionary()
        self.label_dict = Dictionary()
        self.batch_size = batch_size
        self.epoch = epoch

    def train_file(self, data_path, label_path):
        data = tokenize(self.data_dict, data_path)
        _label = tokenize(self.label_dict, label_path)
        label = one_hots(_label, 3)
        return self.train(data, label)

    def train(self, data, label):
        result = self.model.fit(data[:-1], label, batch_size=self.batch_size, epochs=self.epoch)
        print(self.model.predict(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])))
        return result

    def predict(self, x, verbose=0):
        result = self.model.predict(x, batch_size=self.batch_size, verbose=verbose)
        print(result)
        return result

    def evaluate_path(self, data_path, label_path):
        data = tokenize(self.data_dict, data_path)
        _label = tokenize(self.label_dict, label_path)
        label = one_hots(_label, 3)
        return self.evaluate(data, label)

    def evaluate(self, data, label):
        return self.model.evaluate(data, label)

    def get_ai_player(self):
        return AI(self.data_dict, self.label_dict, self.model)
