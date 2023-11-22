from dataset import NamesNationalityDataset
import torch.nn as nn
import utils
import networks
import torch
import yaml
import models
import argparse


class Configs:
    def __init__(self, conf_file=None):
        if conf_file is None:
            self.nations = ["Iranian", "English", "Italian", "Japanese", "Czech", "Arabic"]
            self.raw_data_path = 'data/names/'
            self.data_split = {"train": .7, "test": .2}
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.criterion = nn.NLLLoss()
            self.learning_rate = 0.001  # If you set this too high, it might explode. If too low, it might not learn
            self.optimizer = torch.optim.Adam
            self.arch = "RNN"
            self.n_hidden = 64
            self.n_layers = 1
            self.n_batch = 1
            self.n_max_len = 16
            self.n_inputs = utils.n_letters
            self.n_outputs = len(self.nations)
            self.x_train = .8
            self.x_test = .2
            self.n_test = 100000
        else:
            with open(conf_file, 'r') as cf:
                try:
                    cnf = yaml.safe_load(cf)
                    print(cnf)
                except yaml.YAMLError as exc:
                    print(exc)

            self.nations = cnf["nations"]
            self.raw_data_path = cnf["raw_data_path"]
            self.data_split = {"train": float(cnf["data_split"]["train"]), "test": float(cnf["data_split"]["test"])}
            if cnf["device"] == "default":
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                self.device = cnf["device"]

            self.criterion = nn.NLLLoss() if cnf["criterion"] == "NLLLoss" else nn.MSELoss()
            self.learning_rate = float(cnf["learning_rate"])
            if cnf["optimizer"] == "Adam":
                self.optimizer = torch.optim.Adam
            elif cnf["optimizer"] == "SGD":
                self.optimizer = torch.optim.SGD

            self.arch = cnf["arch"]
            self.n_hidden = int(cnf["n_hidden"])
            self.n_layers = int(cnf["n_layers"])
            self.n_batch = int(cnf["n_batch"])
            self.n_max_len = int(cnf["n_max_len"])
            self.n_inputs = utils.n_letters
            self.n_outputs = len(self.nations)
            self.x_train = float(cnf["x_train"])
            self.x_test = float(cnf["x_test"])
            self.n_test = int(cnf["n_test"])
        self.confusion = torch.zeros(self.n_outputs, self.n_outputs)


class LearningModel:
    def __init__(self):

        self.train_iterations = 25000
        self.train_log_cycle = 5000
        self.train_plot_cycle = 1000
        self.test_iterations = 10000
        self.test_log_cycle = 1000


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", help="increase output verbosity", default=None)
    args = parser.parse_args()

    lm = LearningModel()
    conf = Configs(args.conf)

    ds = NamesNationalityDataset(conf.device, conf.nations, conf.raw_data_path, conf.x_train, conf.x_test)
    if conf.arch == "RNN":
        network = networks.RNN(conf.n_inputs, conf.n_hidden, conf.n_outputs, conf.n_batch, conf.n_layers, conf.device)
    elif conf.arch == "LSTM":
        network = networks.LSTM(conf.n_inputs, conf.n_hidden, conf.n_outputs, conf.n_batch, conf.n_layers, conf.device)
    elif conf.arch == "GRU":
        network = networks.GRU(conf.n_inputs, conf.n_hidden, conf.n_outputs, conf.n_batch, conf.n_layers, conf.device)
    else:
        raise "Model Architecture Error"

    print(conf.device)
    conf.optimizer = torch.optim.Adam(network.parameters(), lr=conf.learning_rate)
    model = models.RecurrentModel(network, ds, conf.criterion, conf.optimizer, lm, conf.confusion, conf.device)

    model.train()
    model.test()
    model.plot_lost()
    model.show_accuracy_matrix()
    model.calculate_accuracy()
    print(model.accuracy)


if __name__ == "__main__":
    run()
