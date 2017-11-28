from data.article_data import graph
from module.keras import Keras
from module.player import *

upbound = 10000
data_path = "text.txt"
label_path = "label.txt"


def run(matrix):
    # 生成训练数据
    p1 = RandomPlayer(matrix)
    p2 = RandomPlayer(matrix)
    test_agent = Agent(p1, p2)
    test_agent.run(data_path, label_path, upbound=upbound)

    # 训练模型
    keras = Keras()
    keras.train_file(data_path, label_path)

    # 测试训练结果
    ai = keras.get_ai_player()
    result_agent = Agent(ai, RandomPlayer(matrix))
    state = result_agent.run(upbound=upbound)
    return state


def test_2():
    print(run(graph[2]))  # {0: 2966, 1: 4364, -1: 2670}


def test_1_1():
    print(run(graph[1.1]))  # {0: 2786, 1: 3699, -1: 3515}


def test_4():
    print(run(graph[4]))  # {0: 2731, 1: 4128, -1: 3141}


def test_9():
    print(run(graph[9]))  # {0: 3050, 1: 4081, -1: 2869}


def test_100():
    print(run(graph[100]))  # {0: 2808, 1: 3710, -1: 3482}


if __name__ == "__main__":
    test_100()
