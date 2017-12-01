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
    test_2()


'''
a = 2时结果：
[[ 0.41320333  0.33116013  0.25563651]
 [ 0.23747079  0.43339899  0.32913026]
 [ 0.23273246  0.39407811  0.37318945]
 [ 0.3606832   0.23351854  0.40579826]
 [ 0.35464913  0.27784342  0.36750743]
 [ 0.23811059  0.22075425  0.54113513]
 [ 0.34295374  0.37168375  0.28536254]
 [ 0.24867342  0.54526824  0.20605834]
 [ 0.55228233  0.21473129  0.23298642]]
dict_items([('p0\n', 1), ('s-1\n', 2), ('s1\n', 5), ('p-1\n', 6), ('r0\n', 0), ('r1\n', 8), ('s0\n', 3), ('r-1\n', 4), ('p1\n', 7)])

{0: 3106, 1: 4342, -1: 2552}



'''