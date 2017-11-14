from module.keras import Keras
from module.player import *

matrix1 = [
    [0.001, 0.004, 0.995],
    [0.063, 0.791, 0.146],
    [0.989, 0.001, 0.01],
]

matrix2 = [
    [0.063, 0.791, 0.146],
    [0.001, 0.004, 0.995],
    [0.989, 0.001, 0.01],
]

matrix3 = [
    [0.5, 0.4, 0.1],
    [0.7, 0.08, 0.22],
    [0.33, 0.33, 0.33]
]

matrix4 = [
    [0.33, 0.33, 0.33],
    [0.33, 0.33, 0.33],
    [0.33, 0.33, 0.33]
]

matrix5 = [
    [0, 0.5, 0.5],
    [0.5, 0, 0.5],
    [0.5, 0.5, 0]
]

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


if __name__ == "__main__":
    print(run(matrix1))
