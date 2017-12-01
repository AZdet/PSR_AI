from module.keras import Keras
from module.player import *

''' 极端情况的矩阵'''
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

'''
极端化情况
[[ 0.09033819  0.06762325  0.84203857]
 [ 0.01072684  0.98736107  0.00191209]
 [ 0.00479867  0.99316299  0.00203837]
 [ 0.00170585  0.00590842  0.99238575]
 [ 0.00195192  0.00481224  0.99323589]
 [ 0.98837602  0.00266861  0.00895541]
 [ 0.99258339  0.00262237  0.00479425]
 [ 0.04564986  0.87561589  0.07873423]
 [ 0.73601609  0.18334256  0.08064132]]
dict_items([('r0\n', 7), ('r1\n', 4), ('s-1\n', 1), ('s0\n', 0), ('p-1\n', 3), ('p0\n', 8), ('s1\n', 6), ('p1\n', 2), ('r-1\n', 5)])

'''