import copy
import time

goal_8 = [[1, 2, 3], 
          [8, 0, 4], 
          [7, 6, 5]]

goal_15 = [[1, 2, 3, 4],
           [5, 6, 7, 8],
           [9, 10, 11, 12],
           [13, 14, 15, 0]]


class Node:         # 用来存储对应节点的状态，到该状态所需的花费，以及到该状态的路径
    def __init__(self, state, cost, path):
        self.state = state
        self.cost = cost
        self.path = path

class PriorityQueue:        # A* 对应的数据结构
    def __init__(self):
        self.list = []

    def push(self, item, priority):     # 放入新的节点信息和总花费
        entry = (item, priority)
        self.list.append(entry)
        self.update()

    def pop(self):                      # 取出花费最少的那个节点
        (item, _) = self.list.pop()
        return item
    
    def update(self):                   # 更新节点列表，使花费最小的节点位于列表最后
        self.list.sort(reverse=True, key=lambda x: x[1])

    def isEmpty(self):                  # 判断节点列表是否为空
        return len(self.list) == 0
    
    def __len__(self):                  # 返回列表长度
        return len(self.list)


class Problem:                          # 问题类
    def __init__(self, state):          
        self.start_state = state

    def getStartState(self):            # 返回问题的最初状态
        return self.start_state

    def getSuccessors(self, state):     # 返回当前节点的所有后继节点的集合
        successors_list = []            # 存储后继节点

        # 找到0的位置，即空格位置
        for y in range(len(state)):
            for x in range(len(state[0])):
                if state[y][x] == 0:
                    zero_location = (x, y)

        # 遍历四种行为（上、下、左、右）
        for (x, y) in zip([1, -1, 0, 0], [0, 0, 1, -1]):
            state_ = copy.deepcopy(state)               # 一定要深拷贝
            new_location = (zero_location[0] + x, zero_location[1] + y)
            if new_location[1] in range(len(state)) and new_location[0] in range(len(state[0])):        # 如果新地址在问题的域上，交换数字
                state_[zero_location[1]][zero_location[0]] = state_[new_location[1]][new_location[0]]
                state_[new_location[1]][new_location[0]] = 0
                successors_list.append(state_)

        return successors_list
                


        

def nullHeuristic(state, goal):       # 默认启发式函数（UCS）
    return 0

def misplacedTiles(state, goal):      # 返回错位块数地启发式函数
    count = 0
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] != goal[i][j]:
                count += 1
    return count

def manhattanDistance(state, goal):
    goal = sum(goal, [])       # 目标降维
    distance = 0
    for i in range(len(state)):
        for j in range(len(state)):
            if state[i][j] != 0:  # 0表示空格，不计算曼哈顿距离
                x_goal, y_goal = divmod(goal.index(state[i][j]), len(state))
                distance += abs(i - x_goal) + abs(j - y_goal)
    return distance




def aStarSearch(problem:Problem, heuristic=nullHeuristic, goal=goal_8):          # A*搜索算法， 返回从初始状态到目标状态的路径，以及扩展的节点个数
    open_list = PriorityQueue()         # 创建一个open表，类型是PriorityQueue
    open_list.push(Node(problem.getStartState(), 0, [problem.getStartState()]), 0)      # 将起始状态放入open表

    closed_list = []        # 创建一个closed表

    # 如果open表不为空，开始循环
    while not open_list.isEmpty():
        cur_node = open_list.pop()      # 从open表中取出代价值最小的节点
        if cur_node.state not in closed_list:       # 如果当前状态不在closed表中，就把它添加到closed表中；否则，进入下一循环
            closed_list.append(cur_node.state)
        else:
            continue

        if cur_node.state == goal:              # 如果当前状态是目标状态，返回path 和 拓展的节点总数
            return cur_node.path, len(open_list) + len(closed_list), cur_node.cost
        else:
            successors = problem.getSuccessors(cur_node.state)      # 获取当前节点的后继节点列表
            for next_state in successors:           # 迭代后继节点列表
                next_cost = cur_node.cost + 1       # 后继节点的代价是当前节点的代价加1
                if next_state not in closed_list:       # 如果下一节点状态不在closed表中，把节点push进open表
                    priority = next_cost + heuristic(next_state, goal)          # 计算估价函数值
                    open_list.push(Node(next_state, next_cost, cur_node.path + [next_state]), priority)

    return

def print_state(state):     # 方便打印每一状态的函数
    for i in range(len(state)):
        print(state[i])
    print('---------')     

if __name__ == "__main__":

    problem_8 = Problem([[2, 8, 3],
                         [1, 6, 4],
                         [7, 0, 5]])
    problem_15 = Problem([[11, 9, 4, 15],
                          [1, 3, 0, 12],
                          [7, 5, 8, 6],
                          [13, 2, 10, 14]])      
    start = time.time()
    answer, length, steps = aStarSearch(problem_15, manhattanDistance, goal_15)       # 用A*搜索算法解决八数码问题
    end = time.time()
    for state in answer:
        print_state(state)
    print(f'得出解的步数：{steps}')
    print(f'搜索时长：{end-start}')
    print(f'拓展的节点数：{length}')
    

    
