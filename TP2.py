import sys
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

# Constantes que atribuem os valores especificados para 
# cada "casa" do Grid de entrada e uma Ação possível do agente 
AGENT = 10
OBSTACLE = -1
GROUND = 0
GOAL = 7
DEFEAT = 4

UP = 'c'
DOWN = 'b'
LEFT = 'e'
RIGHT = 'd'
NONE = 'n'

# Grid de entrada inicializado em escopo global
Grid = np.array([], dtype=int)

# Parâmetros de entrada inicializados em escopo global
nIters, learningRate, discountFactor, stdReward, epsilon = -1, -1, -1, -1, -1

# Inicialização do array de estados do Grid que será utilizado na produção do GIF
data_list = []

# Métodos auxiliares para a criação do GIF
def init(N):
    sns.heatmap(np.zeros((N,N)), square=True, cbar=False)

def animate(i):
    data = data_list[i]
    # if i % 5 == 0: print(i)
    sns.heatmap(data, square=True, cbar=False)

# Classe QSlot encapsula dados e operações úteis para cada estado s do QGrid
class QSlot():
    def __init__(self, slotType, upValue, rightValue, downValue, leftValue, rewardValue):
        # AGENT, OBSTACLE, GROUND, GOAL, DEFEAT
        self.id = slotType
        upValue = np.float128(upValue)
        rightValue = np.float128(rightValue)
        leftValue = np.float128(leftValue)
        downValue = np.float128(downValue)

        # Valores de Q(s,a) para cada ação possível
        self.av = {
            RIGHT: rightValue,
            UP: upValue,
            DOWN: downValue,
            LEFT: leftValue
        }

        # Valor de recompensa para o estado s.
        self.reward = np.float128(rewardValue)
    
    # Atualiza o valor de Q(s,a) para a ação a conforme a equação descrita no algoritmo de Q-Learning
    # Q(s,a) = Q(s,a) + alpha * (reward + gamma * argmax_a(Q(s',a')) - Q(s,a))
    def updateQValue(self, action, sNextBestActionQValue, reward):
        self.av[action] += learningRate * (reward + (discountFactor * sNextBestActionQValue) - self.av[action])

# Classe QLearn define um Grid de QSlots, que armazena os valores de Q(s,a) para cada estado s e ação a
# e implementa o algoritmo de Q-Learning (e métodos auxiliares). 
class QLearn():
    def __init__(self):
        # Grid de QSlots
        self.QGrid = [[0 for _ in range(N)] for _ in range(N)]

        # Estado inicial do agente, uma lista [x, y] com as coordenadas do estado inicial
        self.initialSlot = [-1, -1]

        # Estado corrente do agente, também uma lista [x, y] com as coordenadas no QGrid
        self.s_xy = [-1, -1]

        # Lista de ações possíveis para os estados não terminais e não obstáculos
        self.actionsList = [UP, DOWN, LEFT, RIGHT]

        self.totalReward = 0

    # Inicializa Q(s,a) para todos os estados e ações de acordo com o Grid de entrada
    # e inicializa o estado inicial e corrente do agente
    def initializeQ(self):
        for line in range(N):
            for col in range(N):
                currSlot = Grid[line, col]
                if currSlot == OBSTACLE: self.QGrid[line][col] = QSlot(OBSTACLE, 0, 0, 0, 0, 0)
                elif currSlot == GOAL: self.QGrid[line][col] = QSlot(GOAL, 0, 0, 0, 0, 1)
                elif currSlot == DEFEAT: self.QGrid[line][col] = QSlot(DEFEAT, 0, 0, 0, 0, -1)
                elif currSlot == AGENT:
                    self.initialSlot = [line, col]
                    self.s_xy = self.initialSlot.copy()
                    self.QGrid[line][col] = QSlot(AGENT, 0, 0, 0, 0, stdReward)
                else: self.QGrid[line][col] = QSlot(GROUND, 0, 0, 0, 0, stdReward)

    # Método que decide aleatoriamente se e como o agente irá "escorregar" 
    # a partir de uma ação (action) escolhida. O agente tem 80% de chance de
    # não escorregar, ou seja, a ação escolhida é a ação executada. Caso contrário,
    # o agente escorrega para uma das duas ações adjacentes à escolhida, com 10% de chance cada.
    def slip(self, action):
        toSlip = random.random()
        if toSlip <= 0.8: return action
        elif toSlip < 0.9:
            if action == UP: return LEFT
            elif action == DOWN: return RIGHT
            elif action == LEFT: return DOWN
            elif action == RIGHT: return UP
        else:
            if action == UP: return RIGHT
            elif action == DOWN: return LEFT
            elif action == LEFT: return UP
            elif action == RIGHT: return DOWN
    
    # Método que escolhe aleatoriamente uma ação possível para o estado corrente do agente.
    # Note que o agente pode escorregar após a escolha de uma ação aleatória.
    def randomAction(self):
        action = self.actionsList[random.randint(0, 3)]

        action = self.slip(action)
        return action

    # Método que escolhe a melhor ação possível para o estado corrente do agente.
    # A melhor ação é aquela que maximiza o valor de Q(s,a) para o estado corrente.
    def bestAction(self, s, toSlip):
        action = self.actionsList[random.randint(0, 3)]
        for newBestAction in self.actionsList:
            if s.av[newBestAction] > s.av[action]: action = newBestAction
        
        if toSlip: action = self.slip(action)
        return action
    
    # Método que executa a ação action no estado corrente do agente, trocando os labels
    # das casas do Grid de lugar. Caso a ação seja impossível, ou seja,
    # o agente vai de encontro com uma parede ou um obstáculo retorna -1.
    # Caso contrário, retorna o label da casa para onde o agente se moveu.
    def executeAction(self, action):
        if action == UP:
            if self.s_xy[0] == 0 or Grid[self.s_xy[0] - 1][self.s_xy[1]] == OBSTACLE: return -1
            Grid[self.s_xy[0]][self.s_xy[1]] = GROUND
            self.s_xy[0] -= 1
            agentIn = Grid[self.s_xy[0]][self.s_xy[1]]
            Grid[self.s_xy[0]][self.s_xy[1]] = AGENT
            return agentIn
        elif action == DOWN:
            if self.s_xy[0] == N - 1 or Grid[self.s_xy[0] + 1][self.s_xy[1]] == OBSTACLE: return -1
            Grid[self.s_xy[0]][self.s_xy[1]] = GROUND
            self.s_xy[0] += 1
            agentIn = Grid[self.s_xy[0]][self.s_xy[1]]
            Grid[self.s_xy[0]][self.s_xy[1]] = AGENT
            return agentIn
        elif action == LEFT:
            if self.s_xy[1] == 0 or Grid[self.s_xy[0]][self.s_xy[1] - 1] == OBSTACLE: return -1
            Grid[self.s_xy[0]][self.s_xy[1]] = GROUND
            self.s_xy[1] -= 1
            agentIn = Grid[self.s_xy[0]][self.s_xy[1]]
            Grid[self.s_xy[0]][self.s_xy[1]] = AGENT
            return agentIn
        elif action == RIGHT:
            if self.s_xy[1] == N - 1 or Grid[self.s_xy[0]][self.s_xy[1] + 1] == OBSTACLE: return -1
            Grid[self.s_xy[0]][self.s_xy[1]] = GROUND
            self.s_xy[1] += 1
            agentIn = Grid[self.s_xy[0]][self.s_xy[1]]
            Grid[self.s_xy[0]][self.s_xy[1]] = AGENT
            return agentIn

    # Método que define uma iteração do algoritmo Q-Learning.
    # Primeiro seleciona uma ação randômica (randomAction) ou não (bestAction) de acordo com o fator epsilon.
    # Depois executa a ação escolhida e atualiza os valores de Q(s,a) de acordo com
    # a regra de atualização do algoritmo Q-Learning (updateQValue).
    def QIter(self):
        s = self.QGrid[self.s_xy[0]][self.s_xy[1]]
        if epsilon != -1:
            if random.random() < epsilon: a = self.randomAction()
            else: a = self.bestAction(s, True)
        else: a = self.bestAction(s, True)

        # Executa a ação a no estado s e recupera o label da casa para onde o agente se moveu
        agentIn = self.executeAction(a)
        # Registra o estado do Grid após a execução da ação
        data_list.append(Grid.copy())

        # s'
        sNext = self.QGrid[self.s_xy[0]][self.s_xy[1]]
        # Observa a recompensa r para o novo estado s'
        r = sNext.reward
        self.totalReward += r
        if agentIn == GOAL or agentIn == DEFEAT:
            # s' é terminal. Atualiza Q(s,a) e reinicia o episódio,
            # voltando o agente para a posição inicial.
            # Nesse caso, argmax_a'(Q(s',a')) é a própria recompensa do estado terminal
            s.updateQValue(a, sNext.reward, r)

            Grid[self.s_xy[0]][self.s_xy[1]] = agentIn
            self.s_xy = self.initialSlot.copy()
            Grid[self.s_xy[0]][self.s_xy[1]] = AGENT

            # Registra o frame de reinício do episódio
            data_list.append(Grid.copy())
            return
        
        # s' não é terminal. Atualiza Q(s,a) e continua o episódio.
        # a' é a melhor ação possível para o estado s', sem "escorregar"
        # nextBestActionQValue já recupera o valor de Q(s',a'), que é o máximo.
        sNextBestActionQValue = sNext.av[self.bestAction(sNext, False)]
        s.updateQValue(a, sNextBestActionQValue, r)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 TP2.py <input_file> <output_file_name>")
        sys.exit(1)
    
    input_file_name = sys.argv[1] 
    output_file_name = sys.argv[2]

    f = open(input_file_name, "r")

    # leitura dos parâmetros de entrada
    params = f.readline().split()
    if(len(params) > 4): 
        nIters, learningRate, discountFactor, stdReward, epsilon = map(float, params)
    else:
        nIters, learningRate, discountFactor, stdReward = map(float, params)
    nIters = int(nIters)

    # Dimensão do Grid de entrada
    N = int(f.readline())

    # Leitura do Grid de entrada
    Grid = np.zeros((N,N), dtype=int)
    for i in range(N): Grid[i] = list(map(int, f.readline().split()))
    f.close()

    # Inicialização do array de estados do Grid que será utilizado na produção do GIF
    data_list = [Grid]

    # Inicialização e execução do algoritmo Q-Learning.
    q = QLearn()
    q.initializeQ()
    for _ in range(nIters): q.QIter()

    # print(f"Average Reward: {q.totalReward / nIters}")
    Grid = Grid.astype(np.float128)

    # Grid NxN que indicará a melhor ação para cada casa do Grid
    # e adaptação do Grid de entrada para conter os melhores Q(s,a)
    # para cada casa s
    bestActionsLabels = [[NONE for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            slot = q.QGrid[i][j]
            Grid[i][j] = slot.reward
            if slot.id == AGENT or slot.id == GROUND:
                bestActionsLabels[i][j] = q.bestAction(slot, False)
                Grid[i][j] = slot.av[bestActionsLabels[i][j]]

    # print(Grid)
    # Criação da imagem com as melhores ações para cada estado do Grid
    sns.heatmap(Grid, cbar=True, square=True, annot=bestActionsLabels, fmt='')
    plt.savefig("saidas/" + output_file_name + "_acoes.png")

    # Criação do GIF com dois frames limpos ao final para clarear o fim do GIF
    data_list.append(np.zeros((N, N)))
    data_list.append(np.zeros((N, N)))
    fig = plt.figure()
    anim = animation.FuncAnimation(fig, animate, init_func=init(N), frames=len(data_list), repeat=False)
    pillowwriter = animation.PillowWriter(fps=7)
    anim.save("saidas/" + output_file_name + ".gif", writer=pillowwriter)