import mdptoolbox.example
import numpy as np
from PIL import Image

actions = 4
p = 0.9

frozen_cell_reward = -0.04
hole_reward = -1.0
goal_reward = 1.0

smallMap = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 0, 2],
])
largeMap = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2]
])

eastImage = Image.open('../mdp/assets/right.png')
westImage = Image.open('../mdp/assets/left.png')
northImage = Image.open('../mdp/assets/up.png')
southImage = Image.open('../mdp/assets/down.png')
imageSize = 8

map = largeMap
height, width = map.shape

outputImage = Image.new('RGBA', (width * imageSize, height * imageSize), 'white')

def getRewardFromMap(map_layout, x, y):
    if (map_layout[y][x] == 0):
        return frozen_cell_reward
    elif (map_layout[y][x] == 1):
        return hole_reward
    elif (map_layout[y][x] == 2):
        return goal_reward

def computeTransitionAndRewardMatrix(map_layout):
    P = np.zeros((actions, width * height, width * height))
    R = np.zeros((width * height, actions))

    for y in range(height):
        for x in range(width):
            state = y * width + x
            if (x == 0):
                if (y == 0):
                    stateEast = y * width + x + 1
                    stateNorth = (y + 1) * width + x

                    # East : 0
                    P[0][state][stateEast] = p
                    P[0][state][state] = 2 * (1 - p) / 3
                    P[0][state][stateNorth] = (1 - p) / 3
                    R[state][0] = getRewardFromMap(map_layout, x + 1, y)
                    
                    # West : 1
                    P[1][state][stateEast] = (1 - p) / 3
                    P[1][state][state] = p + (1 - p) / 3
                    P[1][state][stateNorth] = (1 - p) / 3
                    R[state][1] = getRewardFromMap(map_layout, x, y)

                    # North : 2
                    P[2][state][stateEast] = (1 - p) / 3
                    P[2][state][state] = 2 * (1 - p) / 3
                    P[2][state][stateNorth] = p
                    R[state][2] = getRewardFromMap(map_layout, x, y + 1)

                    # South : 3
                    P[3][state][stateEast] = (1 - p) / 3
                    P[3][state][state] = p + (1 - p) / 3
                    P[3][state][stateNorth] = (1 - p) / 3
                    R[state][3] = getRewardFromMap(map_layout, x, y)
                elif (y == height - 1):
                    stateEast = y * width + x + 1
                    stateSouth = (y - 1) * width + x

                    # East : 0
                    P[0][state][stateEast] = p
                    P[0][state][state] = 2 * (1 - p) / 3
                    P[0][state][stateSouth] = (1 - p) / 3
                    R[state][0] = getRewardFromMap(map_layout, x + 1, y)
                    
                    # West : 1
                    P[1][state][stateEast] = (1 - p) / 3
                    P[1][state][state] = p + (1 - p) / 3
                    P[1][state][stateSouth] = (1 - p) / 3
                    R[state][1] = getRewardFromMap(map_layout, x, y)

                    # North : 2
                    P[2][state][stateEast] = (1 - p) / 3
                    P[2][state][state] = p + (1 - p) / 3
                    P[2][state][stateSouth] = (1 - p) / 3
                    R[state][2] = getRewardFromMap(map_layout, x, y)

                    # South : 3
                    P[3][state][stateEast] = (1 - p) / 3
                    P[3][state][state] = 2 * (1 - p) / 3
                    P[3][state][stateSouth] = p
                    R[state][3] = getRewardFromMap(map_layout, x, y - 1)
                else:
                    stateEast = y * width + x + 1
                    stateSouth = (y - 1) * width + x
                    stateNorth = (y + 1) * width + x

                    # East : 0
                    P[0][state][stateEast] = p
                    P[0][state][state] = (1 - p) / 3
                    P[0][state][stateNorth] = (1 - p) / 3
                    P[0][state][stateSouth] = (1 - p) / 3
                    R[state][0] = getRewardFromMap(map_layout, x + 1, y)
                    
                    # West : 1
                    P[1][state][stateEast] = (1 - p) / 3
                    P[1][state][state] = p
                    P[1][state][stateNorth] = (1 - p) / 3
                    P[1][state][stateSouth] = (1 - p) / 3
                    R[state][1] = getRewardFromMap(map_layout, x, y)

                    # North : 2
                    P[2][state][stateEast] = (1 - p) / 3
                    P[2][state][state] = (1 - p) / 3
                    P[2][state][stateNorth] = p
                    P[2][state][stateSouth] = (1 - p) / 3
                    R[state][2] = getRewardFromMap(map_layout, x, y + 1)

                    # South : 3
                    P[3][state][stateEast] = (1 - p) / 3
                    P[3][state][state] = (1 - p) / 3
                    P[3][state][stateNorth] = (1 - p) / 3
                    P[3][state][stateSouth] = p
                    R[state][3] = getRewardFromMap(map_layout, x, y - 1)
            elif (x == width - 1):
                if (y == 0):
                    stateWest = y * width + x - 1
                    stateNorth = (y + 1) * width + x

                    # East : 0
                    P[0][state][stateWest] = (1 - p) / 3
                    P[0][state][state] = p + (1 - p) / 3
                    P[0][state][stateNorth] = (1 - p) / 3
                    R[state][0] = getRewardFromMap(map_layout, x, y)
                    
                    # West : 1
                    P[1][state][stateWest] = p
                    P[1][state][state] = 2 * (1 - p) / 3
                    P[1][state][stateNorth] = (1 - p) / 3
                    R[state][1] = getRewardFromMap(map_layout, x - 1, y)

                    # North : 2
                    P[2][state][stateWest] = (1 - p) / 3
                    P[2][state][state] = 2 * (1 - p) / 3
                    P[2][state][stateNorth] = p
                    R[state][2] = getRewardFromMap(map_layout, x, y + 1)

                    # South : 3
                    P[3][state][stateWest] = (1 - p) / 3
                    P[3][state][state] = p + (1 - p) / 3
                    P[3][state][stateNorth] = (1 - p) / 3
                    R[state][3] = getRewardFromMap(map_layout, x, y)
                elif (y == height - 1):
                    stateWest = y * width + x - 1
                    stateNorth = (y + 1) * width + x

                    # East : 0
                    P[0][state][stateWest] = (1 - p) / 3
                    P[0][state][state] = p + (1 - p) / 3
                    P[0][state][stateSouth] = (1 - p) / 3
                    R[state][0] = getRewardFromMap(map_layout, x, y)
                    
                    # West : 1
                    P[1][state][stateWest] = p
                    P[1][state][state] = 2 * (1 - p) / 3
                    P[1][state][stateSouth] = (1 - p) / 3
                    R[state][1] = getRewardFromMap(map_layout, x - 1, y)

                    # North : 2
                    P[2][state][stateWest] = (1 - p) / 3
                    P[2][state][state] = p + (1 - p) / 3
                    P[2][state][stateSouth] = (1 - p) / 3
                    R[state][2] = getRewardFromMap(map_layout, x, y)

                    # South : 3
                    P[3][state][stateWest] = (1 - p) / 3
                    P[3][state][state] = 2 * (1 - p) / 3
                    P[3][state][stateSouth] = p
                    R[state][3] = getRewardFromMap(map_layout, x, y - 1)
                else:
                    stateWest = y * width + x - 1
                    stateSouth = (y - 1) * width + x
                    stateNorth = (y + 1) * width + x

                    # East : 0
                    P[0][state][stateWest] = (1 - p) / 3
                    P[0][state][state] = p
                    P[0][state][stateNorth] = (1 - p) / 3
                    P[0][state][stateSouth] = (1 - p) / 3
                    R[state][0] = getRewardFromMap(map_layout, x, y)
                    
                    # West : 1
                    P[1][state][stateWest] = p
                    P[1][state][state] = (1 - p) / 3
                    P[1][state][stateNorth] = (1 - p) / 3
                    P[1][state][stateSouth] = (1 - p) / 3
                    R[state][1] = getRewardFromMap(map_layout, x - 1, y)

                    # North : 2
                    P[2][state][stateWest] = (1 - p) / 3
                    P[2][state][state] = (1 - p) / 3
                    P[2][state][stateNorth] = p
                    P[2][state][stateSouth] = (1 - p) / 3
                    R[state][2] = getRewardFromMap(map_layout, x, y + 1)

                    # South : 3
                    P[3][state][stateWest] = (1 - p) / 3
                    P[3][state][state] = (1 - p) / 3
                    P[3][state][stateNorth] = (1 - p) / 3
                    P[3][state][stateSouth] = p
                    R[state][3] = getRewardFromMap(map_layout, x, y - 1)
            else:
                if (y == 0):
                    stateWest = y * width + x - 1
                    stateEast = y * width + x + 1
                    stateNorth = (y + 1) * width + x

                    # East : 0
                    P[0][state][stateWest] = (1 - p) / 3
                    P[0][state][stateEast] = p
                    P[0][state][stateNorth] = (1 - p) / 3
                    P[0][state][state] = (1 - p) / 3
                    R[state][0] = getRewardFromMap(map_layout, x + 1, y)
                    
                    # West : 1
                    P[1][state][stateWest] = p
                    P[1][state][stateEast] = (1 - p) / 3
                    P[1][state][stateNorth] = (1 - p) / 3
                    P[1][state][state] = (1 - p) / 3
                    R[state][1] = getRewardFromMap(map_layout, x - 1, y)

                    # North : 2
                    P[2][state][stateWest] = (1 - p) / 3
                    P[2][state][stateEast] = (1 - p) / 3
                    P[2][state][stateNorth] = p
                    P[2][state][state] = (1 - p) / 3
                    R[state][2] = getRewardFromMap(map_layout, x, y + 1)

                    # South : 3
                    P[3][state][stateWest] = (1 - p) / 3
                    P[3][state][stateEast] = (1 - p) / 3
                    P[3][state][stateNorth] = (1 - p) / 3
                    P[3][state][state] = p
                    R[state][3] = getRewardFromMap(map_layout, x, y)
                elif (y == height - 1):
                    stateWest = y * width + x - 1
                    stateEast = y * width + x + 1
                    stateSouth = (y - 1) * width + x

                    # East : 0
                    P[0][state][stateWest] = (1 - p) / 3
                    P[0][state][stateEast] = p
                    P[0][state][state] = (1 - p) / 3
                    P[0][state][stateSouth] = (1 - p) / 3
                    R[state][0] = getRewardFromMap(map_layout, x + 1, y)
                    
                    # West : 1
                    P[1][state][stateWest] = p
                    P[1][state][stateEast] = (1 - p) / 3
                    P[1][state][state] = (1 - p) / 3
                    P[1][state][stateSouth] = (1 - p) / 3
                    R[state][1] = getRewardFromMap(map_layout, x - 1, y)

                    # North : 2
                    P[2][state][stateWest] = (1 - p) / 3
                    P[2][state][stateEast] = (1 - p) / 3
                    P[2][state][state] = p
                    P[2][state][stateSouth] = (1 - p) / 3
                    R[state][2] = getRewardFromMap(map_layout, x, y)

                    # South : 3
                    P[3][state][stateWest] = (1 - p) / 3
                    P[3][state][stateEast] = (1 - p) / 3
                    P[3][state][state] = (1 - p) / 3
                    P[3][state][stateSouth] = p
                    R[state][3] = getRewardFromMap(map_layout, x, y - 1)
                else:
                    stateEast = y * width + x + 1
                    stateWest = y * width + x - 1
                    stateNorth = (y + 1) * width + x
                    stateSouth = (y - 1) * width + x

                    # East : 0
                    P[0][state][stateEast] = p
                    P[0][state][stateWest] = (1 - p) / 3
                    P[0][state][stateNorth] = (1 - p) / 3
                    P[0][state][stateSouth] = (1 - p) / 3
                    R[state][0] = getRewardFromMap(map_layout, x + 1, y)
                    
                    # West : 1
                    P[1][state][stateEast] = (1 - p) / 3
                    P[1][state][stateWest] = p
                    P[1][state][stateNorth] = (1 - p) / 3
                    P[1][state][stateSouth] = (1 - p) / 3
                    R[state][1] = getRewardFromMap(map_layout, x - 1, y)

                    # North : 2
                    P[2][state][stateEast] = (1 - p) / 3
                    P[2][state][stateWest] = (1 - p) / 3
                    P[2][state][stateNorth] = p
                    P[2][state][stateSouth] = (1 - p) / 3
                    R[state][2] = getRewardFromMap(map_layout, x, y + 1)

                    # South : 3
                    P[3][state][stateEast] = (1 - p) / 3
                    P[3][state][stateWest] = (1 - p) / 3
                    P[3][state][stateNorth] = (1 - p) / 3
                    P[3][state][stateSouth] = p
                    R[state][3] = getRewardFromMap(map_layout, x, y - 1)
    return P, R

P, R = computeTransitionAndRewardMatrix(map)
ql = mdptoolbox.mdp.QLearning(P, R, 0.9, 10000000)
ql.run()

for y in range(height - 1, -1, -1):
    for x in range(width):
        state = y * width + x
        position = (x * imageSize, (height - 1 - y) * imageSize)
        if (ql.policy[state] == 0):
            outputImage.paste(eastImage, position)
        elif (ql.policy[state] == 1):
            outputImage.paste(westImage, position)
        elif (ql.policy[state] == 2):
            outputImage.paste(northImage, position)
        elif (ql.policy[state] == 3):
            outputImage.paste(southImage, position)
outputImage.save('../mdp/output/q.png')