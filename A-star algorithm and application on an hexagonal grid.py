# Hello, and thank you for paying attention to my work !

# To use the code as it is, you just have to executte it, then a 
# graphical interface will show you the initial position in blue, the
# objective position in red and the obstacles in black.
# When you close this window, the A-star algorithm will be computed and
# you will then see the calculated path in green
#If there is just no path, then a message will tell you so.

# Enjoy !

# Here are some parameters that you could want to change according to the size of your screen.

window_width = 1300 # The number of pixels corresponding to the width of the graphical interface
window_height = 600 # The number of pixels corresponding to the height of the graphical interface

# The number of hexagons that you want your grid to contain, horizontally and vertically
# I recommand that you change them so that your graphical interface is full

horizontal_size = 125 
vertical_size = 60

# This parameter determines the size of each hexagon

edge_pixelnumber = 6

# These parameters determine what the map will look like (althrough it is randomly generated)

obstacle_density = .05
obstacle_size = 25

###########################################################################################

# From here, nothing is meant to be modified for a casual use.
# However, you can still have a look at the code and change it as you like !

###########################################################################################

#Imported modules

import numpy as np
from tkinter import *
import random
import time

#Global variables

pi = np.pi
sin = int(edge_pixelnumber / 2)
cos = int(edge_pixelnumber * np.sqrt(3)/2)

hexagon_matrix = [[0 for k in range(vertical_size)] for k in range(horizontal_size)]
start_position = (0, 0)
objective = (0, 0)

#Auxilary functions that will be used later

#Copy from a list
def Copy(L) :
    C = []
    for x in L :
        C.append(x)
    return C

#Copy from a dictionnary
def CopyD(D) :
    C = C = {}
    for x in D :
        C[x] = D[x]
    return C

#sign of a number
def sgn(x) :
    if x == 0 :
        return 0
    elif x > 0 :
        return 1
    else :
        return -1

def pos(x) :
    return x if x > 0 else 0

def fst(couple) :
    (a, b) = couple
    return a

def snd(couple) :
    (a, b) = couple
    return b

def fst3(triplet) :
    (a, b, c) = triplet
    return a

def snd3(triplet) :
    (a, b, c) = triplet
    return b

def trd(triplet) :
    (a, b, c) = triplet
    return c

#returns the indexes of hexagons that are neighbour from the one which indexes are a and b in the matrix
def surroundings(a, b) :
    results = [(a-1, b), (a+1, b), (a, b-1), (a, b+1)]
    if b%2 == 0 :
        results.append((a-1, b+1))
        results.append((a-1, b-1))
    else :
        results.append((a+1, b+1))
        results.append((a+1, b-1))
    results_ = Copy(results)
    for r in results_ :
        if fst(r) < 0 or snd(r) < 0 :
            results.remove(r)
        elif fst(r) > horizontal_size-1 or snd(r) > vertical_size-1 :
            results.remove(r)
    return results

#returns the neighbours that are not obstacles
def accessible_neighbours(a, b) :
    S = surroundings(a, b)
    A = []
    for s in S :
        (i, j) = s
        if hexagon_matrix[i][j] != 1 :
            A.append((i, j))
    return A

#returns the distance between two hexagons (provided that there is no obstacle between them)
def getDistances(pos1, pos2) :
    if pos1 == pos2 :
        return 0, 0, 0
    (a1, b1) = pos1
    (a2, b2) = pos2
    d = abs(b2 - b1)
    d += pos(abs(a2-a1) - d//2)
    if a2 > a1 and (b2-b1)%2 == 1 and abs(b2-b1) <= 2*abs(a2-a1) :
        d -= 1
    if a1%2 == 0 and b1%2 == 0 and b2%2 == 1 :
        d += sgn(a2-a1)
        if abs(b2-b1) >= 2*abs(a2-a1) :
            d += sgn(a1-a2)
    return (a2 - a1), (b2 - b1), d


#Correspondance between the matrix and the graphical interface
    
def get_position(a, b) :
    return 2 *  cos * (1 + a + (b%2)/2), 3/2 * edge_pixelnumber * (1 + b)

def get_corners(x, y) :
    return (x, y - edge_pixelnumber), (x - cos, y - sin), (x - cos, y + sin), (x, y + edge_pixelnumber), (x + cos, y + sin), (x + cos, y - sin), (x, y - edge_pixelnumber)

#returns the neighbour hexagons that are already obstacles
def infected_surrounding(a, b) :
    S = surroundings(a, b)
    for (i, j) in S :
        if hexagon_matrix[i][j] == 1 :
            return True
    return False

#random generation of obstacles onto the given map
def random_obstacle_generation(p, m) :
    global hexagon_matrix
    hexagon_matrix = [[0 for k in range(vertical_size)] for k in range(horizontal_size)]
    cases_list = []
    for a in range(horizontal_size) :
        for b in range(vertical_size) :
            cases_list.append((a, b))
    while cases_list != [] :
        l = len(cases_list)
        ind = random.randint(0, l-1)
        (a, b) = cases_list[ind]
        if infected_surrounding(a, b) :
            if random.random() < m*p :
                hexagon_matrix[a][b] = 1
        else :
            if random.random() < p :
                hexagon_matrix[a][b] = 1
        cases_list.pop(ind)
    return hexagon_matrix

#random choice of a starting position among the non-obstacle hexagons
def random_starting_position() :
    global hexagon_matrix
    global start_position
    cases_list = []
    for a in range(horizontal_size) :
        for b in range(vertical_size) :
            if hexagon_matrix[a][b] == 0 :
                cases_list.append((a, b))
    ind = random.randint(0, len(cases_list))
    (a, b) = cases_list[ind]
    hexagon_matrix[a][b] = 2
    start_position = (a, b)

#random choice of an objective position among the non-obstacle hexagons
def random_objective() :
    global hexagon_matrix
    global objective
    cases_list = []
    for a in range(horizontal_size) :
        for b in range(vertical_size) :
            if hexagon_matrix[a][b] == 0 :
                cases_list.append((a, b))
    ind = random.randint(0, len(cases_list))
    (a, b) = cases_list[ind]
    hexagon_matrix[a][b] = 42
    objective = (a, b)
        
#creation of the graphical interface from the global variable matrix
def create_interface() :
    global horizontal_size
    global vertical_size
    global edge_pixelnumber
    interface = Tk()
    canvas = Canvas(interface, width=window_width, height=window_height, background="white")
    for i in range(horizontal_size) :
        for j in range(vertical_size) :
            x, y = get_position(i, j)
            canvas.create_line((x, y - edge_pixelnumber), (x - cos, y - sin), (x - cos, y + sin), (x, y + edge_pixelnumber), (x + cos, y + sin), (x + cos, y - sin), (x, y - edge_pixelnumber))
    for i in range(horizontal_size) :
        for j in range(vertical_size) :
            if hexagon_matrix[i][j] == 1 :
                color_hexagon(canvas, i, j, 'black')
    (i, j) = start_position
    color_hexagon(canvas, i, j, 'blue')
    (i, j) = objective
    color_hexagon(canvas, i, j, 'red')              
    canvas.grid()
    interface.mainloop()

def color_hexagon(canvas, a, b, color) :
    x, y = get_position(a, b)
    canvas.create_polygon(get_corners(x, y), fill = color)


def color_path(path) :
    global horizontal_size
    global vertical_size
    global edge_pixelnumber
    interface = Tk()
    canvas = Canvas(interface, width=window_width, height=window_height, background="white")
    for i in range(horizontal_size) :
        for j in range(vertical_size) :
            x, y = get_position(i, j)
            canvas.create_line((x, y - edge_pixelnumber), (x - cos, y - sin), (x - cos, y + sin), (x, y + edge_pixelnumber), (x + cos, y + sin), (x + cos, y - sin), (x, y - edge_pixelnumber))
    for i in range(horizontal_size) :
        for j in range(vertical_size) :
            if hexagon_matrix[i][j] == 1 :
                color_hexagon(canvas, i, j, 'black')
    (i, j) = start_position
    color_hexagon(canvas, i, j, 'blue')
    (i, j) = objective
    color_hexagon(canvas, i, j, 'red')
    if path != 42 :
        path.pop(0)
        for ((x, y)) in path :
            color_hexagon(canvas, x, y, 'green')
    canvas.grid()
    interface.mainloop()

#auxilary function to the A-star algorithm
def auxAstar(current_position, road_map, explored, to_explore, start_position, objective, deviation_max) :
    ((a, b)) = current_position
    A = accessible_neighbours(a, b)
    for ((i, j)) in A :
        if ((i, j)) == objective :
            road_map[objective] = current_position
            road = [objective]
            prec = current_position
            while prec != start_position :
                road.append(prec)
                prec = fst3(road_map[prec])
            return len(road), road, road_map, explored, to_explore
        if ((i, j)) not in explored and ((i, j)) not in to_explore :
            distance = trd(road_map[current_position]) + 1
            deviation = snd3(road_map[current_position]) + trd(getDistances((i, j), objective)) - trd(getDistances((a ,b), objective)) + 1
            if (i, j) in road_map :
                if trd(road_map[((i, j))]) > distance :
                    road_map[((i, j))] = current_position, deviation, distance
            else :
                road_map[((i, j))] = current_position, deviation, distance
            if snd3(road_map[(i, j)]) <= deviation_max :
                to_explore.append((i, j))
    if to_explore == [] :
        return 420, [42], road_map, explored, to_explore
    
    return auxAstar(to_explore.pop(0), road_map, explored + [current_position], to_explore, start_position, objective, deviation_max)

#A-star algorithm
def Astar(start_position, objective) :
    road_map = {start_position: ((42, 42), 0, 0)}
    deviation_max = 0
    current_position = start_position
    to_explore = [start_position]
    explored = [start_position]
    while True :
        length, road, road_map, explored, to_explore = auxAstar(current_position, road_map, explored, to_explore, start_position, objective, deviation_max)
        if length != 420 :
            return (length, road)
        explored = sorted(explored, key=lambda x: trd(road_map[x]), reverse=False)
        for ((a, b)) in explored :
            if snd3(road_map[(a, b)]) == deviation_max :
                for nei in accessible_neighbours(a, b) :
                    if nei not in to_explore :
                        to_explore.append(nei)
        if to_explore != [] :
            current_position = to_explore.pop(0)
        else :
            print('{} is not accessible from {}'.format(objective, start_position))
            break
        deviation_max += 1

#initialisation of the matrix and its corresponding interface
def init() :
    global obstacle_density
    global obstacle_size
    random_obstacle_generation(.05, 25)
    random_starting_position()
    random_objective()
    create_interface()
    return start_position, objective

def useAstar() :
    return Astar(start_position, objective)

init()

t0 = time.time()
result = useAstar()
t = time.time() - t0
print("Temps d'exécution de l'algorithme : " + str(t) + " s")
if result == None or snd(result) == 420 :
    print('''The objective can't be reached.''')
else :
    print("Distance between the starting position and the objective : " + str(fst(result)))
    path = snd(result)
    color_path(path)
