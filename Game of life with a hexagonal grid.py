#Hello, and thank you for paying attention to my work !

# To use the code as it is, you just have to execute it, then a 
# graphical animation will show you how the cells (represented in black)
# evolve. You may also want to change the following parameters.

# Enjoy !


# Here are some parameters that you could want to change according to the size of your screen.

window_width = 1300 # The number of pixels corresponding to the width of the graphical interface
window_height = 600 # The number of pixels corresponding to the height of the graphical interface

# The number of hexagons that you want your grid to contain, horizontally and vertically
# You can increase them if your CPU is powerful enough

horizontal_size = 50
vertical_size = 30

# This parameter determines the size of each hexagon

edge_pixelnumber = 20

# These parameters determine where the cells will initially be located 
# (althrough it is a stochastic process)

cell_number_coeff = 0.1 
cell_concentration_coeff = 20

# Parameters that determine how life will evolve on the grid
# I found that 3-1-4 was the most interesting combination,
# but you can also try another one to see how it evolves :)

# The birth coeff is the number at which a cell will be born
# in an empty space if it has this exact number of living neighbours
# death_coeff_1 is the number of living neighbours under which a cell will die
# death_coeff_2 is the number of living neighbours over which a cell will die

birth_coeff = 3
death_coeff_1 = 1
death_coeff_2 = 4

#Finally, you can choose what you want to see.

see_initial_config = False # Shows the initial grid if true

see_game_of_life = True # Shows the evolution of the grid if true

number_of_iterations = 50 # The number of iterations of the game

###########################################################################################

# From here, nothing is meant to be modified for a casual use.
# However, you can still have a look at the code and change it as you like !

###########################################################################################

#Imported modules

import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation
import random
import time as t

#Global variables

pi = np.pi
sin = int(edge_pixelnumber / 2)
cos = int(edge_pixelnumber * np.sqrt(3)/2)

hexagon_matrix = [[0 for k in range(vertical_size)] for k in range(horizontal_size)]
alive_cells = []


# Some basic auxilary functions that will be helpful later

def fst(couple) :
    (a, b) = couple
    return a

def snd(couple) :
    (a, b) = couple
    return b

def Copy(L) :
    C = []
    for x in L :
        C.append(x)
    return C

#Correspondance between the matrix and the graphical interface

def get_position(a, b) :
    return 2 *  cos * (1 + a + (b%2)/2), 3/2 * edge_pixelnumber * (1 + b)

def get_corners(x, y) :
    return (x, y - edge_pixelnumber), (x - cos, y - sin), (x - cos, y + sin), (x, y + edge_pixelnumber), (x + cos, y + sin), (x + cos, y - sin), (x, y - edge_pixelnumber)

# Returns all the neighbour hexagons from the one withs indexes are a and b
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

# Returns the neighbour hexagons that already contain a cell
def infected_surrounding(a, b) :
    S = surroundings(a, b)
    for (i, j) in S :
        if hexagon_matrix[i][j] == 1 :
            return True
    return False

# Generates a random initial configuration for the matrix
# p is a basic probability for a hexagon to contain a cell
# m is a multiplier for the probability when the hexagon already has a neighbor cell
def random_initial_config(p, m) :
    global hexagon_matrix
    global alive_cells
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
            if random.random() < p*m*len(cases_list)/(vertical_size*horizontal_size) :
                hexagon_matrix[a][b] = 1
                alive_cells.append((a, b))
        else :
            if random.random() < p*len(cases_list)/(vertical_size*horizontal_size) :
                hexagon_matrix[a][b] = 1
                alive_cells.append((a, b))
        cases_list.pop(ind)

# Creates a cell in the animation
def create_cell(a, b) :
    x, y = get_position(a, b)
    corners = get_corners(x, y)
    axes.fill([fst(corner) for corner in corners], [snd(corner) for corner in corners], "black")

# Kills a cell in the animation
def kill_cell(a, b) :
    x, y = get_position(a, b)
    corners = get_corners(x, y)
    axes.fill([fst(corner) for corner in corners], [snd(corner) for corner in corners], "white")

# Updates the matrix containing all the cells (1 iteration of the game)
def update_matrix(i) :
    t0 = t.time()
    global hexagon_matrix
    global alive_cells
    if alive_cells == [] :
        return False
    new_alive_cells = []
    next_to_alive_cells = {}
    for cell in alive_cells :
        neighbours = surroundings(fst(cell), snd(cell))
        alive_neighbours = 0
        for nei in neighbours :
            a, b = nei
            if hexagon_matrix[a][b] == 0 :
                if (a, b) in next_to_alive_cells :
                    next_to_alive_cells[(a, b)] += 1
                else :
                    next_to_alive_cells[(a, b)] = 1
            else :
                alive_neighbours += 1
        if alive_neighbours <= death_coeff_1 or alive_neighbours >= death_coeff_2 :
            kill_cell(fst(cell), snd(cell))
        else :
            new_alive_cells.append(cell)
    print("There are {} alive cells.".format(len(alive_cells)))
    for cell in next_to_alive_cells :
        if next_to_alive_cells[cell] == birth_coeff :
            create_cell(fst(cell), snd(cell))
            new_alive_cells.append(cell)
    hexagon_matrix = [[0 for k in range(vertical_size)] for k in range(horizontal_size)]
    alive_cells = new_alive_cells
    for cell in alive_cells :
        hexagon_matrix[fst(cell)][snd(cell)] = 1
    print(t.time() - t0)
    
#initiates the hexagon matrix according to the list of alive cells
def initiate_matrix() :
    global hexagon_matrix
    global alive_cells
    for cell in alive_cells :
        hexagon_matrix[fst(cell)][snd(cell)] = 1

#Initiates the graphical animation according to the list of alive cells
def initiate_graphics() :
    global figure
    global axes
    figure = pl.figure()
    axes = pl.axes()
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.imshow([[(255, 255, 255) for k in range(1300)] for k in range(650)])
    for cell in alive_cells :
        create_cell(fst(cell), snd(cell))

def story(N) :
    initiate_matrix()
    initiate_graphics()
    anim = animation.FuncAnimation(figure, update_matrix, frames=N, interval=1000)
    pl.show()
  
random_initial_config(cell_number_coeff, cell_concentration_coeff)
    
if see_initial_config :
    figure = pl.figure()
    axes = pl.axes()
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.imshow([[(255, 255, 255) for k in range(1300)] for k in range(650)])

if see_game_of_life :
    story(number_of_iterations)  






