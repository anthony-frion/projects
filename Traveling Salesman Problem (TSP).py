'''
Hi, thanks for paying attention to my work !

This file enables to compare TSP heuristics on random heuristics by calculating
the average value of the soutions they give with a few given parameters.

As it requires a lot of time to calculate significative average values,
I have chosen to restrict myself to the cubic-complexity heuristics.
One could hardly restrict to the quadratic one as there are not many of those
except for the nearest neighbour.

This restriction justifies that there are no genetic algorithms on large random
populations. Some might come up later though ;)

You can just update the following variables to make simple tests, and you can
also have a look at the code to understand how things are computed and try
to make some improvements :)
'''

##############################################################################

# Global variables : update them to make different tests

# The number of nodes of the generated problems
numberOfNodes = 20

# The maximum distance between two nodes in the generated problems
maxDistance = 30

# The algorithm that is being tested. First execute the code so that your IDE 
# knows all the heuristics, then uncomment one of the following lines to choose one.
# You can also pick up a heuristic among all those that are defined in this file.

#TestedHeuristic = FourGenAlgorithm
#TestedHeuristic = ThreeGenAlgorithm
#TestedHeuristic = Reproduction_BNN_BH
#TestedHuristic = NearestNeighbour

# Type of test realized : uncomment the line of interest

#test = 'time'
test = 'performance'

# The number of iterations used for the test.
# You could have to make it smaller when the number of nodes increases.
N = 1000

##############################################################################

# Imported modules

from random import randint
from math import inf
import matplotlib.pyplot as pl
import time

#Useful auxilary functions

def fst(couple) :
    a, b = couple
    return a

def snd(couple) :
    a, b = couple
    return b

def SeeMatrix(M) :
    for L in M :
        print(L)

def Copy(M) :
    C = []
    for L in M :
        C.append([])
        for j in L :
            C[-1].append(j)
    return C

# Returns the cycle permutation of L such that the first element is the element number n-k of L
def Translation(L, k) :
    T = []
    if k<0 :
        k += len(L)
    for i in range(len(L)) :
        T.append(L[i-k])
    return T

# Returns the integers between 0 and n-1 that do not belong to 'walk'
def comp(walk, n) :
    L = []
    for k in range(n) :
        if k not in walk :
            L.append(k)
    return L

# Calculates the length of a walk according to the distances matrix 'M'.
def distance(M, walk) :
    distance_traveled = 0
    for k in range(len(walk)-1) :
        distance_traveled += M[walk[k]][walk[k+1]]
    return distance_traveled

# 
def random_from_L(current, L, n) :
    cands = []
    if len(L) == 1 :
        return L[0]
    for k in range(max(n, len(L))) :
        i = randint(0, len(L) -1)
        cands.append(i)
    best = cands[0]
    best_dist = M[current][L[best]]
    for i in cands :
        if M[current][L[i]] < best_dist :
            best = i
            best_dist = M[current][L[i]]
    return L.pop(best)

# Random generation of a matrix containing all the information needed for a TSP.
# n is the number of nodes and m is the maximal distance between 2 nodes
def GenerateProblem(n, m) :
	M = [[0 for k in range (n)] for k in range(n)]
	for i in range(n) :
		for j in range(i) :
			M[i][j] = randint(1, m)
			M[j][i] = M[i][j]
	return M

# The Bruteforce-type algorithms which always return the best solution
    
def auxBF(M, walk, best_walk, distance, best_distance) :
    n = len(M)
    if len(walk) == n :
        distance += M[walk[-1]][0]
        walk.append(0)
        if distance < best_distance :
            best_distance = distance
            best_walk = walk
    else :
        for i in range(n) :
            if i not in walk :
                n_walk = walk + [i]
                n_distance = distance + M[n_walk[-2]][n_walk[-1]]
                c_distance, c_walk = auxBF(M, n_walk, best_walk, n_distance, best_distance)
                if c_distance < best_distance :
                    best_distance = c_distance
                    best_walk = c_walk
    return best_distance, best_walk

def BruteForce(M) :
    best_walk = [k for k in range(len(M))] + [0]
    best_distance = 0
    for k in range(len(M) - 1) :
        best_distance += M[k][k+1]
    best_distance+= M[-1][0]
    return auxBF(M, [0], best_walk, 0, best_distance)

# The backtracking algorithm : also has exponential complexity, but often faster than the naive one

def auxBT(M, walk, best_walk, distance, best_distance) :
    n = len(M)
    if distance >= best_distance :
        return best_distance, best_walk
    elif len(walk) == n :
        distance += M[walk[-1]][0]
        walk.append(0)
        if distance < best_distance :
            best_distance = distance
            best_walk = walk
    else :
        for i in range(n) :
            if i not in walk :
                n_walk = walk + [i]
                n_distance = distance + M[n_walk[-2]][n_walk[-1]]
                c_distance, c_walk = auxBT(M, n_walk, best_walk, n_distance, best_distance)
                if c_distance < best_distance :
                    best_distance = c_distance
                    best_walk = c_walk
    return best_distance, best_walk

def BackTracking(M) :
    best_walk = [k for k in range(len(M))] + [0]
    best_distance = distance(M, best_walk)
    return auxBT(M, [0], best_walk, 0, best_distance)

# Backtracking using the Nearest Neighbour heuristic as a first best walk
def BackTracking_NN(M) :
    best_distance, best_walk = NearestNeighbour(M)
    return auxBT(M, [0], best_walk, 0, best_distance)

# Backtracking using the 03GA heuristic as a first best walk
def BackTracking_O3GA(M) :
    best_distance, best_walk = LocalSearch_3GA(M)
    return auxBT(M, [0], best_walk, 0, best_distance)

##################################################################################

#Nearest-neighbour and insertion heuristics

# The very common Nearest Neighbour basic heuristic.
# Its complexity is quadratic as it compares in average n/2 distances at each step of the solution building
def NearestNeighbour(M) :
    C = Copy(M)
    n = len(C)
    walk = [0]
    for k in range(n-1) :
    	i = walk[-1]
    	cand = 0
    	d_cand = inf
    	for j in range(n) :
    		if C[i][j] != 0 and C[i][j] < d_cand :
    			d_cand = C[i][j]
    			cand = j
    	walk.append(cand)
    	for k in range(n) :
    		C[k][i] = 0
    walk.append(0)
    distance_traveled = distance(M, walk)
    return (distance_traveled, walk)

# Calculates the cheapest addition of k nodes to an incomplete walk.
def best_k_more_moves(M, k, walk) :
    Visited = [False for k in range(len(M))]
    for i in walk :
        Visited[i] = True
    return aux_k_more_moves(M, k, [walk[-1]], [walk[-1]], 0, inf, Visited)

def aux_k_more_moves(M, k, further_walk, best_further_walk, distance, best_distance, Visited) :
    n = len(M)
    if len(further_walk) == k + 1 :
        if distance < best_distance :
            best_distance = distance
            best_further_walk = further_walk
    else :
        for i in range(n) :
            if not Visited[i] :
                n_further_walk = further_walk + [i]
                n_distance = distance + M[n_further_walk[-2]][n_further_walk[-1]]
                n_Visited = Visited
                n_Visited[i] = True
                c_distance, c_walk = aux_k_more_moves(M, k, n_further_walk, best_further_walk, n_distance, best_distance, n_Visited)
                n_Visited[i] = False
                Visited[i] = False
                if c_distance < best_distance :
                    best_distance = c_distance
                    best_further_walk = c_walk
    return best_distance, best_further_walk

# A Nearest Neighbour heuristic that looks for the shortest k-nodes path instead of just the closest node
def k_NearestNeighbour(M, k) :
    C = Copy(M)
    n = len(C)
    distance_traveled = 0
    walk = [0]
    while len(walk) <= n - k :
        further_distance, further_walk = best_k_more_moves(M, k, walk)
        walk = walk + further_walk[1:]
    remaining = comp(walk, n)
    while len(walk) < n :
        i = walk[-1]
        cand = 0
        d_cand = inf
        for j in remaining :
            if C[i][j] != 0 and C[i][j] < d_cand :
                d_cand = C[i][j]
                cand = j
        walk.append(cand)
        remaining.remove(cand)
    walk.append(0)
    distance_traveled = distance(M, walk)
    return distance_traveled, walk

def TwoNearestNeighbour(M) :
    return k_NearestNeighbour(M, 2)

def ThreeNearestNeighbour(M) :
    return k_NearestNeighbour(M,3)

# Calculates the 'nearest neighbour' solution when seeing the node k as the starting point
# Then gets the node 0 back to the status as starting position with a translation of the solution
def NearestNeighbourFrom(M, k) :
    C = Copy(M)
    n = len(C)
    pseudo_walk = [k]
    for k in range(n-1) :
    	i = pseudo_walk[-1]
    	cand = 0
    	d_cand = inf
    	for j in range(n) :
    		if C[i][j] != 0 and C[i][j] < d_cand :
    			d_cand = C[i][j]
    			cand = j
    	pseudo_walk.append(cand)
    	for k in range(n) :
    		C[k][i] = 0
    zero_index = 0
    while pseudo_walk[zero_index] != 0 :
        zero_index += 1
    walk = Translation(pseudo_walk, - zero_index) + [0]
    distance_traveled = distance(M, walk)
    return distance_traveled, walk

# Computes the Nearest Neighbour solutions from every position and returns the best one
# As it computes n times a quadratic algorithm, the global complexity is cubic
def BestNearestNeighbour(M) :
    n = len(M)
    cand = [k for k in range(n)] + [0]
    dist_cand = inf
    for k in range(n) :
        dist, walk = NearestNeighbourFrom(M, k)
        if dist < dist_cand :
            cand = walk
            dist_cand = dist
    return dist_cand, cand
        
# An auxilary function that looks for the best place to insert a node into a path that is being constructed
def Inser(i, walk, M) :
    cand = 1
    extra_dist_cand = inf
    for k in range(len(walk) - 1) :
        previous_dist = M[walk[k]][walk[k+1]]
        new_dist = M[walk[k]][i] + M[i][walk[k+1]]
        extra_dist = new_dist - previous_dist
        if extra_dist < extra_dist_cand :
            cand = k+1
            extra_dist_cand = extra_dist
    return (cand, extra_dist_cand)

# Inserts successively all the nodes using the previous function to created a heuristical path
def Insertion(M) :
    n = len(M)
    distance_traveled = 0
    walk = [0, 0]
    for i in range (1, n) :
        (cand, extra_dist_cand) = Inser(i, walk, M)
        walk.insert(cand, i)
        distance_traveled += extra_dist_cand
        
    return (distance_traveled, walk)

#Hybrids between nearest neighbour and insertion heuristics

# Behaves like the 'Nearest Neighbour' heuristic up to node 'num', then behaves like the 'Insertion' to insert 
def Hybrid(M, num) :
    C = Copy(M)
    n = len(C)
    distance_traveled = 0
    walk = [0]
    for k in range (num) :
        i = walk[-1]
        cand = 0
        d_cand = inf
        for j in range(n) :
                if C[i][j] != 0 and C[i][j] < d_cand :
                    d_cand = C[i][j]
                    cand = j
        walk.append(cand)
        distance_traveled += d_cand
        for k in range(n) :
            C[k][i] = 0
    distance_traveled += M[walk[-1]][0]
    walk.append(0)
    for i in range(n) :
        if i not in walk :
            cand, extra_dist_cand = Inser(i, walk, M)
            walk.insert(cand, i)
            distance_traveled += extra_dist_cand
    return (distance_traveled, walk)

# Inspired by the Insertion algorithm, but instead of inserting the nodes in a given order, it
# always looks for the best insertion among all remaining nodes, which makes its complexity cubic
def CubicInsertion(M) :
    n = len(M)
    distance_traveled = 0
    walk = [0, 0]
    for k in range(n-1) :
        icand = 0
        supercand = 0
        extra_dist_supercand = inf
        for i in range(n) :
            if i not in walk :
                (cand, extra_dist_cand) = Inser(i, walk, M)
                if extra_dist_cand < extra_dist_supercand :
                        supercand = cand
                        extra_dist_supercand = extra_dist_cand
                        icand = i
        walk.insert(supercand, icand)
        distance_traveled += extra_dist_supercand
    return (distance_traveled, walk)
           
# Same as the Hybrid algorithm but using the CubicInsertion instead of the classic Insertion             
def CubicHybrid(M, num) :
    C = Copy(M)
    n = len(C)
    distance_traveled = 0
    walk = [0]
    for k in range (num) :
        i = walk[-1]
        cand = 0
        d_cand = inf
        for j in range(n) :
                if C[i][j] != 0 and C[i][j] < d_cand :
                    d_cand = C[i][j]
                    cand = j
        walk.append(cand)
        distance_traveled += d_cand
        for k in range(n) :
            C[k][i] = 0
    distance_traveled += M[walk[-1]][0]
    walk.append(0)
    for k in range(n-1-num) :
        icand = 0
        supercand = 0
        extra_dist_supercand = inf
        for i in range(n) :
            if i not in walk :
                (cand, extra_dist_cand) = Inser(i, walk, M)
                if extra_dist_cand < extra_dist_supercand :
                        supercand = cand
                        extra_dist_supercand = extra_dist_cand
                        icand = i
        walk.insert(supercand, icand)
        distance_traveled += extra_dist_supercand
    return (distance_traveled, walk)

# Returns the best hybrid solution after computing Hybrid with every possible 'num' parameter
# The complexity is cubic as the Hybrid heuristic has a quadratic complexity
def BestHybrid(M) :
    n = len(M)
    cand = [k for k in range(n)] + [0]
    distance_cand = inf
    for num in range(n) :
        distance, walk = Hybrid(M, num)
        if distance < distance_cand :
            cand = walk
            distance_cand = distance
    return distance_cand, cand

#Reproduction Algorithms (get the child of two Heuristics)

'''
A classic reproduction algorithm
The first part of the child solution is the beginning of the walk1 up to the cut-point
Then the nodes that are after the cut-point in walk2 and still not present are placed
with respect to their order in walk2.
Finally the points that are still not placed in the child solution (i.e. those that
are after cut-point in walk1 and before cut-point in walk2) are placed in the end
of the son walk, with respect to their order in walk2
'''
def auxReproduction(walk1, walk2, cut_point, M) :
    n = len(walk1)
    first_part = walk1[:cut_point]
    child_walk = first_part
    for k in range(cut_point, n) :
        if walk2[k] not in first_part :
            child_walk.append(walk2[k])
    child_walk.append(0)
    for i in range(cut_point) :
        if walk2[i] not in child_walk :
            child_walk.append(walk2[i])
    dist = distance(M, child_walk)
    return dist, child_walk

'''
A special reproduction algorithm with one cut-point
The first part of the child solution is the beginning of the walk1 up to the cut-point
Then the nodes that are after the cut-point in walk2 and still not present are placed
with respect to their order in walk2.
Finally the points that are still not placed in the child solution (i.e. those that
are after cut-point in walk1 and before cut-point in walk2) are inserted in the
best possible place instead of just being placed in the end
'''
def auxBetterReproduction(walk1, walk2, cut_point, M) :
    n = len(walk1)
    first_part = walk1[:cut_point]
    child_walk = first_part
    for k in range(cut_point, n) :
        if walk2[k] not in first_part :
            child_walk.append(walk2[k])
    child_walk.append(0)
    for i in range(cut_point) :
        if walk2[i] not in child_walk :
            cand, extra_dist_cand = Inser(walk2[i], child_walk, M)
            child_walk.insert(cand, walk2[i])
    dist = distance(M, child_walk)
    return dist, child_walk


# Calculates a child solution using the 'auxBetterReproduction' with a random cut point
def RandomReproduction(walk1, walk2, M) :
    n = len(walk1)
    cut_point = randint(2, n-2)
    return auxBetterReproduction(walk1, walk2, cut_point, M)

# Gets the best child solution after calculating the reproduction with every possible cut point
def BestReproduction(walk1, walk2, M) :
    n = len(walk1)
    cand_dist = inf
    cand_walk = walk1
    for k in range(2, n-2) :
        dist, walk = auxReproduction(walk1, walk2, k, M)
        if dist < cand_dist :
            cand_dist = dist
            cand_walk = walk
    return cand_dist, cand_walk

# Gets the best child solution after calculating the better reproduction with every possible cut point
def BestBetterReproduction(walk1, walk2, M) :
    n = len(walk1)
    cand_dist = inf
    cand_walk = walk1
    for k in range(2, n-2) :
        dist, walk = auxBetterReproduction(walk1, walk2, k, M)
        if dist < cand_dist :
            cand_dist = dist
            cand_walk = walk
    return cand_dist, cand_walk

# Calculates the child solution of the solutions obtained by 2 different heuristics
# using a given reproduction algorithm on the problem characterized by the matrix M
def ReproductionHeuristic(M, Heuristic1, Heuristic2, ReproductionAlgorithm) :
    return ReproductionAlgorithm(snd(Heuristic1(M)), snd(Heuristic2(M)), M)

def Reproduction_NN_R(M) :
    return ReproductionHeuristic(M, NearestNeighbour, Insertion, BestBetterReproduction)

def Reproduction_NN_CR(M) :
    return ReproductionHeuristic(M, NearestNeighbour, Insertion, BestBetterReproduction)

def Reproduction_2NN_CR(M) :
    return ReproductionHeuristic(M, TwoNearestNeighbour, CubicInsertion, BestBetterReproduction)

def RReproduction_NN_R(M) :
    return ReproductionHeuristic(M, NearestNeighbour, Insertion, RandomReproduction)

def RBReproduction_NN_R(M) :
    return ReproductionHeuristic(M, NearestNeighbour, Insertion, RandomReproduction)

# A good heuristic, obtained from 4 'grandparents' initial heuristics that are being
# reproduced into 2 parents whose reproduction results in the final solution
def ThreeGenAlgorithm(M) :
    Parent1 = Reproduction_NN_R
    Parent2 = Reproduction_2NN_CR
    return ReproductionHeuristic(M, Parent1, Parent2, BestBetterReproduction)

# The reproduction of the BestNearestNeighbour and the BestHybrid heuristic.
# As those two heuristics are very efficient, this algorithm has better performances than the ThreeGenAlgorithm
def Reproduction_BNN_BH(M) :
    return ReproductionHeuristic(M, BestNearestNeighbour, BestHybrid, BestBetterReproduction)

# The best cubic heuristic I have found so far. It is the reproduction of the
# ThreeeGenAlgorithm and the Reproduction_BNN_BH.
def FourGenAlgorithm(M) :
    Parent1 = Reproduction_BNN_BH
    Parent2 = ThreeGenAlgorithm
    return ReproductionHeuristic(M, Parent1, Parent2, BestBetterReproduction)

#################################################################################

# Functions to optimize an already computed solution using local search
# Have a quadratric complexity

'''
Computes a Heuristic, then tries to permute every couple of nodes in the found solution
and to re-insert every node into every possible position. When an element of the neighbourhood 
has the same cost then it becomes the new solution with a probability 0.5
'''
def LocalSearch(M, Heuristic, nb) :
    dist, walk = Heuristic(M)
    n = len(M)
    for k in range(nb) :
        for i in range(1, n) :
            for j in range(i+1, n) :
                walk = swapOperator(M, walk, i, j)
                walk = insertionOperator(M, walk, i, j)
    new_dist = distance(M, walk)
    return new_dist, walk

def swapOperator(M, walk, i, j) :
    if abs(i-j) >= 2 :
        current_term = M[walk[i-1]][walk[i]] + M[walk[i]][walk[i+1]] + M[walk[j-1]][walk[j]] + M[walk[j]][walk[j+1]]
        cand_term = M[walk[i-1]][walk[j]] + M[walk[j]][walk[i+1]] + M[walk[j-1]][walk[i]] + M[walk[i]][walk[j+1]]
        difference = current_term - cand_term
    else :
            difference = M[walk[i-1]][walk[i]] + M[walk[j]][walk[j+1]] - M[walk[i-1]][walk[j]] - M[walk[i]][walk[j+1]]
    if difference >= 0 :
        if difference > 0 or randint(0, 1) == 1 :
            walk[i], walk[j] = walk[j], walk[i]
    return walk

def insertionOperator(M, walk, i, j) :
    difference = -1
    if abs(i-j) >= 2 :
        current_term = M[walk[i]][walk[i+1]] + M[walk[j-1]][walk[j]] + M[walk[j]][walk[j+1]]
        cand_term = M[walk[i]][walk[j]] + M[walk[j]][walk[i+1]] + M[walk[j-1]][walk[j+1]]
        difference = current_term - cand_term
    if difference >= 0 :
        if difference > 0 or randint(0, 1) == 1 :
            j_node = walk[j]
            for k in range(j, i+1, -1) :
                walk[k] = walk[k-1]
                walk[i+1] = j_node
    return walk

def LocalSearch_NN(M) :
    return LocalSearch(M, NearestNeighbour, 2)

def LocalSearch_3GA(M) :
    return LocalSearch(M, ThreeGenAlgorithm, 2)

def LocalSearch_4GA(M) :
    return LocalSearch(M, FourGenAlgorithm, 2)
     

# Calculates how effective a heuristic is in average with given parameters n and m and compares it to the Nearest Neighbour
def TestPerformance(n, m, N, Heuristic) :
    average = 0
    NN_average = 0
    for k in range(N) :
        M = GenerateProblem(n, m)
        (distance_traveled, walk) = Heuristic(M)
        average += distance_traveled
        (distance_traveled, walk) = NearestNeighbour(M)
        NN_average += distance_traveled
    average /= N
    NN_average /= N
    print("The average distance obtained is " + str(average))
    print("This represents " + str(average/NN_average) + " times the average time obtained with the Nearest Neighbour")
    return average, average/NN_average
    
# Same as the previous one but for hybrid heuristics, which have an extra-argument 'num'
def TestHybridHeuristic(n, m, N, Heuristic, num) :
    average = 0
    for k in range(N) :
        M = GenerateProblem(n, m)
        (distance_traveled, walk) = Heuristic(M, num)
        average += distance_traveled
    average /= N
    return average

# Used to visualise how effective hybrid heuristics are according to their parameter 'num'
def MakeGraph(n, m, N, Heuristic) :
    X = [k for k in range(n)]
    Y = []
    for k in range(n) :
        Y.append(TestHybridHeuristic(n, m, N, Heuristic, k))
    pl.plot(X,Y)
    pl.xlabel('number of choices made with the greedy algorithm')
    pl.ylabel('average distance traveled')
    pl.title('Parameters : n =' + str(n) + ', m =' + str(m))
    pl.show()
    return (min(Y)/Y[-1])

#Algorithm that returns the average time needed to execute an algorithm with parameters n and m
def TestTime(n, m, N, Heuristic) :
    average = 0
    for k in range(N) :
        M = GenerateProblem(n, m)
        t = time.time()
        (distance_traveled, walk) = Heuristic(M)
        average += time.time() - t
    average /= N
    print('The average computing time is ' + str(average) + ' seconds.')
    return average

try :
    if test == 'time' :
        TestTime(numberOfNodes, maxDistance, N, TestedHeuristic)
    elif test == 'performance' :
        TestPerformance(numberOfNodes, maxDistance, N, TestedHeuristic)
except :
    print("TestedHeuristic is not defined yet. Uncomment one of the lines 34 to 37 to define it and you will be able to start your tests ;)")
    

    
    
    
    
    
    
    
    
    