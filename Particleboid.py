import pygame
import math
import random

import numpy as np
import matplotlib as plt
import sklearn
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
import networkx as nx

import tkinter as tk
from tkinter import filedialog
import pickle


class POINT:
    def __init__(self,pos,vel,nodeColor):
        self.pos = Vec2(pos)
        self.vel = Vec2(vel)
        self.nodeColor = nodeColor

class Vec2():
    # Trying something for fun. I wanted to replicate the functionality of the vec2 class found in GLSL, I could have just used numpy
    def __init__(self, x, y = None):
        if isinstance(x, (tuple, list, Vec2)):
            if len(x) != 2:
                raise ValueError("Input tuple or list must have exactly 2 elements")
            self.x, self.y = x
        else:
            if y == None:
                raise ValueError("Vec2() takes 2 values or 1 list/tuple but only 1 was value was given")
            self.x = x
            self.y = y

    # Funky way of making the operator stuff smaller, I would not do this in production code, but this is for fun.
    def selector(self,b,op):
        return Vec2(op(self.x,b.x),op(self.y,b.y)) if type(b) == Vec2 else Vec2(op(self.x,b),op(self.y,b)) 
    
    def __add__(self,b):
        return self.selector(b,lambda x,y: x + y)

    def __sub__(self,b):
        return self.selector(b,lambda x,y: x - y) 
    
    def __mul__(self,b):
        return self.selector(b,lambda x,y: x * y)  
    
    def __truediv__(self,b):
        return self.selector(b,lambda x,y: x / y)  
    
    def __eq__(self,b):
        return type(b) == Vec2 and self.x == b.x and self.y == b.y

    # Allows for list like indexing, example Vec2(2,3)[0], returns 2
    def __getitem__(self, index):
        return self.y if index else self.x
    
    def __str__(self):
        return "[" + str(self.x) + "," + str(self.y) + "]"
    
    def __len__(self):
        return 2
    
    def __iter__(self):
        return iter((self.x,self.y))
    
    def length(self):
        return math.sqrt(self.x**2+self.y**2)
    
    def unit(self):
        return self/self.length()
    

pygame.init()

WIDTH = 1000
HEIGHT = 1000
screen = pygame.display.set_mode((WIDTH,HEIGHT))

# File types
filetypes = [('POS files', '*.pos')]

def generate_colors(n):
    # Create an array of hues evenly spaced in [0, 1]
    hues = np.linspace(0, 1, n, endpoint=False)

    # Set saturation and value to be high and low respectively
    saturation = 0.3
    value = 0.9

    # Create an array of colors in HSV format
    hsv_colors = np.zeros((n, 3))
    hsv_colors[:, 0] = hues
    hsv_colors[:, 1] = saturation
    hsv_colors[:, 2] = value

    # Convert HSV colors to RGB format
    rgb_colors = plt.colors.hsv_to_rgb(hsv_colors)*255

    return rgb_colors

# non-symmetric dataset
relation =[[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]
propagation_cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')

clusterIDs = propagation_cluster.fit_predict(relation)
print('Clusters for Sharpstone network:')
print(clusterIDs)

colors = generate_colors(max(clusterIDs)+1)

population = [POINT(Vec2(random.randint(0,1000),random.randint(0,1000)),Vec2(0,0),colors[clusterIDs[x]]) for x in range(len(relation))]

avoidanceForce = 10000
attractionForce = 0.002
groupAttraction = 2

def RenderCircle():
    font = pygame.font.Font(None, 24)  # Choose the font for the text, None is the default font, 24 is the size
    for i,point in enumerate(population):
        pygame.draw.circle(screen,point.nodeColor,point.pos,20)
        text = font.render(str(i+1), True, (0, 0, 0))  # Render the text. True is for anti-aliasing, (0, 0, 0) is the color (black)
        text_rect = text.get_rect(center=point.pos)  # Get the rectangle that encloses the text
        screen.blit(text, text_rect)  # Draw the text on the screen at the given position




def DrawEdges():
    for i in range(len(relation)):
        for j in range(len(relation[i])):

            if i!=j and relation[i][j] == 1:
                color =  (50,50,50)
                
                start_pos = population[j].pos
                end_pos = population[i].pos
                pygame.draw.line(screen, color, start_pos, end_pos, 2)
                
                # calculate the direction of the arrow
                difference = end_pos - start_pos
                direction = difference.unit()                
                rot90direction = Vec2(direction[1],-direction[0])
                # calculate the positions of the two points forming the arrowhead
                arrowhead_pos1 = end_pos - (direction * 30) + (rot90direction * 5)
                arrowhead_pos2 = end_pos - (direction * 30) - (rot90direction * 5)
                
                # draw the arrowhead
                pygame.draw.polygon(screen, color, [end_pos - (direction * 20), arrowhead_pos1, arrowhead_pos2])

grabbedID = -1

simtoggle = True

while(True):

    screen.fill((255,255,255))
    DrawEdges()
    RenderCircle()

    if simtoggle:
        avgPos = Vec2(0,0)
        for i in range(len(population)):
            avgPos += population[i].pos
            for j in range(i,len(population)):
                if i!=j:
                    difference = population[i].pos-population[j].pos
                    distance = difference.length()
                    Scaling = avoidanceForce/(distance*distance)
                    population[i].vel += difference.unit()*Scaling
                    population[j].vel -= difference.unit()*Scaling
                    if relation[i][j]>0 or clusterIDs[i] == clusterIDs[j]:
                        groupScaling = groupAttraction if (clusterIDs[i] == clusterIDs[j]) else 1
                        Scaling = groupAttraction*(attractionForce)*distance
                        population[i].vel -= difference.unit()*Scaling
                        population[j].vel += difference.unit()*Scaling



        avgPos/=len(population)

        for i in range(len(population)):
            population[i].pos -= avgPos - Vec2(500,500)
            population[i].pos += population[i].vel*0.01
            population[i].vel *= 0.99

    if grabbedID >=0:
        clickPos = pygame.mouse.get_pos()
        population[grabbedID].pos = Vec2(clickPos[0],clickPos[1])
        population[grabbedID].vel = Vec2(0,0)


    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            clickPos = pygame.mouse.get_pos()
            for i in range(len(population)):
                if (population[i].pos - Vec2(clickPos[0],clickPos[1])).length()<20:
                    grabbedID = i
            
        elif event.type == pygame.MOUSEBUTTONUP:
            grabbedID = -1

        elif event.type == pygame.KEYDOWN:
                # detect if ctrl+s or ctrl+o was used:
                keys = pygame.key.get_pressed()
                if event.key == pygame.K_SPACE:
                    simtoggle = not simtoggle
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:  # Check if either Ctrl key is pressed
                    if event.key == pygame.K_s:  # Check if 's' is pressed
                        file_path = filedialog.asksaveasfilename(filetypes=filetypes,defaultextension='.pos')
                        positions = [point.pos for point in population]
                        with open(file_path, 'wb') as f:
                            pickle.dump(positions, f)

                    elif event.key == pygame.K_o:  # Check if 's' is pressed
                        file_path = filedialog.askopenfilename(filetypes=filetypes,defaultextension='.pos')
                        with open(file_path, 'rb') as f:
                            positions = pickle.load(f)
                            simtoggle = False
                        for point, pos in zip(population, positions):
                            point.pos = pos