import pygame
import random
import time

import numpy as np
import warnings
warnings.filterwarnings('ignore')

WIDTH = 1200
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH,HEIGHT))
clock = pygame.time.Clock()

#Random coordinate for apple and snake head spawn location which restricted to width and height
get_random_apple = lambda: [random.randrange(1,79)*15,random.randrange(6,39)*15]
random_coordinate = lambda: [random.randrange(7,79)*15,random.randrange(6,39)*15]

class Brain:
    def __init__(self, weights1, weights2):
        
        self.weights1 = weights1
        self.weights2 = weights2
        
    def softmax(self,x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def swish(self, x):
        return x * self.sigmoid(x)
    
    def predict(self, lokasyon, apple):
        
        self.Data = np.array([lokasyon[0] - apple[0], 
                              lokasyon[1]- apple[1], 
                              np.linalg.norm(np.array(lokasyon) - np.array(apple))])
        
        # Optional.  Normalizing reduces train time. 
        self.Data = self.Data / np.linalg.norm(self.Data)
        
        self.layer1 = self.swish(np.dot(self.Data, self.weights1)) 
        self.layer2 = self.softmax(np.dot(self.layer1, self.weights2)) 
        return self.layer2

class Child:

    def __init__(self, apple, weights1, weights2):
        
        x = random.randint(30, WIDTH-50)
        x -= x % 15 or 1
        #x = (lambda num: num - num % 15 or 1)(random.randint(100, 1000))
        
        y = random.randint(100, HEIGHT-50)
        y -= y % 15 or 1
        
        self.head = [x,y]
        self.kuyruk = [[x+15,y],[x+30,y],[x+45,y]]
        self.fark = np.linalg.norm(np.array(self.head) - np.array(apple))   # fark = difference. This will take euclid distance of snake head and apple
        self.Fitness = 0
        
        self.network = Brain(weights1, weights2)
      
        self.commands = {
        0: [self.sag],
        1: [self.alt],
        2: [self.sol],
        3: [self.ust],
        }
        
    def move(self, apple):
        
        karar = self.network.predict(self.head, apple).argmax()   #Index of neural network prediction. Which mentioned at self.coomands just upside
        
        for func in self.commands[karar]:
            func()
        
        # Clipping the coordinates for keep Snakes head in game screen
        self.head[0] = np.clip(self.head[0], 0, WIDTH-15)
        self.head[1] = np.clip(self.head[1], 75, HEIGHT-15)
        
        # Update current distance
        self.fark = np.linalg.norm(np.array(self.head) - np.array(apple))
        
    # Left command
    def sol(self):
        self.head[0] -= 15
    
    #Right command
    def sag(self):
        self.head[0] += 15
        
    # Go to top command
    def ust(self):
        self.head[1] -= 15
        
    # Go to bottom command
    def alt(self):
        self.head[1] += 15

        
class Env:
   
    def __init__(self,Population_Number):
        
        
        self.run = True
        self.apple = get_random_apple()
        self.Population = []
        self.Population_Number = Population_Number
        self.Died = []
        self.Next_Generation = []
        self.timer = time.time()
        self.epoch = 1
        
        #Creating Agents with their own random initial weights 
        for i in range(self.Population_Number):
            weights1 = np.random.uniform(-1,1,(3, 8))
            weights2 = np.random.uniform(-1,1,(8, 4)) 
            self.Population.append(Child(self.apple, weights1, weights2))
        
        
        self.font = pygame.font.Font(None, 36)
        self.text_surface = self.font.render("", True, (255, 255, 255))
        self.text_rect = self.text_surface.get_rect()
        self.text_rect.center = (100, 40)
    
    
    def check(self):
        if len(self.Population) < 1:                                                             
            self.crossover()
            self.apple = get_random_apple()

            for eleman in self.Population:
                eleman.Fitness = 0
                eleman.head = random_coordinate()
                eleman.fark = np.linalg.norm(np.array(eleman.head) - np.array(self.apple))
            
            
    def eat_apple(self):
        
        self.apple = get_random_apple()
        
        for eleman in self.Population:
            eleman.fark = np.linalg.norm(np.array(eleman.head) - np.array(self.apple))
            
            
    def step(self):
        
        pygame.draw.rect(screen, (252,0,0), [self.apple[0],self.apple[1], 15,15])
        
        for eleman in self.Population:
            
            for pos in eleman.kuyruk:
                pygame.draw.rect(screen, (0,0,120), [pos[0], pos[1], 15,15])
        
            fark = eleman.fark
            eleman.move(self.apple)
            
            eleman.kuyruk.insert(0,list(eleman.head))
            eleman.kuyruk.pop()
            
            if eleman.fark >= fark:  #If snake makes wrong prediction like despite direction to apple, it dies
                
                self.Died.append(eleman)
                self.Population.remove(eleman)
                self.check()
                
            # +score
            if self.apple == eleman.head:
                eleman.kuyruk.insert(0,list(eleman.head))
                self.eat_apple()
            
            
    def crossover(self):
        self.epoch += 1
        self.Died = sorted(self.Died, key=lambda eleman: eleman.Fitness)

        self.Next_Generation = []
        last_best = int((self.Population_Number - 1) * 0.95)
        self.Next_Generation.extend(self.Died[last_best:])
        self.Besties = self.Died[last_best:]

        self.Died.clear()
        
        while True:
            if len(self.Next_Generation) < self.Population_Number:
                member_1 = random.choice(self.Besties)
                member_2 = random.choice(self.Besties)

                member_1_weights_1 = member_1.network.weights1
                member_1_weights_2 = member_1.network.weights2

                member_2_weights_1 = member_2.network.weights1
                member_2_weights_2 = member_2.network.weights2

                chield_weights_1 = []
                chield_weights_2 = []

                for a,b in zip(member_1_weights_1, member_2_weights_1):
                    for c,d in zip(a,b):
                        prob = random.random()
                        if prob < 0.47:
                            chield_weights_1.append(c)
                        elif prob < 0.94:
                            chield_weights_1.append(d)
                        else:
                            chield_weights_1.append(random.uniform(-1, 1))

                for e,f in zip(member_1_weights_2, member_2_weights_2): #7/1
                    for g,h in zip(e,f):
                        prob = random.random()
                        if prob < 0.47:
                            chield_weights_2.append(g)
                        elif prob < 0.94:
                            chield_weights_2.append(h)
                        else:
                            chield_weights_2.append(random.uniform(-1, 1))

                chield_weights_1 = np.array(chield_weights_1).reshape(3,8)
                chield_weights_2 = np.array(chield_weights_2).reshape(8,4)

                self.Next_Generation.append(Child(self.apple, chield_weights_1, chield_weights_2))

            else:
                break

        self.Population = self.Next_Generation
        
    
    def display(self):
        try:
            screen.fill((0,0,0))
            self.drawGrid()
            self.step()
            self.text_surface = self.font.render("Generation / Nesil: "+str(self.epoch), True, (255, 255, 255))
            
            screen.blit(self.text_surface, self.text_rect)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False

            pygame.display.update()
            pygame.time.delay(10)          #Optional delay                                                              

        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            line_number = tb.tb_lineno
            print("Hata!", line_number,". Satırda hata meydana geldi")
            traceback.print_exc()
            self.run = False
            pygame.quit()
            
    def drawGrid(self):
        blockSize = 15 
        for x in range(0, 1200, blockSize):
            for y in range(75, 800, blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                pygame.draw.rect(screen, (25,25,25), rect, 1)

Number_Of_Agent = 100

while True:
    pygame.init()
    try:    
        game = Env(Number_Of_Agent)
        while game.run:
            game.display()
        pygame.quit()
    except Exception as e:
        print("Environment Hatası! Hata = \n",e)
        pygame.quit()
        traceback.print_exc()
    break