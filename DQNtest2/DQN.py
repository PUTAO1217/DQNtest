#Sanil Lala

from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import numpy as np
import math
import timeit
import random


#Set SUMO environment path and import SUMO library and Traci
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)
import sumolib
from sumolib import checkBinary
import traci

#Class for traffic generation
class TrafficGenerator:
    def __init__(self, Max_Steps):
        self.Total_Number_Cars = 500  #Number of cars used in the simulation
        self._max_steps = Max_Steps
    def generate_routefile(self, seed):
        np.random.seed(seed)
        Timing = np.random.poisson(2, self.Total_Number_Cars) #Poisson distribution for the car approach rate to teh intersection
        Timing = np.sort(Timing)
        Car_Generation_Steps = []
        min_old = math.floor(Timing[1])
        max_old = math.ceil(Timing[-1])
        min_new = 0
        max_new = self._max_steps
        
        #Create .xml file for SUMO simulation
        for value in Timing:
            Car_Generation_Steps = np.append(Car_Generation_Steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        Car_Generation_Steps = np.rint(Car_Generation_Steps) 
        with open("test.rou.xml", "w") as routes:
            
            #Generate Routes
            print("""<routes>

    <vType id="traditionCar"  color='1,1,0' carFollowModel= 'IDM' actionStepLength='1' tau='1.4' speedDev='0' accel='5' decel='2' length='5' maxSpeed='20.00'/>
    <route id="W_E" edges="-E0 E2" />
    <route id="W_N" edges="-E0 E1" />
    <route id="W_S" edges="-E0 E3" />
    <route id="E_W" edges="-E2 E0" />
    <route id="E_S" edges="-E2 E3" />
    <route id="E_N" edges="-E2 E1" />
    <route id="N_E" edges="-E1 E2" />
    <route id="N_W" edges="-E1 E0" />
    <route id="N_S" edges="-E1 E3" />
    <route id="S_N" edges="-E3 E1" />
    <route id="S_E" edges="-E3 E2" />
    <route id="S_W" edges="-E3 E0" />""", file=routes)

            #Generate cars to follow routes
            for car_counter, step in enumerate(Car_Generation_Steps):
                Straight_or_Turn = np.random.uniform()
                if Straight_or_Turn < 0.25:
                    Straight = np.random.randint(1, 9)  
                    if Straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif Straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif Straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif Straight == 4:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif Straight == 5:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif Straight == 6:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif Straight == 7:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else: 
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else:
                    Turn = np.random.randint(1, 5) 
                    if Turn == 1:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        
                    elif Turn == 2:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        
                    elif Turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            print("</routes>", file=routes)

  
class RunSimulation:
    def __init__(self, SimSession, model, memory, traffic_gen, total_episodes, gamma, Max_Steps, Green_Duration, Yellow_Duration, SUMO_Command):
        self.Session = SimSession
        self._Model = model
        self._memory = memory
        self._traffic_gen = traffic_gen
        self._total_episodes = total_episodes
        self._gamma = gamma
        self._epsilongreedy = 0 
        self._steps = 0
        self._waiting_times = {}
        self._SUMO_Command = SUMO_Command
        self._max_steps = Max_Steps
        self._green_duration = Green_Duration
        self._yellow_duration = Yellow_Duration
        self._sum_intersection_queue = 0
        self._StoreReward = []
        self._cumulative_wait_store = []
        self._avg_intersection_queue_store = []



    def run(self, episode):
        self._traffic_gen.generate_routefile(episode)
        traci.start(self._SUMO_Command)
        self._epsilongreedy = 1.0 - (episode / self._total_episodes)
        self._steps = 0
        tot_neg_reward = 0
        old_total_queue = 0
        #old_total_wait = 0
        #intersection_queue = 0
        self._waiting_times = {}
        #self._sum_intersection_queue = 0

        while self._steps < self._max_steps:
            current_state = self._get_state()
            current_total_queue = self._get_queuelength()
            reward = old_total_queue - current_total_queue
            #current_total_wait = self._get_waiting_times()
            #reward = old_total_wait - current_total_wait
        

            if self._steps != 0:
                self._memory.Add_Sample((old_state, old_action, reward, current_state))
            action = self._choose_action(current_state)
            
            

            if self._steps != 0 and old_action != action:
                self._Set_YellowPhase(old_action)
                self._simulate(self._yellow_duration) 
            self._Set_GreenPhaseandDuration(action)
            self._simulate(self._green_duration)
            old_state = current_state
            old_action = action
            old_total_queue = current_total_queue
            if reward < 0:
                tot_neg_reward += reward
        

        self._save_stats(tot_neg_reward)
        print("Total reward: {}, Eps: {}".format(tot_neg_reward, self._epsilongreedy))
        traci.close()
        
        
    def _simulate(self, steps_todo):
        if (self._steps + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._steps
        self._steps = self._steps + steps_todo
        while steps_todo > 0:
            traci.simulationStep()
            self._replay() 
            steps_todo -= 1
            #intersection_queue = self._get_stats()
            #self._sum_intersection_queue += intersection_queue
            
    def _get_queuelength(self):
         intersection_queue = 0
         halt_N =traci.lane.getLastStepLength("-E1_0")+traci.lane.getLastStepLength("-E1_1")+traci.lane.getLastStepLength("-E1_2")+traci.lane.getLastStepLength("-E1_3") 
         halt_S = traci.lane.getLastStepLength("-E3_0")+traci.lane.getLastStepLength("-E3_1")+traci.lane.getLastStepLength("-E3_2")+traci.lane.getLastStepLength("-E3_3") 
         halt_E = traci.lane.getLastStepLength("-E2_0")+traci.lane.getLastStepLength("-E2_1")+traci.lane.getLastStepLength("-E2_2")+traci.lane.getLastStepLength("-E2_3") 
         halt_W = traci.lane.getLastStepLength("-E0_0")+traci.lane.getLastStepLength("-E0_1")+traci.lane.getLastStepLength("-E0_2")+traci.lane.getLastStepLength("-E0_3") 
         intersection_queue = halt_N + halt_S + halt_E + halt_W
         return intersection_queue          
     
    #def _get_waiting_times(self):
       # incoming_roads = ["1i", "4i", "2i", "3i"]
        #halt_N = traci.inductionloop.getLastStepOccupancy("Loop4i_0_1") + traci.inductionloop.getLastStepOccupancy("Loop4i_0_2") + traci.inductionloop.getLastStepOccupancy(
            #"Loop4i_0_3") + traci.inductionloop.getLastStepOccupancy("Loop4i_1_1") + traci.inductionloop.getLastStepOccupancy(
               # "Loop4i_1_2")+ traci.inductionloop.getLastStepOccupancy("Loop4i_1_3")
       # halt_S = traci.inductionloop.getLastStepOccupancy("Loop3i_0_1") + traci.inductionloop.getLastStepOccupancy("Loop3i_0_2") + traci.inductionloop.getLastStepOccupancy(
           # "Loop3i_0_3") + traci.inductionloop.getLastStepOccupancy("Loop3i_1_1") + traci.inductionloop.getLastStepOccupancy(
               # "Loop3i_1_2") + traci.inductionloop.getLastStepOccupancy("Loop3i_1_3")
       # halt_E = traci.inductionloop.getLastStepOccupancy("Loop1i_0_1") + traci.inductionloop.getLastStepOccupancy("Loop1i_0_2") + traci.inductionloop.getLastStepOccupancy(
           # "Loop1i_0_3") + traci.inductionloop.getLastStepOccupancy("Loop1i_1_1") + traci.inductionloop.getLastStepOccupancy(
               # "Loop1i_1_2") + traci.inductionloop.getLastStepOccupancy("Loop1i_1_3")
       # halt_W = traci.inductionloop.getLastStepOccupancy("Loop2i_0_1") + traci.inductionloop.getLastStepOccupancy("Loop2i_0_2") + traci.inductionloop.getLastStepOccupancy(
           # "Loop2i_1_3") + traci.inductionloop.getLastStepOccupancy("Loop2i_1_1") + traci.inductionloop.getLastStepOccupancy(
               # "Loop2i_1_2") + traci.inductionloop.getLastStepOccupancy("Loop2i_1_3")
        #wait = halt_N + halt_S + halt_E + halt_W
       # total_waiting_time =wait
        #return total_waiting_time
    
    #choose action
    def _choose_action(self, state):
        if random.uniform(0.0,1.0) < self._epsilongreedy:
            return random.randint(0, self._Model.Number_Actions - 3) #Random action
        else:
            return np.argmax(self._Model.predict_one(state, self.Session) ) #Use memory
    
    #set yellow phase
    def _Set_YellowPhase(self, old_action):
        if old_action == 0 or old_action == 1 or old_action == 2:
          yellow_phase = 1
        elif old_action == 3 or old_action == 4 or old_action == 5:
          yellow_phase = 3
        elif old_action == 6 or old_action == 7 or old_action == 8:
          yellow_phase = 5
        elif old_action == 9 or old_action == 10 or old_action == 11:
          yellow_phase = 7
        traci.trafficlight.setPhase("0",yellow_phase)
        
    #set green phase duration    
    def _Set_GreenPhaseandDuration(self, action):
        if action == 0:
            traci.trafficlight.setPhase("0", 0)
            traci.trafficlight.setPhaseDuration("0", 15)
            self._green_duration = 15
        elif action == 1:
            traci.trafficlight.setPhase("0", 0)
            traci.trafficlight.setPhaseDuration("0", 10)
            self._green_duration = 10
        elif action == 2:
            traci.trafficlight.setPhase("0", 0)
            traci.trafficlight.setPhaseDuration("0", 20)
            self._green_duration = 20
        elif action == 3:
            traci.trafficlight.setPhase("0", 2)
            traci.trafficlight.setPhaseDuration("0", 15)
            self._green_duration = 15
        elif action == 4:
            traci.trafficlight.setPhase("0", 2)
            traci.trafficlight.setPhaseDuration("0", 10)
            self._green_duration = 10
        elif action == 5:
            traci.trafficlight.setPhase("0", 2)
            traci.trafficlight.setPhaseDuration("0", 20)
            self._green_duration = 20
        elif action == 5:
            traci.trafficlight.setPhase("0", 4)
            traci.trafficlight.setPhaseDuration("0", 15)
            self._green_duration = 15
        elif action == 7:
            traci.trafficlight.setPhase("0", 4)
            traci.trafficlight.setPhaseDuration("0", 10)
            self._green_duration = 10
        elif action == 8:
            traci.trafficlight.setPhase("0", 4)
            traci.trafficlight.setPhaseDuration("0", 20)
            self._green_duration = 20
        elif action == 9:
            traci.trafficlight.setPhase("0", 6)
            traci.trafficlight.setPhaseDuration("0", 15)
            self._green_duration = 15
        elif action == 10:
            traci.trafficlight.setPhase("0", 6)
            traci.trafficlight.setPhaseDuration("0", 10)
            self._green_duration = 10
        elif action == 11:
            traci.trafficlight.setPhase("0", 6)
            traci.trafficlight.setPhaseDuration("0", 20)
            self._green_duration = 20
            
    #obtain queue stats from simulation        
   
       
    #obtain state after action
    def _get_state(self):
        #Create 16 x 3 Matrix for Vehicle Positions and Velocities
        Position_Matrix = []
        Velocity_Matrix = []
        for i in range(16):
                Position_Matrix.append([])
                Velocity_Matrix.append([])
                for j in range(3):
                    Position_Matrix[i].append(0)
                    Velocity_Matrix[i].append(0)
        Position_Matrix = np.array(Position_Matrix)
        Velocity_Matrix = np.array(Velocity_Matrix)

        Loop1i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_0_1" )
        Loop1i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_0_2" )
        Loop1i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_0_3" )
        Loop1i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_1_1" )
        Loop1i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_1_2" )
        Loop1i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_1_3" )
        Loop1i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_2_1" )
        Loop1i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_2_2" )
        Loop1i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_2_3" )
        Loop1i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_3_1" )
        Loop1i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_3_2" )
        Loop1i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_3_3" )


        if len(Loop1i_0_1) != 0:
           Velocity_Matrix[0,0] = traci.vehicle.getSpeed(Loop1i_0_1[0])
           Loop1i_0_1 = 1
        else:
           Loop1i_0_1 = 0
           
        if len(Loop1i_0_2) != 0:
           Velocity_Matrix[0,1] = traci.vehicle.getSpeed(Loop1i_0_2[0])
           Loop1i_0_2 = 1
        else:
           Loop1i_0_2 = 0
           
        if len(Loop1i_0_3) != 0:
           Velocity_Matrix[0,2] = traci.vehicle.getSpeed(Loop1i_0_3[0])
           Loop1i_0_3 = 1
        else:
           Loop1i_0_3 = 0   
           
        if len(Loop1i_1_1) != 0:
           Velocity_Matrix[1,0] = traci.vehicle.getSpeed(Loop1i_1_1[0])
           Loop1i_1_1 = 1
        else:
           Loop1i_1_1 = 0 
           
        if len(Loop1i_1_2) != 0:
           Velocity_Matrix[1,1] = traci.vehicle.getSpeed(Loop1i_1_2[0])
           Loop1i_1_2 = 1
        else:
           Loop1i_1_2 = 0 
         
        if len(Loop1i_1_3) != 0:
           Velocity_Matrix[1,2] = traci.vehicle.getSpeed(Loop1i_1_3[0])
           Loop1i_1_3 = 1
        else:
           Loop1i_1_3 = 0 
        
        if len(Loop1i_2_1) != 0:
           Velocity_Matrix[2,0] = traci.vehicle.getSpeed(Loop1i_2_1[0])
           Loop1i_2_1 = 1
        else:
           Loop1i_2_1 = 0 
           
        if len(Loop1i_2_2) != 0:
           Velocity_Matrix[2,1] = traci.vehicle.getSpeed(Loop1i_2_2[0])
           Loop1i_2_2 = 1
        else:
           Loop1i_2_2 = 0 
         
        if len(Loop1i_2_3) != 0:
           Velocity_Matrix[2,2] = traci.vehicle.getSpeed(Loop1i_2_3[0])
           Loop1i_2_3 = 1
        else:
           Loop1i_2_3 = 0 
          
        if len(Loop1i_3_1) != 0:
           Velocity_Matrix[3,0] = traci.vehicle.getSpeed(Loop1i_3_1[0])
           Loop1i_3_1 = 1
        else:
           Loop1i_3_1 = 0 
               
        if len(Loop1i_3_2) != 0:
           Velocity_Matrix[3,1] = traci.vehicle.getSpeed(Loop1i_3_2[0])
           Loop1i_3_2 = 1
        else:
           Loop1i_3_2 = 0 
             
        if len(Loop1i_3_3) != 0:
           Velocity_Matrix[3,2] = traci.vehicle.getSpeed(Loop1i_2_3[0])
           Loop1i_2_3 = 1
        else:
           Loop1i_2_3 = 0 
       
        Position_Matrix[0,0] = Loop1i_0_1
        Position_Matrix[0,1] = Loop1i_0_2
        Position_Matrix[0,2] = Loop1i_0_3
        Position_Matrix[1,0] = Loop1i_1_1
        Position_Matrix[1,1] = Loop1i_1_2
        Position_Matrix[1,2] = Loop1i_1_3
        Position_Matrix[2,0] = Loop1i_2_1
        Position_Matrix[2,1] = Loop1i_2_2
        Position_Matrix[2,2] = Loop1i_2_3
        Position_Matrix[3,0] = Loop1i_3_1
        Position_Matrix[3,1] = Loop1i_3_2
        Position_Matrix[3,2] = Loop1i_3_3
        
        Loop2i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_0_1" )
        Loop2i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_0_2" )
        Loop2i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_0_3" )
        Loop2i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_1_1" )
        Loop2i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_1_2" )
        Loop2i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_1_3" )
        Loop2i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_2_1" )
        Loop2i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_2_2" )
        Loop2i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_2_3" )
        Loop2i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_3_1" )
        Loop2i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_3_2" )
        Loop2i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_3_3" )


        if len(Loop2i_0_1) != 0:
           Velocity_Matrix[4,0] = traci.vehicle.getSpeed(Loop2i_0_1[0])
           Loop2i_0_1 = 1
        else:
           Loop2i_0_1 = 0
           
        if len(Loop2i_0_2) != 0:
           Velocity_Matrix[4,1] = traci.vehicle.getSpeed(Loop2i_0_2[0])
           Loop2i_0_2 = 1
        else:
           Loop2i_0_2 = 0
           
        if len(Loop2i_0_3) != 0:
           Velocity_Matrix[4,2] = traci.vehicle.getSpeed(Loop2i_0_3[0])
           Loop2i_0_3 = 1
        else:
           Loop2i_0_3 = 0   
           
        if len(Loop2i_1_1) != 0:
           Velocity_Matrix[5,0] = traci.vehicle.getSpeed(Loop2i_1_1[0])
           Loop2i_1_1 = 1
        else:
           Loop2i_1_1 = 0 
           
        if len(Loop2i_1_2) != 0:
           Velocity_Matrix[5,1] = traci.vehicle.getSpeed(Loop2i_1_2[0])
           Loop2i_1_2 = 1
        else:
           Loop2i_1_2 = 0 
         
        if len(Loop2i_1_3) != 0:
           Velocity_Matrix[5,2] = traci.vehicle.getSpeed(Loop2i_1_3[0])
           Loop2i_1_3 = 1
        else:
           Loop2i_1_3 = 0 
        
        if len(Loop2i_2_1) != 0:
           Velocity_Matrix[6,0] = traci.vehicle.getSpeed(Loop2i_2_1[0])
           Loop2i_2_1 = 1
        else:
           Loop2i_2_1 = 0 
           
        if len(Loop2i_2_2) != 0:
           Velocity_Matrix[6,1] = traci.vehicle.getSpeed(Loop2i_2_2[0])
           Loop2i_2_2 = 1
        else:
           Loop2i_2_2 = 0 
         
        if len(Loop2i_2_3) != 0:
           Velocity_Matrix[6,2] = traci.vehicle.getSpeed(Loop2i_2_3[0])
           Loop2i_2_3 = 1
        else:
           Loop2i_2_3 = 0 
          
        if len(Loop2i_3_1) != 0:
           Velocity_Matrix[7,0] = traci.vehicle.getSpeed(Loop2i_3_1[0])
           Loop2i_3_1 = 1
        else:
           Loop2i_3_1 = 0 
               
        if len(Loop2i_3_2) != 0:
           Velocity_Matrix[7,1] = traci.vehicle.getSpeed(Loop2i_3_2[0])
           Loop2i_3_2 = 1
        else:
           Loop2i_3_2 = 0 
             
        if len(Loop2i_3_3) != 0:
           Velocity_Matrix[7,2] = traci.vehicle.getSpeed(Loop2i_2_3[0])
           Loop2i_2_3 = 1
        else:
           Loop2i_2_3 = 0 
       
        Position_Matrix[4,0] = Loop2i_0_1
        Position_Matrix[4,1] = Loop2i_0_2
        Position_Matrix[4,2] = Loop2i_0_3
        Position_Matrix[5,0] = Loop2i_1_1
        Position_Matrix[5,1] = Loop2i_1_2
        Position_Matrix[5,2] = Loop2i_1_3
        Position_Matrix[6,0] = Loop2i_2_1
        Position_Matrix[6,1] = Loop2i_2_2
        Position_Matrix[6,2] = Loop2i_2_3
        Position_Matrix[7,0] = Loop2i_3_1
        Position_Matrix[7,1] = Loop2i_3_2
        Position_Matrix[7,2] = Loop2i_3_3
        
        Loop3i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_0_1" )
        Loop3i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_0_2" )
        Loop3i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_0_3" )
        Loop3i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_1_1" )
        Loop3i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_1_2" )
        Loop3i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_1_3" )
        Loop3i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_2_1" )
        Loop3i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_2_2" )
        Loop3i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_2_3" )
        Loop3i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_3_1" )
        Loop3i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_3_2" )
        Loop3i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_3_3" )


        if len(Loop3i_0_1) != 0:
           Velocity_Matrix[8,0] = traci.vehicle.getSpeed(Loop3i_0_1[0])
           Loop3i_0_1 = 1
        else:
           Loop3i_0_1 = 0
           
        if len(Loop3i_0_2) != 0:
           Velocity_Matrix[8,1] = traci.vehicle.getSpeed(Loop3i_0_2[0])
           Loop3i_0_2 = 1
        else:
           Loop3i_0_2 = 0
           
        if len(Loop3i_0_3) != 0:
           Velocity_Matrix[8,2] = traci.vehicle.getSpeed(Loop3i_0_3[0])
           Loop3i_0_3 = 1
        else:
           Loop3i_0_3 = 0   
           
        if len(Loop3i_1_1) != 0:
           Velocity_Matrix[9,0] = traci.vehicle.getSpeed(Loop3i_1_1[0])
           Loop3i_1_1 = 1
        else:
           Loop3i_1_1 = 0 
           
        if len(Loop3i_1_2) != 0:
           Velocity_Matrix[9,1] = traci.vehicle.getSpeed(Loop3i_1_2[0])
           Loop3i_1_2 = 1
        else:
           Loop3i_1_2 = 0 
         
        if len(Loop3i_1_3) != 0:
           Velocity_Matrix[9,2] = traci.vehicle.getSpeed(Loop3i_1_3[0])
           Loop3i_1_3 = 1
        else:
           Loop3i_1_3 = 0 
        
        if len(Loop3i_2_1) != 0:
           Velocity_Matrix[10,0] = traci.vehicle.getSpeed(Loop3i_2_1[0])
           Loop3i_2_1 = 1
        else:
           Loop3i_2_1 = 0 
           
        if len(Loop3i_2_2) != 0:
           Velocity_Matrix[10,1] = traci.vehicle.getSpeed(Loop3i_2_2[0])
           Loop3i_2_2 = 1
        else:
           Loop3i_2_2 = 0 
         
        if len(Loop3i_2_3) != 0:
           Velocity_Matrix[10,2] = traci.vehicle.getSpeed(Loop3i_2_3[0])
           Loop3i_2_3 = 1
        else:
           Loop3i_2_3 = 0 
          
        if len(Loop3i_3_1) != 0:
           Velocity_Matrix[11,0] = traci.vehicle.getSpeed(Loop3i_3_1[0])
           Loop3i_3_1 = 1
        else:
           Loop3i_3_1 = 0 
               
        if len(Loop3i_3_2) != 0:
           Velocity_Matrix[11,1] = traci.vehicle.getSpeed(Loop3i_3_2[0])
           Loop3i_3_2 = 1
        else:
           Loop3i_3_2 = 0 
             
        if len(Loop3i_3_3) != 0:
           Velocity_Matrix[11,2] = traci.vehicle.getSpeed(Loop3i_2_3[0])
           Loop3i_2_3 = 1
        else:
           Loop3i_2_3 = 0 
       
        Position_Matrix[8,0] = Loop3i_0_1
        Position_Matrix[8,1] = Loop3i_0_2
        Position_Matrix[8,2] = Loop3i_0_3
        Position_Matrix[9,0] = Loop3i_1_1
        Position_Matrix[9,1] = Loop3i_1_2
        Position_Matrix[9,2] = Loop3i_1_3
        Position_Matrix[10,0] = Loop3i_2_1
        Position_Matrix[10,1] = Loop3i_2_2
        Position_Matrix[10,2] = Loop3i_2_3
        Position_Matrix[11,0] = Loop3i_3_1
        Position_Matrix[11,1] = Loop3i_3_2
        Position_Matrix[11,2] = Loop3i_3_3
        
        Loop4i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_0_1" )
        Loop4i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_0_2" )
        Loop4i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_0_3" )
        Loop4i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_1_1" )
        Loop4i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_1_2" )
        Loop4i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_1_3" )
        Loop4i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_2_1" )
        Loop4i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_2_2" )
        Loop4i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_2_3" )
        Loop4i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_3_1" )
        Loop4i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_3_2" )
        Loop4i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_3_3" )


        if len(Loop4i_0_1) != 0:
           Velocity_Matrix[12,0] = traci.vehicle.getSpeed(Loop4i_0_1[0])
           Loop4i_0_1 = 1
        else:
           Loop4i_0_1 = 0
           
        if len(Loop4i_0_2) != 0:
           Velocity_Matrix[12,1] = traci.vehicle.getSpeed(Loop4i_0_2[0])
           Loop4i_0_2 = 1
        else:
           Loop4i_0_2 = 0
           
        if len(Loop4i_0_3) != 0:
           Velocity_Matrix[12,2] = traci.vehicle.getSpeed(Loop4i_0_3[0])
           Loop4i_0_3 = 1
        else:
           Loop4i_0_3 = 0   
           
        if len(Loop4i_1_1) != 0:
           Velocity_Matrix[13,0] = traci.vehicle.getSpeed(Loop4i_1_1[0])
           Loop4i_1_1 = 1
        else:
           Loop4i_1_1 = 0 
           
        if len(Loop4i_1_2) != 0:
           Velocity_Matrix[13,1] = traci.vehicle.getSpeed(Loop4i_1_2[0])
           Loop4i_1_2 = 1
        else:
           Loop4i_1_2 = 0 
         
        if len(Loop4i_1_3) != 0:
           Velocity_Matrix[13,2] = traci.vehicle.getSpeed(Loop4i_1_3[0])
           Loop4i_1_3 = 1
        else:
           Loop4i_1_3 = 0 
        
        if len(Loop4i_2_1) != 0:
           Velocity_Matrix[14,0] = traci.vehicle.getSpeed(Loop4i_2_1[0])
           Loop4i_2_1 = 1
        else:
           Loop4i_2_1 = 0 
           
        if len(Loop4i_2_2) != 0:
           Velocity_Matrix[14,1] = traci.vehicle.getSpeed(Loop4i_2_2[0])
           Loop4i_2_2 = 1
        else:
           Loop4i_2_2 = 0 
         
        if len(Loop4i_2_3) != 0:
           Velocity_Matrix[14,2] = traci.vehicle.getSpeed(Loop4i_2_3[0])
           Loop4i_2_3 = 1
        else:
           Loop4i_2_3 = 0 
          
        if len(Loop4i_3_1) != 0:
           Velocity_Matrix[15,0] = traci.vehicle.getSpeed(Loop4i_3_1[0])
           Loop4i_3_1 = 1
        else:
           Loop4i_3_1 = 0 
               
        if len(Loop4i_3_2) != 0:
           Velocity_Matrix[15,1] = traci.vehicle.getSpeed(Loop4i_3_2[0])
           Loop4i_3_2 = 1
        else:
           Loop4i_3_2 = 0 
             
        if len(Loop4i_3_3) != 0:
           Velocity_Matrix[15,2] = traci.vehicle.getSpeed(Loop4i_2_3[0])
           Loop4i_2_3 = 1
        else:
           Loop4i_2_3 = 0 
       
        Position_Matrix[12,0] = Loop4i_0_1
        Position_Matrix[12,1] = Loop4i_0_2
        Position_Matrix[12,2] = Loop4i_0_3
        Position_Matrix[13,0] = Loop4i_1_1
        Position_Matrix[13,1] = Loop4i_1_2
        Position_Matrix[13,2] = Loop4i_1_3
        Position_Matrix[14,0] = Loop4i_2_1
        Position_Matrix[14,1] = Loop4i_2_2
        Position_Matrix[14,2] = Loop4i_2_3
        Position_Matrix[15,0] = Loop4i_3_1
        Position_Matrix[15,1] = Loop4i_3_2
        Position_Matrix[15,2] = Loop4i_3_3
        

        
        #Create 4 x 1 matrix for phase state
        Phase = []
        if traci.trafficlight.getPhase('0') == 0 or traci.trafficlight.getPhase('0') == 1 or traci.trafficlight.getPhase('0') == 2:
            Phase = [1, 0, 0, 0]
        elif traci.trafficlight.getPhase('0') == 3 or traci.trafficlight.getPhase('0') == 4 or traci.trafficlight.getPhase('0') == 5:
            Phase = [0, 1, 0, 0]
        elif traci.trafficlight.getPhase('0') == 6 or traci.trafficlight.getPhase('0') == 7 or traci.trafficlight.getPhase('0') == 8:
            Phase = [0, 0, 1, 0]
        elif traci.trafficlight.getPhase('0') == 9 or traci.trafficlight.getPhase('0') == 10 or traci.trafficlight.getPhase('0') == 11:
            Phase = [0, 0, 0, 1]
      
        Phase = np.array(Phase)
        Phase = Phase.flatten()
        
        state = np.concatenate((Position_Matrix,Velocity_Matrix), axis=0)
        state = state.flatten()
        state =  np.concatenate((state,Phase), axis=0)
        
        #Create matrix for duration
        Duration_Matrix = [traci.trafficlight.getPhaseDuration('0')]

        Duration_Matrix = np.array(Duration_Matrix)
        Duration_Matrix = Duration_Matrix.flatten()
        state =  np.concatenate((state,Duration_Matrix), axis=0)
       

        return state 


    #Replay memory     
    def _replay(self):
        Batch = self._memory.Get_Samples(self._Model.batch_size)
        if len(Batch) > 0:  
            states = np.array([val[0] for val in Batch])  
            next_states = np.array([val[3] for val in Batch])  
            QSA = self._Model.predict_batch(states, self.Session)  
            QSATarget = self._Model.predict_batch(next_states, self.Session) 
            x = np.zeros((len(Batch), self._Model.Number_States))
            y = np.zeros((len(Batch), self._Model.Number_Actions))
            for i, b in enumerate(Batch):
                state, action, reward, next_state = b[0], b[1], b[2], b[3]
                Current_Q = QSA[i] 
                Current_Q[action] = reward + self._gamma * np.amax(QSATarget[i])  
                x[i] = state
                y[i] = Current_Q  
            self._Model.train_batch(self.Session, x, y) 


    def _save_stats(self, tot_neg_reward):
            self._StoreReward.append(tot_neg_reward) 
            self._cumulative_wait_store.append(self._sum_intersection_queue) 

    @property
    def reward_store(self):
        return self._StoreReward

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_intersection_queue_store(self):
        return self._avg_intersection_queue_store
        
#Reinforcement learning model        
class Model:
    def __init__(self, Number_States, Number_Actions, batch_size):
        self._Number_States = Number_States
        self._Number_Actions = Number_Actions
        self._batch_size = batch_size

        self._states = None
        self._actions = None

        self._logits = None
        self._optimizer = None
        self._var_init = None

        self._define_model()
    
    #Create neural network
    def _define_model(self):
      
        self._states = tf.placeholder(shape=[None, self._Number_States], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._Number_Actions], dtype=tf.float32)

        fc1 = tf.layers.dense(self._states, 33, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 33, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 33, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc3, self._Number_Actions)

        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, SimSession):
        return SimSession.run(self._logits, feed_dict={self._states: state.reshape(1, self.Number_States)})

    def predict_batch(self, states, SimSession):
        return SimSession.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, SimSession, x_batch, y_batch):
        SimSession.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

    @property
    def Number_States(self):
        return self._Number_States

    @property
    def Number_Actions(self):
        return self._Number_Actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init
        
#Class for storying and receiving memory
class Memory:
    def __init__(self, Memory_Size):
        self._Memory_Size = Memory_Size
        self.Samples = []

    def Get_Samples(self, Number_Samples):
        if Number_Samples > len(self.Samples):
            return random.sample(self.Samples, len(self.Samples))
        else:
            return random.sample(self.Samples, Number_Samples) 

    def Add_Sample(self, sample):
        self.Samples.append(sample)
        if len(self.Samples) > self._Memory_Size:
            self.Samples.pop(0)


#Saving and ploting graphs 
def save_graphs(sim_runner, total_episodes, plot_path):

    plt.rcParams.update({'font.size': 24})

    # reward
    data = sim_runner.reward_store
    plt.plot(data)
    plt.ylabel("Cumulative negative reward")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val + 0.05 * min_val, max_val - 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'reward.png', dpi=96)
    plt.close("all")
    with open(plot_path + 'reward_data.txt', "w") as file:
        for item in data:
                file.write("%s\n" % item)

    data = sim_runner.cumulative_wait_store
    plt.plot(data)
    plt.ylabel("Cumulative Qccupancy (s)")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'delay.png', dpi=96)
    plt.close("all")
    with open(plot_path + 'delay_data.txt', "w") as file:
        for item in data:
                file.write("%s\n" % item)


if __name__ == "__main__":

    gui = True
    total_episodes = 100
    gamma = 0.75
    batch_size = 32
    Memory_Size = 3200
    path = "./model/model_1g/"
    # ----------------------

    Number_States = 53
    Number_Actions = 12
    Max_Steps = 3600  
    Green_Duration = 10
    Yellow_Duration = 7
    
    #Change to False if Simulation GUI must be shown
    if gui == True:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    model = Model(Number_States, Number_Actions, batch_size)
    memory = Memory(Memory_Size)
    traffic_gen = TrafficGenerator(Max_Steps)
    SUMO_Command = [sumoBinary, "-c", "project.sumocfg", '--start', "--no-step-log", "true", "--waiting-time-memory", str(Max_Steps)]
    saver = tf.train.Saver()

    with tf.Session() as SimSession:
        print("PATH:", path)
        print("----- Start time:", datetime.datetime.now())
        SimSession.run(model.var_init)
        sim_runner = RunSimulation(SimSession, model, memory, traffic_gen, total_episodes, gamma, Max_Steps, Green_Duration, Yellow_Duration, SUMO_Command)
        episode = 0

        while episode < total_episodes:
            print('----- Episode {} of {}'.format(episode+1, total_episodes))
            start = timeit.default_timer()
            sim_runner.run(episode)  # run the simulation
            stop = timeit.default_timer()
            print('Time: ', round(stop - start, 1))
            episode += 1

        os.makedirs(os.path.dirname(path), exist_ok=True)
        saver.save(SimSession, path + "my_tlcs_model.ckpt") 
        print("----- End time:", datetime.datetime.now())
        print("PATH:", path)
        save_graphs(sim_runner, total_episodes, path)
