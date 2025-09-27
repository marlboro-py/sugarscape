import mesa
import numpy as np
from agents import Trader
from resources import Sugar, Spice
from utils import geometric_mean, get_trade, flatten

class SugarscapeG1mt(mesa.Model):
    """
    Manager class to run Sugarscape with traders
    """
    
    def __init__(
        self, width=50, height=50, initial_population=200, endowment_min=25,
        endowment_max=50, metabolism_min=1, metabolism_max=5, vision_min=1, vision_max=5
    ):
        super().__init__()
        # Initiate width and height of sugarscape
        self.width = width
        self.height = height
        # Initiate population attributes
        self.initial_population = initial_population
        self.endowment_min = endowment_min
        self.endowment_max = endowment_max
        self.metabolism_min = metabolism_min
        self.metabolism_max = metabolism_max
        self.vision_min = vision_min
        self.vision_max = vision_max
        self.running = True
        
        # initiate activation schedule - agents are randomly activated by type
        self.schedule = mesa.time.RandomActivationByType(self)
        
        # initiate mesa grid class - the grid has several impacts on the ABM, since it determines with who agents can interact
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
        
        # initiate datacollector
        # TODO: collect if agent is buyer or seller  
        self.datacollector = mesa.DataCollector(
            model_reporters = {
                "Trader": lambda m: m.schedule.get_type_count(Trader),
                "Trade Volume": lambda m: sum(len(a.trade_partners) for a in m.schedule.agents_by_type[Trader].values()),
                "Price": lambda m: geometric_mean(flatten(a.prices for a in m.schedule_agents_by_type[Trader].values()))
            },
            agent_reporters = {
                "Trade Network": lambda a: get_trade(a)
            }
        )
        
        # sugar distribution - spice distribution is the inverse
        sugar_distribution = np.genfromtxt("./data/sugarmap.txt")
        spice_distribution = np.flip(sugar_distribution, 1)
        
        # initiates agent id with first id being 0
        agent_id = 0
        
        # iters through the grid and places sugar and spice classes where they should exist (populates the environment)
        for value, (x, y) in self.grid.coord_iter():
            
            max_sugar = sugar_distribution[x, y]
            if max_sugar > 0:
                sugar = Sugar(agent_id, self, (x, y), max_sugar)
                self.schedule.add(sugar)
                self.grid.place_agent(sugar, (x, y))
                agent_id += 1
                
            max_spice = spice_distribution[x, y]
            if max_spice > 0:
                spice = Spice(agent_id, self, (x, y), max_spice)
                self.schedule.add(spice)
                self.grid.place_agent(spice, (x, y))
                agent_id += 1
                
        # populates the grid with trader agents       
        for i in range(self.initial_population):
            
            # get agent position
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            
            # initial endowment - the +1 here is due to how generating uniform pseudorandom numbers work in python
            sugar = int(self.random.uniform(self.endowment_min, self.endowment_max + 1))
            spice = int(self.random.uniform(self.endowment_min, self.endowment_max + 1))
            
            # initial metabolism (how much sugar and spice he needs to consume)
            metabolism_sugar = int(self.random.uniform(self.metabolism_min, self.metabolism_max))
            metabolism_spice = int(self.random.uniform(self.metabolism_min, self.metabolism_max))
            
            # gives agent vision (how far he can see on the grid)
            vision = int(self.random.uniform(self.vision_min, self.vision_max + 1))
            
            # creates a trader object
            trader = Trader(
                agent_id, 
                self, 
                (x, y), 
                moore = False,
                sugar = sugar,
                spice = spice, 
                metabolism_sugar = metabolism_sugar,
                metabolism_spice = metabolism_spice,
                vision = vision 
            )
            
            # places the agent
            self.grid.place_agent(trader, (x, y))
            self.schedule.add(trader)
            agent_id += 1

        def step(self):
            """
            Function that does staget activation of sugar and spice and then
            randomly activates trader
            """
            
            # step sugar agents
            for sugar in self.schedule.agents_by_type[Sugar].values():
                sugar.step()
                
            # step spice agents
            for spice in self.schedule.agents_by_type[Spice].values():
                spice.step()
            
            # step trader agents
            # to account for agent death and removal we need a separate data structure to iterate
            
            # this puts traders in a randomized list
            trader_shuffle = list(self.schedule.agents_by_type[Trader].values())
            self.random.shuffle(trader_shuffle)
            
            for agent in trader_shuffle:
                agent.prices = []
                agent.trade_partners = []
                agent.bought_or_sold = []
                agent.move()
                agent.eat()
                agent.die()
                
            # randomize traders again
            trader_shuffle = list(self.schedule.agents_by_type[Trader].values())
            self.random.shuffle(trader_shuffle)
            
            for agent in trader_shuffle:
                agent.trade_with_neighbors()
            
            # tracks the number of steps taken in the simulation
            self.schedule.steps += 1
            
            # collect model level data
            self.datacollector.collect(self)
            
            # TODO: see if this is still true
            # remove excess data, as Mesa does not have a datacollection by agent type feature
            
            agent_trades = self.datacollector._agent_records[self.schedule.steps]
            
            # leverage None aspect of no data in data collector
            agent_trades = [agent for agent in agent_trades if agent[2] is not None]
            
            # reassign that step in the dictionary with lean trade data
            self.datacollector._agent_records[self.schedule.steps] = agent_trades
            
        def run_model(self, step_count = 10):
            """
            Runs the model
            """
            for i in range(step_count):
                self.step()
            
            