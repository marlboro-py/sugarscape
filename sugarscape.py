import mesa
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.discrete_space.property_layer import PropertyLayer

from agents import Trader
from utils import geometric_mean, flatten, np

def get_trade(agent):
  """
  For agent reports in data collector

  Return list of trade partners and None for other agents
  """
  
  if isinstance(agent, Trader):
    return agent.trade_partners
  else: 
      return None
  
def get_trader_type(agent):
    """
    For agent reports in data collector
    
    Return list of instances where agent was buyer or seller
    """
    if isinstance(agent, Trader):
        return agent.bought_or_sold
    else:
        return None

class SugarscapeG1mt(mesa.Model):
    """
    Manager class to run Sugarscape with traders
    """
    
    def __init__(
        self, 
        width=50,
        height=50,
        initial_population=200,
        endowment_min=25,
        endowment_max=50,
        metabolism_min=1,
        metabolism_max=5,
        vision_min=1,
        vision_max=5,
        enable_trade=True,
        seed=None
    ):
        super().__init__(seed=seed)
        # Initiate width and height of sugarscape
        self.width = width
        self.height = height
        
        # Initiate population attributes
        self.enable_trade = enable_trade
        self.running = True
        
        # initiate mesa grid class - the grid has several impacts on the ABM, since it determines with who agents can interact
        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height), torus=False, random=self.random
        )
        
        # initiate datacollector
        # TODO: collect if agent is buyer or seller  
        self.datacollector = mesa.DataCollector(
            model_reporters = {
                "Traders": lambda m: len(m.agents),
                "Trade Volume": lambda m: sum(len(a.trade_partners) for a in m.agents),
                "Price": lambda m: geometric_mean(flatten([a.prices for a in m.agents]))
            },
            agent_reporters = {
                "Trade Network": lambda a: get_trade(a),
                "Buyer or seller": lambda a: get_trader_type(a)
            }
        )
        
        # sugar distribution - spice distribution is the inverse
        self.sugar_distribution = np.genfromtxt("/home/ketzer/repos/sugarscape/data/sugarmap.txt")
        self.spice_distribution = np.flip(self.sugar_distribution, 1)
        
        # treats sugar and spice as a cell property layer in the grid
        self.grid.add_property_layer(
            PropertyLayer.from_data("sugar", self.sugar_distribution)
        )
        
        self.grid.add_property_layer(
            PropertyLayer.from_data("spice", self.spice_distribution)
        )
        
        # populates the grid with trader agents       
        Trader.create_agents(
            self,
            initial_population,
            self.random.choices(self.grid.all_cells.cells, k=initial_population), # randomly populates cells on the grid with traders
            sugar=self.rng.integers(
                endowment_min, endowment_max, (initial_population, ), endpoint=True
            ), # sugar endowment
            spice=self.rng.integers(
                endowment_min, endowment_max, (initial_population, ), endpoint=True
            ), # spice endowment
            metabolism_sugar=self.rng.integers(
                metabolism_min, metabolism_max, (initial_population, ), endpoint=True
            ), # randomly attributes sugar metabolism
            metabolism_spice=self.rng.integers(
                metabolism_min, metabolism_max, (initial_population, ), endpoint=True
            ), # randomly attributes spice metabolism
            vision=self.rng.integers(
                vision_min, vision_max, (initial_population, ), endpoint=True
            ),
        )

    def step(self):
        """
        Function that does staget activation of sugar and spice and then
        randomly activates trader
        """
        
        # step resource agents
        self.grid.sugar.data = np.minimum(
            self.grid.sugar.data + 1, self.sugar_distribution
        )
        
        self.grid.spice.data = np.minimum(
            self.grid.spice.data + 1, self.spice_distribution
        )
        
        # step trader agents
        # to account for agent death and removal we need a separate data structure to iterate
        
        # this puts traders in a randomized list
        trader_shuffle = self.agents_by_type[Trader].shuffle()
        
        for agent in trader_shuffle:
            agent.prices = []
            agent.trade_partners = []
            agent.bought_or_sold = []
            agent.move()
            agent.eat()
            agent.die()
            
        if not self.enable_trade:
            # return early if trade is not enabled
            self.datacollector.collect(self)
            return
    
        # randomize traders again
        trader_shuffle = self.agents_by_type[Trader].shuffle()
        
        for agent in trader_shuffle:
            agent.trade_with_neighbors()
        
        # collect model level data
        self.datacollector.collect(self)
        
        # TODO: see if this is still true
        # remove excess data, as Mesa does not have a datacollection by agent type feature
        
        agent_trades = self.datacollector._agent_records[self.steps]
        
        # leverage None aspect of no data in data collector
        agent_trades = [agent for agent in agent_trades if agent[2] is not None]
        
        # reassign that step in the dictionary with lean trade data
        self.datacollector._agent_records[self.steps] = agent_trades
        
    def run_model(self, step_count=1000):
        
        """
        Runs the model
        """
        for _ in range(step_count):
            self.step()
            
        