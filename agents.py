import mesa
import math
from utils import get_distance
from resources import Sugar, Spice

class Trader(mesa.Agent):
    """
    Trader:
    - has a metabolism of sugar and spice
    - harvest and trade sugar and spice to survive
    """
    
    def __init__(
        self, unique_id, model, pos, moore=False,
        sugar=0, spice=0, metabolism_sugar=0, metabolism_spice=0, vision=0
    ):
        super().__init__(model)
        self.unique_id = unique_id
        self.pos = pos
        self.moore = moore
        self.sugar = sugar
        self.spice = spice
        self.metabolism_sugar = metabolism_sugar 
        self.metabolism_spice = metabolism_spice
        self.vision = vision
        self.bought_or_sold = []
        self.prices = []
        self.trade_partners = []
    
    def get_resource(self, pos):
        """
        Gets amount of resources
        """
        this_cell = self.model.grid.get_cell_list_contents(pos)
        
        agents = {
            type(agent):agent for agent in this_cell
        }

        sugar = agents.get(Sugar)
        spice = agents.get(Spice)
        
        return {"sugar":sugar, "spice": spice}
    
    def get_trader(self, pos):
        """
        Identify if agent is of type trader, used in self.trade_with_neighbors()
        """
        this_cell = self.model.grid.get_cell_list_contents(pos)
        
        for agent in this_cell:
            if isinstance(agent, Trader):
                return agent 
    
    def is_occupied(self, pos):
        """
        Identify if cell is occupied by another agent
        """        
        a = self.get_trader(pos)
        if a and not pos == self.pos:
            return True
        else:
            return False
        
    def calc_welfare(self, sugar, spice):
        """
        Calculates current welfare of the agent
        """
        # total resources
        m = self.metabolism_sugar + self.metabolism_spice
        
        # Cobb-Douglas
        return sugar**(self.metabolism_sugar/m) * spice**(self.metabolism_spice/m)
    
    def calc_mrs(self, sugar, spice):
        """
        Calculates the marginal rate of substitution for the agent
        Used in trade() and sell_resource()
        """
        
        return (spice / self.metabolism_spice) / (sugar / self.metabolism_sugar)
    
    def calc_sell_amount(self, price):
        """
        Defines how much resources the agent will sell
        
        In Sugarscape, price is defined as the exchange rate in units of spice per units of sugar (price = spice/sugar),
        
        Agents only trade in whole units - in order to avoid issues with fractional spice, when price is below 1, 
        we change the numeraire good to spice
        
        Ex: when price = 3, 3 spice : 1 sugar
        When price = 0.25, 0.25 spice : 1 sugar, or, in other terms, 1 spice = 4 sugar (which the code belows does) 
        """
        
        if price >= 1:
            sugar = 1
            spice = int(price)
        else:
            sugar = int(1 / price)
            spice = 1
        return sugar, spice
    
    def exchange_resources(self, other, sugar, spice):
        """
        Exchanges sugar and spice between traders
        
        Notice that the function is such that the agent will
        buy sugar from the other agent, and sell it's spice,
        which reflects how price was determined previously 
        (he pays in spice to buy sugar)
        """
        
        self.sugar += sugar
        self.spice -= spice
        other.sugar -= sugar
        other.spice += spice
        
    def sell_spice(
        self, other, price,
        welfare_self, welfare_other
    ):
        """
        Helper function for self.trade()
        """
        
        sugar_exchanged, spice_exchanged = self.calc_sell_amount(price)
        
        # Assess new sugar and spice amount
        
        self_sugar = self.sugar + sugar_exchanged
        self_spice = self.spice  - spice_exchanged
        
        other_sugar = other.sugar - sugar_exchanged
        other_spice = other.spice + spice_exchanged
        
        
        # check if both agents have resources
        if any([self_sugar <= 0, self_spice <= 0, other_sugar <= 0, other_spice <= 0]):
            return False
        
        # Trade criteria #1 - both agents need to be better off
        both_better_off = (
            (welfare_self < self.calc_welfare(self_sugar, self_spice)) and
            (welfare_other < other.calculate_welfare(other_sugar, other_spice))
        )
        
        # Trade criteria #2 - mrs not crossing - this comes from the Edgeworth Box, where trade happens until the mrs is the same for all agents
        mrs_not_crossing = self.calc_mrs(self_sugar, self_spice) > other.calculate(other_sugar, other_spice)
        
        if not (both_better_off and mrs_not_crossing):
            return False

        self.exchange_resources(other, sugar_exchanged, spice_exchanged)
        
        return True
    
    def trade(self, other):
        """
        Helper function that agent uses to trade with neighbors   
        """
        
        # sanity check
        assert self.sugar > 0
        assert self.spice > 0
        assert other.sugar > 0
        assert other.spice > 0
        
        # calculate mrs for both agents
        mrs_self = self.calc_mrs(self.sugar, self.spice)
        mrs_other = self.calc_mrs(other.sugar, other.spice)
        
        # calculate welfare for both agents
        welfare_self = self.calc_welfare(self.sugar, self.spice)
        welfare_other = other.calc_welfare(other.sugar, other.spice)
        
        # condition for trade to happen - if MRS of both is equal, no beneficial trade can happen, stops trade
        if math.isclose(mrs_self, mrs_other):
            return
        
        # calculates price, which is the geometric mean between the two mrs
        price = math.sqrt(mrs_self * mrs_other)
        buyer_or_seller = "seller" if mrs_self > mrs_other else "buyer"
        
        # self sugar buyer, spice seller
        if buyer_or_seller == "seller":
            sold = self.sell_spice(other, price, welfare_self, welfare_other)
            if not sold: # criteria not met - stop trade
                return
        # self is spice buyer, sugar seller
        else:
            sold = other.sell_spice(self, price, welfare_other, welfare_self)
            if not sold: # criteria not met - stop trade
                return
        
        # capture data
        self.prices.append(price)
        self.trade_partners.append(other.unique_id)
        self.bought_or_sold.append(buyer_or_seller)
        
        # continue trading - recursion guarantees that trades occurr until all benefits are exhausted
        self.trade(other)
        
    def move(self):
        """
        Trader agent identifies its optimal move for each step in 4 parts:
        1 - Identify all possible moves
        2 - Determine which move maximizes welfare
        3 - Find closest best option
        4 - Move
        """
        
        # step 1
        neighbors = [
            i for i in self.model.grid.get_neighborhood(
                self.pos, self.moore, True, self.vision
            ) if not self.is_occupied(i)
        ]
        
        # step 2
        welfares = [
            self.calc_welfare(
                self.sugar + self.get_resource(pos).get("sugar").amount,
                self.spice + self.get_resource(pos).get("spice").amount
            ) for pos in neighbors
        ]
        
        # step 3
        # maximize welfare
        max_welfare = max(welfares)
        
        # get index of cells closest to the maximum welfare
        candidates_idx = [
            i for i in range(len(welfares)) if math.isclose(welfares[i], max_welfare)
        ]
        
        # convert index to positions
        candidates = [neighbors[i] for i in candidates_idx]
        
        # get the minimum euclidean distance between itself and the considered positions
        min_dist = min(get_distance(self.pos, pos) for pos in candidates)
        
        # final candidates based on if the distance is close enough
        final_candidates = [
            pos for pos in candidates if math.isclose(get_distance(self.pos, pos), min_dist)
        ]
        
        # shuffle candidates randomly
        self.random.shuffle(final_candidates)
        
        # step 4 - the agent moves
        self.model.grid.move_agent(self, final_candidates[0])
    
    def eat(self):
        """
        The agent eats in order to survive, according to his metabolism
        """
        sugar_patch = self.get_resource(self.pos).get("sugar")
        
        if sugar_patch:
            self.sugar += sugar_patch.amount
            sugar_patch.amount = 0
        self.sugar -= self.metabolism_sugar
        
        spice_patch = self.get_resource(self.pos).get("spice")
        
        if spice_patch:
            self.spice += spice_patch
            spice_patch.amount = 0
        self.spice -= self.metabolism_spice

    def die(self):
        
        """
        Remove Trader that has consumed all their sugar or spice
        """
        # condition: agent is starved
        if (self.sugar <= 0) or (self.spice <= 0):
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            
    def trade_with_neighbors(self):
        """
        Function for trader agents to decide who to trade with
        
        1. Identify neighbors who can trade
        2. Trade
        3. Collect data
        """
        
        neighbor_agents = [
            self.get_trader(pos) for pos in self.model.grid.get_neighborhood(
                self.pos, self.moore, False, self.vision
            ) if self.is_occupied(pos)
        ]
        
        if len(neighbor_agents) == 0:
            return
        
        for a in neighbor_agents:
            if a:
                self.trade(a)
        
        return