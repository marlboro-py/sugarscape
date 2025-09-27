import math
from utils import get_distance
from mesa.discrete_space import CellAgent


class Trader(CellAgent):
    """
    Trader:
    - has a metabolism of sugar and spice
    - harvest and trade sugar and spice to survive
    """
    
    def __init__(
        self, 
        model,
        cell,
        sugar=0,
        spice=0,
        metabolism_sugar=0,
        metabolism_spice=0,
        vision=0
    ):
        super().__init__(model)
        self.cell = cell
        self.sugar = sugar
        self.spice = spice
        self.metabolism_sugar = metabolism_sugar 
        self.metabolism_spice = metabolism_spice
        self.vision = vision
        self.bought_or_sold = []
        self.prices = []
        self.trade_partners = []
    
    def get_trader(self, cell):
        """
        Identify if agent is of type trader, used in self.trade_with_neighbors()
        """    
        for agent in cell.agents:
            if isinstance(agent, Trader):
                return agent 
            
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
        if any(x <= 0 for x in [self_sugar, self_spice, other_sugar, other_spice]):
            return False
        
        # Trade criteria #1 - both agents need to be better off
        both_better_off = (
            (welfare_self < self.calc_welfare(self_sugar, self_spice)) and
            (welfare_other < other.calc_welfare(other_sugar, other_spice))
        )
        
        # Trade criteria #2 - mrs not crossing - this comes from the Edgeworth Box, where trade happens until the mrs is the same for all agents
        mrs_not_crossing = self.calc_mrs(self_sugar, self_spice) > other.calc_mrs(other_sugar, other_spice)
        
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
        mrs_other = other.calc_mrs(other.sugar, other.spice)
        
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
           cell for cell in self.cell.get_neighborhood(self.vision, include_center=True) if cell.is_empty
        ]
        
        # step 2
        welfares = [
            self.calc_welfare(self.sugar + cell.sugar, self.spice + cell.spice) for cell in neighbors
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
        min_dist = min(get_distance(self.cell, cell) for cell in candidates)
        
        # final candidates based on if the distance is close enough
        final_candidates = [
            cell for cell in candidates if math.isclose(get_distance(self.cell, cell), min_dist, rel_to=1e-02)
        ]
        
        # step 4
        self.cell = self.random.choice(final_candidates) 
    
    def eat(self):
        """
        The agent eats in order to survive, according to his metabolism
        """
        self.sugar += self.cell.sugar
        self.cell.sugar = 0
        self.sugar -= self.metabolism_sugar
        
        self.spice += self.cell.spice
        self.cell.spice = 0
        self.spice -= self.metabolism_spice

    def die(self):
        
        """
        Remove Trader that has consumed all their sugar or spice
        """
        # condition: agent is starved
        if (self.sugar <= 0) or (self.spice <= 0):
            self.remove()
            
    def trade_with_neighbors(self):
        """
        Function for trader agents to decide who to trade with
        
        1. Identify neighbors who can trade
        2. Trade
        3. Collect data
        """
        for a in self.cell.get_neighborhood(radius=self.vision).agents:
            self.trade(a)
        
        return