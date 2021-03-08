import math
from datetime import datetime
"""
Convex Pricing Strategy
Formula: log(standard delivery  time/ confirmed lead time) * 0.5
"""
class ConvexBiddingAgent():
    def _load_config(self):
        """
        Add configurations if needed
        """

    def __init__(self):
        self._load_config()
        self.wins_e = 0 #wins in each episode
        self.total_wins =0 #total wins
        self.total_rewards = 0.0
        self.bid_count = 0.0

    def act(self,state,reward, cost):
        # Formula written at the top of the file

        # Step 1: Calculate number of days between order entry date and
        # confirmed delivery date
        confirmed_lead_time = state['confirmed_orderLT']
        original_price = state['order_quantity'] * state['sales_product']
        order_length_days = self.days_between(state['orderentry_date'], state['confirmed_delivery_date']) + confirmed_lead_time
        percentage = order_length_days/confirmed_lead_time
        print("percentage is {}".format(percentage))
        price_premium = math.log2(percentage) * 0.5 * original_price


        action = price_premium

        if reward > 0:
            self.wins_e += 1
            self.total_wins += 1
            self.total_rewards += 1

        self.bid_count += 1

        return action

    def done(self):
        return self.bid_count < 7000

    def days_between(self,d1, d2):
        d1 = datetime.strptime(d1, "%m.%Y")
        d2 = datetime.strptime(d2, "%m.%Y")
        return abs((d2 - d1).days)