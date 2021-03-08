from datetime import datetime

"""
Price Buckets - Percentage of how much delivery time is faster - increase in price
for example: 0 < x < 10% (faster delivery) -- 0 % increase
10% < x < 20% (faster delivery) -- 2% increase
.....

"""
class PriceBucketsBiddingAgent():
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
        confirmed_lead_time = state['confirmed_orderLT']
        order_length_days = self.days_between(state['orderentry_date'], state['confirmed_delivery_date']) + confirmed_lead_time
        original_price = state['order_quantity'] * state['sales_product']
        percentage = confirmed_lead_time/order_length_days
        calculated_price = original_price

        if percentage > 0.0 and percentage <= 0.1:
            calculated_price += original_price * 0
        elif percentage > 0.1 and percentage <= 0.2:
            calculated_price += original_price * 0.02
        elif percentage > 0.2 and percentage <= 0.35:
            calculated_price += original_price * 0.05
        elif percentage > 0.35 and percentage <= 0.5:
            calculated_price += original_price * 0.1
        elif percentage > 0.5 and percentage <= 0.65:
            calculated_price += original_price * 0.2
        elif percentage > 0.65 and percentage <= 0.8:
            calculated_price += original_price * 0.4
        elif percentage > 0.8 and percentage <= 0.9:
            calculated_price += original_price * 0.8
        elif percentage > 0.9 and percentage < 1.0:
            calculated_price += original_price * 0.9
        else:
            calculated_price += original_price

        action = calculated_price

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
