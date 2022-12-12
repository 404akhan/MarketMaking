from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
import math

from simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions


class BestPosStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, hold_time:Optional[float] = None) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                #place order
                bid_order = sim.place_order( receive_ts, 0.001, 'BID', best_bid )
                ask_order = sim.place_order( receive_ts, 0.001, 'ASK', best_ask )
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders


class StoikovStrategy:
    def __init__(self) -> None:
        self.q = 0
        self.gamma = 0.1
        self.sigma = 2
        self.delay = pd.Timedelta(1, 's').delta
        self.hold_time = pd.Timedelta(10, 's').delta
        self.T = pd.Timedelta(24, 'h').delta
        self.k = 1.5

        self.base_time = -1


    def ask_bid(self, s, ts):
        T_t = (self.T - (ts - self.base_time)) / self.T
        theta_2 = self.gamma * (self.sigma ** 2) * T_t
        reservation_price = s - self.q * theta_2
        spread = theta_2 + 2 / self.gamma * math.log(1 + self.gamma / self.k)

        return reservation_price + spread / 2, reservation_price - spread / 2


    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        # market data list
        md_list: List[MdUpdate] = []
        # executed trades list
        trades_list: List[OwnTrade] = []
        # all updates list
        updates_list = []
        # current best positions
        best_bid = -np.inf
        best_ask = np.inf
        mid_price = 0

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []

        while True:
            # get update from simulator
            receive_ts, updates = sim.tick()
            # starting reference time
            if self.base_time < 0:
                self.base_time = receive_ts
            if updates is None:
                break
            # save updates
            updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    mid_price = (best_bid + best_ask) / 2
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else:
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts

                # # cancel old
                # to_cancel = []
                # for ID, order in ongoing_orders.items():
                #     sim.cancel_order(receive_ts, ID)
                #     to_cancel.append(ID)
                # for ID in to_cancel:
                #     ongoing_orders.pop(ID)

                # place order
                ask, bid = self.ask_bid(mid_price, receive_ts)
                bid_order = sim.place_order(receive_ts, 0.001, 'BID', bid)
                ask_order = sim.place_order(receive_ts, 0.001, 'ASK', ask)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]


        return trades_list, md_list, updates_list, all_orders