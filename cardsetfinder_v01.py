import numpy as np
import gym
import random

class CardSetFinder(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        cardset_tot = [10,10],
        cardset_goal = [3,3],
        hand_limit = 3):
        
        self.reward_for_set = 3
        self.reward_for_invalid_action = -5
        self.reward_for_win = 10
        self.reward_for_lose = -10
        
        self.cardset_tot = cardset_tot
        self.cardset_goal = cardset_goal
        self.hand_limit = hand_limit
        self.sets_num = len(self.cardset_tot)
        
        self.won = False
        self.lost = False
        
        # actions: to discard one of the cards
        # we will code it as a discrete 0 to sets_num
        # so an action is to discard a type of card, not a specific card
        # NB there will be invalid actions, not all of the card types will be in hand all the time
        self.action_space = gym.spaces.Discrete(self.sets_num)
        
        # states are going to be coded in a matrix with sets_num number of rows and  columns
        # each row corresponds to a type of card
        # col 0: number of cards in the deck, min: 0, max: cardset_tot
        # col 1: ... in hand, min: 0, max: hand limit
        # col 2: 1 if we already have the set for that type (in which case the cards are useless), 0 if not
        self.observation_space = gym.spaces.Box(
            low = np.tile(np.array([0,0,0]),(self.sets_num,1)), 
            high = np.tile(np.array([max(self.cardset_tot),self.hand_limit,1]),(self.sets_num,1)), 
            dtype = int)
        
        # note: this is slightly inefficient, we could limit each row's 1st col by the corresponding card's number
        # not goin to bother now
    
    def create_deck(self):
        # creates a list of 0, 1, 2, ... etc in random order
        
        for i in range(0, self.sets_num):
            self.deck += [i] * self.cardset_tot[i]
            
        random.shuffle(self.deck)
        
    def draw_cards(self):
        # draws cards until hand limit is met, or until we run out of cards
        curr_handsize = sum(self.hand)
        if curr_handsize < self.hand_limit:
            for i in range(curr_handsize, min(self.hand_limit,curr_handsize+len(self.deck))):
                self.draw_card()
        
    def draw_card(self):
        # draws the top card from the dack to hand
        current_card = self.deck[-1]
        self.hand[current_card] += 1
        self.deck.pop()

    def step(self, action):
        
        reward = 0
        done = False
        info = {}

        # incoming action is a number between 0 and sets_num
        if self.hand[action] == 0:
            # does not change anything, just returns invalid action penalty
            reward = self.reward_for_invalid_action
        else:
            # deletes a card from hand
            self.discard_card(action)

            # checks for set (at this point too, we can find new sets immediately, need to loop
            # e.g. as soon as we discard one set, we draw new cards, and it could be that we draw a set            
            found_sets = 1
            while found_sets > 0:
                self.draw_cards()
                found_sets = self.check_hand_for_sets()
                reward += found_sets * self.reward_for_set
            
            self.calc_state()
            
            # at this point, also needs to check if the game is done
            self.check_if_won()
            if self.won:
                reward += self.reward_for_win
            self.check_if_lost()
            if self.lost:
                reward += self.reward_for_lose

            done = self.won | self.lost
            
        return self.state, reward, done, info

    def reset(self):
        
        # set parameters to starting position
        
        # state: in the format of observation_space, 3 columns, sets_num rows
        self.state = np.zeros((self.sets_num, 3), dtype = int)
        # deck: a list of randomly arranged integers, from 0 to 
        self.deck = []
        self.create_deck()
        # hand: a list of sets_num, each element is an integer with the number of cards hold 
        self.hand = np.zeros(self.sets_num, dtype = int)
        
        found_sets = 1
        while found_sets > 0:
            self.draw_cards()
            found_sets = self.check_hand_for_sets()
            # here, we are not giving rewards, this is the first draw of the game, and sets are automatic
        
        self.calc_state()
        
        return self.state
    
    def check_hand_for_sets(self):
        # in the hand set, checks each card type
        # sees if we have any that is equal to the limit
        # only those sets that are not found yet
        
        found_sets = 0
        
        for i in range(0, self.sets_num):
            if self.state[i,2]==0:
                if self.check_hand_for_set(i):
                    found_sets +=1
                    
        return found_sets
                
    def check_hand_for_set(self, set_num):
        set_found = False
        if self.hand[set_num] >= self.cardset_goal[set_num]:
            # discard those cards from hand
            self.hand[set_num] -= self.cardset_goal[set_num]
            # set the last element of state as DONE (to 1 from 0)
            self.state[set_num,2]=1
            set_found = True
            
        return set_found
            
    def calc_state(self):
        # calculates the state variable based on deck list and hand set
        # only changes first two columns
        # the third one, whether we have already found the set, is handled in the check_hand_for_set
        for i in range(0, self.sets_num):
            self.state[i,0] = self.deck.count(i)
            self.state[i,1] = self.hand[i]
            
    def discard_card(self, set_to_discard):
        self.hand[set_to_discard] -= 1
    
    def check_if_won(self):
        # the game is won if all the sets are found
        self.won = min(self.state[:,2]) == 1
    
    def check_if_lost(self):
        # the game is lost if not all the sets are found, but we have nothing in the deck
        if self.won == False:
            self.lose = len(self.deck)==0
    
    