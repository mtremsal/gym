import logging, sys
from six import StringIO

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

def get_hand(deck, player):
    hand = []
    for i in range(len(deck)):
        if deck[i] == player:
            hand.append(i)
    return hand

def card_name(n):
    assert type(n) == int
    assert n>=0 and n<=51
    res = "".join([["ace","2","3","4","5","6","7","8","9","10","jack","queen","king"][n%13],
        " of ",
        ["clubs", "diamonds", "hearts", "spades"][n//13]])
    return res

def hand_name(deck, player):
    hand = get_hand(deck, player)
    cards = [card_name(c) for c in hand]
    return ", ".join(cards)

def board_name(deck):
    return 'Dealer Hand: {} | Player Hand: {}'.format(hand_name(deck, -1), hand_name(deck, 1))

def draw_card(deck, player):
    new_deck = deck
    i = np.random.randint(0, 52)
    while new_deck[i] != 0:
        i = np.random.randint(0, 52)
    new_deck[i] = player
    return new_deck

def card_value(n):
    if n%13 in [10, 11, 12]:
        return 10
    elif n%13 == 0:
        return 11
    else:
        return n%13 + 1

def generate_deck():
    deck = np.zeros(52)
    deck = draw_card(deck, 1)
    deck = draw_card(deck, 1)
    deck = draw_card(deck, -1)
    return deck

def eval_hand(deck, player):
    hand = get_hand(deck, player)
    score = sum([card_value(c) for c in hand])
    n_aces = len([c for c in hand if c%13 == 0])

    for i in range(4):
        if score > 21 and n_aces > 0:
            score -= 10
            n_aces -= 1

    if score > 21:
        return score, "bust"
    elif score == 21 and len(hand) == 2:
        return 21, "blackjack"
    elif n_aces > 0:
        return score, "soft"
    else:
        return score, ""

def provide_actions(deck, player):
    if player == -1:
        return [0] # Stand
    else:
        score, qual = eval_hand(deck, 1)
        if score < 21 and qual != "bust" and qual != "blackjack":
            return [0, 1]
        else:
            return [0]

def resolve(deck):
    reward = 0
    info = ""
    d_score, d_qual = eval_hand(deck, -1)
    p_score, p_qual = eval_hand(deck, 1)

    if p_qual == "bust":
        reward = -1
        info = "Player busts"
    elif p_qual == "blackjack" and d_qual == "blackjack":
        reward = 0
        info = "Player pushes"
    elif p_qual == "blackjack":
        reward = 1.5
        info = "Player wins (blackjack)"
    elif d_qual == "blackjack":
        reward = -1
        info = "Player loses (blackjack)"
    elif d_qual == "bust":
        reward = 1
        info = "Player wins"
    elif p_score == d_score:
        reward = 0
        info = "Player pushes"
    elif p_score > d_score:
        reward = 1
        info = "Player wins"
    else:
        reward = -1
        info = "Player loses"

    return reward, info

class BJEnv(gym.Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        # Stand == 0, Hit == 1. On bust or after initial stand the agent is forced to stand.
        self.action_space = [0,1]
        # 52 cards with 0 == in the deck, 1 == in player's hand and -1 == in dealer's hand
        # self.observation_space = spaces.MultiDiscrete([ [0,3] * 52 ])
        self.observation_space = np.zeros(52)
        self._seed()
        self._reset()

    def _step(self, action):
        assert action in self.action_space
        info = ""

        if action == 1: # hit
            # draw a card
            self.observation = draw_card(self.observation, 1)
            p_score, p_qual = eval_hand(self.observation, 1)

            info = board_name(self.observation)

            if p_score > 21:
                done = True
                reward, info = resolve(self.observation)
            else:
                done = False
                reward = 0
            self.allowed_actions = provide_actions(self.observation, 1)

        else: # stand
            self.observation = draw_card(self.observation, -1)

            p_score, p_qual = eval_hand(self.observation, 1)
            d_score, d_qual = eval_hand(self.observation, -1)

            info = board_name(self.observation)

            if d_score > 16:
                done = True
                reward, info = resolve(self.observation)
            else:
                done = False
                reward = 0
            self.allowed_actions = provide_actions(self.observation, -1)
        return self.observation, reward, done, info

    def _reset(self):
        self.observation = generate_deck()
        self.allowed_actions = provide_actions(self.observation, 1)
        return self.observation

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(board_name(self.observation))
        outfile.write("\n")
        return outfile

        # No need to return anything for human
        if mode != 'human':
            return outfile

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
