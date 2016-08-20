import nose

import gym

from gym import envs
from gym.envs.mlady import bjenv
from gym.envs.mlady.bjenv import eval_hand, resolve, provide_actions

import numpy as np

def build_deck(player_hand, dealer_hand):
    deck = np.zeros(52)
    for card in player_hand:
        deck[card] = 1
    for card in dealer_hand:
        deck[card] = -1
    return deck

def setup_func():
    test_decks = {}
    # unresolved boards
    test_decks["5"] = build_deck([1,2],[50])
    test_decks["9"] = build_deck([1,2,3],[50])
    test_decks["12"] = build_deck([1,10],[50])
    test_decks["20"] = build_deck([10,11],[50])
    test_decks["soft 13"] = build_deck([0,1],[50])
    test_decks["soft 20"] = build_deck([0,8],[50])
    test_decks["blackjack"] = build_deck([0,10],[50])
    test_decks["21"] = build_deck([0,10,11],[50])
    test_decks["bust"] = build_deck([4,6,10],[50])
    test_decks["bust with 1 ace"] = build_deck([0,4,5,10],[50])
    test_decks["bust with 2 aces"] = build_deck([0,13,3,5,10],[50])
    test_decks["hard 21 with 2 aces"] = build_deck([0,13,3,4,10],[50])
    # resolved boards
    test_decks["bj vs bj"] = build_deck([0,10],[13,23])
    test_decks["bj vs 21"] = build_deck([13,23],[0,10,11])
    test_decks["21 vs bj"] = build_deck([0,10,11],[13,23])
    test_decks["5 vs bust"] = build_deck([1,2],[4,6,10])
    test_decks["21 vs 20"] = build_deck([0,10,11],[23,24])
    test_decks["20 vs 21"] = build_deck([23,24],[0,10,11])
    test_decks["soft 20 vs 19"] = build_deck([0,8],[21,22])
    test_decks["soft 20 vs 21"] = build_deck([0,8],[13,23,24])
    test_decks["soft 13 vs 17"] = build_deck([0,1],[6,9])
    # impossible boards
    test_decks["bust vs bust"] = build_deck([10,11,12],[23,24,25]) # cannot happen in practice as the player loses first
    test_decks["15 vs 16"] = build_deck([4,9],[18,23]) # cannot happen in practice as the dealer must hit
    return test_decks

def eval_hand_test():
    test_decks = setup_func()
    assert eval_hand(test_decks["5"], 1) == (5, "")
    assert eval_hand(test_decks["9"], 1) == (9, "")
    assert eval_hand(test_decks["12"], 1) == (12, "")
    assert eval_hand(test_decks["20"], 1) == (20, "")
    assert eval_hand(test_decks["soft 13"], 1) == (13, "soft")
    assert eval_hand(test_decks["soft 20"], 1) == (20, "soft")
    assert eval_hand(test_decks["blackjack"], 1) == (21, "blackjack")
    assert eval_hand(test_decks["21"], 1) == (21, "")
    assert eval_hand(test_decks["bust"], 1) == (22, "bust")
    assert eval_hand(test_decks["bust with 1 ace"], 1) == (22, "bust")
    assert eval_hand(test_decks["bust with 2 aces"], 1) == (22, "bust")
    assert eval_hand(test_decks["hard 21 with 2 aces"], 1) == (21, "")

def resolve_test():
    test_decks = setup_func()
    assert 1.5 in resolve(test_decks["blackjack"])
    assert -1 in resolve(test_decks["bust"])
    assert -1 in resolve(test_decks["bust with 1 ace"])
    assert -1 in resolve(test_decks["bust with 2 aces"])
    assert 0 in resolve(test_decks["bj vs bj"])
    assert 1.5 in resolve(test_decks["bj vs 21"])
    assert -1 in resolve(test_decks["21 vs bj"])
    assert 1 in resolve(test_decks["5 vs bust"])
    assert 1 in resolve(test_decks["21 vs 20"])
    assert -1 in resolve(test_decks["20 vs 21"])
    assert 1 in resolve(test_decks["soft 20 vs 19"])
    assert -1 in resolve(test_decks["soft 20 vs 21"])
    assert -1 in resolve(test_decks["soft 13 vs 17"])
    assert -1 in resolve(test_decks["bust vs bust"])
    assert -1 in resolve(test_decks["15 vs 16"])

def provide_actions_test():
    test_decks = setup_func()
    assert 1 in provide_actions(test_decks["5"], 1)
    assert 1 in provide_actions(test_decks["9"], 1)
    assert 1 in provide_actions(test_decks["12"], 1)
    assert 1 in provide_actions(test_decks["20"], 1)
    assert 1 in provide_actions(test_decks["soft 13"], 1)
    assert 1 in provide_actions(test_decks["soft 20"], 1)
    assert not 1 in provide_actions(test_decks["blackjack"], 1)
    assert not 1 in provide_actions(test_decks["21"], 1)
    assert not 1 in provide_actions(test_decks["bust"], 1)
    assert not 1 in provide_actions(test_decks["bust with 1 ace"], 1)
    assert not 1 in provide_actions(test_decks["bust with 2 aces"], 1)
    assert not 1 in provide_actions(test_decks["hard 21 with 2 aces"], 1)
    for deck in test_decks:
        assert not 1 in provide_actions(deck, -1)
