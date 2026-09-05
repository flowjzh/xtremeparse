"""Plain-data trace reads: budget aggregation and router-tripwire math."""

from xtremeparse.evalkit import pair_at, per_item_budgets, router_overlap, whole_units


def test_pair_at_clamps_to_the_last_value():
    assert pair_at([10, 20], 0) == 10 and pair_at([10, 20], 1) == 20
    assert pair_at([10, 20], 5) == 20  # a short list's last value covers beyond


def test_whole_units_names_whole_strategy_groups_with_budgets():
    groups = [{'unit': 'jobs', 'budget': [10, 20], 'item': None, 'strategy': 'whole'},
              {'unit': 'edu', 'budget': 50, 'item': 0, 'strategy': 'per-item'},
              {'unit': 'misc', 'budget': 5, 'item': None, 'strategy': None}]
    assert whole_units(groups) == {'jobs'}


def test_per_item_budgets_zips_batch_lists_with_their_items():
    groups = [{'unit': 'jobs', 'budget': 100, 'item': 0},
              {'unit': 'jobs', 'budget': 600, 'item': 1},
              {'unit': 'jobs', 'budget': [150, 300], 'item': None, 'batch': [2, 3]}]
    assert per_item_budgets(groups) == {'jobs': [100, 600, 150, 300]}


def test_per_item_budgets_keeps_a_whole_array_declaration_as_declared():
    groups = [{'unit': 'jobs', 'budget': [10, 200], 'item': None},
              {'unit': 'edu', 'budget': 50, 'item': None}]
    assert per_item_budgets(groups) == {'jobs': [10, 200], 'edu': 50}


def test_per_item_budgets_skips_unbudgeted_groups():
    assert per_item_budgets([{'unit': 'jobs', 'budget': None, 'item': 0},
                             {'unit': 'edu', 'item': 0}]) == {}


def test_router_overlap_counts_duplicates_and_lost_items():
    groups = [{'unit': 'a', 'chunk_ids': [0, 1], 'item': None, 'strategy': None},
              {'unit': 'a', 'item': 0, 'chunk_ids': [2], 'strategy': 'per-item'}]
    assert router_overlap({'assignments': [{'unit': 'a', 'chunks': [0, 1]},
                                           {'unit': 'a', 'item': 0, 'chunks': [1, 2]}]},
                          groups) == (1, 0)
    # item 0 declared but never executed (no group) -> lost; its chunk 5
    # also rides no call -> overlap counts it
    assert router_overlap({'assignments': [{'unit': 'a', 'item': 0, 'chunks': [5]},
                                           {'unit': 'a', 'item': 1, 'chunks': [6, 7]}]},
                          [{'unit': 'a', 'item': 1, 'chunk_ids': [6, 7],
                            'strategy': 'per-item'}]) == (1, 1)
    assert router_overlap(None, []) == (0, 0)


def test_router_overlap_tolerates_whole_strategy_coalescing():
    # routed as items 0..3, executed as one whole-array call: not a loss
    coalesced = [{'unit': 'jobs', 'item': None, 'strategy': 'whole',
                  'chunk_ids': [0, 1, 2, 3]}]
    assignments = [{'unit': 'jobs', 'item': i, 'chunks': [i]} for i in range(4)]
    assert router_overlap({'assignments': assignments}, coalesced) == (0, 0)
