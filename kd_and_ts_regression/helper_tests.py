import helpers

locs = [10, 53, 50, 50, 50, 6, 6, 53]
assert(helpers.priority_order(locs,0) == [50,53,6,10])
assert(helpers.priority_order(locs,3) == [50,6,10])
assert(helpers.priority_order(locs,4) == [50,6])