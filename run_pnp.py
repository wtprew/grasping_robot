#!/usr/bin/env python
from tasks.pnp import PickAndPlace

if __name__ == '__main__':
    pnp = PickAndPlace(
        place_position=[0.18, 0, 0.14],
    )
    pnp.run()
