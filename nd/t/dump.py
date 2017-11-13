import numpy as np
import msgpack
import msgpack_numpy as m
m.patch()

def dump(path, x):
    with open(path, 'wb') as f:
        f.write(msgpack.packb(x, default=m.encode))


dump("int_2_3.mp", np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        ])
    )
     
dump("float_3_2.mp", np.array(
    [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        ])
    )


