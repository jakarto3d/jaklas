from struct import unpack
from typing import BinaryIO


class Header:
    def __init__(self, file_object: BinaryIO):
        def get_prop(offset, format_, size):
            file_object.seek(offset)
            return list(unpack(format_, file_object.read(size)))

        self.scale = get_prop(131, "ddd", 8 * 3)
        self.offset = get_prop(155, "ddd", 8 * 3)

        min_max = get_prop(179, "dddddd", 8 * 6)

        self.min = [min_max[1], min_max[3], min_max[5]]
        self.max = [min_max[0], min_max[2], min_max[4]]
