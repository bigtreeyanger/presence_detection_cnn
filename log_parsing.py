#!/usr/bin/env python3

import numpy as np
from struct import unpack, calcsize
import collections

   
class ParseDataFile:
    def __init__(self):
        self.fmt = '<HQHHBBBBBBBBBBBH'
        self.fmt_sz = calcsize(self.fmt)
        self.bit_per_byte = 8
        self.bit_per_symbol = 10 # highest bit is the sign
        self.packet_format = collections.namedtuple('packet_format', 'field_len, timestamp, csi_len, tx_channel, '
                                                                'err_info, noise_floor, rate, bw, num_tones,'
                                                                'nr, nc, rssi, rssi1, rssi2, rssi3, payload_len')

    def parse(self, filename, has_payload=False):
        print("reading file {}".format(filename))
        byte_file = np.fromfile(filename, np.uint8)
        file_size = byte_file.shape[0]
        print('file size '+str(file_size))
        frame_data = []
        count = 0
        offset = 0
        while offset < file_size-self.fmt_sz:
            current_format = self.packet_format._make(unpack(self.fmt,
                                                             byte_file[offset:offset+self.fmt_sz]))
            if current_format.field_len == 0:
                # print("field length is 0")
                offset += 2
                continue
            if (offset + current_format.field_len) > file_size-4:
                break
            offset += self.fmt_sz
            num_tones = current_format.num_tones
            nc = current_format.nc
            nr = current_format.nr
            csi_len = current_format.csi_len
            if csi_len > 0:
                if csi_len != int(nc * nr *num_tones * 2 * self.bit_per_symbol/8):
                    print("incorrect csi len "+str(csi_len))
                    break
                csi_byte = byte_file[offset:offset+csi_len]
                csi_bits = np.unpackbits(csi_byte)
                csi_bits = np.reshape(csi_bits, (8, int(len(csi_bits)/8)), order='F')
                permutation = range(7, -1, -1)
                csi_bits = csi_bits[permutation, :]
                csi_bits = np.reshape(csi_bits, (self.bit_per_symbol, int(csi_len*8/self.bit_per_symbol)), order='F')
                csi_num = np.zeros((1, nr*nc*num_tones*2), np.float32)
                csi_bits = csi_bits.astype(np.uint16)
                for i in range(self.bit_per_symbol):
                    csi_num[0, :] += (csi_bits[i, :] << i)
                csi_num[0, :] -= (csi_bits[self.bit_per_symbol-1, :]*(1 << self.bit_per_symbol))
                csi = csi_num[0, 1::2] + 1j*csi_num[0, 0::2]
                csi = np.reshape(csi, (nr, nc, num_tones), order='F')
                offset += csi_len
                # skip H which has zero entry (in amplitude)
                min_abs = np.amin(np.abs(csi))
                if min_abs >= 1:
                    this_frame = {"format": current_format,
                                  "csi": csi,
                                  "rssi": current_format.rssi}
                    frame_data.append(this_frame)
                    count += 1

            offset += current_format.payload_len if has_payload else 0

            if offset + 420 > file_size:
                break

        # print("finished parsing file, total number of valid frame (has csi): "+str(len(frame_data)))
        return frame_data

