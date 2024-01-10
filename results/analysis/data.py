import os
import numpy as np


class Data:
    INFO = '[INFO]'
    END_INFO = '[/INFO]'
    DATA = '[DATA]'
    END_DATA = '[/DATA]'
    L = '[L]'
    D = '[D]'

    def __init__(self, base_path, data_file_name):
        path = os.getcwd() + base_path + data_file_name
        with open(path) as file:
            lines = file.readlines()
            lines = [l.rstrip('\n') for l in lines]

        start_info = lines.index(self.INFO) + 1
        end_info = lines.index(self.END_INFO)
        start_data = lines.index(self.DATA) + 1
        end_data = lines.index(self.END_DATA)

        self.params = {}
        for p in lines[start_info:end_info]:
            pp = p.split(' ')
            if pp[1].isnumeric():
                self.params[pp[0]] = int(pp[1])
            elif self.is_float(pp[1]):
                self.params[pp[0]] = float(pp[1])
            else:
                self.params[pp[0]] = " ".join(pp[1:])

        temp = []
        for p in lines[start_data:end_data]:
            pp = p.split(' ')
            if pp[0] == self.L:
                temp.append([' '.join(pp[1:])])
            elif pp[0] == self.D:
                temp[-1].append([int(x) for x in pp[1:]])

        self.data = {}
        for x in temp:
            self.data[x[0]] = np.array(x[1])

    def is_float(self, num_str):
        return num_str.replace('.', "").isnumeric()

    @staticmethod
    def break_lines(string, line_length):
        parts = string.split(' ')
        res = ""
        last = 0
        lines = 1
        for p in parts:
            if len(res) + len(p) - last > line_length:
                res += '\n'
                last = len(res) - lines
                lines += 1
            elif len(res) > 0:
                res += ' '
            res += p
        return res