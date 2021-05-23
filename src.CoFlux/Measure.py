import pandas as pd
import math
from tqdm import tqdm


class CorrelationMeasurement():
    coTHR = 0.8

    def __init__(self, *args):
        if len(args) == 1:
            self.coTHR = args[0]
        else:
            self.coTHR = 0

    def amplification(self, a, b, x):
        temp_result = []
        for v in x:
            if v >= 0:
                temp = math.pow(math.exp(1), a * min(v, b)) - 1
                temp_result.append(temp)
            else:
                temp = -math.pow(math.exp(1), a * min(-v, b)) + 1
                temp_result.append(temp)
        result = pd.Series(temp_result)
        return result

    def R(self, Gs, H):
        summ = 0
        for index in range(len(Gs)):
            summ += Gs[index] * H[index]
        return summ

    # RuntimeWarning: invalid value encountered in double_scalars
    def CC(self, G, H, Gs):
        return self.R(Gs, H) / math.sqrt(self.R(G, G) * self.R(H, H))

    def FCC(self, afx, afy, max_dis):
        """
        :param afx:
        :param afy:
        :param max_dis: the maximum distance that the algorithm will check
        :return:
        """
        maxCC = 0
        minCC = 0
        s1 = 0
        s2 = 0
        l = len(afx) if not max_dis else max_dis
        for i in range(-l, l + 1):
            Gs = [0] * l
            if i < 0:
                for j in range(l - abs(i)):
                    Gs[j] = afx[j - i]
            else:
                for j in range(l - abs(i)):
                    Gs[j + i] = afx[j]

            tempcc = self.CC(afx, afy, Gs)
            temparg = i
            if tempcc > maxCC:
                maxCC = tempcc
                s2 = temparg
            if tempcc < minCC:
                minCC = tempcc
                s1 = temparg

        if abs(maxCC) < abs(minCC):
            return minCC, s1
        else:
            return maxCC, s2

    def correlation_measurement(self, afx_set, afy_set, max_dis=None):
        result_set = []
        cnt = 0
        for afx in tqdm(afx_set):
            for afy in afy_set:
                if len(afx) == 0 or len(afy) == 0:
                    continue
                else:
                    result_set.append(self.FCC(self.amplification(0.5, 10, afx), self.amplification(0.5, 10, afy), max_dis))
                cnt += 1
        maxv = 0
        minv = 0
        if len(result_set) == 0:
            print("no effective result")
            return
        for result in result_set:
            maxv = max(maxv, result[0])
            minv = min(minv, result[0])
        if abs(maxv) > abs(minv):
            ccV, shiftV = max(result_set)
        else:
            ccV, shiftV = min(result_set)

        if abs(ccV) >= self.coTHR:
            if ccV >= 0:
                return (1, shiftV), ccV
            else:
                return (-1, shiftV), ccV
        else:
            return (0, 0), ccV


if __name__ == '__main__':
    afx_set = [[0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0]]

    afy_set = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0]],
               [[0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0]],
               [[0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0]],
               [[0, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0, 0, 0, 0]],
               [[0, 0, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0, 0, 0]],
               [[0, 0, 0, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0, 0]],
               [[-5, -4, -3, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]

    for item in afy_set:
        cm = CorrelationMeasurement()
        (pos_neg, shift), ccv = cm.correlation_measurement(afx_set=afx_set, afy_set=item)
        if pos_neg == 1:
            print("[pos] ", end="")
        elif pos_neg == -1:
            print("[neg] ", end="")
        else:
            print("no correlation ", end="")
        print("shift: {}, ccv: {:.4f}".format(shift, ccv))
