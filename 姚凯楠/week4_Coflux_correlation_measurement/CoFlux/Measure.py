import pandas as pd
import math


class CorrelationMeasurement():
    coTHR = 0.8

    # def __init__(self):
    #     self.coTHR = 0.8

    def __init__(self, *args):
        if len(args) == 1:
            self.coTHR = args
        else:
            self.coTHR = 0.8

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

    def CC(self, G, H, Gs):
        return self.R(Gs, H) / math.sqrt(self.R(G, G) * self.R(H, H))

    def FCC(self, afx, afy):
        maxCC = 0
        minCC = 0
        s1 = 0
        s2 = 0
        l = len(afx)
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

    def correlation_measurement(self, afx_set, afy_set):
        result_set = []
        for afx in afx_set:
            for afy in afy_set:
                result_set.append(self.FCC(self.amplification(0.5, 10, afx), self.amplification(0.5, 10, afy)))

        maxv = 0
        minv = 0
        for result in result_set:
            maxv = max(maxv, result[0])
            minv = min(minv, result[0])
        if abs(maxv) > abs(minv):
            ccV, shiftV = max(result_set)
        else:
            ccV, shiftV = min(result_set)

        if abs(ccV) >= self.coTHR:
            if shiftV == 0:
                if ccV >= 0:
                    print('x<-+->y')
                else:
                    print('x<--->y')
            elif shiftV < 0:
                if ccV >= 0:
                    print('x-+->y')
                else:
                    print('x--->y')
            else:
                if ccV >= 0:
                    print('x<-+-y')
                else:
                    print('x<---y')
        else:
            print('x=||=y')
        return 0


if __name__ == '__main__':
    afx_set = [[0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0]]
    afy_set = [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0]],
               [[0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0]],
               [[0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0]],
               [[0, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0, 0, 0, 0]],
               [[0, 0, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0, 0, 0]],
               [[0, 0, 0, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0, 0]],
               [[-5, -4, -3, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]

    for item in afy_set:
        cm = CorrelationMeasurement()
        cm.correlation_measurement(afx_set=afx_set, afy_set=item)
