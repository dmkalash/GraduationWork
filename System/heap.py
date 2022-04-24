
import heapq


class LimitedHeap:

    def __init__(self, limit=50, order='max'):
        """  max - удаляем максимальный по значению элемент"""
        self.heap = []
        self.order = order
        self.limit = limit
        self.counter = 1e-6
        self.counter_step = 1e-6

    def size(self):
        return len(self.heap)

    def push(self, value, elem):
        if self.order == 'max':
            value = -value
        if self.size() < self.limit:
            heapq.heappush(self.heap, (value, self.counter, elem))
            self.counter += self.counter_step
        else:
            heapq.heapreplace(self.heap, (value, self.counter, elem))
            self.counter += self.counter_step

    def values(self):
        return [elem[-1].value().unsqueeze(0) for elem in self.heap]
        #return [elem[-1].value().unsqueeze(0) for elem in self.heap]

    def get_text(self):
        return [elem[-1].text() for elem in self.heap]

    def items(self):
        return self.heap

    def reset(self, init_val=None):
        if init_val is None:
            self.heap = []
        else:
            self.heap = [(elem[0], self.counter, elem[1]) for elem in init_val]
            self.counter += self.counter_step

    def init_by_list(self, init_arr):
        if self.order == 'max':
            self.heap = [(-elem[0], self.counter_step, elem[1]) for elem in init_arr]
        heapq.heapify(self.heap)
