import torch

from config import Config
from heap import LimitedHeap

class DataHolder:
    def __init__(self, max_text_window_size, max_series_window_size=30):
        self.last_ts = None
        self.config = Config()

        self.y_pred = []
        self.y_true = []
        self.losses = []

        self.last_pred = None
        self.last_ts_hidden = None
        self.last_doc_hidden = None

        self.max_text_window_size = max_text_window_size
        self.max_series_window_size = max_series_window_size

        self.doc_elems = LimitedHeap(self.max_text_window_size)

    def get_result(self, crop=True):
        if crop:
            if len(self.y_true) > self.max_series_window_size:
                self.y_true = self.y_true[-self.max_series_window_size:]
                self.y_pred = self.y_pred[-self.max_series_window_size:]
        return self.y_true[1:], self.y_pred[:-1], self.doc_elems.items()

    def save_doc_step(self, y_pred):
        self.y_pred[-1].append(y_pred.item())

    def save_last_ts(self, last_ts, last_ts_hidden):
        self.last_ts_hidden = last_ts_hidden
        self.last_ts = last_ts
        self.y_true.append(last_ts.item())

    def get_last_ts_hidden(self):
        return self.last_ts_hidden

    def save_loss(self, loss):
        self.losses.append(loss.item())

    def reinit_doc_sequence(self, init_arr):
        if len(init_arr) > self.max_text_window_size:
            init_arr = init_arr[:self.max_text_window_size]

        #init_arr = [(elem[0], elem[1].unsqueeze(0)) for elem in init_arr]
        self.doc_elems.init_by_list(init_arr)
        self.y_pred.append([])

    def update_doc_sequence(self, value, doc_elem):
        #doc_emb = doc_elem.value().unsqueeze(0)
        self.doc_elems.push(value, doc_elem) # doc_emb

    def get_doc_sequence(self):
        return torch.cat(self.doc_elems.values(), axis=0)