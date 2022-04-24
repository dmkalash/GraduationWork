import json
import torch


class SeriesElem(object):
    def __init__(self):
        self.timestamp = None

    def time(self):
        return self.timestamp

    def elem_type(self):
        raise NotImplemented()

    def value(self):
        raise NotImplemented()


class DocElem(SeriesElem):
    def __init__(self, text, embedding, timestamp):
        super(DocElem, self).__init__()
        self.text = text
        if isinstance(embedding, str):
            replaced = embedding.replace('tensor(', '').replace(')', '')
            replaced = replaced.replace('0.,', '0.0,').replace('0. ', '0.0 ').replace('0.]', '0.0]')
            try:
                self.embedding = torch.tensor(json.loads(replaced)).float()
            except Exception as e:
                print(e, timestamp, embedding)
        else:
            self.embedding = embedding
        self.timestamp = timestamp

    def elem_type(self):
        return 'doc'

    def value(self):
        return self.embedding

    def get_text(self):
        return self.text


class TsElem(SeriesElem):
    def __init__(self, ts, timestamp):
        super(TsElem, self).__init__()
        self.ts = torch.tensor([ts]).float()
        self.timestamp = timestamp

    def elem_type(self):
        return 'ts'

    def value(self):
        return self.ts


class Concater:
    def __init__(self):
        pass

    def __call__(self, a, b):
        if isinstance(a, TsElem) and isinstance(b, DocElem):
            return torch.cat([a.value(), b.value()])
        if isinstance(a, TsElem) and isinstance(b, torch.Tensor):
            return torch.cat([a.value(), b])
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if len(a.shape) < len(b.shape):
                a = a.expand_as(b)
            elif len(a.shape) > len(b.shape):
                b = a.expand_as(a)
            return torch.cat([a, b])
        else:
            raise ValueError(f'Unexpected types for a and b: a is {type(a)} and {type(b)}')

