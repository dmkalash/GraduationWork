import os
import pandas as pd
from elements import DocElem, TsElem


class SeriesLoader:
    def __init__(self, base_text_dir, ts_path):
        self.base_text_dir = base_text_dir
        self.ts_path = ts_path

        names = list(map(lambda w: w.split('.')[0], os.listdir(self.base_text_dir)))
        sorted_dates = sorted(pd.to_datetime(names, format='impr_%Y%m%d'))
        self.sorted_names = list(map(lambda w: 'impr_' + str(w).split()[0].replace('-', '') + '.tsv', sorted_dates))

        self.ts_data = pd.read_csv(ts_path, sep='\t')

        self.last_doc = None
        self.last_ts = None

        self.doc_gen = self.generate_new_doc()
        self.ts_gen = self.generate_new_ts()

    def generate_new_doc(self):
        for fname in self.sorted_names:
            try:
                df = pd.read_csv(self.base_text_dir + fname)
            except Exception as e:
                continue
            try:
                df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H:%M')  # %p %Z
            except ValueError:
                df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H:%M %p %Z')  # %p %Z
            df.sort_values(by='ts', inplace=True)
            for i, row in df.iterrows():
                elem = DocElem(row['title'], row['embedding'], row['ts'])
                yield elem
        yield None

    def generate_new_ts(self):
        df = pd.read_csv(self.ts_path)
        df['date'] = pd.to_datetime(df['date'] + ' 16:00:00', format='%Y%m%d %H:%M:%S')
        for i, row in df.iterrows():
            elem = TsElem(row['close'], row['date'])
            yield elem
        yield None

    def get_next_value(self):

        if self.doc_gen is None and self.ts_gen is None:
            return None

        if self.doc_gen is not None and self.last_doc is None:
            self.last_doc = next(self.doc_gen)

        if self.ts_gen is not None and self.last_ts is None:
            self.last_ts = next(self.ts_gen)

        if self.doc_gen is not None and (self.last_ts is None or self.last_doc.time() < self.last_ts.time()):
            last_doc = next(self.doc_gen)
            if last_doc is None:
                self.doc_gen = None
            last_doc, self.last_doc = self.last_doc, last_doc
            return last_doc

        if self.ts_gen is not None and (self.last_doc is None or self.last_doc.time() >= self.last_ts.time()):
            last_ts = next(self.ts_gen)
            if last_ts is None:
                self.ts_gen = None
            last_ts, self.last_ts = self.last_ts, last_ts
            return last_ts

        return None