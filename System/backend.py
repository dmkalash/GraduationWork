import numpy as np
import pandas as pd

from trainer import Trainer

class BackEnd:
    def __init__(self):
        self.trainer = Trainer()
        self.base_df = pd.DataFrame( columns=['value', 'timestamp','forecast'])
        self.base_doc_df = pd.DataFrame( columns=['text', 'error'])

    def get_forecast(self, y_pred):
        return y_pred[-1][-1] #[pred[-1] for pred in y_pred]

    def step(self):
        y_true, y_pred, documents = self.trainer.step()

        if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
            return self.base_df, self.base_doc_df

        #print('sizes:', len(y_true), len(y_pred), len(documents), flush=True)

        # assert isinstance(y_true, list) and isinstance(y_pred, list)
        forecast = self.get_forecast(y_pred)
        value = y_true + [y_true[-1], forecast]
        tms = list(range(len(y_true)))
        tms += [tms[-1], tms[-1] + 1]
        df = pd.DataFrame( {'value' : value,
                        'timestamp' : tms,
                        'forecast' : ['previous values'] * len(y_true)  +  ['forecast'] * 2
                       })

        doc_df = pd.DataFrame({
            'text' : [elem[-1].get_text() for elem in documents],
            'error': [elem[0] for elem in documents],
        })

        return df, doc_df