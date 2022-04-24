import torch
from torch import nn

from loader import SeriesLoader
from data_holder import DataHolder
from config import Config
from elements import TsElem, DocElem, Concater

from models import DocModel, KfModel, ScoreModel

from embedder import FastTextEmbedder

class Trainer:
    def __init__(self):
        self.config = Config()

        self.learning_rate = self.config.learning_rate
        self.max_text_window_size = self.config.max_text_window_size
        self.max_series_window_size = self.config.max_series_window_size

        self.loader = SeriesLoader(self.config.text_path, self.config.ts_path)
        self.data_holder = DataHolder(self.max_text_window_size, self.max_series_window_size)
        self.concater = Concater()

        self.embedder = FastTextEmbedder()
        self.pad_embedding = self.embedder.set_pad_embedding()

        self.doc_model = DocModel(self.config.embedding_size, self.config.attention_num_heads,
                             self.config.attention_dropout, self.config.kv_dim,
                                  self.pad_embedding, self.config.doc_output_size)

        self.ts_model = KfModel(self.config.ts_feature_size,
                                self.config.ts_hidden)
        self.head_model = nn.Sequential(nn.Linear(self.config.doc_hidden + self.config.ts_hidden, self.config.cat_linear),
                                        nn.ReLU(),
                                        nn.Linear(self.config.cat_linear, self.config.output_size))

        self.score_model = ScoreModel(self.config.embedding_size,
                                      self.config.ts_hidden,
                                      self.config.score_linear,
                                      self.config.output_size)

        self.optimizer = torch.optim.Adam(
            sum( [list(self.doc_model.parameters()),
                  list(self.ts_model.parameters()),
                  list(self.head_model.parameters()),
                 list(self.score_model.parameters())], [] ),
            lr = self.learning_rate
        )

        self.loss_fn = torch.nn.MSELoss()
        self.empty_doc_elem = DocElem('', torch.zeros(self.config.embedding_size), None)

        self.border = 0


    def step(self):
        while True:
            elem = self.loader.get_next_value()
            #if isinstance(elem, TsElem) or torch.rand(1).item() < 0.05:
            if self.data_holder.get_last_ts_hidden() is None and isinstance(elem, DocElem):
                continue
            if not isinstance(elem, DocElem) or self.border < 7:
                break

        if elem is None:
            print('<>', flush=True)
            return None

        if isinstance(elem, DocElem):
            self.border += 1
            #print('#', end='', sep='', flush=True)
            self.doc_step(elem)
            return self.data_holder.get_result()

        elif isinstance(elem, TsElem):
            self.border = 0
            print('@', end='', sep='', flush=True)
            self.time_series_step(elem)
            self.next_init(elem)
            return self.data_holder.get_result()

        else:
            raise ValueError(f'Unknown type: {type(elem)}')

    def doc_step(self, elem):
        last_ts_hidden = self.data_holder.get_last_ts_hidden()
        if last_ts_hidden is not None:
            self.doc_model.eval()
            self.ts_model.eval()
            self.head_model.eval()
            self.score_model.eval()

            with torch.no_grad():
                doc_score = self.score_model(last_ts_hidden, elem.value().unsqueeze(0)).item()
                self.data_holder.update_doc_sequence(doc_score, elem)
                cur_doc_seq = self.data_holder.get_doc_sequence()
                last_doc_hidden = self.doc_model(cur_doc_seq)

                cat_hidden = self.concater(last_ts_hidden, last_doc_hidden)
                y_pred = self.head_model(cat_hidden)

                self.data_holder.save_doc_step(y_pred)

    def time_series_step(self, elem):
        last_ts_hidden = self.data_holder.get_last_ts_hidden()
        if last_ts_hidden is not None:

            self.head_model.train()
            self.doc_model.train()
            self.ts_model.train()
            self.score_model.eval()

            # main model training
            cur_doc_seq = self.data_holder.get_doc_sequence()

            last_doc_hidden = self.doc_model(cur_doc_seq)

            cat_hidden = self.concater(last_ts_hidden, last_doc_hidden)
            y_pred = self.head_model(cat_hidden)

            loss = self.loss_fn(y_pred, elem.value())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.data_holder.save_loss(loss)

            # score model training
            self.head_model.eval()
            self.doc_model.eval()
            self.ts_model.eval()
            self.score_model.train()

            cur_doc_seq = self.data_holder.get_doc_sequence()

            scores_true = []
            for doc_tensor in cur_doc_seq:
                last_doc_hidden = self.doc_model(doc_tensor.unsqueeze(0))
                cat_hidden = self.concater(last_ts_hidden, last_doc_hidden)
                y_pred = self.head_model(cat_hidden)
                score = (y_pred - elem.value()) ** 2
                scores_true.append(score.item())

            scores_true = torch.tensor(scores_true).float()
            doc_preds = self.score_model(last_ts_hidden, cur_doc_seq)
            doc_scores = (doc_preds - elem.value()) ** 2

            loss = self.loss_fn(doc_scores.squeeze(1), scores_true)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)


    def next_init(self, elem):
        self.doc_model.eval()
        self.ts_model.eval()
        self.head_model.eval()
        self.score_model.eval()

        with torch.no_grad():
            self.data_holder.reinit_doc_sequence([(1e9, self.empty_doc_elem)])

            cur_doc_seq = self.data_holder.get_doc_sequence()
            last_doc_hidden = self.doc_model(cur_doc_seq)

            last_ts = elem.value()
            last_ts_hidden = self.ts_model(last_ts)

            cat_hidden = self.concater(last_ts_hidden, last_doc_hidden)
            y_pred = self.head_model(cat_hidden)

            self.data_holder.save_doc_step(y_pred)
            self.data_holder.save_last_ts(elem.value(), last_ts_hidden)
