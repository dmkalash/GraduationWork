class Config:
    def __init__(self):
        self.text_path = '../../главные датасеты/dji_v1/reuters_fasttext_real_embedded_trial3_replaced/'
        self.ts_path = '../../главные датасеты/dji_v1/dji.csv'

        self.random_seed = 2022
        self.learning_rate = 0.01

        self.ts_hidden = 128
        self.cat_linear = 128
        self.ts_feature_size = 1
        self.output_size = 1
        self.score_linear = 128

        self.max_window_size = 50
        self.embedding_size = 300
        self.attention_num_heads = 1
        self.kv_dim = 64
        self.doc_output_size = 16
        self.doc_hidden = self.doc_output_size
        self.attention_dropout = 0.0

        self.max_text_window_size = 3
        self.max_series_window_size = 30

        #self.graph_freq = 30
