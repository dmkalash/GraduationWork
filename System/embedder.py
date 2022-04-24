import torch
import fasttext as ft


class FastTextEmbedder:
    def __init__(self):
        pass

    def load(self, model_file='../fasttext/cc.en.300.bin'):
        self.model = ft.load_model(model_file)
        return self

    def reset(self):
        del self.model

    def preprocess(self, doc):
        if not isinstance(doc, str):
            print(f'Text type is {type(doc)}, need str. Move empty insteed')
            return ''
        return doc.replace('\n', ' ')

    def __call__(self, doc):
        doc = self.preprocess(doc)
        sentence = self.model.get_sentence_vector(doc)
        sentence = torch.tensor(sentence)
        return sentence

    def embedding_size(self):
        return 300

    def set_pad_embedding(self):
        # self.load()
        self.pad_embedding = torch.zeros((self.embedding_size(),)) # self('<cls>')
        # self.reset()
        return self.pad_embedding