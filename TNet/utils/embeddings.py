import pickle
import numpy as np

class Glove:

    def __init__(self, fname):
        self.fname = fname
        self.embeddings_dim = 300

        self._read_data()
        self._build_embeddings()

    def __getitem__(self, value):
        try:
            return self.embeddings[value]

        except KeyError:
            self.embeddings[value] = np.random.uniform(-0.25, 0.25, size=self.embeddings_dim)
            return self.embeddings[value]

    def _read_data(self):
        with open(self.fname, 'r') as file:
            self._raw_data = [
                line for line in file.readlines()
            ]
            file.close()

    def _build_embeddings(self):
        """
        This function will spend a lot of time on building (depends on your computer spec. and embeddings model size.)
        """
        self.embeddings = {}

        for line in self._raw_data:
            splitted_line = line.split()
            concatenated_word = ' '.join(splitted_line[:-self.embeddings_dim])
            self.embeddings[concatenated_word] = np.array(splitted_line[-self.embeddings_dim:], dtype=np.float32)

        self.embeddings[''] = np.random.uniform(-0.25, 0.25, size=self.embeddings_dim)

        # memory recycle
        del self._raw_data
    