from settings.common import DefaultSettings
import os


class NERSettings(DefaultSettings):
    num_labels = 9
    entity_vocab_file = os.path.join(super().data_path, "entity_vocab.txt")
