import miditok
from miditok import REMI, TokenizerConfig


def get_tokenizer_config():
    return TokenizerConfig(
        nb_velocities=16,
        use_chords=False,
        use_programs=False,
        pitch_range=miditok.constants.PITCH_RANGE,
        use_tempos=False,
        use_sustain_pedals=False,
    )


def get_tokenizer(config: TokenizerConfig):
    return REMI(config)


def get_pretrained_tokenizer(path: str = "tokenizer.json"):
    return miditok.REMI.from_pretrained(path)
