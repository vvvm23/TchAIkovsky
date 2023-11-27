import miditok
from miditok import REMI, TokenizerConfig


def get_tokenizer_config(
    num_velocities: int = 16,
    use_chords: bool = False,
    use_tempos: bool = False,
    use_sustain_pedal: bool = False,
):
    return TokenizerConfig(
        nb_velocities=num_velocities,
        use_chords=use_chords,
        use_programs=False,
        pitch_range=miditok.constants.PITCH_RANGE,
        use_tempos=use_tempos,
        use_sustain_pedals=use_sustain_pedal,
    )


def get_tokenizer(config: TokenizerConfig):
    return REMI(config)


def get_pretrained_tokenizer(path: str = "tokenizer.json"):
    return miditok.REMI.from_pretrained(path)
