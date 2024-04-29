import os
import phonemizer


from phonemizer.separator import Separator
from phonemizer.utils import list2str, str2list
from typing import List, Union
phonemizer_backend = None
separator = None
def _phonemize(  # pylint: disable=too-many-arguments
        backend,
        text: Union[str, List[str]],
        separator: Separator,
        strip: bool,
        njobs: int,
        prepend_text: bool,
        preserve_empty_lines: bool):
    """Auxiliary function to phonemize()

    Does the phonemization and returns the phonemized text. Raises a
    RuntimeError on error.

    """
    # remember the text type for output (either list or string)
    text_type = type(text)

    # force the text as a list
    text = [line.strip(os.linesep) for line in str2list(text)]

    # if preserving empty lines, note the index of each empty line
    if preserve_empty_lines:
        empty_lines = [n for n, line in enumerate(text) if not line.strip()]

    # ignore empty lines
    text = [line for line in text if line.strip()]

    if (text):
        # phonemize the text
        phonemized = backend.phonemize(
            text, separator=separator, strip=strip, njobs=njobs)
    else:
        phonemized = []

    # if preserving empty lines, reinsert them into text and phonemized lists
    if preserve_empty_lines:
        for i in empty_lines: # noqa
            if prepend_text:
                text.insert(i, '')
            phonemized.insert(i, '')

    # at that point, the phonemized text is a list of str. Format it as
    # expected by the parameters
    if prepend_text:
        return list(zip(text, phonemized))
    if text_type == str:
        return list2str(phonemized)
    return phonemized

def g2p(text, language='en-us'):
    global tokenizer, phonemizer_backend, separator
    if phonemizer_backend is None:
        phonemizer_backend = phonemizer.backend.espeak.espeak.EspeakBackend(language=language,
                                                                            preserve_punctuation=True, with_stress=False,
                                                                            language_switch="remove-flags")
    if separator is None:
        seperator = Separator(phone='', syllable='', word=' ')

    phones = _phonemize(phonemizer_backend, text, seperator, strip=True, njobs=1, prepend_text=False, preserve_empty_lines=False)
    return phones


if __name__ == "__main__":
    # print(get_dict())
    # print(eng_word_to_phoneme("hello"))
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))
    # all_phones = set()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    #         for ph in group:
    #             all_phones.add(ph)
    # print(all_phones)
