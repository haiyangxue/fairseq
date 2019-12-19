import numpy as np
import torch

from .. import data_utils, LanguagePairDataset
import torchaudio
import torchaudio.compliance.kaldi as kaldi
# from . import kaldi as kaldi

import os


def collate_frames(frames):
    """Convert a list of 2d frames into a padded 3d tensor
    Args:
        frames (list): list of 2d frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3d tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    len_max = max(frame.size(0) for frame in frames)
    f_dim = frames[0].size(1)
    res = frames[0].new(len(frames), len_max, f_dim).fill_(0.0)

    for i, v in enumerate(frames):
        res[i, : v.size(0)] = v

    return res

def collate(
    samples,pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    # src_lengths, sort_order = src_lengths.sort(descending=True)
    # print(id)
    # id = id.index_select(0, sort_order)
    # src_tokens = src_tokens.index_select(0, sort_order)
    # print(id)
    # exit()
    frames = collate_frames([s["audio"] for s in samples])
    # sort samples by descending number of frames
    frames_lengths = torch.LongTensor([s["audio"].size(0) for s in samples])
    # frames_lengths, sort_order = frames_lengths.sort(descending=True)
    # id = id.index_select(0, sort_order)
    # frames = frames.index_select(0, sort_order)
    # frames_lengths = frames_lengths.index_select(0, sort_order)
    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            # prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            "audio": frames, "audio_lengths": frames_lengths
        },
        'target': target,
    }

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch

class AudioLanguagePairDataset(LanguagePairDataset):

    def __init__(
        self, src, src_sizes, src_dict,audio,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False
    ):
        super().__init__(
         src, src_sizes, src_dict,tgt, tgt_sizes=tgt_sizes, tgt_dict=tgt_dict,
        left_pad_source=left_pad_source, left_pad_target=left_pad_target,
        max_source_positions=max_source_positions, max_target_positions=max_target_positions,
        shuffle=shuffle, input_feeding=input_feeding,
        remove_eos_from_source=remove_eos_from_source, append_eos_to_target=append_eos_to_target,
        align_dataset=align_dataset,
        append_bos=append_bos)

        self.audio = audio
        self.num_mel_bins = 80
        self.frame_length = 25.0
        self.frame_shift = 10.0
    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])
            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])
        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
        # print(self.audio[])
        path = self.audio[str(index)]['input']['path']
        if not os.path.exists(path):
            raise FileNotFoundError("Audio file not found: {}".format(path))

        # print(path)
        # print(index)
        # exit()
        sound, sample_rate = torchaudio.load_wav(path)

        # print(self.num_mel_bins)
        # print(self.frame_length)
        # print(self.frame_shift)
        # print("&&&&&&&&&&&&&&&&&")
        # exit()
        # if "20170001P00053I0108" in path:
        #     pp=True
        # else:
        #     pp=False
        output = kaldi.fbank(
            sound,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=0.0,
            energy_floor=1.0
        )
        output_cmvn = data_utils.apply_mv_norm(output)
        # if "20170001P00053I0108" in path:
        #     print(path)
        #     print(sound)
        #     print(sample_rate)
        #     print("*******")
        #     print("output")
        #     print(output)
        #     print("output_cmvn")
        #     print(output_cmvn)
        # self.s2s_collater = Seq2SeqCollater(
        #     0, 1, pad_index=self.tgt_dict.pad(),
        #     eos_index=self.tgt_dict.eos(), move_eos_to_beginning=True
        # )

        # return {"id": index, "data": [output_cmvn.detach(), tgt_item]}
        example = {
            'id': index,
            'audio':output_cmvn.detach(),
            'source': src_item,
            'target': tgt_item,
        }
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )



