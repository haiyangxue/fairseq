#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput

import torch
import os
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders,data_utils
import torchaudio
import torchaudio.compliance.kaldi as kaldi

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    # print(args)
    print()
    print("*******************")
    print(args.task)
    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    def collate_frames(frames):
        """Convert a list of 2d frames into a padded 3d tensor
        Args:
            frames (list): list of 2d frames of size L[i]*f_dim. Where L[i] is
                length of i-th frame and f_dim is static dimension of features
        Returns:
            3d tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
        """
        len_max = max(frame.size(0) for frame in frames)
        print(frames.size())
        print(frames[0].size())
        f_dim = frames[0].size(1)
        res = frames[0].new(len(frames), len_max, f_dim).fill_(0.0)
        for i, v in enumerate(frames):
            res[i, : v.size(0)] = v
        return res

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    # if args.buffer_size > 1:
    #     print('| Sentence buffer size:', args.buffer_size)
    # print('| Type the input sentence and press return:')
    start_id = 0
    audio_root_path="/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/train/train2"
    with open(args.input)as inp:
        input = inp.readline().strip()
        while input:
            print()
            audio_path=audio_root_path+"/"+input.split(" ")[0]+".wav"
            inputs = [" ".join(input.split(" ")[1:])]
            results = []
            for batch in make_batches(inputs, args, task, max_positions, encode_fn):
                src_tokens = batch.src_tokens
                src_lengths = batch.src_lengths
                if use_cuda:
                    src_tokens = src_tokens.cuda()
                    src_lengths = src_lengths.cuda()
                if args.task=="translation":
                    sample = {
                        'net_input': {
                            'src_tokens': src_tokens,
                            'src_lengths': src_lengths,
                        },
                    }
                else:
                    if not os.path.exists(audio_path):
                        raise FileNotFoundError("Audio file not found: {}".format(audio_path))
                    sound, sample_rate = torchaudio.load_wav(audio_path)
                    num_mel_bins ,frame_length,frame_shift= 80,25.0,10.0

                    output = kaldi.fbank(
                        sound,
                        num_mel_bins=num_mel_bins,
                        frame_length=frame_length,
                        frame_shift=frame_shift,
                        dither=0.0,
                        energy_floor=1.0
                    )

                    frames = data_utils.apply_mv_norm(output).detach()[None,:,:].type(torch.cuda.FloatTensor)
                    # print(output_cmvn)
                    # frames = collate_frames(output_cmvn)
                    # sort samples by descending number of frames
                    # frames_lengths = torch.cuda.LongTensor(frames.size()[1])
                    frames_lengths = torch.LongTensor([s.size(0) for s in frames])

                    sample = {
                    'net_input': {
                        'src_tokens': src_tokens,
                        'src_lengths': src_lengths,
                        "audio": frames, "audio_lengths": frames_lengths
                    },}
                translations = task.inference_step(generator, models, sample)
                for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                    src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                    results.append((start_id + id, src_tokens_i, hypos))

            # sort output to match input order
            for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                    print('S-{}\t{}'.format(id, src_str))

                # Process top predictions
                for hypo in hypos[:min(len(hypos), args.nbest)]:
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )
                    hypo_str = decode_fn(hypo_str)
                    print('H-{}\t{}'.format(id, hypo_str))

                    # print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                    # print('P-{}\t{}'.format(
                    #     id,
                    #     ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                    # ))
                    if args.print_alignment:
                        alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                        print('A-{}\t{}'.format(
                            id,
                            alignment_str
                        ))
            input = inp.readline().strip()

        # update running id counter
        start_id += len(inputs)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
