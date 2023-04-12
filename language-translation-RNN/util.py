import torch
import math
from torch.nn.init import kaiming_uniform
from core import GRU, StackedGRU, Attention, OutputLayer, Seq2SeqModel, Seq2SeqAttentionModel
import pickle
from typing import Dict, List, Tuple


def load_data() -> (Tuple[List[int], List[int]], Tuple[List[int], List[int]], Dict[int, str], Dict[int, str]):
    """ Load the dataset.

    :return: (1) train_sentences: list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) test_sentences : list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) source_vocab   : dictionary which maps from source word index to source word
             (3) target_vocab   : dictionary which maps from target word index to target word
    """
    with open('data/translation_data.bin', 'rb') as f:
        corpus, source_vocab, target_vocab = pickle.load(f)
        test_sentences = corpus[:1000]
        train_sentences = corpus[1000:]
        print("# source vocab: {}\n"
              "# target vocab: {}\n"
              "# train sentences: {}\n"
              "# test sentences: {}\n".format(len(source_vocab), len(target_vocab), len(train_sentences),
                                              len(test_sentences)))
        return train_sentences, test_sentences, source_vocab, target_vocab


def uniform_tensor(shape, a, b):
    return torch.FloatTensor(*shape).uniform_(a, b)


def normal_tensor(shape):
    return torch.FloatTensor(*shape).normal_()


def kaiming_tensor(shape):
    tensor = torch.FloatTensor(*shape)
    torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
    return tensor


def initialize_bias(weight, bias):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    torch.nn.init.uniform_(bias, -bound, bound)


# hyperparameters

SOURCE_VOCAB_SIZE = 3796
TARGET_VOCAB_SIZE = 2788
SOURCE_EMBEDDING_DIM = 400
TARGET_EMBEDDING_DIM = 400
HIDDEN_DIM = 500


def initialize_seq2seq_params() -> Dict[str, torch.tensor]:
    """ Random initialization of weights for a Seq2Seq model.
    :return: model_params, a dictionary Dict[str, torch.tensor] mapping from parameter name to parameter value
    """
    stdv = 1.0 / math.sqrt(HIDDEN_DIM)
    model_params = {
        'enc_1_W': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_U': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_1_Wr': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Ur': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_1_Wz': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Uz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'enc_2_W': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_U': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Wr': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Ur': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Wz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Uz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'dec_1_W': uniform_tensor((HIDDEN_DIM, TARGET_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_U': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_1_Wr': uniform_tensor((HIDDEN_DIM, TARGET_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Ur': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_1_Wz': uniform_tensor((HIDDEN_DIM, TARGET_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Uz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'dec_2_W': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_U': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Wr': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Ur': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Wz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Uz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        "source_embedding_matrix": normal_tensor((SOURCE_VOCAB_SIZE, SOURCE_EMBEDDING_DIM)),
        "target_embedding_matrix": normal_tensor((TARGET_VOCAB_SIZE, TARGET_EMBEDDING_DIM)),

        "output_weight": kaiming_tensor((TARGET_VOCAB_SIZE, HIDDEN_DIM)),
        "output_bias": torch.FloatTensor(TARGET_VOCAB_SIZE),
    }

    initialize_bias(model_params["output_weight"], model_params["output_bias"])
    return model_params


def initialize_seq2seq_attention_params() -> Dict[str, torch.tensor]:
    """ Random initialization of weights for a Seq2SeqAttention model.
    :return: model_params, a dictionary Dict[str, torch.tensor] mapping from parameter name to parameter value"""
    stdv = 1.0 / math.sqrt(HIDDEN_DIM)
    model_params = {
        'enc_1_W': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_U': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_1_Wr': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Ur': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_1_Wz': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Uz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'enc_2_W': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_U': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Wr': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Ur': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Wz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Uz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'dec_1_W': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM + TARGET_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_U': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_1_Wr': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM + TARGET_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Ur': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_1_Wz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM + TARGET_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Uz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'dec_2_W': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_U': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Wr': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Ur': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Wz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Uz': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        "source_embedding_matrix": normal_tensor((SOURCE_VOCAB_SIZE, SOURCE_EMBEDDING_DIM)),
        "target_embedding_matrix": normal_tensor((TARGET_VOCAB_SIZE, TARGET_EMBEDDING_DIM)),

        "output_weight": kaiming_tensor((TARGET_VOCAB_SIZE, 2 * HIDDEN_DIM)),
        "output_bias": torch.FloatTensor(TARGET_VOCAB_SIZE),

        "attention": torch.eye(HIDDEN_DIM)
    }

    initialize_bias(model_params["output_weight"], model_params["output_bias"])
    return model_params


def build_seq2seq_model(model_params: Dict[str, torch.tensor]) -> Seq2SeqModel:
    """ Build a Seq2SeqModel object from a model_params dict """

    encoder_gru = StackedGRU([
        GRU(model_params['enc_1_W'], model_params['enc_1_U'], model_params['enc_1_Wr'], model_params['enc_1_Ur'],
            model_params['enc_1_Wz'], model_params['enc_1_Uz']),
        GRU(model_params['enc_2_W'], model_params['enc_2_U'], model_params['enc_2_Wr'], model_params['enc_2_Ur'],
            model_params['enc_2_Wz'], model_params['enc_2_Uz']),
    ])
    decoder_gru = StackedGRU([
        GRU(model_params['dec_1_W'], model_params['dec_1_U'], model_params['dec_1_Wr'], model_params['dec_1_Ur'],
            model_params['dec_1_Wz'], model_params['dec_1_Uz']),
        GRU(model_params['dec_2_W'], model_params['dec_2_U'], model_params['dec_2_Wr'], model_params['dec_2_Ur'],
            model_params['dec_2_Wz'], model_params['dec_2_Uz']),
    ])
    source_embedding_matrix = model_params['source_embedding_matrix']
    target_embedding_matrix = model_params['target_embedding_matrix']
    output_layer = OutputLayer(model_params['output_weight'], model_params['output_bias'])
    return Seq2SeqModel(HIDDEN_DIM, encoder_gru, decoder_gru, source_embedding_matrix,
                        target_embedding_matrix, output_layer)


def build_seq2seq_attention_model(model_params: Dict[str, torch.tensor]) -> Seq2SeqAttentionModel:
    """ Build a Seq2SeqAttentionModel object from a model_params dict """

    for key in model_params:
        model_params[key].requires_grad_(True)

    encoder_gru = StackedGRU([
        GRU(model_params['enc_1_W'], model_params['enc_1_U'], model_params['enc_1_Wr'], model_params['enc_1_Ur'],
            model_params['enc_1_Wz'], model_params['enc_1_Uz']),
        GRU(model_params['enc_2_W'], model_params['enc_2_U'], model_params['enc_2_Wr'], model_params['enc_2_Ur'],
            model_params['enc_2_Wz'], model_params['enc_2_Uz']),
    ])
    decoder_gru = StackedGRU([
        GRU(model_params['dec_1_W'], model_params['dec_1_U'], model_params['dec_1_Wr'], model_params['dec_1_Ur'],
            model_params['dec_1_Wz'], model_params['dec_1_Uz']),
        GRU(model_params['dec_2_W'], model_params['dec_2_U'], model_params['dec_2_Wr'], model_params['dec_2_Ur'],
            model_params['dec_2_Wz'], model_params['dec_2_Uz']),
    ])
    source_embedding_matrix = model_params['source_embedding_matrix']
    target_embedding_matrix = model_params['target_embedding_matrix']
    attention = Attention(model_params['attention'])
    output_layer = OutputLayer(model_params['output_weight'], model_params['output_bias'])
    return Seq2SeqAttentionModel(HIDDEN_DIM, encoder_gru, decoder_gru, source_embedding_matrix,
                                 target_embedding_matrix, attention, output_layer)
