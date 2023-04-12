import argparse
import time
import torch
from typing import *
from util import load_data, initialize_seq2seq_params, build_seq2seq_model
from time import time
from torch import optim
from core import Seq2SeqModel, encode_all, SOS_token, EOS_token
from math import exp

def decode(prev_hidden: torch.tensor, input: int, model: Seq2SeqModel) -> (torch.tensor, torch.tensor):
    """ Run the decoder AND the output layer for a single step.

    (This function will be used in both log_likelihood and translate_greedy_search.)

    :param prev_hidden: tensor of shape [L, hidden_dim] - the decoder's previous hidden state, denoted H^{dec}_{t-1}
                        in the assignment
    :param input: int - the word being inputted to the decoder.
                            during log-likelihood computation, this is y_{t-1}
                            during greedy decoding, this is yhat_{t-1}
    :param model: a Seq2Seq model
    :return: (1) a tensor `probs` of shape [target_vocab_size], denoted p(y_t | x_1 ... x_S, y_1 .. y_{t-1})
             (2) a tensor `hidden` of shape [L, hidden_dim], denoted H^{dec}_t in the assignment
    """
    # call the decoder_gru to do one forward pass with given input
    # and previous hidden
    # first convert input to embedding
    cur_embedding = model.target_embedding_matrix[input]
    hidden = model.decoder_gru.forward(cur_embedding, prev_hidden)
    
    # forward the top hidden unit to output_layer for probability
    probs = model.output_layer.forward(hidden[-1])

    return probs, hidden


def log_likelihood(source_sentence: List[int], target_sentence: List[int], model: Seq2SeqModel) -> torch.Tensor:
    """ Compute the log-likelihood for a (source_sentence, target_sentence) pair.

    :param source_sentence: the source sentence, as a list of words
    :param target_sentence: the target sentence, as a list of words
    :return: conditional log-likelihood of the (source_sentence, target_sentence) pair
    """
    # encode the source sentence
    source_hiddens = encode_all(source_sentence, model)
 
    # initial hidden = previous last one
    init_hidden = source_hiddens[-1]

    # feed in the first, get the hidden state to pass on AND the output prob
    # at current
    # next_probs: [target_vocab_size]
    next_probs, next_hidden = decode(init_hidden, SOS_token, model)
    
    # initialize final result with p(y1|x1,...,xs)
    # (basically find the one with the target_sentence index)
    log_likelihood = torch.log(next_probs[target_sentence[0]])
    #print(log_likelihood)

    # feed the next target sentence one by one
    # add target_sentence[i] probability to log_likelihood
    for i in range(len(target_sentence) - 1):
        next_probs, next_hidden = decode(next_hidden, target_sentence[i], model)
        log_likelihood += torch.log(next_probs[target_sentence[i + 1]])
        #print(log_likelihood)

    # return the total log likelihood, which is the sum of all outputs
    #print(log_likelihood)
    return log_likelihood
    

def translate_greedy_search(source_sentence: List[int], model: Seq2SeqModel, max_length=10) -> List[int]:
    """ Translate a source sentence using greedy decoding.

    :param source_sentence: the source sentence, as a list of words
    :param max_length: the maximum length that the target sentence could be
    :return: the translated sentence as a list of words
    """
    # first encode_all the source_sentence
    source_hiddens = encode_all(source_sentence, model)

    # initial H0 to decoder -> last one of source hiddens
    init_hidden = source_hiddens[-1]
    
    # get the init prob and 1st hidden state
    next_probs, next_hidden = decode(init_hidden, SOS_token, model)

    # predict the first word with highest prob "next_probs"
    prediction = []    
    prediction.append(int(torch.argmax(next_probs)))

    # continually call decoder for next word prediction till max_length OR EOS
    while (len(prediction) <= max_length):
        # if the last prediction is EOS, break from loop
        if prediction[-1] == EOS_token:
            break

        # call for another round of decode with previous prediction as key
        next_probs, next_hidden = decode(next_hidden, prediction[-1], model)
        prediction.append(int(torch.argmax(next_probs)))
    
    # return the final predicted list of words (int)
    return prediction


def perplexity(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqModel):
    """ Compute the perplexity of an entire dataset under a seq2seq model.  Refer to the write-up for the
    definition of perplexity.

    :param sentences: list of (source_sentence, target_sentence) pairs
    :param model: seq2seq model
    :return: perplexity of the dataset
    """
    ppl = 0.
    ll_total = torch.tensor(0.)   # total conditional log-likelihood of the dataset
    t_total = 0    # total number of target language words in the dataset

    # iterate through the sentences to get all the parameters right
    for i in range(len(sentences)):
        # add target language word length to t_total
        t_total += len(sentences[i][1])
        
        # compute log-likelihood for the pair (source, target)_i
        ll_total += log_likelihood(sentences[i][0], sentences[i][1], model)

    # compute perplexity
    ppl = torch.exp(-1. * ll_total/t_total)
    #print(ll_total, t_total)
    return ppl.item()


def train_epoch(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqModel,
                epoch: int, print_every: int = 100, learning_rate: float = 0.0001, gradient_clip=5):
    """ Train the model for an epoch.

    :param sentences: list of (source_sentence, target_sentence) pairs
    :param model: a Seq2Seq model
    :param epoch: which epoch we're at
    """
    print("epoch\titer\tavg loss\telapsed secs")
    total_loss = 0
    start_time = time()
    optimizer = optim.Adam(model_params.values(), lr=learning_rate)
    for i, (source_sentence, target_sentence) in enumerate(sentences):
        optimizer.zero_grad()
        theloss = -log_likelihood(source_sentence, target_sentence, model)
        total_loss += theloss
        theloss.backward()

        torch.nn.utils.clip_grad_norm_(model_params.values(), gradient_clip)

        optimizer.step()

        if i % print_every == 0:
            avg_loss = total_loss / print_every
            total_loss = 0
            elapsed_secs = time() - start_time
            print("{}\t{}\t{:.3f}\t{:.3f}".format(epoch, i, avg_loss, elapsed_secs))

    return model_params


def print_translations(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqModel,
                       source_vocab: Dict[int, str], target_vocab: Dict[int, str]):
    """ Iterate through a dataset, printing (1) the source sentence, (2) the actual target sentence, and (3)
    the translation according to our model.

    :param sentences: a list of (source sentence, target sentence) pairs
    :param model: a Seq2Seq model
    :param source_vocab: the mapping from word index to word string, in the source language
    :param target_vocab: the mapping from word index to word string, in the target language
    """
    for (source_sentence, target_sentence) in sentences:
        translation = translate_greedy_search(source_sentence, model)

        print("source sentence:" + " ".join([source_vocab[word] for word in source_sentence]))
        print("target sentence:" + " ".join([target_vocab[word] for word in target_sentence]))
        print("translation:\t" + " ".join([target_vocab[word] for word in translation]))
        print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Seq2Seq Homework Assignment')
    parser.add_argument("action", type=str,
                        choices=["train", "finetune", "train_perplexity", "test_perplexity",
                                 "print_train_translations", "print_test_translations"])
    parser.add_argument("--load_model", type=str,
                        help="path to saved model on disk.  if this arg is unset, the weights are initialized randomly")
    parser.add_argument("--save_model_prefix", type=str, help="prefix to save model with, if you're training")
    args = parser.parse_args()

    # load train/test data, and source/target vocabularies
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()

    # load model weights (if path is specified) or else initialize weights randomly
    model_params = initialize_seq2seq_params() if args.load_model is None \
        else torch.load(args.load_model)  # type: Dict[str, torch.Tensor]

    # build a Seq2SeqModel object
    model = build_seq2seq_model(model_params)  # type: Seq2SeqModel

    if args.action == 'train':
        for epoch in range(10):
            train_epoch(train_sentences, model, epoch)
            torch.save(model_params, '{}_{}.pth'.format(args.save_model_prefix, epoch))
    elif args.action == 'finetune':
        train_epoch(train_sentences[:1000], model, 0, learning_rate=1e-5)
        torch.save(model_params, "{}.pth".format(args.save_model_prefix))
    elif args.action == "print_train_translations":
        print_translations(train_sentences, model, source_vocab, target_vocab)
    elif args.action == "print_test_translations":
        print_translations(test_sentences, model, source_vocab, target_vocab)
    elif args.action == "train_perplexity":
        print("perplexity: {}".format(perplexity(train_sentences[:1000], model)))
    elif args.action == "test_perplexity":
        print("perplexity: {}".format(perplexity(test_sentences, model)))
