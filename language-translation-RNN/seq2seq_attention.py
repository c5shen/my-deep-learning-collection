import argparse
import time
import torch
import sys
from typing import *
from util import initialize_seq2seq_attention_params, build_seq2seq_attention_model, load_data
from time import time
from torch import optim
from core import Seq2SeqAttentionModel, encode_all
from math import exp
from core import SOS_token, EOS_token


def decode(prev_hidden: torch.tensor, source_hiddens: torch.tensor, prev_context: torch.tensor,
           input: int, model: Seq2SeqAttentionModel) -> (
        torch.tensor, torch.tensor, torch.tensor, torch.tensor):
    """ Run the decoder AND the output layer for a single step.

    :param: prev_hidden: tensor of shape [L, hidden_dim] - the decoder's previous hidden state, denoted H^{dec}_{t-1}
                          in the assignment
    :param: source_hiddens: tensor of shape [source sentence length, L, hidden_dim] - the encoder's hidden states,
                            denoted H^{enc}_1 ... H^{enc}_T in the assignment
    :param: prev_context: tensor of shape [hidden_dim], denoted c_{t-1} in the assignment
    :param input: int - the word being inputted to the decoder.
                            during log-likelihood computation, this is y_{t-1}
                            during greedy decoding, this is yhat_{t-1}
    :param model: a Seq2SeqAttention model
    :return: (1) a tensor `probs` of shape [target_vocab_size], denoted p(y_t | x_1 ... x_S, y_1 .. y_{t-1})
             (2) a tensor `hidden` of shape [L, hidden_size], denoted H^{dec}_t in the assignment
             (3) a tensor `context` of shape [hidden_size], denoted c_t in the assignment
             (4) a tensor `attention_weights` of shape [source_sentence_length], denoted \alpha in the assignment
    """
    # convert input to embedding matrix
    cur_embedding = model.target_embedding_matrix[input]

    # combine the previous context (t-1) with the current embedding
    # of size [embedding_dim + hidden_dim]
    new_input = torch.cat((cur_embedding, prev_context))
 
    # get the next hidden of size [L, hidden_dim]
    hidden = model.decoder_gru.forward(new_input, prev_hidden)

    # get attention weights
    # feed in the top layer of source AND top of the current hidden layer (decoder)
    # of size [source_sentence_length]
    source_top = source_hiddens[:,-1,:]     # of size [source_len, hidden_dim]
    attention_weights = model.attention.forward(source_top, hidden[-1])

    # get new context vector of size [hidden_dim]
    context = torch.zeros(hidden.shape[1])
    for i in range(attention_weights.shape[0]):
        context = context + attention_weights[i] * source_top[i]

    # get the output_layer output probs of size [target_vocab_size]
    new_input = torch.cat((hidden[-1], context))
    probs = model.output_layer.forward(new_input)

    return probs, hidden, context, attention_weights


def log_likelihood(source_sentence: List[int],
                   target_sentence: List[int],
                   model: Seq2SeqAttentionModel) -> torch.Tensor:
    """ Compute the log-likelihood for a (source_sentence, target_sentence) pair.

    :param source_sentence: the source sentence, as a list of words
    :param target_sentence: the target sentence, as a list of words
    :return: log-likelihood of the (source_sentence, target_sentence) pair
    """
    # encode the source sentence
    source_hiddens = encode_all(source_sentence, model)

    # initial hidden for decoder
    init_hidden = source_hiddens[-1]
    #print(init_hidden.shape)

    # initial context vector of all 0s
    init_context = torch.zeros(init_hidden.shape[1])

    # get the initial one
    probs, hidden, context, attention_weights = decode(init_hidden,
                    source_hiddens, init_context, SOS_token, model)

    # get the first log likelihood
    log_likelihood = torch.log(probs[target_sentence[0]])

    # iterate through the target sentence till 2nd last position
    for i in range(len(target_sentence) - 1):
        probs, hidden, context, attention_weights = decode(hidden,
                    source_hiddens, context, target_sentence[i], model)
        log_likelihood += torch.log(probs[target_sentence[i+1]])
    
    return log_likelihood


def perplexity(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqAttentionModel) -> float:
    """ Compute the perplexity of an entire dataset under a seq2seq model.  Refer to the write-up for the
    definition of perplexity.

    :param sentences: list of (source_sentence, target_sentence) pairs
    :param model: seq2seq attention model
    :return: perplexity of the translation
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


def translate_greedy_search(source_sentence: List[int],
                            model: Seq2SeqAttentionModel, max_length=10) -> (List[int], torch.tensor):
    """ Translate a source sentence using greedy decoding.

    :param source_sentence: the source sentence, as a list of words
    :param max_length: the maximum length that the target sentence could be
    :return: (1) the translated sentence as a list of ints
             (2) the attention matrix, a tensor of shape [target_sentence_length, source_sentence_length]

    """
    # encode the source sentence
    source_hiddens = encode_all(source_sentence, model)
    
    # initial hidden for decoder
    init_hidden = source_hiddens[-1]

    # initial context vector
    init_context = torch.zeros(init_hidden.shape[1])

    # get initial outputs
    probs, hidden, context, attention_weights = decode(init_hidden,
                source_hiddens, init_context, SOS_token, model)

    # record all attention weights for each output position
    attention_matrix = []
    attention_matrix.append(attention_weights)
    # record max conditional prob output index
    prediction = []
    prediction.append(int(torch.argmax(probs)))

    # iterate all
    while (len(prediction) <= max_length):
        if (prediction[-1] == EOS_token):
            break

        probs, hidden, context, attention_weights = decode(hidden,
                source_hiddens, context, prediction[-1], model)
        prediction.append(int(torch.argmax(probs)))
        attention_matrix.append(attention_weights)

    attention_matrix = torch.stack(attention_matrix)
    return prediction, attention_matrix

# function to find the top b indexes in log-probs
def top_log_likelihood(probs: torch.tensor, b: int) -> (List[List[int]], List[int]):
    working = torch.tensor(probs)  # assume we have log-probs
    indexes = []
    idx_probs = []

    # iteratively find the top probs
    for i in range(b):
        indexes.append([int(torch.argmax(working))])
        idx_probs.append(working[indexes[-1][0]].item())
        #print("inserting this: {}".format(idx_probs[-1]))
        # set the previous largest to smallest float point
        working[indexes[-1][0]] = torch.tensor(-1000.)   # arbitrary minimum log-likelihood
    return indexes, idx_probs

def translate_beam_search(source_sentence: List[int], model: Seq2SeqAttentionModel,
                          beam_width: int, max_length=10) -> Tuple[List[int], float]:
    """ Translate a source sentence using beam search decoding.

    :param beam_width: the number of translation candidates to keep at each time step
    :param max_length: the maximum length that the target sentence could be
    :return: (1) the target sentence (translation),
             (2) sum of conditional log-likelihood of the translation, i.e., log p(target sentence|source sentence)
    """
    # encode the source
    source_hiddens = encode_all(source_sentence, model)
    B = beam_width  # easier access

    # initialize first hidden of size [L, hidden_dim]
    init_hidden = source_hiddens[-1]

    # initialize context vector of size [hidden_dim]
    init_context = torch.zeros(init_hidden.shape[1])

    # get the first batch of result from decoding
    probs, hidden, context, attention_weights = decode(init_hidden,
            source_hiddens, init_context, SOS_token, model)

    # target sentence and log-likelihood
    final_prediction = []
    final_probs = []
    # store the top B probs/indexes temporarily (or B if initial)
    temp_beam = []
    temp_probs = []
    # temp storage of decode output
    temp_hidden = []
    temp_context = []
 
    # function to return top B indexes in a list
    temp_beam, temp_probs = top_log_likelihood(torch.log(probs), B)
    #print("source is {}".format(source_sentence))

    #print("init batch of probs: {}".format(temp_probs))
    #print("init beam: {}".format(temp_beam))
    # stop searching if beam width == 0, the max length criteria is checked
    # when searching through current candidates for final_predictions
    while (B > 0):
        #######################
        # STEP 1: run through all posibilities for B x B, store all working items
        working_hidden = []
        working_context = []
        working_probs = []
        for i in range(0, len(temp_beam)):
            # we continue searching down
            if len(temp_hidden) == 0:
                probs, hidden, context, attention_weights = decode(hidden,
                        source_hiddens, context, temp_beam[i][-1], model)
            else:
                probs, hidden, context, attention_weights = decode(temp_hidden[i],
                        source_hiddens, temp_context[i], temp_beam[i][-1], model)
            # append for each beam in temp_beam
            working_hidden.append(hidden)
            # the conditional log-likelihood for all [B, vocab_size] of them
            #print(temp_probs[i], temp_probs[i] * len(temp_beam[i]))
            working_probs.append((torch.log(probs) + temp_probs[i] * len(temp_beam[i])) / (len(temp_beam[i]) + 1))
            working_context.append(context)
        # ----DEBUG---- #
        #print("working hidden size: {}\tworking probs size: {}\tworking context size: {}\t".format(
        #            len(working_hidden), len(working_probs), len(working_context)))

        #######################
        # STEP 2: after all the runs, find the next top B items and update
        # temp_storages correspondingly

        # ----DEBUG---- #
        #print("temp_beam before update: {}".format(temp_beam))
        
        working_beam = []
        vocab_size = working_probs[0].shape[0]  # vocab size
        working_probs = torch.cat(working_probs)    # size [B x target_vocab_size]
        indexes, idx_probs = top_log_likelihood(working_probs, B)

        # convert indexes to tuple of deeper indexes (i \in [0,B-1], j \in [vocab_size])
        indexes = [(item[0] // vocab_size, item[0] % vocab_size) for item in indexes]
        
        # ----DEBUG---- #
        #print("corresponding indexes: {}".format(indexes))

        # correspondingly update temp storages
        temp_hidden = []
        temp_context = []
        for i in range(0, len(indexes)):   # should be B of them
            #print("working on: {}".format(indexes[i]))
            prev_idx = indexes[i][0]    # index points to the index of prev beam
            cur = indexes[i][1]         # the current word: int
            working_beam.append(temp_beam[prev_idx] + [cur])

            # update the temp hidden, context, probs to corresponding ones
            temp_hidden.append(working_hidden[prev_idx])
            temp_context.append(working_context[prev_idx])
            #print(temp_probs[prev_idx])
            temp_probs[i] = idx_probs[i]
        
        # finally assign temp_beam to the working beam
        temp_beam = working_beam
        
        #print("next temp_probs (avg): {}".format(temp_probs))
        #print("next temp_probs: {}".format([temp_probs[x] * len(temp_beam[x]) for x in range(len(temp_probs))]))
        #print("next temp_beam: {}\n".format(temp_beam))
        # ----DEBUG---- #
        #print("temp_beam after update: {}".format(temp_beam))
        
        #######################
        # STEP 3: after successful update, "clean up"/"collect" those ends with
        # EOS token or length > 10, decrease size of B correspondingly
        remove = []
        
        for i in range(0, len(temp_beam)):
            # if any of the top B beams ends with EOS, add it to final_prediction
            # and reduce B width
            if temp_beam[i][-1] == EOS_token or len(temp_beam[i]) > max_length:
                # ----DEBUG---- #
                remove.append(i)    # record the index to remove
                final_prediction.append(temp_beam[i])
                final_probs.append(temp_probs[i] * len(temp_beam[i]))
                #print("prepare to add {} to final. Probs {}".format(final_prediction[-1], final_probs[-1]))                
                B -= 1

        ########################
        # STEP 4: remove the corresponding indexes from all temp_storages
        # finish up updates, and go for another loop
        # sort the removal list to make sure the deletion doesn't mess up order
        for item in sorted(remove, reverse=True):
            del temp_beam[item]
            del temp_probs[item]
            del temp_hidden[item]
            del temp_context[item]

    # finally find the max log-likelihood out of the final_predictions
    # normalized weights largest
    len_vec = [len(x) for x in final_prediction]
    #normalized_final_probs = [final_probs[i] / len_vec[i] for i in range(len(len_vec))]
    #largest = int(torch.argmax(torch.tensor(normalized_final_probs)))
    largest = int(torch.argmax(torch.tensor(final_probs)))
    # ----DEBUG---- #
    #print("final_probs: {}".format(final_probs))
    #print("final_prediction: {}".format(final_prediction[largest]))
    
    return (final_prediction[largest], final_probs[largest])

def train_epoch(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqAttentionModel,
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

# function to find the average conditional log-likelihood of first 20 test sentences
# using beam search
def avg_log_likelihood(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqAttentionModel,
        source_vocab: Dict[int, str], target_vocab: Dict[int, str], beam_width, num=20) -> float:
    
    count = 1
    avg_ll = 0.
    for (source_sentence, target_sentence) in sentences:
        # first 20/or defined test sentences
        if count > num:
            break
        if beam_width > 0:
            translation, ll = translate_beam_search(source_sentence, model, beam_width)
        else:
            translation, _ = translate_greedy_search(source_sentence, model)

        avg_ll += ll
        count += 1
    return (avg_ll / num)

def print_translations(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqAttentionModel,
                       source_vocab: Dict[int, str], target_vocab: Dict[int, str], beam_width):
    """ Iterate through a dataset, printing (1) the source sentence, (2) the actual target sentence, and (3)
    the translation according to our model.

    :param sentences: a list of (source sentence, target sentence) pairs
    :param model: a Seq2Seq model
    :param source_vocab: the mapping from word index to word string, in the source language
    :param target_vocab: the mapping from word index to word string, in the target language
    """
    for (source_sentence, target_sentence) in sentences:
        if beam_width > 0:
            translation, _ = translate_beam_search(source_sentence, model, beam_width)
        else:
            translation, _ = translate_greedy_search(source_sentence, model)

        print("source sentence:" + " ".join([source_vocab[word] for word in source_sentence]))
        print("target sentence:" + " ".join([target_vocab[word] for word in target_sentence]))
        print("translation:\t" + " ".join([target_vocab[word] for word in translation]))
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Seq2Seq Homework Assignment')
    parser.add_argument("action", type=str,
                        choices=["train", "finetune", "train_perplexity", "test_perplexity",
                                 "print_train_translations", "print_test_translations", "visualize_attention",
                                 "beam_avg_likelihood"])
    parser.add_argument("--beam_width", type=int, default=-1,
                        help="number of translation candidates to keep at each time step. "
                             "this option is used for beam search translation (greedy search decoding is used by default).")
    parser.add_argument("--load_model", type=str,
                        help="path to saved model on disk.  if this arg is unset, the weights are initialized randomly")
    parser.add_argument("--save_model_prefix", type=str, help="prefix to save model with, if you're training")
    args = parser.parse_args()

    # load train/test data, and source/target vocabularies
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()

    # load model weights (if path is specified) or else initialize weights randomly
    model_params = initialize_seq2seq_attention_params() if args.load_model is None \
        else torch.load(args.load_model)  # type: Dict[str, torch.Tensor]

    # build a Seq2SeqAttentionModel object
    model = build_seq2seq_attention_model(model_params)  # type: Seq2SeqAttentionModel

    if args.action == 'train':
        for epoch in range(10):
            train_epoch(train_sentences, model, epoch)
            torch.save(model_params, '{}_{}.pth'.format(args.save_model_prefix, epoch))
    elif args.action == 'finetune':
        train_epoch(train_sentences[:1000], model, 0, learning_rate=1e-5)
        torch.save(model_params, "{}.pth".format(args.save_model_prefix))
    elif args.action == "print_train_translations":
        print_translations(train_sentences, model, source_vocab, target_vocab, args.beam_width)
    elif args.action == "print_test_translations":
        print_translations(test_sentences, model, source_vocab, target_vocab, args.beam_width)
    elif args.action == "train_perplexity":
        print("perplexity: {}".format(perplexity(train_sentences[:1000], model)))
    elif args.action == "test_perplexity":
        print("perplexity: {}".format(perplexity(test_sentences, model)))
    elif args.action == "beam_avg_likelihood":
        avg = avg_log_likelihood(test_sentences, model, source_vocab, target_vocab, args.beam_width)
        print("Beam width {}\nAverage conditional log-likelihood of first 20: {}".format(args.beam_width, avg))
    elif args.action == "visualize_attention":
        from plotting import visualize_attention
        # visualize the attention matrix for the first 5 test set sentences
        for i in range(5):
            source_sentence = test_sentences[i][0]
            predicted_sentence, attention_matrix = translate_greedy_search(source_sentence, model)
            source_sentence_str = [source_vocab[w] for w in source_sentence]
            predicted_sentence_str = [target_vocab[w] for w in predicted_sentence]
            visualize_attention(source_sentence_str, predicted_sentence_str,
                                attention_matrix.detach().numpy(), "images/{}.png".format(i))
