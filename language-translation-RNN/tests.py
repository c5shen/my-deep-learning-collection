import unittest
from core import encode_all
from util import build_seq2seq_model, build_seq2seq_attention_model
import torch
from seq2seq_attention import translate_greedy_search as translate_greedy_search_attention, \
    log_likelihood as log_likelihood_attention, decode as decode_attention, translate_beam_search, perplexity
from seq2seq import translate_greedy_search, log_likelihood, decode
from numpy.testing import assert_allclose

TOLERANCE = 1e-4

seq2seq_model = build_seq2seq_model(torch.load("pretrained/seq2seq.pth"))
seq2seq_attention_model = build_seq2seq_attention_model(torch.load("pretrained/seq2seq_attention.pth"))

# to run one test: python -m unittest tests.TestGRU
# to run all tests: python -m unittest tests


class TestGRU(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_gru.pth")
        hidden = seq2seq_attention_model.encoder.grus[0].forward(data["cur_input"], data["prev_hidden"])
        assert_allclose(hidden.detach().numpy(), data["hidden"].detach().numpy(), atol=TOLERANCE)


class TestStackedGRU(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_stacked_gru.pth")
        hidden = seq2seq_attention_model.encoder.forward(data["cur_input"], data["prev_hidden"])
        assert_allclose(hidden.detach().numpy(), data["hidden"].detach().numpy(), atol=TOLERANCE)


class TestOutputLayer(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_output.pth")
        output = seq2seq_attention_model.output_layer.forward(data["input"])
        assert_allclose(output.detach().numpy(), data["output"].detach().numpy(), atol=TOLERANCE)


class TestEncodeAll(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_encode_all.pth")
        source_hiddens = encode_all(data["source_sentence"], seq2seq_attention_model)
        assert_allclose(source_hiddens.detach().numpy(), data["source_hiddens"].detach().numpy(), atol=TOLERANCE)


class TestDecode(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_decode.pth")
        out_probs, out_hidden = decode(data["hidden"], data["prev_word"], seq2seq_model)
        assert_allclose(out_probs.detach().numpy(), data["out_probs"].detach().numpy(), atol=TOLERANCE)
        assert_allclose(out_hidden.detach().numpy(), data["out_hidden"].detach().numpy(), atol=TOLERANCE)


class TestLogLikelihood(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_log_likelihood.pth")
        ll = log_likelihood(data["source_sentence"], data["target_sentence"], seq2seq_model)
        self.assertAlmostEqual(ll.item(), data["log_likelihood"].item(), places=4)


class TestTranslate(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_translate.pth")
        target_sentence = translate_greedy_search(data["source_sentence"], seq2seq_model)
        self.assertEqual(target_sentence, data["target_sentence"])


class TestAttention(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_attention.pth")
        attention_weights = seq2seq_attention_model.attention.forward(data["source_top_hiddens"],
                                                                      data["target_top_hidden"])
        assert_allclose(attention_weights.detach().numpy(), data["attention_weights"].detach().numpy(), atol=TOLERANCE)


class TestDecodeAttention(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_decode_attention.pth")
        out_probs, out_hidden, out_context, out_attention_weights = decode_attention(data["hidden"],
                                                                                     data["source_hiddens"],
                                                                                     data["context"], data["prev_word"],
                                                                                     seq2seq_attention_model)
        assert_allclose(out_probs.detach().numpy(), data["out_probs"].detach().numpy(), atol=TOLERANCE),
        assert_allclose(out_hidden.detach().numpy(), data["out_hidden"].detach().numpy(), atol=TOLERANCE)
        assert_allclose(out_context.detach().numpy(), data["out_context"].detach().numpy(), atol=TOLERANCE)
        assert_allclose(out_attention_weights.detach().numpy(),
                        data["out_attention_weights"].detach().numpy(), atol=TOLERANCE)


class TestLogLikelihoodAttention(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_log_likelihood_attention.pth")
        ll = log_likelihood_attention(data["source_sentence"], data["target_sentence"], seq2seq_attention_model)
        assert_allclose(ll.item(), data["log_likelihood"].item(), atol=TOLERANCE)


class TestTranslateAttention(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_translate_attention.pth")
        target_sentence, attention_matrix = translate_greedy_search_attention(data["source_sentence"],
                                                                              seq2seq_attention_model)
        self.assertEqual(target_sentence, data["target_sentence"])
        assert_allclose(attention_matrix.detach().numpy(), data["attention_matrix"].detach().numpy(), atol=TOLERANCE)


class TestTranslateBeamSearch(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_translate_beam_search.pth")
        target_sentence, sum_log_likelihood = translate_beam_search(data["source_sentence"], seq2seq_attention_model,
                                                                    beam_width=4, max_length=10)
        #print(data["source_sentence"], data["target_sentence"])

        # autolab test mimic
        #data["source_sentence"] = [25, 26, 700, 1319, 457, 5, 1]
        #data["target_sentence"] = [15, 16, 98, 819, 297, 4, 1]
        #data["sum_log_likelihood"] = torch.tensor(-0.8507395386695862);
        #print(data["sum_log_likelihood"])
        self.assertEqual(target_sentence, data["target_sentence"])
        try:
            assert_allclose(sum_log_likelihood, data["sum_log_likelihood"].detach().numpy(), atol=TOLERANCE)
        except Exception:
            assert_allclose(sum_log_likelihood.detach().numpy(), data["sum_log_likelihood"].detach().numpy(), atol=TOLERANCE)


class TestPerplexity(unittest.TestCase):
    def test(self):
        data = torch.load("test_data/test_perplexity.pth")
        ppl = perplexity(data["sentences"], seq2seq_attention_model)
        assert_allclose(ppl, data["ppl"], atol=TOLERANCE)
