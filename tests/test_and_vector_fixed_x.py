"""CPU checks for the fixed-x experiment's target and frozen-encoder gradient path."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import unittest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

spec = importlib.util.spec_from_file_location('fixed_x', Path(__file__).parents[1] / '5-text-embedding-third-expert/scripts/and_vector_fixed_x.py')
fixed_x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed_x)


def test_correction_exactly_recovers_guided_joint():
    torch.manual_seed(2)
    cat, dog, joint, empty = [torch.randn(1, 4, 8, 8) for _ in range(4)]
    x = fixed_x.correction_target(cat, dog, joint, empty)
    actual = fixed_x.guided_sum(empty, [cat, dog, x])
    expected = fixed_x.guided_sum(empty, [joint])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fixed_x.guided_sum(empty, [cat, dog, empty]),
                               fixed_x.guided_sum(empty, [cat, dog]))


class Tokenizer:
    model_max_length = 10

    def __call__(self, text, **kwargs):
        if kwargs.get('add_special_tokens') is False:
            return SimpleNamespace(input_ids=[1, 2, 3, 4, 5, 6])
        return SimpleNamespace(input_ids=torch.tensor([[28, 1, 2, 3, 4, 5, 6, 29, 0, 0]]))


def models():
    cfg = CLIPTextConfig(vocab_size=30, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2,
                         max_position_embeddings=10, eos_token_id=29, projection_dim=12)
    return dict(tokenizer=Tokenizer(), tokenizer_2=Tokenizer(),
                text_encoder=CLIPTextModel(cfg).eval().requires_grad_(False),
                text_encoder_2=CLIPTextModelWithProjection(cfg).eval().requires_grad_(False))


def check_private_rows(tokens):
    torch.manual_seed(7)
    m = models()
    original = {key: {n: p.detach().clone() for n, p in m[key].named_parameters()}
                for key in ('text_encoder', 'text_encoder_2')}
    x = fixed_x.PrivatePrompt(m, tokens)
    ids = m['tokenizer']('phrase').input_ids
    a = m['text_encoder'](ids, output_hidden_states=True)
    b = m['text_encoder_2'](ids, output_hidden_states=True)
    seq, pool = x.encode()
    torch.testing.assert_close(seq, torch.cat([a.hidden_states[-2], b.hidden_states[-2]], -1), rtol=0, atol=0)
    torch.testing.assert_close(pool, b.text_embeds, rtol=0, atol=0)
    opt = torch.optim.Adam(x.rows, lr=0.001)
    # Arbitrary detached targets exercise both sequence and projected pooled paths.
    loss = (seq-torch.randn_like(seq)).square().mean() + (pool-torch.randn_like(pool)).square().mean()
    loss.backward()
    assert all(row.grad is not None and row.grad.norm() > 0 for row in x.rows)
    opt.step()
    changed_seq, changed_pool = x.encode()
    assert not torch.equal(seq, changed_seq) and not torch.equal(pool, changed_pool)
    for key in original:
        for name, parameter in m[key].named_parameters():
            assert not parameter.requires_grad and parameter.grad is None
            torch.testing.assert_close(parameter, original[key][name], rtol=0, atol=0)
    # Hooks were removed: ordinary prompt encodings retain their original output.
    torch.testing.assert_close(m['text_encoder_2'](ids).text_embeds, b.text_embeds, rtol=0, atol=0)


class FixedXTests(unittest.TestCase):
    def test_target_identity(self):
        test_correction_exactly_recovers_guided_joint()

    def test_private_rows(self):
        for tokens in (1, 4, 6):
            with self.subTest(tokens=tokens):
                check_private_rows(tokens)


if __name__ == '__main__':
    unittest.main()
