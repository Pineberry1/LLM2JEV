import unittest

import torch

from llm2jev.attention import shared_prefix_attention
from llm2jev.reference import reference_attention


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class AttentionTests(unittest.TestCase):
    def test_shared_prefix_and_independent_baseline_match_reference(self):
        torch.manual_seed(17)
        for dtype in (torch.float16, torch.bfloat16):
            q = torch.randn(17, 8, 64, device="cuda", dtype=dtype) / 4
            pk = torch.randn(131, 2, 64, device="cuda", dtype=dtype) / 4
            pv = torch.randn_like(pk)
            sk = torch.randn(17, 5, 2, 64, device="cuda", dtype=dtype) / 4
            sv = torch.randn_like(sk)
            lengths = torch.tensor([i % 6 for i in range(17)], device="cuda", dtype=torch.int32)
            expected = reference_attention(q, pk, pv, sk, sv, lengths)
            for branches_per_program in (1, 16):
                actual = shared_prefix_attention(
                    q, pk, pv, sk, sv, lengths,
                    branches_per_program=branches_per_program,
                )
                torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)

    def test_prefix_only_matches_reference(self):
        q = torch.randn(3, 8, 64, device="cuda", dtype=torch.float16) / 4
        pk = torch.randn(67, 2, 64, device="cuda", dtype=torch.float16) / 4
        pv = torch.randn_like(pk)
        torch.testing.assert_close(
            shared_prefix_attention(q, pk, pv),
            reference_attention(q, pk, pv), atol=0.015, rtol=0.015,
        )


class ValidationTests(unittest.TestCase):
    def test_invalid_head_count(self):
        q = torch.empty(2, 3, 64)
        pk = torch.empty(32, 2, 64)
        with self.assertRaises(ValueError):
            shared_prefix_attention(q, pk, pk)

    def test_invalid_suffix_shape(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA GPU required")
        q = torch.empty(2, 8, 64, device="cuda", dtype=torch.float16)
        pk = torch.empty(32, 2, 64, device="cuda", dtype=torch.float16)
        sk = torch.empty(3, 2, 2, 64, device="cuda", dtype=torch.float16)
        lengths = torch.tensor([1, 1], device="cuda", dtype=torch.int32)
        with self.assertRaises(ValueError):
            shared_prefix_attention(q, pk, pk, sk, sk, lengths)
