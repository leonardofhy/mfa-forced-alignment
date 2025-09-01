
import pytest
import random

from solution import OptimizedForcedAlignmentCorrector


@pytest.fixture()
def corrector():
    return OptimizedForcedAlignmentCorrector()


class TestBasicFunctionality:
    """Basic functionality tests"""

    @pytest.mark.parametrize(
        "ground_truth,mfa,expected",
        [
            # 1. Single <unk> in middle
            ("我 喜歡 學習 演算法", "我 喜歡 <unk> 演算法", "我 喜歡 學習 演算法"),
            # 2. Multiple separated <unk>
            (
                "今天 天氣 真的 很 好",
                "今天 <unk> 真的 <unk> 好",
                "今天 天氣 真的 很 好",
            ),
            # 3. Consecutive <unk>
            ("我 要 去 北京 旅遊", "我 <unk> <unk> 北京 旅遊", "我 要 去 北京 旅遊"),
            # 4. <unk> at start
            ("我 喜歡 程式 設計", "<unk> 喜歡 程式 設計", "我 喜歡 程式 設計"),
            # 5. <unk> at end
            ("今天 天氣 很 好", "今天 天氣 很 <unk>", "今天 天氣 很 好"),
            # 6. All tokens are <unk>
            ("甲 乙 丙 丁", "<unk> <unk> <unk> <unk>", "甲 乙 丙 丁"),
            # 7. Repeated token ambiguity
            ("我 喜歡 喜歡 你", "我 <unk> 喜歡 你", "我 喜歡 喜歡 你"),
        ],
    )
    def test_wildcard_matching_basic(self, corrector, ground_truth, mfa, expected):
        """Validate wildcard matching for typical scenarios"""
        assert corrector.correct_wildcard_matching(ground_truth, mfa) == expected


class TestEdgeCases:
    """Edge case tests"""

    def test_empty_strings(self, corrector):
        """Test with empty strings"""
        assert corrector.correct_wildcard_matching("", "") == ""

    def test_single_token(self, corrector):
        """Test with single token"""
        assert corrector.correct_wildcard_matching("我", "我") == "我"
        assert corrector.correct_wildcard_matching("我", "<unk>") == "我"

    def test_only_unk_tokens(self, corrector):
        """Test when MFA has only <unk> tokens"""
        gt = "這 是 一 個 測試"
        mfa = "<unk> <unk> <unk> <unk> <unk>"
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert result == gt

    def test_no_unk_tokens(self, corrector):
        """Test when MFA has no <unk> tokens"""
        text = "今天 天氣 很 好"
        assert corrector.correct_wildcard_matching(text, text) == text

    def test_more_unk_than_ground_truth(self, corrector):
        """Test when there are more <unk> than ground truth tokens"""
        gt = "我 愛"
        mfa = "<unk> <unk> <unk>"
        result = corrector.correct_wildcard_matching(gt, mfa)
        # Should handle gracefully without crashing
        assert len(result.split()) <= 3

    def test_alternating_unk(self, corrector):
        """Test alternating <unk> and normal tokens"""
        gt = "一 二 三 四 五"
        mfa = "<unk> 二 <unk> 四 <unk>"
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert result == "一 二 三 四 五"


class TestComplexPatterns:
    """Complex pattern tests"""

    def test_multiple_consecutive_unk_groups(self, corrector):
        """Test multiple groups of consecutive <unk>"""
        gt = "我 愛 學 習 中 文 和 英 文"
        mfa = "我 <unk> <unk> 習 <unk> <unk> 和 英 文"
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert "愛" in result and "學" in result and "中" in result and "文" in result

    def test_long_consecutive_unk(self, corrector):
        """Test very long consecutive <unk> sequence"""
        gt = " ".join([f"字{i}" for i in range(10)])
        mfa = "字0 " + " ".join(["<unk>"] * 8) + " 字9"
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert result.startswith("字0") and result.endswith("字9")

    def test_repeated_patterns(self, corrector):
        """Test with repeated patterns in text"""
        gt = "測試 測試 一 二 三 測試 測試"
        mfa = "測試 <unk> 一 二 三 <unk> 測試"
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert result.count("測試") == 4

    def test_ambiguous_context(self, corrector):
        """Test with ambiguous context that could match multiple positions"""
        gt = "我 喜歡 你 我 喜歡 他"
        mfa = "我 <unk> 你 我 喜歡 <unk>"
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert "喜歡" in result and "他" in result


class TestMixedContent:
    """Tests with mixed Chinese/English content"""

    def test_mixed_chinese_english(self, corrector):
        """Test with mixed Chinese and English tokens"""
        gt = "我 love 學習 Python 和 機器 learning"
        mfa = "我 <unk> 學習 Python <unk> 機器 learning"
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert "love" in result and "和" in result

    def test_english_only(self, corrector):
        """Test with English only text"""
        gt = "I love machine learning algorithms"
        mfa = "I <unk> machine <unk> algorithms"
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert result == gt

    def test_numbers_and_symbols(self, corrector):
        """Test with numbers and symbols"""
        gt = "價格 是 100 元 或 $20"
        mfa = "價格 <unk> 100 元 <unk> $20"
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert "是" in result and "或" in result


class TestPerformance:
    """Performance and stress tests"""

    def test_large_text(self, corrector):
        """Test with large text (near maximum constraint)"""
        tokens = [f"字{i}" for i in range(500)]
        gt = " ".join(tokens)

        # Replace every 10th token with <unk>
        mfa_tokens = tokens.copy()
        for i in range(0, len(mfa_tokens), 10):
            mfa_tokens[i] = "<unk>"
        mfa = " ".join(mfa_tokens)

        result = corrector.correct_wildcard_matching(gt, mfa)
        assert len(result.split()) == len(tokens)

    def test_many_unk(self, corrector):
        """Test with many <unk> tokens (50% of text)"""
        gt = " ".join([f"字{i}" for i in range(100)])
        tokens = gt.split()
        mfa_tokens = []
        for i, token in enumerate(tokens):
            if i % 2 == 0:
                mfa_tokens.append("<unk>")
            else:
                mfa_tokens.append(token)
        mfa = " ".join(mfa_tokens)

        result = corrector.correct_wildcard_matching(gt, mfa)
        result_tokens = result.split()

        # Check that non-<unk> tokens are preserved
        for i in range(1, len(tokens), 2):
            assert result_tokens[i] == tokens[i]


class TestIdempotence:
    """Idempotence and consistency tests"""

    def test_idempotent_correction(self, corrector):
        """Running correction twice should not change the result"""
        gt = "我 要 去 北京 旅遊"
        mfa = "我 <unk> <unk> 北京 旅遊"
        once = corrector.correct_wildcard_matching(gt, mfa)
        twice = corrector.correct_wildcard_matching(gt, once)
        assert once == twice

    def test_already_correct(self, corrector):
        """Correcting already correct text should not change it"""
        text = "這 是 正確 的 文本"
        assert corrector.correct_wildcard_matching(text, text) == text


class TestAllMethods:
    """Test all correction methods"""

    @pytest.mark.parametrize(
        "method_name",
        [
            "correct_wildcard_matching",
            "correct_kmp_variant",
            "correct_rolling_hash",
            "correct_suffix_array",
            "correct_with_fsa",
        ],
    )
    def test_method_consistency(self, corrector, method_name):
        """All methods should handle basic case without crashing"""
        gt = "我 喜歡 學習 演算法"
        mfa = "我 喜歡 <unk> 演算法"

        method = getattr(corrector, method_name)
        result = method(gt, mfa)

        # Should not crash and should return something
        assert result is not None
        assert isinstance(result, str)

        # Should have same number of tokens or close to it
        result_tokens = result.split()
        gt_tokens = gt.split()
        assert abs(len(result_tokens) - len(gt_tokens)) <= 2

    @pytest.mark.parametrize(
        "method_name",
        [
            "correct_wildcard_matching",
            "correct_kmp_variant",
            "correct_rolling_hash",
            "correct_suffix_array",
            "correct_with_fsa",
        ],
    )
    def test_method_no_unk(self, corrector, method_name):
        """All methods should handle text without <unk> correctly"""
        text = "這 是 測試 文本"
        method = getattr(corrector, method_name)
        assert method(text, text) == text


class TestRobustness:
    """Robustness and error handling tests"""

    def test_malformed_input(self, corrector):
        """Test with malformed input"""
        # Extra spaces
        gt = "我  喜歡   學習"
        mfa = "我   <unk>  學習"
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert result is not None

    def test_special_characters(self, corrector):
        """Test with special characters"""
        gt = "這是「引號」和（括號）"
        mfa = "這是「<unk>」和（括號）"
        result = corrector.correct_wildcard_matching(gt, mfa)
        # Current algorithm can't infer inside paired punctuation; just ensure no crash
        assert result.startswith("這是")

    def test_unicode_characters(self, corrector):
        """Test with various Unicode characters"""
        gt = "😀 表情 符號 ♠ ♥ ♦ ♣"
        mfa = "😀 <unk> 符號 ♠ <unk> ♦ ♣"
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert "表情" in result and "♥" in result


class TestFuzzing:
    """Fuzzing tests with random inputs"""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1001])
    def test_random_unk_placement(self, corrector, seed):
        """Test with random <unk> placement"""
        random.seed(seed)

        # Generate random ground truth
        tokens = [f"字{i}" for i in range(random.randint(10, 50))]
        gt = " ".join(tokens)

        # Randomly replace some tokens with <unk>
        mfa_tokens = tokens.copy()
        num_unk = random.randint(1, len(tokens) // 2)
        unk_positions = random.sample(range(len(tokens)), num_unk)

        for pos in unk_positions:
            mfa_tokens[pos] = "<unk>"
        mfa = " ".join(mfa_tokens)

        # Should not crash
        result = corrector.correct_wildcard_matching(gt, mfa)
        assert result is not None
        assert len(result.split()) == len(tokens)

    def test_stress_consecutive_unk(self, corrector):
        """Stress test with many consecutive <unk>"""
        for num_consecutive in [2, 5, 10, 20]:
            tokens = [f"字{i}" for i in range(30)]
            gt = " ".join(tokens)

            # Place consecutive <unk> in the middle
            start = 10
            mfa_tokens = (
                tokens[:start]
                + ["<unk>"] * num_consecutive
                + tokens[start + num_consecutive :]
            )
            mfa = " ".join(mfa_tokens)

            result = corrector.correct_wildcard_matching(gt, mfa)
            assert result is not None


# Benchmark fixture for performance testing
@pytest.fixture
def benchmark_data():
    """Generate benchmark test data"""
    return [
        # Small text
        ("我 喜歡 學習", "我 <unk> 學習"),
        # Medium text
        (
            " ".join([f"字{i}" for i in range(50)]),
            " ".join([f"字{i}" if i % 5 != 0 else "<unk>" for i in range(50)]),
        ),
        # Large text
        (
            " ".join([f"字{i}" for i in range(200)]),
            " ".join([f"字{i}" if i % 10 != 0 else "<unk>" for i in range(200)]),
        ),
    ]


@pytest.mark.parametrize(
    "method_name",
    [
        "correct_wildcard_matching",
        "correct_kmp_variant",
        "correct_rolling_hash",
        "correct_suffix_array",
        "correct_with_fsa",
    ],
)
def test_method_performance(corrector, benchmark_data, method_name):
    """Basic performance test for all methods"""
    import time

    method = getattr(corrector, method_name)
    total_time = 0

    for gt, mfa in benchmark_data:
        start = time.perf_counter()
        result = method(gt, mfa)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        # Basic validation
        assert result is not None
        assert isinstance(result, str)

    # Just ensure it completes in reasonable time (< 1 second for all test cases)
    assert total_time < 1.0, f"{method_name} took too long: {total_time:.3f}s"
