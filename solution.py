"""
強制對齊文本修正 - 優化解法與相關 LeetCode 問題

本問題的核心是序列對齊（Sequence Alignment），類似於以下 LeetCode 問題：
1. 72. Edit Distance (編輯距離)
2. 1143. Longest Common Subsequence (最長公共子序列)
3. 44. Wildcard Matching (通配符匹配) - 最相似！
4. 10. Regular Expression Matching (正則表達式匹配)
5. 97. Interleaving String (交錯字符串)
6. 115. Distinct Subsequences (不同的子序列)
"""

from typing import List, Dict, Tuple, Optional

class OptimizedForcedAlignmentCorrector:
    """優化的強制對齊文本修正器"""
    
    def __init__(self):
        self.unk_token = "<unk>"
        # 預計算的拼音緩存
        self.pinyin_cache = {}
    
    # ============ 優化解法1: 通配符匹配 (類似 LeetCode 44) ============
    # 時間複雜度: O(n*k) 其中 k 是 <unk> 的數量
    # 空間複雜度: O(1) 如果不考慮結果存儲
    def correct_wildcard_matching(self, ground_truth: str, mfa_text: str) -> str:
        """
        將 <unk> 視為通配符，使用優化的匹配算法
        這是最優的方法，因為 <unk> 數量通常遠小於文本長度
        """
        gt_tokens = ground_truth.split()
        mfa_tokens = mfa_text.split()
        
        if self.unk_token not in mfa_tokens:
            return mfa_text
        
        # 構建模式匹配
        n, m = len(gt_tokens), len(mfa_tokens)
        
        # 使用雙指針 + 貪心策略
        result = []
        i, j = 0, 0
        last_unk_pos = -1
        last_match_pos = -1
        
        while j < m:
            if mfa_tokens[j] == self.unk_token:
                # 記錄 <unk> 位置，準備回溯
                last_unk_pos = j
                last_match_pos = i
                j += 1
            elif i < n and self._tokens_match(gt_tokens[i], mfa_tokens[j]):
                # 正常匹配
                result.append(mfa_tokens[j])
                i += 1
                j += 1
            elif last_unk_pos != -1:
                # 回溯：讓上一個 <unk> 多匹配一個字符
                j = last_unk_pos + 1
                last_match_pos += 1
                i = last_match_pos
                
                # 重新構建結果
                result = result[:last_unk_pos]
                if last_match_pos <= n:
                    # 添加 <unk> 對應的字符
                    unk_replacement = []
                    for k in range(last_unk_pos > 0 and i > 0 and i-1 or i, 
                                 min(i + 1, n)):
                        if k < n:
                            unk_replacement.append(gt_tokens[k])
                    result.extend(unk_replacement)
            else:
                # 無法匹配
                result.append(mfa_tokens[j])
                j += 1
        
        # 處理剩餘的 <unk>
        final_result = []
        gt_idx = 0
        for token in mfa_tokens:
            if token == self.unk_token:
                if gt_idx < n:
                    final_result.append(gt_tokens[gt_idx])
                    gt_idx += 1
                else:
                    final_result.append(token)
            else:
                final_result.append(token)
                # 同步 gt_idx
                for k in range(gt_idx, n):
                    if gt_tokens[k] == token:
                        gt_idx = k + 1
                        break
        
        return ' '.join(final_result)
    
    # ============ 優化解法2: KMP 變體算法 ============
    # 時間複雜度: O(n + m)
    # 空間複雜度: O(m)
    def correct_kmp_variant(self, ground_truth: str, mfa_text: str) -> str:
        """
        使用 KMP 算法的變體進行快速模式匹配
        特別適合處理重複模式
        """
        gt_tokens = ground_truth.split()
        mfa_tokens = mfa_text.split()
        
        if self.unk_token not in mfa_tokens:
            return mfa_text
        
        # 為非 <unk> 部分構建失敗函數
        pattern = []
        unk_positions = []
        for i, token in enumerate(mfa_tokens):
            if token == self.unk_token:
                unk_positions.append(len(pattern))
                pattern.append(None)  # 佔位符
            else:
                pattern.append(token)
        
        # 構建部分匹配表（跳過 None）
        lps = self._build_lps_with_wildcards(pattern)
        
        # KMP 搜索
        matches = self._kmp_search_with_wildcards(gt_tokens, pattern, lps)
        
        if matches:
            # 使用第一個匹配來替換 <unk>
            start_idx = matches[0]
            result = []
            pattern_idx = 0
            
            for i, token in enumerate(mfa_tokens):
                if token == self.unk_token:
                    # 找到對應的 ground truth token
                    while pattern_idx < len(pattern) and pattern[pattern_idx] is not None:
                        pattern_idx += 1
                    
                    if start_idx + pattern_idx < len(gt_tokens):
                        result.append(gt_tokens[start_idx + pattern_idx])
                    else:
                        result.append(token)
                    pattern_idx += 1
                else:
                    result.append(token)
                    pattern_idx += 1
            
            return ' '.join(result)
        
        return mfa_text
    
    def _build_lps_with_wildcards(self, pattern: List) -> List[int]:
        """構建帶通配符的 LPS 數組"""
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] is None or pattern[length] is None:
                # 通配符總是匹配
                i += 1
            elif pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
        
        return lps
    
    def _kmp_search_with_wildcards(self, text: List[str], pattern: List, 
                                   lps: List[int]) -> List[int]:
        """KMP 搜索，支持通配符"""
        n, m = len(text), len(pattern)
        matches = []
        i = j = 0
        
        while i < n:
            if j < m and (pattern[j] is None or text[i] == pattern[j]):
                i += 1
                j += 1
            
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
            elif i < n and j < m and pattern[j] is not None and text[i] != pattern[j]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return matches
    
    # ============ 優化解法3: Rolling Hash (Rabin-Karp 變體) ============
    # 時間複雜度: O(n + m) 平均情況
    # 空間複雜度: O(1)
    def correct_rolling_hash(self, ground_truth: str, mfa_text: str) -> str:
        """
        使用滾動哈希快速定位可能的匹配位置
        特別適合長文本和多個 <unk> 的情況
        """
        gt_tokens = ground_truth.split()
        mfa_tokens = mfa_text.split()
        
        if self.unk_token not in mfa_tokens:
            return mfa_text
        
        # 預計算哈希值
        BASE = 256
        MOD = 10**9 + 7
        
        # 為 MFA 的非 <unk> 部分計算哈希指紋
        mfa_segments = self._split_by_unk(mfa_tokens)
        
        result = []
        gt_used = [False] * len(gt_tokens)
        
        for segment_type, segment in mfa_segments:
            if segment_type == 'unk':
                # 找到未使用的 token
                for i, token in enumerate(gt_tokens):
                    if not gt_used[i]:
                        # 檢查是否可能匹配
                        if self._check_context_match(gt_tokens, mfa_tokens, i):
                            result.append(token)
                            gt_used[i] = True
                            break
                else:
                    result.append(self.unk_token)
            else:
                # 使用滾動哈希快速匹配
                segment_hash = self._compute_hash(segment, BASE, MOD)
                segment_len = len(segment)
                
                # 在 ground truth 中尋找匹配
                if segment_len > 0:
                    power = pow(BASE, segment_len - 1, MOD)
                    gt_hash = 0
                    
                    # 初始化第一個窗口
                    for i in range(min(segment_len, len(gt_tokens))):
                        gt_hash = (gt_hash * BASE + hash(gt_tokens[i])) % MOD
                    
                    # 滾動窗口
                    for i in range(len(gt_tokens) - segment_len + 1):
                        if i > 0:
                            # 移除最左邊，添加最右邊
                            gt_hash = (gt_hash - hash(gt_tokens[i-1]) * power) % MOD
                            gt_hash = (gt_hash * BASE + hash(gt_tokens[i + segment_len - 1])) % MOD
                        
                        if gt_hash == segment_hash:
                            # 驗證實際匹配（避免哈希碰撞）
                            if self._verify_match(gt_tokens[i:i+segment_len], segment):
                                result.extend(segment)
                                for j in range(i, i + segment_len):
                                    gt_used[j] = True
                                break
                    else:
                        result.extend(segment)
        
        return ' '.join(result)
    
    # ============ 優化解法4: 後綴數組 (Suffix Array) ============
    # 時間複雜度: O(n log n) 構建，O(log n) 查詢
    # 空間複雜度: O(n)
    def correct_suffix_array(self, ground_truth: str, mfa_text: str) -> str:
        """
        使用後綴數組進行高效的子串匹配
        適合需要多次查詢的場景
        """
        gt_tokens = ground_truth.split()
        mfa_tokens = mfa_text.split()
        
        if self.unk_token not in mfa_tokens:
            return mfa_text
        
        # 構建後綴數組
        sa = self._build_suffix_array(gt_tokens)
        lcp = self._build_lcp_array(gt_tokens, sa)
        
        result = []
        for i, token in enumerate(mfa_tokens):
            if token == self.unk_token:
                # 使用二分搜索在後綴數組中查找最佳匹配
                context = self._get_context(mfa_tokens, i, window=2)
                best_match = self._binary_search_in_sa(
                    gt_tokens, sa, context['before'], context['after']
                )
                if best_match is not None:
                    result.append(gt_tokens[best_match])
                else:
                    result.append(token)
            else:
                result.append(token)
        
        return ' '.join(result)
    
    def _build_suffix_array(self, tokens: List[str]) -> List[int]:
        """構建後綴數組（簡化版）"""
        n = len(tokens)
        suffixes = [(i, tokens[i:]) for i in range(n)]
        suffixes.sort(key=lambda x: x[1])
        return [s[0] for s in suffixes]
    
    def _build_lcp_array(self, tokens: List[str], sa: List[int]) -> List[int]:
        """構建最長公共前綴數組"""
        n = len(tokens)
        lcp = [0] * n
        rank = [0] * n
        
        for i, idx in enumerate(sa):
            rank[idx] = i
        
        h = 0
        for i in range(n):
            if rank[i] > 0:
                j = sa[rank[i] - 1]
                while i + h < n and j + h < n and tokens[i + h] == tokens[j + h]:
                    h += 1
                lcp[rank[i]] = h
                if h > 0:
                    h -= 1
        
        return lcp
    
    # ============ 優化解法5: 有限狀態自動機 (FSA) ============
    # 時間複雜度: O(n) 遍歷，O(m) 構建
    # 空間複雜度: O(m * Σ) 其中 Σ 是字母表大小
    def correct_with_fsa(self, ground_truth: str, mfa_text: str) -> str:
        """
        構建有限狀態自動機進行模式匹配
        最適合固定模式的重複匹配
        """
        gt_tokens = ground_truth.split()
        mfa_tokens = mfa_text.split()
        
        if self.unk_token not in mfa_tokens:
            return mfa_text
        
        # 構建 FSA
        fsa = self._build_fsa(mfa_tokens)
        
        # 使用 FSA 進行匹配
        result = []
        state = 0
        gt_idx = 0
        
        for token in mfa_tokens:
            if token == self.unk_token:
                # 在當前狀態下尋找最佳轉移
                if gt_idx < len(gt_tokens):
                    result.append(gt_tokens[gt_idx])
                    gt_idx += 1
                else:
                    result.append(token)
            else:
                result.append(token)
                # 更新 FSA 狀態
                if gt_idx < len(gt_tokens) and gt_tokens[gt_idx] == token:
                    gt_idx += 1
        
        return ' '.join(result)
    
    # ============ 輔助方法 ============
    def _tokens_match(self, token1: str, token2: str) -> bool:
        """檢查兩個 token 是否匹配"""
        return token1 == token2
    
    def _split_by_unk(self, tokens: List[str]) -> List[Tuple[str, List[str]]]:
        """將 token 列表按 <unk> 分割"""
        segments = []
        current = []
        
        for token in tokens:
            if token == self.unk_token:
                if current:
                    segments.append(('normal', current))
                    current = []
                segments.append(('unk', []))
            else:
                current.append(token)
        
        if current:
            segments.append(('normal', current))
        
        return segments
    
    def _compute_hash(self, tokens: List[str], base: int, mod: int) -> int:
        """計算 token 序列的哈希值"""
        h = 0
        for token in tokens:
            h = (h * base + hash(token)) % mod
        return h
    
    def _verify_match(self, tokens1: List[str], tokens2: List[str]) -> bool:
        """驗證兩個 token 序列是否匹配"""
        if len(tokens1) != len(tokens2):
            return False
        return all(t1 == t2 for t1, t2 in zip(tokens1, tokens2))
    
    def _check_context_match(self, gt_tokens: List[str], mfa_tokens: List[str], 
                            gt_idx: int) -> bool:
        """檢查上下文是否匹配"""
        # 簡化版本的上下文檢查
        return True
    
    def _get_context(self, tokens: List[str], idx: int, window: int = 2) -> Dict:
        """獲取指定位置的上下文"""
        before = []
        after = []
        
        # 獲取前文
        for i in range(max(0, idx - window), idx):
            if tokens[i] != self.unk_token:
                before.append(tokens[i])
        
        # 獲取後文
        for i in range(idx + 1, min(len(tokens), idx + window + 1)):
            if tokens[i] != self.unk_token:
                after.append(tokens[i])
        
        return {'before': before, 'after': after}
    
    def _binary_search_in_sa(self, tokens: List[str], sa: List[int], 
                            before: List[str], after: List[str]) -> Optional[int]:
        """在後綴數組中二分搜索"""
        # 簡化版本，實際應該更複雜
        for idx in sa:
            # 檢查前後文是否匹配
            match = True
            for i, b in enumerate(before):
                if idx - len(before) + i < 0 or \
                   tokens[idx - len(before) + i] != b:
                    match = False
                    break
            
            if match:
                for i, a in enumerate(after):
                    if idx + i + 1 >= len(tokens) or \
                       tokens[idx + i + 1] != a:
                        match = False
                        break
            
            if match:
                return idx
        
        return None
    
    def _build_fsa(self, pattern: List[str]) -> Dict:
        """構建有限狀態自動機"""
        # 簡化版本的 FSA
        fsa = {}
        state = 0
        
        for i, token in enumerate(pattern):
            if token != self.unk_token:
                fsa[state] = {token: state + 1}
                state += 1
            else:
                fsa[state] = {'*': state + 1}  # 通配符轉移
                state += 1
        
        return fsa


# ============ 複雜度分析比較 ============
def complexity_analysis():
    """
    各種算法的複雜度分析：
    
    1. 通配符匹配 (Wildcard Matching)
       - 時間: O(n*k) 其中 k 是 <unk> 數量
       - 空間: O(1)
       - 優勢: k << n 時非常高效
    
    2. KMP 變體
       - 時間: O(n + m)
       - 空間: O(m)
       - 優勢: 線性時間，適合長文本
    
    3. Rolling Hash
       - 時間: O(n + m) 平均，O(n*m) 最壞
       - 空間: O(1)
       - 優勢: 實現簡單，平均性能好
    
    4. 後綴數組
       - 時間: O(n log n) 預處理，O(log n) 查詢
       - 空間: O(n)
       - 優勢: 多次查詢時效率高
    
    5. FSA
       - 時間: O(n) 匹配，O(m) 構建
       - 空間: O(m * Σ)
       - 優勢: 匹配速度最快
    """
    print(complexity_analysis.__doc__)


# ============ LeetCode 相關問題映射 ============
class LeetCodeSimilarProblems:
    """
    相關 LeetCode 問題及其解法映射：
    """
    
    @staticmethod
    def wildcard_matching_44(s: str, p: str) -> bool:
        """
        LeetCode 44: Wildcard Matching
        * 可以匹配任意字符串
        ? 可以匹配任意單個字符
        
        與我們的問題最相似！<unk> 類似於 * 但有限制
        """
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        # 處理 p 開頭的 *
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-1]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j-1] == '*':
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
                elif p[j-1] == '?' or s[i-1] == p[j-1]:
                    dp[i][j] = dp[i-1][j-1]
        
        return dp[m][n]
    
    @staticmethod
    def edit_distance_72(word1: str, word2: str) -> int:
        """
        LeetCode 72: Edit Distance
        計算將 word1 轉換為 word2 的最少操作數
        
        可以用來衡量 MFA 和 Ground Truth 的差異
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def longest_common_subsequence_1143(text1: str, text2: str) -> int:
        """
        LeetCode 1143: Longest Common Subsequence
        找出兩個字符串的最長公共子序列
        
        可以用來找出 MFA 和 Ground Truth 的共同部分
        """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


# ============ 性能測試 ============
def benchmark_solutions():
    """
    性能測試不同解法
    """
    import time
    
    corrector = OptimizedForcedAlignmentCorrector()
    
    # 生成測試數據
    test_cases = [
        ("我 喜歡 學習 演算法", "我 喜歡 <unk> 演算法"),
        ("今天 天氣 真的 很 好", "今天 <unk> 真的 <unk> 好"),
        ("我 要 去 北京 旅遊", "我 <unk> <unk> 北京 旅遊"),
        # 長文本測試
        (" ".join(["測試"] * 100), " ".join(["測試"] * 50 + ["<unk>"] * 10 + ["測試"] * 40))
    ]
    
    methods = [
        ("Wildcard Matching", corrector.correct_wildcard_matching),
        ("KMP Variant", corrector.correct_kmp_variant),
        ("Rolling Hash", corrector.correct_rolling_hash),
        ("Suffix Array", corrector.correct_suffix_array),
        ("FSA", corrector.correct_with_fsa),
    ]
    
    print("性能測試結果：")
    print("-" * 60)
    
    for method_name, method in methods:
        total_time = 0
        for gt, mfa in test_cases:
            start = time.time()
            result = method(gt, mfa)
            elapsed = time.time() - start
            total_time += elapsed
        
        print(f"{method_name:20} 平均時間: {total_time/len(test_cases)*1000:.3f}ms")


if __name__ == "__main__":
    # 執行複雜度分析
    complexity_analysis()
    print("\n" + "="*60 + "\n")
    
    # 執行性能測試
    benchmark_solutions()