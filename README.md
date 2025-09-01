# MFA Forced Alignment Text Correction

本專案示範如何修正 MFA (Montreal Forced Aligner) 輸出中出現的 `<unk>`，利用已知的 Ground Truth 文本進行對齊與替換。核心類別與多種演算法實現在 `solution.py`。

## 使用說明

1. 直接執行（列出複雜度說明並跑簡易效能測試）：

   ```bash
   python solution.py
   ```

2. 程式庫方式：

   ```python
   from solution import OptimizedForcedAlignmentCorrector

   gt = "我 喜歡 學習 演 算法"
   mfa = "我 喜歡 <unk> 算法"

   c = OptimizedForcedAlignmentCorrector()
   out_text = c.correct_wildcard_matching(gt, mfa)
   print(out_text)
   ```

3. 可呼叫的方法（介面一致）：

   - `correct_wildcard_matching`
   - `correct_kmp_variant`
   - `correct_rolling_hash`
   - `correct_suffix_array`
   - `correct_with_fsa`

## 測試（若已建立 tests/）

```bash
pytest -q
```

尚未建立測試可自行新增 `tests/test_xxx.py`，比對不同方法輸出是否一致。

## 檔案簡述

| 檔案          | 說明                                  |
| ------------- | ------------------------------------- |
| `Problem.md`  | 問題原始敘述                          |
| `solution.py` | 各種 `<unk>` 修正演算法與效能測試入口 |
| `README.md`   | 使用說明                              |

## 擴充方向（簡述）

- 加入拼音轉換再比對
- 支援連續多個 `<unk>`（可變長填補）
- 導入語言模型或機率評分
- 使用 DTW / Needleman–Wunsch 進行完整序列對齊

## 授權

尚未指定，可視需要新增 `LICENSE`。

## 貢獻

歡迎送 PR：改善演算法、加入測試或拼音層。
