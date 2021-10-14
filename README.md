# 課題

## 禁則事項
`test_main.py` を修正した場合、不正行為があったと見なされ本講義を含む今学期の全ての単位を失う可能性があります。

## 内容

1. `fit` を完成させよ。
    - 入力:
	  - `X`:`sample_size` x `dim` の `numpy.ndarray`
	- 出力: なし
	- 中でやること
      - `self.mean` に `X` で計算した最尤推定量を代入する
	  - `self.cov` に `X` で計算した最尤推定量を代入する
1. `sample` を完成させよ。
    - 入力:
	  - `sample_size`: 整数
    - 出力:
	  - `X`: `sample_size` x `dim` の `numpy.ndarray` で、各行は平均 `self.mean`、分散`self.cov` の正規分布に従う乱数

## 実行環境の作り方
`pip3 install -r requirements.txt`

## 実行コマンド
`python3 test_main.py`
