# DL基礎講座2024　最終課題「Visual Question Answering（VQA）」


## 環境構築
### Conda
```bash
conda create -n dl_competition python=3.10
pip install -r requirements.txt
```
### Docker
- Dockerイメージのcudaバージョンについては，ご自身が利用するGPUに合わせて変更してください．
```bash
docker build -t <イメージ名> .
docker run -it -v $PWD:/workspace -w /workspace <イメージ名> bash
```

## ベースラインモデルを動かす
### データのダウンロード
- [こちら](https://drive.google.com/drive/folders/1QTcWMATZ_iGsHnxq6-3aXa7D5VZAzs5T?usp=sharing)から各データをダウンロードしてください．
  - train.json: 訓練データのjsonファイル．画像のパス，質問文，回答文がjson形式でまとめられている．
  - valid.json: テストデータのjsonファイル．画像のパス，質問文がjson形式でまとめられている．
  - train.zip: 訓練データの画像ファイル．
  - valid.zip: テストデータの画像ファイル．
- ダウンロード後，train.zipとvalid.zipを解凍し，各データをdataディレクトリ下に置いてください．
### 訓練・提出ファイル作成
```bash
python3 main.py
```
- `main.py`と同様のディレクトリ内に，学習したモデルのパラメータ`model.pth`とテストデータに対する予測結果`submission.npy`ファイルが作成されます．
- ベースラインは非常に単純な手法のため，改善の余地が多くあります．VQAでは**Omnicampusにおいてベースラインのtest accuracy=41%を超えた提出のみ，修了要件として認めることとします．**


  - 画像の前処理には形状を同じにするためのResizeのみを利用しています．第5回の演習で紹介したようなデータ拡張を追加することで，疑似的にデータを増やし汎化性能の向上が見込めます．
