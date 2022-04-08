# 前提条件

下記がインストールされていること

- python 3.9.10
- poetry 1.1.11

# インストール方法
gaiiフォルダに移動し、下記のコマンドを実行

```
./gaii setup
```

# 実験の実行
```
./gaii exec_experiment
```

# 実験がうまくいっているかの確認

1. experiment.logファイルのアクセス日時が直近であることを確認する。
2. experiment.logの内容で、下記のi,j,kが順次増えていることを確認する。
 data: {データ名} ({i}/36)
 partation: {パーテーション名} ({j}/3)
 model: {モデル名} ({k}/36)