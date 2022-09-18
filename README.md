# 前提条件

下記がインストールされていること

- bash
- python 3.9.13
- poetry 1.1.11

# インストール方法
gaiiフォルダに移動し、下記のコマンドを実行

```
./gaii setup
```

# 実験の実行
```
./gaii create_param_file

export latest=$(ls -d -rt data/exp_gaii_div/*/ | tail -n 1)
./gaii exec_experiment $latest

./gaii summary $latest
```

# 実験がうまくいっているかの確認
1. experiment.logファイルのアクセス日時が直近であることを確認する。
2. experiment.logの内容で、下記の記述が順次増えていることを確認する。
 data: {データ名}
 partation: {パーテーション名}
 model: {モデル名}
3. data/exp_gaii_div/{実験開始日}-{実験開始時刻}/{実験条件}/フォルダにそれぞれファイルが10個あることを確認する。