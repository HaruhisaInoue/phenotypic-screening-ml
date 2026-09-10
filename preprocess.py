#前処理コード
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np

# CSVファイルの読み込み
file_path_test = './filter.csv'
file_path_data = './V1_2040129.csv'
data_test = pd.read_csv(file_path_test)
data_original = pd.read_csv(file_path_data)

# 'Number of Active Electrodes'が6未満のwell番号を抽出
exclude_wells = {}
for index, row in data_test.items():
    print(row)
    csv_name = row['CSVname']
    for well, value in row.items():
        if well != 'CSVname' and value < 6:
            if csv_name not in exclude_wells:
                exclude_wells[csv_name] = []
            exclude_wells[csv_name].append(well)

# 削除されるwellのリストをCSVに出力
exclude_wells_df = pd.DataFrame([(csv_name, well) for csv_name, wells in exclude_wells.items() for well in wells],
                                columns=['CSVname', 'Excluded Well'])
exclude_wells_csv_path = './content/excluded_wells.csv'
exclude_wells_df.to_csv(exclude_wells_csv_path, index=False)

# # Dで始まるwell番号のデータを完全に除外
# data_original = data_original[~data_original['well'].str.startswith('D')]

# exclude_wellsに含まれるwell番号のレコードを削除
for csv_name, wells in exclude_wells.items():
    for well in wells:
        # CSVnameが一致し、かつwell番号がexclude_wellsに含まれるレコードを削除
        data_original = data_original[~((data_original['CSVname'] == csv_name) & (data_original['well'] == well))]

# 電極カラムを削除（'Burst Duration - Avg (s)_11'以降右のカラム）
# 最後の列 'DiseaseStatus(0 or 1)' は残す
electrode_columns_end = data_original.columns.get_loc("Burst Duration - Avg (s)_11")
data_processed = data_original.drop(data_original.columns[electrode_columns_end:-1], axis=1)

# 前処理後のデータセットをCSVに出力
processed_data_csv_path = './content/processed_data.csv'
data_processed.to_csv(processed_data_csv_path, index=False)

# 特定のカラム「○○」と「○○」を削除
data_processed = data_processed.drop(columns=['Network Normalized Duration IQR', 'Network IBI Coefficient of Variation'])

# 欠損値を含むレコードを削除
data_processed = data_processed.dropna()

# データを学習データ（1~4）と推論データ（5~8）に分割
train_data = data_processed[data_processed['CSVname'].isin([1, 2, 3, 4])]
test_data = data_processed[data_processed['CSVname'].isin([5, 6, 7, 8])]

# 4レコード未満のwellを学習データから除外
train_complete_wells = train_data.groupby(['CSVname', 'well']).size()
train_incomplete_wells = train_complete_wells[train_complete_wells < 4].reset_index()['well']
train_data = train_data[~train_data['well'].isin(train_incomplete_wells)]

# 同様に4レコード未満のwellをテストデータから除外
test_complete_wells = test_data.groupby(['CSVname', 'well']).size()
test_incomplete_wells = test_complete_wells[test_complete_wells < 4].reset_index()['well']
test_data = test_data[~test_data['well'].isin(test_incomplete_wells)]


# 分割したデータセットをCSVに出力
train_data_csv_path = './content/train_data.csv'
test_data_csv_path = './content/test_data.csv'
train_data.to_csv(train_data_csv_path, index=False)
test_data.to_csv(test_data_csv_path, index=False)
