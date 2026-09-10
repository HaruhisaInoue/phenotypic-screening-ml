#予測、SHAP表示コード
import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


# CSVファイルの読み込み
train_data = pd.read_csv('./input/train_data_new_fix.csv')
test_data = pd.read_csv('./input/test_data_new_fix.csv')

# 学習データの準備
X_train = train_data.drop(['disease_status', 'index', 'CSVname', 'well', 'time', 'Number of Active Electrodes'], axis=1)
y_train = train_data['disease_status']

# Train/validation分割のためにwell番号を抽出
train_data['well_id'] = train_data['well'].str.extract('(\d+)').astype(int)
HC_wells = train_data[train_data['well'].str.startswith(('A', 'B', 'C', 'D', 'E'))]['well_id'].unique()
disease_wells = train_data[train_data['well'].str.startswith(('F', 'G', 'H'))]['well_id'].unique()

# HCおよび疾患データのwell単位での分割
HC_train_wells, HC_valid_wells = train_test_split(HC_wells, test_size=0.2, random_state=42)
disease_train_wells, disease_valid_wells = train_test_split(disease_wells, test_size=0.2, random_state=42)

###########################分布確認用
# 分割したwell IDに基づいて、分割タイプを示すカラムを作成
train_data['Split Type'] = 'Not Used'
train_data.loc[train_data['well_id'].isin(HC_train_wells), 'Split Type'] = 'HC Train'
train_data.loc[train_data['well_id'].isin(HC_valid_wells), 'Split Type'] = 'HC Valid'
train_data.loc[train_data['well_id'].isin(disease_train_wells), 'Split Type'] = 'Disease Train'
train_data.loc[train_data['well_id'].isin(disease_valid_wells), 'Split Type'] = 'Disease Valid'

# 必要なカラムのみを選択してCSVに出力
split_info_df = train_data[['CSVname', 'well', 'Split Type']]
split_info_df.to_csv('./content/train_valid_split_info.csv', index=False)
###############################


# 分割したwell番号に基づいて、訓練データセットと検証データセットを作成
train_indices = train_data['well_id'].isin(HC_train_wells) | train_data['well_id'].isin(disease_train_wells)
valid_indices = ~train_indices

X_train_split = X_train[train_indices]
y_train_split = y_train[train_indices]
X_valid_split = X_train[valid_indices]
y_valid_split = y_train[valid_indices]

# モデルの定義と訓練
# ポジティブクラス（少数クラス）とネガティブクラス（多数クラス）の比率を計算
# 偽陰性を減らすために、ポジティブクラスに高い重みを割り当てる
scale_pos_weight = 1.0*sum(y_train == 0) / sum(y_train == 1)
# モデルの定義
# ハイパーパラメータの調整例：max_depth, learning_rateなど
xgb_classifier = xgb.XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    max_depth=3,  # モデルの深さを制限して過学習を防ぐ
    learning_rate=0.1,  # 学習率を小さくして、よりゆっくり学習
    n_estimators=1000,  # より多くの決定木で学習
    subsample=0.8,  # 学習データのサブサンプリング
    colsample_bytree=0.8,  # 特徴量のサブサンプリング
    scale_pos_weight=scale_pos_weight
)

# 訓練時に訓練データと検証データの評価を行う設定
eval_set = [(X_train_split, y_train_split), (X_valid_split, y_valid_split)]

# モデルの訓練（早期停止を設定して過学習を防ぐ）
xgb_classifier.fit(
    X_train_split,
    y_train_split,
    eval_set=eval_set,
    early_stopping_rounds=10,  # 検証データの性能が10ラウンド改善されなければ停止
    verbose=True
)

# 学習曲線のプロット
results = xgb_classifier.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Validation')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()



# XGBoostモデルの決定木を可視化
xgb.plot_tree(xgb_classifier, num_trees=2, rankdir='LR')
# プロットのサイズを大きく設定
fig = plt.gcf()
fig.set_size_inches(30, 15)
# プロットを表示
plt.show()


# XGBoostモデルでの検証データセットに対する予測
xgb_predictions = xgb_classifier.predict(X_valid_split)

# 予測結果を保存するためのDataFrameを作成
predictions_df = pd.DataFrame({
    'Index': X_valid_split.index,
    'CSVname': train_data.loc[X_valid_split.index, 'CSVname'],
    'Well': train_data.loc[X_valid_split.index, 'well'],
    'Time': train_data.loc[X_valid_split.index, 'time'],
    'TrueLabel': y_valid_split,
    'Prediction': xgb_predictions
})

# 混同行列と評価指標の計算
conf_matrix = confusion_matrix(y_valid_split, xgb_predictions)
precision = precision_score(y_valid_split, xgb_predictions)
recall = recall_score(y_valid_split, xgb_predictions)
f1 = f1_score(y_valid_split, xgb_predictions)
accuracy = accuracy_score(y_valid_split, xgb_predictions)

# テストデータの準備と予測
X_test = test_data.drop(['disease_status', 'index', 'CSVname', 'well', 'time', 'Number of Active Electrodes'], axis=1)
y_test = test_data['disease_status']
xgb_test_predictions = xgb_classifier.predict(X_test)
feature_names = X_test.columns.tolist()  # 特徴量の名前を取得

# 予測確率を計算（1クラス目の確率を取得）
pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]

# テストデータフレームに予測確率を追加
test_data['PredictionProb'] = pred_proba
# 予測結果を元のテストデータフレームに追加
test_data['Prediction'] = xgb_test_predictions
# 予測結果を含むテストデータフレームをCSVファイルに出力
test_data_with_predictions_path = './content/test_data_with_predictions.csv'
test_data.to_csv(test_data_with_predictions_path, index=False)
print(f"Test data with predictions saved to: {test_data_with_predictions_path}")

# TP, FP, TN, FN に該当するレコードのインデックスを取得
tp_indices = test_data[(y_test == 1) & (test_data['Prediction'] == 1)].index
fp_indices = test_data[(y_test == 0) & (test_data['Prediction'] == 1)].index
tn_indices = test_data[(y_test == 0) & (test_data['Prediction'] == 0)].index
fn_indices = test_data[(y_test == 1) & (test_data['Prediction'] == 0)].index

# テスト結果の評価
test_conf_matrix = confusion_matrix(y_test, xgb_test_predictions)
test_precision = precision_score(y_test, xgb_test_predictions)
test_recall = recall_score(y_test, xgb_test_predictions)
test_f1 = f1_score(y_test, xgb_test_predictions)
test_accuracy = accuracy_score(y_test, xgb_test_predictions)

# 予測結果のDataFrameをCSVに出力
test_predictions_df = pd.DataFrame({
    'Index': X_test.index,
    'CSVname': test_data['CSVname'],
    'Well': test_data['well'],
    'Time': test_data['time'],
    'TrueLabel': y_test,
    'Prediction': xgb_test_predictions
})
test_predictions_csv_path = './content/xgb_test_predictions.csv'
test_predictions_df.to_csv(test_predictions_csv_path, index=False)

# 出力結果の表示
print(f"Test Confusion Matrix:\n{test_conf_matrix}\n")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

test_predictions_csv_path

# モデルを JSON 形式で保存
path_to_json_file = './saved_model/xgb_model.json'  # 保存先のパスを指定
xgb_classifier.save_model(path_to_json_file)

print(f"Model saved to: {path_to_json_file}")

#######以下well毎での評価#####
def majority_vote(predictions):
    """多数決に基づいて予測結果を決定する。50%以上（含む）が1の場合は1を予測する。"""
    counts = np.bincount(predictions)
    # 1のカウントが存在するか確認し、存在する場合のみ割合を計算
    if len(counts) > 1 and counts[1] >= len(predictions) / 2:
        return 1
    else:
        return 0

# well毎の予測結果と評価指標の計算関数（TrueLabelを含むように修正）
def calculate_well_based_metrics(df, true_label_col, prediction_col, csvname_col, well_col):
    grouped = df.groupby([csvname_col, well_col])
    well_true = grouped[true_label_col].agg(lambda x: x.mode()[0])
    well_predictions = grouped[prediction_col].agg(majority_vote)

    # 混同行列と評価指標の計算
    conf_matrix = confusion_matrix(well_true, well_predictions)
    precision = precision_score(well_true, well_predictions)
    recall = recall_score(well_true, well_predictions)
    f1 = f1_score(well_true, well_predictions)
    accuracy = accuracy_score(well_true, well_predictions)

    return well_true, well_predictions, conf_matrix, precision, recall, f1, accuracy

# テストデータに対するwell毎の予測結果と評価指標の計算（修正版）
test_well_true, test_well_predictions, test_well_conf_matrix, test_well_precision, test_well_recall, test_well_f1, test_well_accuracy = calculate_well_based_metrics(
    test_predictions_df, 'TrueLabel', 'Prediction', 'CSVname', 'Well')


# 結果の表示
print(f"Test Well-Based Confusion Matrix:\n{test_well_conf_matrix}\n")
print(f"Test Well-Based Precision: {test_well_precision:.4f}")
print(f"Test Well-Based Recall: {test_well_recall:.4f}")
print(f"Test Well-Based F1 Score: {test_well_f1:.4f}")
print(f"Test Well-Based Accuracy: {test_well_accuracy:.4f}")

# well毎の予測結果をDataFrameに保存（TrueLabelを含むように修正）
test_well_predictions_df = pd.DataFrame({
    'CSVname': test_well_true.index.get_level_values('CSVname'),
    'Well': test_well_true.index.get_level_values('Well'),
    'TrueLabel': test_well_true.values,
    'WellPrediction': test_well_predictions.values
})

# well毎の予測結果のDataFrameをCSVに出力
test_well_predictions_csv_path = './content/xgb_test_well_predictions.csv'
test_well_predictions_df.to_csv(test_well_predictions_csv_path, index=False)

test_well_predictions_csv_path

# 可視化関数
def plot_confusion_matrix(conf_matrix, title):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# 混同行列のラベルを定義（0から始まるように）
labels = [0, 1]

# 混同行列を計算し直す（ラベルの順序を指定）
conf_matrix_record_level = confusion_matrix(y_test, xgb_test_predictions, labels=labels)
conf_matrix_well_level = confusion_matrix(test_well_true.values, test_well_predictions.values, labels=labels)

# 混同行列の可視化
plot_confusion_matrix(conf_matrix_record_level, "Test Data - Record Level")
plot_confusion_matrix(conf_matrix_well_level, "Test Data - Well Level")

# well毎の予測が外れた箇所の抽出とCSV出力
incorrect_well_predictions = test_well_predictions_df[test_well_predictions_df['TrueLabel'] != test_well_predictions_df['WellPrediction']]
incorrect_well_predictions.to_csv('./content/incorrect_well_predictions.csv', index=False)


##### SHAPフェーズ #####
# SHAP解析の準備
explainer = shap.Explainer(xgb_classifier, X_train_split)
shap_values = explainer(X_test)

# SHAPの初期化
shap.initjs()
shap.plots.bar(shap_values, max_display=15)

# 混同行列から各ケースのデータを抽出
FP = X_test[(xgb_test_predictions == 1) & (y_test == 0)]
FN = X_test[(xgb_test_predictions == 0) & (y_test == 1)]
TP = X_test[(xgb_test_predictions == 1) & (y_test == 1)]
TN = X_test[(xgb_test_predictions == 0) & (y_test == 0)]


###########################################
# モデル全体のSHAP値の平均を計算（絶対値を使わず）
shap_values_mean = shap_values.values.mean(axis=0)

# 特徴量の重要度をDataFrameに変換
shap_values_mean_df = pd.DataFrame({
    'feature': feature_names,
    'shap_mean': shap_values_mean
})

# 特徴量の重要度を降順にソート（絶対値の大きさでソート）
shap_values_mean_df['abs_mean'] = shap_values_mean_df['shap_mean'].abs()
shap_values_mean_df = shap_values_mean_df.sort_values(by='abs_mean', ascending=False).drop('abs_mean', axis=1)

# 特徴量の重要度を可視化（正と負の影響を含む）
plt.figure(figsize=(10, 8))
sns.barplot(x='shap_mean', y='feature', data=shap_values_mean_df)
plt.title('モデル全体のSHAP値の平均による特徴量の重要度（正負含む）')
plt.xlabel('平均SHAP値')
plt.ylabel('特徴量')
plt.show()

# DataFrameをCSVに出力
shap_values_mean_df.to_csv('./content/shap_mea_feature_importance_ranking.csv', index=False)
print("特徴量の重要度ランキングが 'shap_feature_importance_ranking.csv' に出力されました。")


# 結果を表示
print("mean共通特徴量（特徴量リストAのランキング順）:", shap_values_mean_df)
###########################################

###########################################
# SHAP値の絶対値の平均を計算
shap_values_abs_mean = np.abs(shap_values.values).mean(axis=0)
# 特徴量の名前と絶対値の平均SHAP値を含むDataFrameの作成
shap_values_abs_mean_df = pd.DataFrame({
    'feature': feature_names,
    'abs_shap_mean': shap_values_abs_mean
}).sort_values(by='abs_shap_mean', ascending=False)

# 絶対値を用いたSHAPランキングの可視化
plt.figure(figsize=(10, 8))
sns.barplot(x='abs_shap_mean', y='feature', data=shap_values_abs_mean_df)
plt.title('Feature Importance based on Mean Absolute SHAP values')
plt.xlabel('Mean Absolute SHAP Value (Impact on model output)')
plt.show()


# DataFrameをCSVに出力
shap_values_abs_mean_df.to_csv('./content/shap_feature_importance_ranking.csv', index=False)
print("特徴量の重要度ランキングが 'shap_feature_importance_ranking.csv' に出力されました。")


###########################################


###########################################
# モデル全体のSHAP値の平均を計算
shap_values_mean_positive = shap_values.values.mean(axis=0)

# 特徴量の重要度をDataFrameに変換（正のSHAP値に基づく）
shap_values_mean_positive_df = pd.DataFrame({
    'feature': feature_names,
    'shap_mean_positive': shap_values_mean_positive
})

# 特徴量の重要度を降順にソート（正の影響が強い順）
shap_values_mean_positive_df_sorted = shap_values_mean_positive_df.sort_values(by='shap_mean_positive', ascending=False)

# 特徴量の重要度を可視化（正の影響のみ）
plt.figure(figsize=(10, 8))
sns.barplot(x='shap_mean_positive', y='feature', data=shap_values_mean_positive_df_sorted)
plt.title('モデル全体のSHAP値の平均による特徴量の重要度（正の影響のみ）')
plt.xlabel('平均SHAP値（正の影響）')
plt.ylabel('特徴量')
plt.show()

###########################################


def calculate_shap_ranking(shap_values, segment, feature_names, top_k=15, bottom_k=15):
    segment_index = segment.index
    segment_shap_values = shap_values.values[segment_index]
    mean_shap_values = segment_shap_values.mean(axis=0)
    ranking_mean = pd.Series(mean_shap_values, index=feature_names).sort_values(ascending=False)
    return ranking_mean



# 各セグメントごとの特徴量ランキングを抽出して可視化
def plot_feature_importance_for_segment(segment, feature_names, segment_name):
    ranking = calculate_shap_ranking(shap_values, segment, feature_names)
    plt.figure(figsize=(10, 6))
    ranking.plot(kind='barh', color='blue')
    plt.title(f"Feature Importance for {segment_name}")
    plt.gca().invert_yaxis()
    plt.show()


# 特徴量名の取得
feature_names = X_test.columns


# 各セグメントごとの特徴量ランキングと可視化
for segment, name in zip([TP, FP, TN, FN], ["True Positives", "False Positives", "True Negatives", "False Negatives"]):
    plot_feature_importance_for_segment(segment, feature_names, name)

# SHAP値の分析と特徴量のランキングに基づく共通および特殊な特徴量の抽出
rankings = {
    "TP": calculate_shap_ranking(shap_values, TP, feature_names),
    "FP": calculate_shap_ranking(shap_values, FP, feature_names),
    "TN": calculate_shap_ranking(shap_values, TN, feature_names),
    "FN": calculate_shap_ranking(shap_values, FN, feature_names)
}


# 上位/下位10個の特徴量を取得
top_features_up = {key: rankings[key].head(6) for key in rankings}
top_features_down = {key: rankings[key].tail(6) for key in rankings}
# 上位/下位10個の特徴量を取得（FN/FP）
print("top_features_up:", top_features_up["FP"].index)
print("top_features_down:", top_features_down["FN"].index)

# 共通特徴量の抽出
common_features = set(top_features_up["TP"].index) & set(top_features_down["TN"].index) & set(top_features_up["FP"].index) & set(top_features_down["FN"].index)
print("Common Features across all quadrants:", common_features)

# TPとFPの共通特徴量の計算
tp_fp_common_features = set(top_features_up["TP"].index) & set(top_features_up["FP"].index)
print("TPとFPの共通特徴量:", tp_fp_common_features)

# TNとFNの共通特徴量の計算
tn_fn_common_features = set(top_features_down["TN"].index) & set(top_features_down["FN"].index)
print("TNとFNの共通特徴量:", tn_fn_common_features)

# FP/FN固有の特徴量の抽出
tp_fn_unique_features1 = set(top_features_up["FP"].index)- tp_fp_common_features
tn_fn_unique_features2 = set(top_features_down["FN"].index)- tn_fn_common_features
print("TP/FPで特殊な特徴量(FP):", tp_fn_unique_features1)
print("FP/FNで特殊な特徴量(FN):", tn_fn_unique_features2)

# TP/TN共通
z1 = set(top_features_up["TP"].index) & set(top_features_down["TN"].index)
print("Common Features across all quadrants:", z1)


#########
# SHAP値の要約プロット（モデル全体）
print("SHAP Summary Plot for Entire Model:")
shap.summary_plot(shap_values.values, X_test, plot_type="dot", feature_names=feature_names)

# 各象限毎にSHAP summary_plotを表示
for segment, label in zip([TP, FP, TN, FN], ["True Positives", "False Positives", "True Negatives", "False Negatives"]):
    print(f"SHAP Summary Plot for {label}:")
    segment_shap_values = shap_values.values[segment.index]
    shap.summary_plot(segment_shap_values, segment, plot_type="dot", feature_names=feature_names)

# SHAP Force Plotの表示用関数（モデル全体）
print("SHAP Force Plot for Entire Model:")
force_plot = shap.force_plot(explainer.expected_value, shap_values.values, X_test, feature_names=feature_names)
shap.save_html("./content/force_plot_entire_model.html", force_plot)
display(HTML(filename="force_plot_entire_model.html"))

# 各象限毎にSHAP Force Plotを表示
for segment, label in zip([TP, FP, TN, FN], ["True Positives", "False Positives", "True Negatives", "False Negatives"]):
    print(f"SHAP Force Plot for {label}:")
    segment_shap_values = shap_values.values[segment.index]
    force_plot_case = shap.force_plot(explainer.expected_value, segment_shap_values, segment, feature_names=feature_names)
    html_file = f"force_plot_{label}.html"
    shap.save_html(html_file, force_plot_case)
    display(HTML(filename=html_file))

### コードの追記部分
####混同行列の集計結果####
# TP, FP, TN, FN に該当するレコードのインデックスを取得
tp_indices = X_test[(y_test == 1) & (xgb_test_predictions == 1)].index
fp_indices = X_test[(y_test == 0) & (xgb_test_predictions == 1)].index
tn_indices = X_test[(y_test == 0) & (xgb_test_predictions == 0)].index
fn_indices = X_test[(y_test == 1) & (xgb_test_predictions == 0)].index

# 各象限ごとにテストデータの予測結果と元のデータをCSVに出力する関数
def save_full_predictions_by_quadrant(quadrant_indices, filename):
    # 元のテストデータから対応するインデックスのレコードを抽出
    quadrant_full_df = test_data.loc[quadrant_indices]
    # 予測結果を追加
    quadrant_full_df['Prediction'] = test_predictions_df.loc[quadrant_indices, 'Prediction']
    # CSVに保存
    quadrant_csv_path = f'./content/{filename}_full.csv'
    quadrant_full_df.to_csv(quadrant_csv_path, index=False)
    print(f"{filename}_full saved to: {quadrant_csv_path}")

# 各象限のデータを保存
save_full_predictions_by_quadrant(tp_indices, "tp_predictions")
save_full_predictions_by_quadrant(fp_indices, "fp_predictions")
save_full_predictions_by_quadrant(tn_indices, "tn_predictions")
save_full_predictions_by_quadrant(fn_indices, "fn_predictions")

# 各象限ごとにテストデータの予測結果と予測確率を含むCSVを出力する関数
def save_predictions_by_quadrant_with_proba(quadrant_indices, filename):
    quadrant_df = test_data.loc[quadrant_indices]
    quadrant_csv_path = f'./content/{filename}_with_proba.csv'
    quadrant_df.to_csv(quadrant_csv_path, index=False)
    print(f"{filename} with proba saved to: {quadrant_csv_path}")

# 各象限のデータを保存
save_predictions_by_quadrant_with_proba(tp_indices, "tp_predictions")
save_predictions_by_quadrant_with_proba(fp_indices, "fp_predictions")
save_predictions_by_quadrant_with_proba(tn_indices, "tn_predictions")
save_predictions_by_quadrant_with_proba(fn_indices, "fn_predictions")
