## py_ml

```
docker-compose up -d
```
## chapter

[algorithm](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch02/ch02.ipynb)

[scikit-learn](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch03/ch03.ipynb)

[preprocessing](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch04/ch04.ipynb)

[dimensionality_reduction](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch05/ch05.ipynb)

[hyperparameter_tuning](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/ch06.ipynb)

[ensemble learning](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch07/ch07.ipynb)

[Sentiment Analysis](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch08/ch08.ipynb)

[web_app](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch09/ch09.ipynb)

[regression_analysis](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch10/ch10.ipynb)

[clustering_analysis](https://github.com/rasbt/python-machine-learning-book/tree/master/code/ch11/ch11.ipynb)

## 参考文献

[hirai.me](http://hirai.me/notes_pyml.html#)
[データサイエンスプロジェクトのディレクトリ構成どうするか問題](https://takuti.me/note/data-science-project-structure/)

## index

- [x] [分類問題 -- 機械学習アルゴリズムのトレーニング](https://github.com/KentFujii/pmlb/tree/master/algorithm)
    - [x] [パーセプトロン](https://github.com/KentFujii/pmlb/blob/master/algorithm/perceptron.ipynb)
    - [x] [Adaline](https://github.com/KentFujii/pmlb/blob/master/algorithm/adaline.ipynb)
    - [x] [確率的勾配法](https://github.com/KentFujii/pmlb/blob/master/algorithm/adaline_stochastic_gradient_descent.ipynb)
- [x] [分類問題 -- 機械学習ライブラリscikit-learnの活用](https://github.com/KentFujii/pmlb/tree/master/scikit-learn)
    - [x] [scikit-learnによるパーセプトロン](https://github.com/KentFujii/pmlb/blob/master/scikit-learn/sklearn_perceptron.ipynb)
    - [x] [ロジスティック回帰](https://github.com/KentFujii/pmlb/blob/master/scikit-learn/logistic_regression.ipynb)
    - [x] [サポートベクトルマシン](https://github.com/KentFujii/pmlb/blob/master/scikit-learn/svm.ipynb)
    - [x] [決定木](https://github.com/KentFujii/pmlb/blob/master/scikit-learn/decision_tree.ipynb)
    - [x] [k近傍法](https://github.com/KentFujii/pmlb/blob/master/scikit-learn/k_neighbors.ipynb)
- [x] データ前処理 -- よりよいトレーニングセットの構築
    - [x] [欠損値](https://github.com/KentFujii/pmlb/blob/master/preprocessing/missing_value.ipynb)
    - [x] [カテゴリーデータ](https://github.com/KentFujii/pmlb/blob/master/preprocessing/category_data.ipynb)
    - [x] [データセットとトレーニングデータ](https://github.com/KentFujii/pmlb/blob/master/preprocessing/train_test_split.ipynb)
    - [x] [スケーリング](https://github.com/KentFujii/pmlb/blob/master/preprocessing/scaler.ipynb)
    - [x] [有益な特徴量の選択](https://github.com/KentFujii/pmlb/blob/master/preprocessing/feature_selections.ipynb)
    - [x] [ランダムフォレストで特徴量の重要度にアクセス](https://github.com/KentFujii/pmlb/blob/master/preprocessing/feature_selection_random_forest.ipynb)
- [x] [次元削減でデータを圧縮する](https://github.com/KentFujii/pmlb/tree/master/dimensionality_reduction)
    - [x] [PCA](https://github.com/KentFujii/pmlb/blob/master/dimensionality_reduction/pca.ipynb)
    - [x] [LDA](https://github.com/KentFujii/pmlb/blob/master/dimensionality_reduction/lda.ipynb)
    - [x] [カーネルPCA](https://github.com/KentFujii/pmlb/blob/master/dimensionality_reduction/kernel_pca.ipynb)
- [x] モデル評価とハイパーパラメータのチューニングのベストプラクティス
    - [x] [パイプライン](https://github.com/KentFujii/pmlb/blob/master/hyperparameter_tuning/pipeline.ipynb)
    - [x] [k分割交差検証](https://github.com/KentFujii/pmlb/blob/master/hyperparameter_tuning/k_fold_cross_validation.ipynb)
    - [x] [学習曲線と検証曲線](https://github.com/KentFujii/pmlb/blob/master/hyperparameter_tuning/learning_curve_validation_curve.ipynb)
    - [x] [グリッドサーチ](https://github.com/KentFujii/pmlb/blob/master/hyperparameter_tuning/grid_search.ipynb)
    - [x] [性能指標](https://github.com/KentFujii/pmlb/blob/master/hyperparameter_tuning/evaluation.ipynb)
- [x] アンサンブル学習 -- 異なるモデルの組み合わせ
    - [x] [アンサンブル学習の原理](https://github.com/KentFujii/pmlb/blob/master/ensemble/ensemble_error.ipynb)
    - [x] [多数決分類器](https://github.com/KentFujii/pmlb/blob/master/ensemble/majority_voting.ipynb)
    - [x] [アンサンブル分類器の評価とチューニング](https://github.com/KentFujii/pmlb/blob/master/ensemble/evaluating_and_tuning_the_ensemble_classifier.ipynb)
    - [x] [バギング](https://github.com/KentFujii/pmlb/blob/master/ensemble/bagging.ipynb)
    - [x] [ブースティング](https://github.com/KentFujii/pmlb/blob/master/ensemble/boosting.ipynb)
- [x] 機械学習の適用1 -- 感情分析
    - [x] [映画レビューデータセット](https://github.com/KentFujii/pmlb/blob/master/sentiment_analysis/acl_imdb.ipynb)
    - [x] [Bag-of-Wordsモデル](https://github.com/KentFujii/pmlb/blob/master/sentiment_analysis/bag_of_words_model.ipynb)
    - [x] [アウトオブコア学習](https://github.com/KentFujii/pmlb/blob/master/sentiment_analysis/out_of_core_learning.ipynb)
- [x] 機械学習の適用2 -- Webアプリケーション
    - [x] [学習済みの推定器をシリアライズする](https://github.com/KentFujii/pmlb/blob/master/web_app/serialize_trained_classifier.ipynb)
- [x] 回帰分析 -- 連続値をとる目的変数の予測
    - [x] [Housingデータセット](https://github.com/KentFujii/pmlb/blob/master/regression_analysis/housing_dataset.ipynb)
    - [x] [最小二乗線形回帰モデル](https://github.com/KentFujii/pmlb/blob/master/regression_analysis/ordinary%20_least_squares_linear_regression.ipynb)
    - [x] [RANSACを使ったロバスト回帰モデルの学習](https://github.com/KentFujii/pmlb/blob/master/regression_analysis/ransac.ipynb)
    - [x] [線形回帰モデルの性能評価](https://github.com/KentFujii/pmlb/blob/master/regression_analysis/evaluating_the_performance_of_linear_regression.ipynb)
    - [x] [回帰に正則化手法を使用する](https://github.com/KentFujii/pmlb/blob/master/regression_analysis/regularized_methods_for_regression.ipynb)
    - [x] [多項式回帰：線形回帰モデルから曲線を見出す](https://github.com/KentFujii/pmlb/blob/master/regression_analysis/polynomial_regression.ipynb)
- [x] クラスタ分析 -- ラベルなしデータの分析
    - [x] [k-mean](https://github.com/KentFujii/pmlb/blob/master/clustering_analysis/k-means.ipynb)
    - [x] [クラスタを階層木として構成する](https://github.com/KentFujii/pmlb/blob/master/clustering_analysis/hierarchical_tree.ipynb)
    - [x] [DBSCAN](https://github.com/KentFujii/pmlb/blob/master/clustering_analysis/DBSCAN.ipynb)
