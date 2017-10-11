# Python版本：

Python 3.6.2

# 相关库：

numpy --- 1.12.1

pandas --- 0.20.3

xgboost --- 0.60

sklearn --- 0.19.0

matplotlib --- 2.0.2

seaborn --- 0.8

# 相关文件描述：

- sales_prediction.ipynb : 记录运算过程的代码，其中在运行过程中会生成4个json文件（path : `./json/`）以保存数据以便在graph.ipynb中绘图用：

  - cvresults_random.json : 记录RandomizedSearchCV的搜索数据
  - learn_curve_default.json: 记录初始模型的learning curve数据
  - learn_curve : 记录优化模型的learning curve数据
  - feature_importance.json : 记录优化模型训练完成后所导出的各特征重要性数据

  同时还会生成2个csv文件，以记录不同模型的预测结果（path : `./score_sheet/`）：

  - sample_submission_linear.csv : 报告2.4中的基准模型的预测结果
  - sample_submission_optimized.csv : 报告4.1中的优化模型的预测结果

- graph.ipynb : 会用生产report.pdf中的各种可视化，并将图片导入path : `./pic/`中

# 所需文件：

1. train.csv : 用于训练模型的数据集
2. test.csv : 用于测试模型表现的数据集，其中并不包含标签，testset预测的结果需上传至[Kaggle---Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)才能得到评分
3. store.csv : 每家门店的补充信息

# sales_prediction.ipynb 运行所需时间：

- RandomizedSearchCV --- 9.5小时
- learning curve + 模型训练+预测 --- 0.8小时

预计运行时间 $\approx \ $**$10.5 $** 小时

因使用ipynb文件，运行成功的记录都记录在ipynb中，可点开进行查看。