import akshare as ak
import pandas as pd

# 设置Pandas的显示选项，以便打印所有列
pd.set_option('display.max_columns', None)


stock_financial_hk_report_em_df = ak.stock_financial_hk_report_em(stock="02333", symbol="资产负债表", indicator="年度")

print(stock_financial_hk_report_em_df.columns)

filtered_df = stock_financial_hk_report_em_df.loc[stock_financial_hk_report_em_df['REPORT_DATE'] == '2023-12-31 00:00:00']
print(filtered_df)

lr = ak.stock_financial_hk_report_em(stock="02333", symbol="利润表", indicator="年度")
print(lr.columns)
print(lr.head())
anlyasis = ak.stock_financial_hk_analysis_indicator_em(symbol="02333", indicator="报告期")
# print(anlyasis)
# gpdm = ak.stock_hk_hist()
# print(gpdm)