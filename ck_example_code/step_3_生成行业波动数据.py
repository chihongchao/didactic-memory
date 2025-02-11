import os
import json
from quants_00_hk_stock_code import stocks_dict

# 定义数据存储结构
industry_data = {}

# 定义数据目录
cache_dir = './data'

# 遍历数据目录中的所有JSON文件
for filename in os.listdir(cache_dir):
    if filename.endswith('_profile.json'):
        stock_code = filename.split('_')[0]
        json_file_path = os.path.join(cache_dir, filename)

        # 读取JSON文件
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 提取数据
        stocks_data = data['year_profile']

        # 遍历每个股票的数据
        for stock in stocks_data:
            industry = stocks_dict.get(stock_code, "未知行业").split('|')[1]
            year = stock['data_date']

            # 提取市盈率、市净率、负债率和ROE
            pe_ratio = stock['evaluation']['动态市盈率TTM']
            pb_ratio = stock['evaluation']['市净率']
            debt_ratio = stock['growth_rates']['资产负债率']
            roe = stock['growth_rates']['净资产收益率（ROE）']

            # 将数据归入相应的行业
            if industry not in industry_data:
                industry_data[industry] = {}
            if year not in industry_data[industry]:
                industry_data[industry][year] = []
            industry_data[industry][year].append((pe_ratio, pb_ratio, debt_ratio, roe))

# 计算每个行业每年的平均值
industry_averages = {}

for industry, years_data in industry_data.items():
    industry_averages[industry] = {}
    for year, data in years_data.items():
        valid_data = [(d[0], d[1], d[2], d[3]) for d in data if None not in d]
        if valid_data:
            pe_ratio_avg = round(sum(d[0] for d in valid_data) / len(valid_data), 3)
            pb_ratio_avg = round(sum(d[1] for d in valid_data) / len(valid_data), 3)
            debt_ratio_avg = round(sum(d[2] for d in valid_data) / len(valid_data), 3)
            roe_avg = round(sum(d[3] for d in valid_data) / len(valid_data), 3)
        else:
            pe_ratio_avg, pb_ratio_avg, debt_ratio_avg, roe_avg = None, None, None, None
        industry_averages[industry][year] = {
            "平均市盈率": pe_ratio_avg,
            "平均市净率": pb_ratio_avg,
            "平均负债率": debt_ratio_avg,
            "平均ROE": roe_avg
        }

# 计算近5年、近4年、近3年、近2年、近1年的平均值
for industry, years_data in industry_averages.items():
    years = sorted(years_data.keys(), key=int, reverse=True)  # 按年份降序排列
    for year in years:
        for n in [5, 4, 3, 2, 1]:
            recent_years = [yr for yr in years if int(yr) >= int(year) - n and int(yr) <= int(year)]
            valid_recent_years_data = [years_data[yr] for yr in recent_years if
                                       None not in [years_data[yr]["平均市盈率"], years_data[yr]["平均市净率"],
                                                    years_data[yr]["平均负债率"], years_data[yr]["平均ROE"]]]
            if valid_recent_years_data:
                pe_ratio_avg = round(
                    sum(d["平均市盈率"] for d in valid_recent_years_data) / len(valid_recent_years_data), 3)
                pb_ratio_avg = round(
                    sum(d["平均市净率"] for d in valid_recent_years_data) / len(valid_recent_years_data), 3)
                debt_ratio_avg = round(
                    sum(d["平均负债率"] for d in valid_recent_years_data) / len(valid_recent_years_data), 3)
                roe_avg = round(sum(d["平均ROE"] for d in valid_recent_years_data) / len(valid_recent_years_data), 3)
            else:
                pe_ratio_avg, pb_ratio_avg, debt_ratio_avg, roe_avg = None, None, None, None
            years_data[year][f'近{n}年平均市盈率'] = pe_ratio_avg
            years_data[year][f'近{n}年平均市净率'] = pb_ratio_avg
            years_data[year][f'近{n}年平均负债率'] = debt_ratio_avg
            years_data[year][f'近{n}年平均ROE'] = roe_avg

# 保存结果到JSON文件
output_file = os.path.join(cache_dir, 'industry_type_analysis.json')
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(industry_averages, f, ensure_ascii=False, indent=4)

print(f"结果已保存到 {output_file}")