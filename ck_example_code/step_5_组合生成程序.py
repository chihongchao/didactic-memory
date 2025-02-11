from quants_00_hk_stock_code import stocks_dict
import random
import json
import os
import numpy as np

# 定义被剔除的股票代码数组
excluded_codes = [
    "02800", "02801", "02825", "02828", "02837", "03032", "03033", "03037", "03040", "03067",
    "03069", "03070", "03088", "03110", "03115", "03403"
]

# 定义分析的年份数组
analysis_years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]


# 从stocks_dict中剔除excluded_codes中的股票代码
filtered_stocks = {code: name for code, name in stocks_dict.items() if code not in excluded_codes}

# 定义缓存目录
cache_dir = './data'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# 读取每只股票的单年收益信息
def read_stock_profile(stock_code, year):

    profile_file_path = os.path.join(cache_dir, f"{stock_code}_profile.json")
    if not os.path.exists(profile_file_path):
        return None
    with open(profile_file_path, mode='r', encoding='utf-8') as jsonfile:
        profile_data = json.load(jsonfile)
    for year_profile in profile_data['year_profile']:
        if year_profile['data_date'] == year:
            return year_profile, profile_data['industry_type']
    return None

# 读取每只股票的分类信息
def read_stock_industry_type(stock_code):
    profile_file_path = os.path.join(cache_dir, f"{stock_code}_profile.json")
    if not os.path.exists(profile_file_path):
        return None
    with open(profile_file_path, mode='r', encoding='utf-8') as jsonfile:
        profile_data = json.load(jsonfile)
        return profile_data['industry_type'], profile_data['industry_sub_type']


  # 导入 numpy 库

def calculate_combination_attributes(combination, year):
    next_total_return = 0.0
    pre_total_return = 0.0


    industries = set()
    industries_sub = set()

    returns = []  # 用于存储每个股票的收益
    roes = []  # 用于存储每个股票的ROE
    revenue_growths = []  # 用于存储每个股票的营业额增长率
    profit_growths = []  # 用于存储每个股票的利润增长率
    caps = []  # 用于存储每个股票的市值
    pe_ttms = []  # 用于存储每个股票的动态市盈率TTM
    pes = []  # 用于存储每个股票的静态市盈率
    pbs = []  # 用于存储每个股票的市净率
    pcfs = []  # 用于存储每个股票的市现率

    sum_codes = 0
    for code in combination:
        result = read_stock_profile(code, year)

        industry_result = read_stock_industry_type(code)
        if industry_result is None:
            continue  # 如果没有数据，跳过该股票
        industry_type, industry_sub_type = industry_result

        current_year = int(year)
        next_year = current_year + 1

        if result is None:
            continue  # 如果没有数据，跳过该股票
        sum_codes += 1
        stock_profile, _ = result

        next_stock_return = 0
        if str(next_year) in analysis_years:
            next_result = read_stock_profile(code, str(next_year))
            if next_result is not None:
                next_stock_profile, next_industry_type = next_result
                next_stock_return = next_stock_profile['exchange'].get('近一年股价增长幅度', 0)

        pre_stock_return = 0
        pre_year = current_year - 1
        if str(pre_year) in analysis_years:
            pre_result = read_stock_profile(code, str(pre_year))
            if pre_result is not None:
                pre_stock_profile, pre_industry_type = pre_result
                pre_stock_return = pre_stock_profile['exchange'].get('近一年股价增长幅度', 0)

        industries.add(industry_type)
        industries_sub.add(industry_sub_type)
        stock_return = stock_profile['exchange'].get('近一年股价增长幅度', 0)  # 默认值为0
        roe = stock_profile['growth_rates'].get('净资产收益率（ROE）', 0)  # 默认值为0
        revenue_growth = stock_profile['growth_rates'].get('营业额同比增长率', 0)  # 默认值为0
        profit_growth = stock_profile['growth_rates'].get('净利润同比增长率', 0)  # 默认值为0

        cap_value = stock_profile['evaluation'].get('市值', 0)  # 默认值为0
        pe_ttm_value = stock_profile['evaluation'].get('动态市盈率TTM', 0)  # 默认值为0
        pe_value = stock_profile['evaluation'].get('静态市盈率', 0)  # 默认值为0
        pb_value = stock_profile['evaluation'].get('市净率', 0)  # 默认值为0
        pcf_value = stock_profile['evaluation'].get('市现率', 0)  # 默认值为0

        returns.append(stock_return / 100.0 if stock_return is not None else 0)
        roes.append(roe if roe is not None else 0)
        revenue_growths.append(revenue_growth if revenue_growth is not None else 0)
        profit_growths.append(profit_growth if profit_growth is not None else 0)
        caps.append(cap_value if cap_value is not None else 0)
        pe_ttms.append(pe_ttm_value if pe_ttm_value is not None else 0)
        pes.append(pe_value if pe_value is not None else 0)
        pbs.append(pb_value if pb_value is not None else 0)
        pcfs.append(pcf_value if pcf_value is not None else 0)
        next_total_return += next_stock_return/100.0  if next_stock_return is not None else 0
        pre_total_return += pre_stock_return/100.0  if pre_stock_return is not None else 0
    if sum_codes == 0:
        return None  # 如果没有有效的股票数据，返回None

    return {
        "组合收益": f"{np.mean(returns) * 100:.2f}%",
        "组合收益标准差": f"{np.std(returns) * 100:.2f}%",
        "组合前年收益": f"{(pre_total_return / sum_codes) * 100:.2f}%",
        "组合次年收益": f"{(next_total_return / sum_codes) * 100:.2f}%",
        "平均ROE": f"{np.mean(roes):.2f}%",
        "ROE标准差": f"{np.std(roes):.2f}%",
        "平均营业额增长率": f"{np.mean(revenue_growths):.2f}%",
        "营业额增长率标准差": f"{np.std(revenue_growths):.2f}%",
        "平均利润增长率": f"{np.mean(profit_growths):.2f}%",
        "利润增长率标准差": f"{np.std(profit_growths):.2f}%",
        "平均市值": f"{np.mean(caps):.2f}",
        "市值标准差": f"{np.std(caps):.2f}",
        "平均动态市盈率TTM": f"{np.mean(pe_ttms):.2f}",
        "动态市盈率TTM标准差": f"{np.std(pe_ttms):.2f}",
        "平均静态市盈率": f"{np.mean(pes):.2f}",
        "静态市盈率标准差": f"{np.std(pes):.2f}",
        "平均市净率": f"{np.mean(pbs):.2f}",
        "市净率标准差": f"{np.std(pbs):.2f}",
        "平均市现率": f"{np.mean(pcfs):.2f}",
        "市现率标准差": f"{np.std(pcfs):.2f}",
        "有交易的股票数": f"{sum_codes}",
        "行业列表": list(industries),
        "行业子类": list(industries_sub)
    }

# 生成随机组合并计算组合属性
combinations_data = []
num_combinations = 1000  # 生成1000个随机组合
while len(combinations_data) < num_combinations:
    combination = random.sample(list(filtered_stocks.keys()), 10)
    industries = set()
    for code in combination:
        industry = filtered_stocks[code].split("|")[1]
        industries.add(industry)

    # 如果行业数量不少于8个，则保留该组合
    if len(industries) >= 8:
        combination_id = f"Combination_{len(combinations_data) + 1}"
        combination_profile = {
            "Combination ID": combination_id,
            "Combination": combination,
            "组合属性": []
        }

        # 计算组合属性
        for year in analysis_years:
            attributes = calculate_combination_attributes(combination, year)
            if attributes:
                combination_profile["组合属性"].append({
                    "data_date": year,
                    **attributes
                })

        combinations_data.append(combination_profile)

# 将组合信息保存到JSON文件中
output_file_path = os.path.join(cache_dir, "combinations.json")
with open(output_file_path, mode='w', encoding='utf-8') as jsonfile:
    json.dump(combinations_data, jsonfile, ensure_ascii=False, indent=4)

print(f"All combination data has been saved to {output_file_path}")



# 计算每个组合每年的次年组合收益排名
def calculate_ranking(combinations_data):
    # 按年对组合进行排名
    for year in analysis_years:
        # 提取该年所有组合的次年收益
        year_returns = []
        for combination in combinations_data:
            for attribute in combination["组合属性"]:
                if attribute["data_date"] == year:
                    next_year_return = float(attribute["组合次年收益"].strip("%"))
                    year_returns.append((combination["Combination ID"], next_year_return))

        # 根据次年收益进行排序（降序）
        year_returns.sort(key=lambda x: x[1], reverse=True)

        # 为每个组合分配排名
        ranking = {}
        for idx, (combination_id, _) in enumerate(year_returns):
            ranking[combination_id] = idx + 1  # 排名从1开始

        # 将排名信息添加到每个组合的属性中
        for combination in combinations_data:
            combination_id = combination["Combination ID"]
            for attribute in combination["组合属性"]:
                if attribute["data_date"] == year:
                    attribute["次年组合收益排名"] = ranking[combination_id]

    return combinations_data


# 调用计算排名的方法
combinations_data = calculate_ranking(combinations_data)

# 保存更新后的组合信息到原有的JSON文件中
with open(output_file_path, mode='w', encoding='utf-8') as jsonfile:
    json.dump(combinations_data, jsonfile, ensure_ascii=False, indent=4)

print(f"Updated combination data with ranking has been saved to {output_file_path}")