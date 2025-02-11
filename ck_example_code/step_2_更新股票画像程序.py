import os
import json
from collections import defaultdict
from quants_00_hk_stock_code import stocks_dict  # 假设这是导入股票字典的模块

# 定义缓存目录
cache_dir = './data'
os.makedirs(cache_dir, exist_ok=True)

# 读取所有股票的JSON文件并加载数据
def load_all_stock_data(stocks_dict, cache_dir):
    all_stock_data = []
    for stock_code, stock_info in stocks_dict.items():
        stock_name, industry_type = stock_info.split("|")
        json_file_path = os.path.join(cache_dir, f"{stock_code}_profile.json")
        if os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as f:
                stock_data = json.load(f)
                stock_data["stock_code"] = stock_code
                stock_data["stock_name"] = stock_name.strip()
                all_stock_data.append(stock_data)
    return all_stock_data

# 计算行业地位
def calculate_industry_ranking(all_stock_data):
    industry_ranking = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for stock_data in all_stock_data:
        industry_type = stock_data["industry_type"]
        for year_profile in stock_data["year_profile"]:
            year = year_profile["data_date"]
            try:
                # 确保 evaluation 和 income 字段存在
                evaluation = year_profile.get("evaluation", {})
                income = year_profile.get("income", {})
                balance = year_profile.get("balance", {})
                growth_rates = year_profile.get("growth_rates", {})

                # 提取数据，确保数据存在且有效
                market_cap = float(evaluation.get("市值", 0)) if evaluation.get("市值") is not None else 0
                turnover = float(income.get("营业额", 0)) if income.get("营业额") is not None else 0
                total_assets = float(balance.get("总资产", 0)) if balance.get("总资产") is not None else 0
                roe = float(growth_rates.get("净资产收益率（ROE）", 0)) if growth_rates.get("净资产收益率（ROE）") is not None else 0
                debt_ratio = float(growth_rates.get("资产负债率", 0)) if growth_rates.get("资产负债率") is not None else 0
                pe_static = float(evaluation.get("静态市盈率", 0)) if evaluation.get("静态市盈率") is not None else 0

                # 只有当数据大于 0 时，才加入排名列表
                if market_cap > 0:
                    industry_ranking[industry_type][year]["市值"].append((stock_data["stock_code"], market_cap))
                if turnover > 0:
                    industry_ranking[industry_type][year]["营业额"].append((stock_data["stock_code"], turnover))
                if total_assets > 0:
                    industry_ranking[industry_type][year]["总资产"].append((stock_data["stock_code"], total_assets))
                if roe > 0:
                    industry_ranking[industry_type][year]["净资产收益率（ROE）"].append((stock_data["stock_code"], roe))
                if debt_ratio > 0:
                    industry_ranking[industry_type][year]["资产负债率"].append((stock_data["stock_code"], debt_ratio))
                if pe_static > 0:
                    industry_ranking[industry_type][year]["静态市盈率"].append((stock_data["stock_code"], pe_static))
            except (KeyError, ValueError) as e:
                print(f"Warning: Missing or invalid data for stock {stock_data['stock_code']} in year {year}: {e}")
                continue

    # 排名
    for industry, years in industry_ranking.items():
        for year, metrics in years.items():
            for metric, values in metrics.items():
                # 排序并添加排名信息
                sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
                ranked_values = []
                for rank, (code, value) in enumerate(sorted_values, start=1):
                    ranked_values.append((code, rank, len(sorted_values)))
                metrics[metric] = ranked_values  # 更新为排名后的数据

    return industry_ranking

# 更新JSON数据
def update_json_with_ranking(stock_data, industry_ranking):
    stock_code = stock_data["stock_code"]
    industry_type = stock_data["industry_type"]
    for year_profile in stock_data["year_profile"]:
        year = year_profile["data_date"]
        year_profile["industry_top"] = {}
        for metric in ["市值", "营业额", "总资产", "净资产收益率（ROE）", "资产负债率", "静态市盈率"]:
            try:
                # 直接从排名数据中获取排名信息
                rank_info = next(item for item in industry_ranking[industry_type][year][metric] if item[0] == stock_code)
                year_profile["industry_top"][f"{metric}行业地位"] = f"{rank_info[1]}/{rank_info[2]}"
            except (StopIteration, KeyError) as e:
                print(f"Warning: Missing ranking data for stock {stock_code} in year {year} for metric {metric}: {e}")
                year_profile["industry_top"][f"{metric}行业地位"] = "N/A"

# 保存到文件
def save_to_file(stock_data, cache_dir):
    stock_code = stock_data["stock_code"]
    json_file_path = os.path.join(cache_dir, f"{stock_code}_profile.json")
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(stock_data, f, ensure_ascii=False, indent=4)

# 主程序
def main():
    # 加载所有股票数据
    all_stock_data = load_all_stock_data(stocks_dict, cache_dir)

    # 计算行业排名
    industry_ranking = calculate_industry_ranking(all_stock_data)

    # 更新JSON数据并保存
    for stock_data in all_stock_data:
        update_json_with_ranking(stock_data, industry_ranking)
        save_to_file(stock_data, cache_dir)

if __name__ == "__main__":
    main()