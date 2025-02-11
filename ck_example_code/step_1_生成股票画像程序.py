from quants_00_hk_stock_code import stocks_dict
import akshare as ak
import pandas as pd
import os
import csv
import json

# 定义分析年份
analy_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']

def generate_stock_profile():
    # 定义缓存目录
    cache_dir = './data'

    for stock_code, stock_name in stocks_dict.items():
        json_data = {
            "stock_code": "",
            "stock_name": "",
            "industry_type": "",
            "industry_sub_type": "",
            "year_profile": []
        }

        name, category = stock_name.split('|')

        # 检查是否包含括号
        if '（' in category and '）' in category:
            # 提取一级分类（括号之前的内容）
            primary_category = category[:category.index('（')].strip()
            # 提取二级分类（括号内的内容）
            secondary_category = category[category.index('（') + 1:category.index('）')].strip()
        else:
            # 如果没有括号，一级分类为整个分类部分，二级分类为 None
            primary_category = category.strip()
            secondary_category = None


        json_data['stock_code'] = stock_code
        json_data['stock_name'] = name
        json_data['industry_type'] = primary_category
        json_data['industry_sub_type'] = secondary_category

        # 确保缓存目录存在
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # step 1  读取资产负债表 -----
        # 定义缓存文件路径
        cache_bl_file_path = os.path.join(cache_dir, f"{stock_code}_report_em.csv")
        report_em_df = None
        if os.path.exists(cache_bl_file_path):
            # 从缓存文件读取数据
            report_em_df = pd.read_csv(cache_bl_file_path)
        else:
            try:

                report_em_df = ak.stock_financial_hk_report_em(stock=stock_code, symbol="资产负债表", indicator="年度")
                if report_em_df is not None:
                    report_em_df.to_csv(cache_bl_file_path, encoding='utf-8', index=False)
                    print(f"资产数据已保存到本地缓存文件：{cache_bl_file_path}")
            except TypeError as e:
                print(f"接口调用返回了 None，无法访问数据：{e}")
            except Exception as e:
                print(f"处理过程中出现异常：{e}")

        # 读取CSV文件
        if os.path.exists(cache_bl_file_path):
            with open(cache_bl_file_path, mode='r', encoding='utf-8') as csvfile:
                csvreader = csv.DictReader(csvfile)
                # 将所有记录读入内存，避免多次打开文件
                records = list(csvreader)

                for year in analy_years:
                    year_profile = {"data_date": year, "balance": {}, "income": {}, "crash_flow": {}}  # 初始化 "income" 字段
                    for row in records:
                        # 提取年份
                        fiscal_year = row['REPORT_DATE'].split('-')[0]
                        if fiscal_year == year:
                            item_name = row['STD_ITEM_NAME']
                            amount = row['AMOUNT']
                            year_profile["balance"][item_name] = amount

                    if year_profile["balance"]:
                        json_data['year_profile'].append(year_profile)

        # step 2  读取利润表 -----
        # 定义缓存文件路径
        cache_re_file_path = os.path.join(cache_dir, f"{stock_code}_report_lr.csv")
        report_em_df = None
        if os.path.exists(cache_re_file_path):
            # 从缓存文件读取数据
            report_em_df = pd.read_csv(cache_re_file_path)
        else:
            try:
                report_em_df = ak.stock_financial_hk_report_em(stock=stock_code, symbol="利润表", indicator="年度")
                if report_em_df is not None:
                    report_em_df.to_csv(cache_re_file_path, encoding='utf-8', index=False)
                    print(f"数据已保存到本地缓存文件：{cache_re_file_path}")
            except TypeError as e:
                print(f"接口调用返回了 None，无法访问数据：{e}")
            except Exception as e:
                print(f"处理过程中出现异常：{e}")

        # 读取CSV文件
        if os.path.exists(cache_re_file_path):
            with open(cache_re_file_path, mode='r', encoding='utf-8') as csvfile:
                csvreader = csv.DictReader(csvfile)
                # 将所有记录读入内存，避免多次打开文件
                records = list(csvreader)

                # 遍历每个年份，追加新的字段
                for year_profile in json_data['year_profile']:
                    year = year_profile['data_date']
                    for row in records:
                        # 提取年份
                        fiscal_year = row['REPORT_DATE'].split('-')[0]
                        if fiscal_year == year:
                            item_name = row['STD_ITEM_NAME']
                            amount = row['AMOUNT']
                            year_profile["income"][item_name] = amount




        # step 3  读取现金流表 -----
        # 定义缓存文件路径
        cache_cf_file_path = os.path.join(cache_dir, f"{stock_code}_report_cf.csv")
        report_cf_df = None
        if os.path.exists(cache_cf_file_path):
            # 从缓存文件读取数据
            report_cf_df = pd.read_csv(cache_cf_file_path)
        else:
            try:
                # SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,ORG_CODE,REPORT_DATE,DATE_TYPE_CODE,FISCAL_YEAR,START_DATE,STD_ITEM_CODE,STD_ITEM_NAME,AMOUNT"
                report_cf_df = ak.stock_financial_hk_report_em(stock=stock_code, symbol="现金流量表", indicator="年度")
                if report_cf_df is not None:
                    report_cf_df.to_csv(cache_cf_file_path, encoding='utf-8', index=False)
                    print(f"现金流数据已保存到本地缓存文件：{cache_cf_file_path}")
            except TypeError as e:
                print(f"接口调用返回了 None，无法访问数据：{e}")
            except Exception as e:
                print(f"处理过程中出现异常：{e}")

        # 读取CSV文件
        if os.path.exists(cache_cf_file_path):
            with open(cache_cf_file_path, mode='r', encoding='utf-8') as csvfile:
                csvreader = csv.DictReader(csvfile)
                # 将所有记录读入内存，避免多次打开文件
                records = list(csvreader)

                # 遍历每个年份，追加新的字段
                for year_profile in json_data['year_profile']:
                    year = year_profile['data_date']
                    for row in records:
                        # 提取年份
                        #
                        fiscal_year = row['REPORT_DATE'].split('-')[0]
                        if fiscal_year == year:
                            item_name = row['STD_ITEM_NAME']
                            amount = row['AMOUNT']
                            year_profile["crash_flow"][item_name] = amount

        # 将数据写入JSON文件
        json_file_path = os.path.join(cache_dir, f"{stock_code}_profile.json")
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)

        print(f"资产、利润、现金流的数据以JSON数据形式已保存到 {json_file_path}")

        # step 4  计算成长属性并保存画像 -----
        # 检查JSON文件是否存在
        if not os.path.exists(json_file_path):
            print(f"JSON文件不存在：{json_file_path}")
            continue

        # 读取现有的JSON文件
        with open(json_file_path, mode='r', encoding='utf-8') as jsonfile:
            json_data = json.load(jsonfile)

        # 遍历每个年份，计算增长率
        for i, year_profile in enumerate(json_data['year_profile']):
            year = year_profile['data_date']
            balance = year_profile.get('balance', {})
            income = year_profile.get('income', {})

            # 获取当前年份和上一年的数据
            current_net_profit = float(income.get("除税后溢利", 0)) if income.get("除税后溢利") else 0
            current_revenue = float(income.get("营业额", 0)) if income.get("营业额") else 0
            current_cash_flow = float(balance.get("现金及等价物", 0)) if balance.get("现金及等价物") else 0
            current_total_liabilities = float(balance.get("总负债", 0)) if balance.get("总负债") else 0
            current_total_assets = float(balance.get("总资产", 0)) if balance.get("总资产") else 0
            current_shareholders_equity = float(balance.get("股东权益", 0)) if balance.get("股东权益") else 0

            previous_net_profit = 0
            previous_revenue = 0
            previous_cash_flow = 0
            previous_total_liabilities = 0

            if i > 0:
                previous_year_profile = json_data['year_profile'][i - 1]
                previous_income = previous_year_profile.get('income', {})
                previous_balance = previous_year_profile.get('balance', {})

                previous_net_profit = float(previous_income.get("除税后溢利", 0)) if previous_income.get("除税后溢利") else 0
                previous_revenue = float(previous_income.get("营业额", 0)) if previous_income.get("营业额") else 0
                previous_cash_flow = float(previous_balance.get("现金及等价物", 0)) if previous_balance.get("现金及等价物") else 0
                previous_total_liabilities = float(previous_balance.get("总负债", 0)) if previous_balance.get("总负债") else 0
                previous_shareholders_equity = float(previous_balance.get("股东权益", 0)) if previous_balance.get(
                "股东权益") else 0
            # 计算增长率
            net_profit_growth_rate = calculate_growth_rate(current_net_profit, previous_net_profit)
            revenue_growth_rate = calculate_growth_rate(current_revenue, previous_revenue)
            cash_flow_growth_rate = calculate_growth_rate(current_cash_flow, previous_cash_flow)
            total_liabilities_growth_rate = calculate_growth_rate(current_total_liabilities, previous_total_liabilities)
            # 计算平均股东权益
            avg_shareholders_equity = (current_shareholders_equity + previous_shareholders_equity) / 2 if i > 0 else current_shareholders_equity

            # 计算净资产收益率（ROE）
            roe = calculate_roe(current_net_profit, avg_shareholders_equity)

            # 计算资产负债率
            debt_to_asset_ratio = calculate_d2Asset_raito(current_total_liabilities, current_total_assets)

            # 添加到JSON数据中
            year_profile['growth_rates'] = {
                "净利润同比增长率": net_profit_growth_rate,
                "营业额同比增长率": revenue_growth_rate,
                "现金流同比增长率": cash_flow_growth_rate,
                "总负债增长率": total_liabilities_growth_rate,
                "净资产收益率（ROE）": roe,
                "资产负债率": debt_to_asset_ratio
            }
            # step 5  计算市场交易画像，并保存 -----
            # 获取交易数据
            exchange_data = get_exchange_data(stock_code, year, cache_dir)
            # 将当年交易数据插入到画像中
            year_profile['exchange'] = exchange_data

            # step 6  计算市场的市值评估数据，并保存 -----
            # 获取市值评估数据
            evaluation_data = get_evaluation_data(stock_code, year, cache_dir)
            # 将当年最后市值评估数据插入到画像中
            year_profile['evaluation'] = evaluation_data

        # 保存更新后的JSON文件
        with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)

        print(f"更新后的JSON数据已保存到 {json_file_path}")


def calculate_growth_rate(current_value, previous_value):
    """计算增长率"""
    if previous_value == 0:
        return None  # 避免除以零的情况
    return ((current_value - previous_value) / previous_value) * 100

def calculate_roe(net_profit, avg_shareholders_equity):
    """计算净资产收益率（ROE）"""
    if avg_shareholders_equity == 0:
        return None  # 避免除以零的情况
    return (net_profit / avg_shareholders_equity) * 100

def calculate_d2Asset_raito(current_total_liabilities, current_total_assets):
    """计算资产负载率"""
    if current_total_assets == 0:
        return None  # 避免除以零的情况
    return (current_total_liabilities / current_total_assets) * 100

def get_exchange_data(stock_code, year, cache_dir):
    """获取指定年份的交易数据，并保存到缓存文件中"""

    # 定义缓存文件路径
    cache_file_path = os.path.join(cache_dir, f"{stock_code}_exchange.csv")

    # 检查缓存文件是否存在
    if os.path.exists(cache_file_path):
        # 从缓存文件读取数据
        stock_data = pd.read_csv(cache_file_path)
    else:
        # 获取全量历史数据
        stock_data = ak.stock_hk_daily(symbol=stock_code, adjust="qfq")

        if stock_data.empty:
            return {
                "年初值": None,
                "年末值": None,
                "年内最高值": None,
                "年内最低值": None,
                "近三年股价增长幅度": None,
                "近二年股价增长幅度": None,
                "近一年股价增长幅度": None
            }

        # 转换日期格式
        stock_data['date'] = pd.to_datetime(stock_data['date'])

        # 保存到缓存文件
        stock_data.to_csv(cache_file_path, index=False, encoding='utf-8')

    # 定义日期范围
    start_date = pd.to_datetime(f"{year}-01-01")
    end_date = pd.to_datetime(f"{year}-12-31")

    # 确保 'date' 列是 datetime 类型
    stock_data['date'] = pd.to_datetime(stock_data['date'])

    # 过滤特定日期范围内的数据
    filtered_data = stock_data[(stock_data['date'] >= start_date) & (stock_data['date'] <= end_date)]

    if filtered_data.empty:
        return {
            "年初值": None,
            "年末值": None,
            "年内最高值": None,
            "年内最低值": None,
            "近三年股价增长幅度": None,
            "近二年股价增长幅度": None,
            "近一年股价增长幅度": None
        }

    # 获取年初和年末的股价
    year_start_value = filtered_data.iloc[0]['close']  # 年初值
    year_end_value = filtered_data.iloc[-1]['close']  # 年末值
    year_high_value = filtered_data['high'].max()  # 年内最高值
    year_low_value = filtered_data['low'].min()  # 年内最低值

    # 计算近三年、近二年和近一年的股价增长幅度
    current_year = int(year)
    two_years_ago = current_year - 2
    three_years_ago = current_year - 3

    # 获取近三年的最后一个交易日的股价

    two_years_ago_end_date = pd.to_datetime(f"{two_years_ago}-12-31")
    three_years_ago_end_date = pd.to_datetime(f"{three_years_ago}-12-31")


    # 过滤特定日期范围内的数据
    two_years_filtered_data = stock_data[(stock_data['date'] > two_years_ago_end_date) & (stock_data['date'] <= end_date)]
    two_years_ago_start_value = two_years_filtered_data.iloc[0]['close']

    # 过滤特定日期范围内的数据
    three_years_filtered_data = stock_data[(stock_data['date'] > three_years_ago_end_date) & (stock_data['date'] <= end_date)]
    three_years_ago_start_value = three_years_filtered_data.iloc[0]['close']

    # 计算增长幅度
    one_year_growth = calculate_growth_rate(year_start_value, year_end_value) if year_start_value else None
    two_years_growth = calculate_growth_rate(two_years_ago_start_value, year_end_value) if two_years_ago_start_value else None
    three_years_growth = calculate_growth_rate(three_years_ago_start_value, year_end_value) if three_years_ago_start_value else None

    return {
        "年初值": year_start_value,
        "年末值": year_end_value,
        "年内最高值": year_high_value,
        "年内最低值": year_low_value,
        "近三年股价增长幅度": three_years_growth,
        "近二年股价增长幅度": two_years_growth,
        "近一年股价增长幅度": one_year_growth
    }


def get_evaluation_data(stock_code, year, cache_dir):
    """获取指定年份的市值评估数据，并保存到缓存文件中"""

    # 定义缓存文件路径
    cache_file_path = os.path.join(cache_dir, f"{stock_code}_evaluation.csv")

    # 检查缓存文件是否存在
    if os.path.exists(cache_file_path):
        # 从缓存文件读取数据
        stock_data = pd.read_csv(cache_file_path)
    else:
        # 获取全量历史评估数据
        stock_data = get_evaluation_df(stock_code)

        if stock_data.empty:
            print(f"未获取到 {stock_code} 的估值数据，跳过保存。")
            return {
                "市值": None,
                "动态市盈率TTM": None,
                "静态市盈率": None,
                "市净率": None,
                "市现率": None
            }

        # 转换日期格式
        stock_data['date'] = pd.to_datetime(stock_data['date'])

        # 保存到缓存文件
        stock_data.to_csv(cache_file_path, index=False, encoding='utf-8')

    # 定义日期范围
    start_date = pd.to_datetime(f"{year}-01-01")
    end_date = pd.to_datetime(f"{year}-12-31")

    # 确保 'date' 列是 datetime 类型
    stock_data['date'] = pd.to_datetime(stock_data['date'])

    # 过滤特定日期范围内的数据
    filtered_data = stock_data[(stock_data['date'] >= start_date) & (stock_data['date'] <= end_date)]

    if filtered_data.empty:
        print(f"未找到 {year} 年的估值数据。")
        return {
            "市值": None,
            "动态市盈率TTM": None,
            "静态市盈率": None,
            "市净率": None,
            "市现率": None
        }

    # 获取年初和年末的股价
    cap_value = filtered_data.iloc[-1]['CAP']  # 总资产
    pe_ttm_value = filtered_data.iloc[-1]['PE_TTM']  # 动态市盈率
    pe_value = filtered_data.iloc[-1]['PE']   # 市盈率
    pb_value = filtered_data.iloc[-1]['PB']   # 市净率
    pcf_value = filtered_data.iloc[-1]['PCF']   # 市现率

    return {
        "市值": cap_value,
        "动态市盈率TTM": pe_ttm_value,
        "静态市盈率": pe_value,
        "市净率": pb_value,
        "市现率": pcf_value
    }


def get_evaluation_df(stock_code):
    # 获取港股市盈率(TTM)数据、等市值评估数据
    try:
        CAP_df = ak.stock_hk_valuation_baidu(symbol=stock_code, indicator="总市值", period="全部")
        PE_TTM_df = ak.stock_hk_valuation_baidu(symbol=stock_code, indicator="市盈率(TTM)", period="全部")
        PE_df = ak.stock_hk_valuation_baidu(symbol=stock_code, indicator="市盈率(静)", period="全部")
        PB_df = ak.stock_hk_valuation_baidu(symbol=stock_code, indicator="市净率", period="全部")
        PCF_df = ak.stock_hk_valuation_baidu(symbol=stock_code, indicator="市现率", period="全部")
    except Exception as e:
        print(f"获取估值数据时发生错误：{e}")
        return pd.DataFrame()  # 返回空的 DataFrame

    # 检查每个 DataFrame 是否为空
    if CAP_df.empty or PE_TTM_df.empty or PE_df.empty or PB_df.empty or PCF_df.empty:
        print("某些估值数据为空，无法完成合并。")
        return pd.DataFrame()  # 返回空的 DataFrame

    # 将所有数据集放入一个列表
    valuation_dfs = [CAP_df, PE_TTM_df, PE_df, PB_df, PCF_df]

    # 初始化合并后的数据框
    merged_df = valuation_dfs[0]

    # 逐个合并数据集
    for i, df in enumerate(valuation_dfs[1:], start=1):
        merged_df = pd.merge(merged_df, df, on='date', suffixes=('', f'-{i + 1}'))

    # 重命名列
    merged_df.columns = ['date', 'CAP', 'PE_TTM', 'PE', 'PB', 'PCF']

    return merged_df
# 调用函数
generate_stock_profile()
