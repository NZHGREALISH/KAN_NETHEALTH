import pandas as pd
import numpy as np

def inspect_data_files():
    """
    检查数据文件的结构和内容
    """
    
    # 检查Fitbit数据
    print("=" * 60)
    print("检查Fitbit数据文件")
    print("=" * 60)
    
    fitbit_files = ["fitbit_data.csv", "BasicSurvey3620.csv", "data.csv"]  # 可能的文件名
    fitbit_df = None
    
    for filename in fitbit_files:
        try:
            print(f"尝试读取: {filename}")
            fitbit_df = pd.read_csv(filename)
            print(f"✓ 成功读取 {filename}")
            break
        except FileNotFoundError:
            print(f"✗ 文件不存在: {filename}")
        except Exception as e:
            print(f"✗ 读取失败 {filename}: {e}")
    
    if fitbit_df is not None:
        print(f"\nFitbit数据基本信息:")
        print(f"  行数: {len(fitbit_df)}")
        print(f"  列数: {len(fitbit_df.columns)}")
        print(f"  列名: {list(fitbit_df.columns)}")
        
        # 检查可能的ID列
        id_candidates = []
        for col in fitbit_df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'ego', 'user', 'participant']):
                id_candidates.append(col)
                unique_count = fitbit_df[col].nunique()
                print(f"  可能的ID列 '{col}': {unique_count} 个唯一值")
        
        # 显示前几行
        print(f"\n前5行数据:")
        print(fitbit_df.head())
        
        # 检查数据类型
        print(f"\n数据类型:")
        print(fitbit_df.dtypes)
        
    else:
        print("无法读取Fitbit数据文件")
        return None, None
    
    # 检查量表数据
    print("\n" + "=" * 60)
    print("检查量表数据文件")
    print("=" * 60)
    
    scales_files = ["/home/grealish/summer/KAN_NET/check/extracted_scales_data.csv", "FitbitActivity(1-30-20).csv"]
    scales_df = None
    
    for filename in scales_files:
        try:
            print(f"尝试读取: {filename}")
            scales_df = pd.read_csv(filename)
            print(f"✓ 成功读取 {filename}")
            break
        except FileNotFoundError:
            print(f"✗ 文件不存在: {filename}")
        except Exception as e:
            print(f"✗ 读取失败 {filename}: {e}")
    
    if scales_df is not None:
        print(f"\n量表数据基本信息:")
        print(f"  行数: {len(scales_df)}")
        print(f"  列数: {len(scales_df.columns)}")
        
        # 检查ID列
        if 'egoid' in scales_df.columns:
            print(f"  ID列 'egoid': {scales_df['egoid'].nunique()} 个唯一值")
        elif 'ID' in scales_df.columns:
            print(f"  ID列 'ID': {scales_df['ID'].nunique()} 个唯一值")
        
        # 显示列名（前20个）
        print(f"  前20个列名: {list(scales_df.columns[:20])}")
        
        # 检查目标变量
        target_cols = ['BDI_T2', 'BAI_T2', 'PSQI_T2', 'SelfEsteem_T2', 
                      'CESD_T2', 'STAI_T2']
        
        print(f"\n目标变量检查:")
        for col in target_cols:
            if col in scales_df.columns:
                valid_count = scales_df[col].notna().sum()
                total_count = len(scales_df)
                pct = (valid_count / total_count) * 100
                print(f"  ✓ {col}: {valid_count}/{total_count} ({pct:.1f}%)")
            else:
                print(f"  ✗ {col}: 不存在")
        
        # 显示前几行
        print(f"\n前5行数据:")
        print(scales_df.head())
        
    else:
        print("无法读取量表数据文件")
        return fitbit_df, None
    
    # 检查数据匹配性
    print("\n" + "=" * 60)
    print("检查数据匹配性")
    print("=" * 60)
    
    if fitbit_df is not None and scales_df is not None:
        # 找到ID列
        fitbit_id_col = None
        for col in fitbit_df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'ego']):
                fitbit_id_col = col
                break
        
        scales_id_col = 'egoid' if 'egoid' in scales_df.columns else 'ID'
        
        if fitbit_id_col and scales_id_col in scales_df.columns:
            fitbit_ids = set(fitbit_df[fitbit_id_col].dropna())
            scales_ids = set(scales_df[scales_id_col].dropna())
            
            common_ids = fitbit_ids.intersection(scales_ids)
            
            print(f"Fitbit数据中的唯一ID数: {len(fitbit_ids)}")
            print(f"量表数据中的唯一ID数: {len(scales_ids)}")
            print(f"共同ID数: {len(common_ids)}")
            print(f"匹配率: {len(common_ids)/min(len(fitbit_ids), len(scales_ids))*100:.1f}%")
            
            if len(common_ids) > 0:
                print(f"✓ 数据可以匹配")
                print(f"共同ID示例: {list(common_ids)[:5]}")
            else:
                print(f"✗ 没有共同的ID，无法匹配数据")
                print(f"Fitbit ID示例: {list(fitbit_ids)[:5]}")
                print(f"量表ID示例: {list(scales_ids)[:5]}")
        else:
            print("无法确定ID列进行匹配")
    
    return fitbit_df, scales_df

def create_test_data():
    """
    如果数据文件不存在，创建测试数据
    """
    print("\n" + "=" * 60)
    print("创建测试数据")
    print("=" * 60)
    
    # 创建测试Fitbit数据
    np.random.seed(42)
    n_participants = 100
    n_days_per_participant = 30
    
    test_fitbit_data = []
    
    for participant_id in range(44869, 44869 + n_participants):
        for day in range(n_days_per_participant):
            test_fitbit_data.append({
                'egoid': participant_id,
                'datadate': f"2023-01-{day+1:02d}",
                'complypercent': np.random.normal(85, 15),
                'meanrate': np.random.normal(75, 10),
                'sdrate': np.random.normal(12, 3),
                'steps': np.random.normal(8000, 2000),
                'floors': np.random.normal(10, 5),
                'sedentaryminutes': np.random.normal(600, 100),
                'lightlyactiveminutes': np.random.normal(200, 50),
                'fairlyactiveminutes': np.random.normal(30, 15),
                'veryactiveminutes': np.random.normal(15, 10),
                'lowrangemins': np.random.normal(400, 100),
                'fatburnmins': np.random.normal(100, 30),
                'cardiomins': np.random.normal(20, 10),
                'peakmins': np.random.normal(5, 3),
                'lowrangecal': np.random.normal(800, 200),
                'fatburncal': np.random.normal(300, 100),
                'cardiocal': np.random.normal(100, 50),
                'peakcal': np.random.normal(50, 25)
            })
    
    test_fitbit_df = pd.DataFrame(test_fitbit_data)
    test_fitbit_df.to_csv('test_fitbit_data.csv', index=False)
    print("✓ 创建测试Fitbit数据: test_fitbit_data.csv")
    
    # 创建测试量表数据
    test_scales_data = []
    
    for participant_id in range(44869, 44869 + n_participants):
        test_scales_data.append({
            'egoid': participant_id,
            'BDI_T2': np.random.normal(10, 8),
            'BAI_T2': np.random.normal(8, 6),
            'PSQI_T2': np.random.normal(6, 3),
            'SelfEsteem_T2': np.random.normal(30, 5),
            'BigFive_Extraversion_T1': np.random.normal(3.5, 0.8),
            'BigFive_Agreeableness_T1': np.random.normal(3.8, 0.7),
            'BigFive_Conscientiousness_T1': np.random.normal(3.6, 0.8),
            'BigFive_Neuroticism_T1': np.random.normal(2.8, 0.9),
            'BigFive_Openness_T1': np.random.normal(3.7, 0.8),
            'SelfEsteem_T1': np.random.normal(32, 6),
            'Trust_T1': np.random.normal(20, 4)
        })
    
    test_scales_df = pd.DataFrame(test_scales_data)
    test_scales_df.to_csv('test_extracted_scales_data.csv', index=False)
    print("✓ 创建测试量表数据: test_extracted_scales_data.csv")
    
    return test_fitbit_df, test_scales_df

if __name__ == "__main__":
    print("数据文件检查工具")
    print("=" * 60)
    
    # 检查现有数据文件
    fitbit_df, scales_df = inspect_data_files()
    
    # 如果没有找到数据文件，询问是否创建测试数据
    if fitbit_df is None or scales_df is None:
        response = input("\n是否创建测试数据用于测试模型？(y/n): ")
        if response.lower() == 'y':
            create_test_data()
            print("\n测试数据已创建，你可以使用以下文件名运行模型:")
            print("  Fitbit数据: test_fitbit_data.csv")
            print("  量表数据: test_extracted_scales_data.csv")
        else:
            print("\n请准备好数据文件后再运行模型")