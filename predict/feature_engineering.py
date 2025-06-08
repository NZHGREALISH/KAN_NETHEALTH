import pandas as pd

def preprocess_fitbit_data(fitbit_df):
    print("正在预处理Fitbit数据...")
    print(f"原始Fitbit数据列名: {list(fitbit_df.columns)}")
    required_id_col = None
    possible_id_cols = ['egoid', 'ID', 'id', 'user_id', 'participant_id']
    for col in possible_id_cols:
        if col in fitbit_df.columns:
            required_id_col = col
            break
    if required_id_col is None:
        raise ValueError(f"未找到ID列。可能的ID列名: {possible_id_cols}")
    print(f"使用ID列: {required_id_col}")
    fitbit_df = fitbit_df.copy()
    fitbit_df = fitbit_df[fitbit_df[required_id_col].notna()]
    if required_id_col != 'egoid':
        fitbit_df = fitbit_df.rename(columns={required_id_col: 'egoid'})
    expected_numeric_cols = ['complypercent', 'meanrate', 'sdrate', 'steps', 'floors', 
                           'sedentaryminutes', 'lightlyactiveminutes', 'fairlyactiveminutes',
                           'veryactiveminutes', 'lowrangemins', 'fatburnmins', 'cardiomins',
                           'peakmins', 'lowrangecal', 'fatburncal', 'cardiocal', 'peakcal']
    available_numeric_cols = []
    for col in expected_numeric_cols:
        if col in fitbit_df.columns:
            available_numeric_cols.append(col)
            fitbit_df[col] = pd.to_numeric(fitbit_df[col], errors='coerce')
        else:
            print(f"警告: 列 '{col}' 不存在于Fitbit数据中")
    print(f"找到 {len(available_numeric_cols)} 个数值列")
    if len(available_numeric_cols) == 0:
        raise ValueError("没有找到可用的数值列")
    agg_functions = {}
    for col in available_numeric_cols:
        agg_functions[col] = ['mean', 'std']
        if col in ['steps', 'floors']:
            agg_functions[col].append('max')
    print(f"聚合函数: {list(agg_functions.keys())}")
    try:
        fitbit_agg = fitbit_df.groupby('egoid').agg(agg_functions)
        fitbit_agg.columns = [f"{col[0]}_{col[1]}" for col in fitbit_agg.columns]
        fitbit_agg = fitbit_agg.reset_index()
        print(f"聚合后数据形状: {fitbit_agg.shape}")
        activity_cols = ['lightlyactiveminutes_mean', 'fairlyactiveminutes_mean', 'veryactiveminutes_mean']
        available_activity_cols = [col for col in activity_cols if col in fitbit_agg.columns]
        if len(available_activity_cols) >= 2:
            fitbit_agg['total_active_mins_mean'] = fitbit_agg[available_activity_cols].sum(axis=1)
            if 'veryactiveminutes_mean' in fitbit_agg.columns:
                fitbit_agg['activity_intensity_ratio'] = (
                    fitbit_agg['veryactiveminutes_mean'] / 
                    (fitbit_agg['total_active_mins_mean'] + 1e-6)
                )
            if 'sedentaryminutes_mean' in fitbit_agg.columns:
                fitbit_agg['sedentary_active_ratio'] = (
                    fitbit_agg['sedentaryminutes_mean'] / 
                    (fitbit_agg['total_active_mins_mean'] + 1e-6)
                )
        print(f"最终聚合数据列数: {len(fitbit_agg.columns)}")
        return fitbit_agg
    except Exception as e:
        print(f"聚合数据时出错: {e}")
        print("尝试简化聚合...")
        simple_agg = {col: 'mean' for col in available_numeric_cols}
        fitbit_agg = fitbit_df.groupby('egoid').agg(simple_agg)
        fitbit_agg = fitbit_agg.reset_index()
        return fitbit_agg

def prepare_features_targets(merged_df):
    print("正在准备特征和目标变量...")
    fitbit_features = [col for col in merged_df.columns if col != 'egoid' and 
                      not any(scale in col for scale in ['CESD', 'BDI', 'BAI', 'BigFive', 
                                                        'SelfEsteem', 'STAI', 'PSQI', 'Trust', 
                                                        'SELSA', 'Stress'])]
    scale_features = [
        'BigFive_Extraversion_T1', 'BigFive_Agreeableness_T1', 
        'BigFive_Conscientiousness_T1', 'BigFive_Neuroticism_T1', 'BigFive_Openness_T1',
        'SelfEsteem_T1', 'Trust_T1'
    ]
    fitbit_features = [col for col in fitbit_features if col in merged_df.columns]
    scale_features = [col for col in scale_features if col in merged_df.columns]
    feature_columns = fitbit_features + scale_features
    target_columns = {
        'depression_bdi': 'BDI_T2',
        'anxiety_bai': 'BAI_T2',
        'sleep_psqi': 'PSQI_T2',
        'selfesteem': 'SelfEsteem_T2'
    }
    available_targets = {}
    for task_name, col_name in target_columns.items():
        if col_name in merged_df.columns:
            available_targets[task_name] = col_name
    X = merged_df[feature_columns].copy()
    y = merged_df[[available_targets[task] for task in available_targets]].copy()
    y.columns = list(available_targets.keys())
    return X, y, available_targets 