import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    input_file = "/home/grealish/summer/KAN_NET/predict/processed_data.csv"
    train_file = "/home/grealish/summer/KAN_NET/predict/train.csv"
    test_file = "/home/grealish/summer/KAN_NET/predict/test.csv"
    test_size = 0.2  # 20% 作为测试集
    random_state = 42

    df = pd.read_csv(input_file)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"训练集已保存到: {train_file}，样本数: {len(train_df)}")
    print(f"测试集已保存到: {test_file}，样本数: {len(test_df)}")

if __name__ == "__main__":
    main() 