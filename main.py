from fastapi import FastAPI
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from typing import List
from pydantic import BaseModel

app = FastAPI()


CSV_FILE_PATH = "./stockout_analysis.csv"
PREDICTION_CSV_FILE_PATH = "predictions.csv"


class StockOutToAnalysis(BaseModel):
    categoryName: str
    itemName: str
    itemCode: str
    itemPrice: float
    dates: dict

class CategoryAnalysisEntity(BaseModel):
    categoryName: str
    firstCount: int
    secondCount: int
    thirdCount: int
    fourthCount: int
    fifthCount: int
    sixthCount: int
    seventhCount: int
    eighthCount: int
    ninthCount: int
    tenthCount: int
    eleventhCount: int
    twelfthCount: int
    thirteenthCount: int
    fourteenthCount: int
    fifteenthCount: int


def create_lstm_model(X_train_shape):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train_shape[1], X_train_shape[2])),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def preprocess_data_for_lstm(df):
    # MinMaxScaler 적용
    scale_cols = ["origin_price", "product_price", "stock_price", "stock_out"]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # X와 y 데이터 분할
    X = []
    y = []
    time_steps = 30
    for i in range(len(df) - time_steps):
        X.append(df.iloc[i:i + time_steps].drop(columns=["stock_out", "date", "Year", "Month", "Day"]).values)
        y.append(df.iloc[i + time_steps]["stock_out"])
    X, y = np.array(X), np.array(y)

    return X, y, scaler

def save_predictions_to_csv(future_predictions, CSV_FILE_PATH):
    # 예측값을 DataFrame 형식으로 변환
    predictions_data = []

    for category, prediction in future_predictions.items():
        # 예측된 값들을 1주일 단위로 분리
        prediction_dict = {
            "categoryName": category,
            "week1": prediction[0:7],
            "week2": prediction[7:14],
            "week3": prediction[14:21],
            "week4": prediction[21:28],
            "week5": prediction[28:35],
            "week6": prediction[35:42],
            "week7": prediction[42:49],
            "week8": prediction[49:56],
            "week9": prediction[56:63],
            "week10": prediction[63:70],
            "week11": prediction[70:77],
            "week12": prediction[77:84],
            "week13": prediction[84:91],
            "week14": prediction[91:98],
            "week15": prediction[98:105]
        }
        predictions_data.append(prediction_dict)

    # DataFrame으로 변환
    predictions_df = pd.DataFrame(predictions_data)

    # 예측값을 CSV 파일에 저장
    predictions_df.to_csv(CSV_FILE_PATH, index=False, mode='w', header=True)

    print(f"예측 결과가 {CSV_FILE_PATH}에 저장되었습니다.")

@app.post("/stockout-analysis")
async def analyze_stock_out_data(stock_out_data: List[StockOutToAnalysis]):
    try:
        # List[StockOutToAnalysis] 형태로 받은 데이터를 처리
        data = []
        for item in stock_out_data:
            for date, count in item.dates.items():
                # 날짜를 string으로 변환 (예: '2024-07-01')
                formatted_date = date.strftime('%Y-%m-%d')
                data.append({
                    "categoryName": item.categoryName,
                    "itemName": item.itemName,
                    "itemCode": item.itemCode,
                    "itemPrice": item.itemPrice,
                    "originPrice": item.itemPrice * 0.9,  # itemPrice의 0.9배로 originPrice 계산
                    "stockOut": count,
                    "date": formatted_date,
                    "stockPrice": item.stockPrice  # stockPrice 유지
                })

        # pandas DataFrame으로 변환
        df = pd.DataFrame(data)

        # CSV 파일이 존재하지 않으면 새로 생성하고, 존재하면 덧붙여서 저장
        df.to_csv(CSV_FILE_PATH, index=False, mode='a', header=not pd.io.common.file_exists(CSV_FILE_PATH))

        # 데이터 전처리 및 모델 학습
        df["date"] = pd.to_datetime(df["date"])
        df["Year"] = df["date"].dt.year
        df["Month"] = df["date"].dt.month
        df["Day"] = df["date"].dt.day
        df = df.sort_values(by=["Year", "Month", "Day"])

        # 각 product_code에 대해 가장 오래된 데이터를 찾고, 해당 데이터를 업데이트
        unique_product_codes = df["itemCode"].unique()
        updated_data = []

        for product_code in unique_product_codes:
            # product_code가 일치하는 데이터들 필터링
            df_product = df[df["itemCode"] == product_code]

            # 가장 오래된 date를 가진 데이터 찾기
            oldest_data = df_product.loc[df_product["date"].idxmin()]

            # 해당 항목들을 업데이트
            updated_data.append({
                "categoryName": oldest_data["categoryName"],
                "itemName": oldest_data["itemName"],
                "itemCode": oldest_data["itemCode"],
                "itemPrice": oldest_data["itemPrice"],
                "originPrice": oldest_data["itemPrice"] * 0.9,  # itemPrice의 0.9배로 originPrice 업데이트
                "stockOut": oldest_data["stockOut"],  # stockOut 업데이트
                "date": oldest_data["date"].strftime('%Y-%m-%d'),
                "stockPrice": oldest_data["stockPrice"]  # stockPrice 유지
            })

        # 업데이트된 데이터를 DataFrame으로 변환
        updated_df = pd.DataFrame(updated_data)

        # 최종 업데이트된 데이터를 CSV 파일에 저장 (헤더 포함)
        updated_df.to_csv(CSV_FILE_PATH, index=False, mode='w', header=True)

        # 데이터 전처리 및 모델 학습 (이후 코드 유지)
        category_list = df.filter(like="category_").columns
        future_predictions = {}

        for category in category_list:
            df_category = df[df[category] == 1]

            # LSTM 모델 학습
            X, y, scaler = preprocess_data_for_lstm(df_category)

            # LSTM 모델 생성
            model = create_lstm_model(X.shape)

            # 모델 훈련
            model.fit(X, y, epochs=50, batch_size=16)

            # 미래 예측
            future_X = X[-1].reshape(1, 30, X.shape[2])
            future_pred = []
            for _ in range(15 * 7):  # 105일 예측
                pred = model.predict(future_X)
                future_pred.append(pred[0, 0])
                future_X = np.roll(future_X, shift=-1, axis=1)
                future_X[0, -1, -1] = pred[0, 0]

            future_predictions[category] = future_pred
            print(f"{category} 예측 완료!")

        # 예측된 결과 반환
        save_predictions_to_csv(future_predictions, "predictions.csv")
        return {"message": "Data successfully saved to CSV and predictions completed."}

    except Exception as e:
        # 에러가 발생한 경우
        return {"error": str(e)}



@app.get("/stockout-analysis/results", response_model=List[CategoryAnalysisEntity])
async def get_category_analysis_results():
    try:
        # CSV 파일에서 예측 결과 읽기
        prediction_df = pd.read_csv(PREDICTION_CSV_FILE_PATH)

        # 카테고리별로 예측 결과를 그룹화
        category_analysis = []

        for category in prediction_df["category"].unique():
            category_data = prediction_df[prediction_df["category"] == category]

            # 7일 단위로 묶어서 15주치 예측값 계산
            week_data = []
            for i in range(0, len(category_data), 7):
                week_counts = category_data.iloc[i:i+7]["predicted_stock_out"].values.tolist()
                week_data.extend(week_counts)

                # 부족한 부분은 0으로 채우기 (15*7 일 예측이므로)
                while len(week_data) < 7:
                    week_data.append(0)

            # 15주치 데이터로 조정
            while len(week_data) < 15 * 7:
                week_data.append(0)

            # 반환할 CategoryAnalysisEntity 객체 생성
            analysis_results = CategoryAnalysisEntity(
                categoryName=category,
                firstCount=week_data[0] if len(week_data) > 0 else 0,
                secondCount=week_data[1] if len(week_data) > 1 else 0,
                thirdCount=week_data[2] if len(week_data) > 2 else 0,
                fourthCount=week_data[3] if len(week_data) > 3 else 0,
                fifthCount=week_data[4] if len(week_data) > 4 else 0,
                sixthCount=week_data[5] if len(week_data) > 5 else 0,
                seventhCount=week_data[6] if len(week_data) > 6 else 0,
                eighthCount=week_data[7] if len(week_data) > 7 else 0,
                ninthCount=week_data[8] if len(week_data) > 8 else 0,
                tenthCount=week_data[9] if len(week_data) > 9 else 0,
                eleventhCount=week_data[10] if len(week_data) > 10 else 0,
                twelfthCount=week_data[11] if len(week_data) > 11 else 0,
                thirteenthCount=week_data[12] if len(week_data) > 12 else 0,
                fourteenthCount=week_data[13] if len(week_data) > 13 else 0,
                fifteenthCount=week_data[14] if len(week_data) > 14 else 0,
            )
            category_analysis.append(analysis_results)

        return category_analysis

    except Exception as e:
        # 에러가 발생한 경우
        return {"error": str(e)}