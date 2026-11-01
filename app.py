# -*- coding: utf-8 -*-
"""APP DỰ ĐOÁN GIÁ NHÀ - NHÓM 7
Chạy:  streamlit run app.py
Cần để cùng thư mục: Nhom7_GiaNha_train.xlsx, Nhom7_GiaNha_test.xlsx
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "Gia_ty_dong"

# ====== ĐỔI 2 THUẬT TOÁN CỦA NHÓM TẠI ĐÂY ======
MODELS = {
    "Thuật toán 1: Linear Regression": LinearRegression(),
    "Thuật toán 2: Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
}
# ================================================

st.set_page_config(page_title="Dự đoán giá nhà - Nhóm 7", page_icon="🏠")


@st.cache_resource
def train_models():
    train = pd.read_excel("Nhom7_GiaNha_train.xlsx")
    test = pd.read_excel("Nhom7_GiaNha_test.xlsx")
    X_train = pd.get_dummies(train.drop(columns=[TARGET]), columns=["Quan"])
    X_test = pd.get_dummies(test.drop(columns=[TARGET]), columns=["Quan"])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_train, y_test = train[TARGET], test[TARGET]

    fitted, metrics = {}, []
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics.append({
            "Thuật toán": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred),
        })
        fitted[name] = model
    return fitted, pd.DataFrame(metrics), list(X_train.columns), sorted(train["Quan"].unique())


models, metrics_df, columns, districts = train_models()

st.title("🏠 DỰ ĐOÁN GIÁ NHÀ - NHÓM 7")
st.write("Nhập thông tin căn nhà để mô hình dự đoán giá (đơn vị: tỷ đồng).")

with st.form("form_nhap"):
    quan = st.selectbox("Quận", districts)
    c1, c2 = st.columns(2)
    dien_tich = c1.number_input("Diện tích (m²)", 10, 500, 80)
    chieu_rong_hem = c2.number_input("Chiều rộng hẻm (m)", 1, 30, 6)
    so_phong_ngu = c1.number_input("Số phòng ngủ", 1, 10, 3)
    so_phong_tam = c2.number_input("Số phòng tắm", 1, 10, 2)
    so_tang = c1.number_input("Số tầng", 1, 10, 3)
    co_so_do = c2.radio("Có sổ đỏ", ["Có", "Không"], horizontal=True)
    submit = st.form_submit_button("Dự đoán giá")

if submit:
    row = pd.DataFrame([{
        "Dien_tich_m2": dien_tich, "So_phong_ngu": so_phong_ngu,
        "So_phong_tam": so_phong_tam, "So_tang": so_tang,
        "Chieu_rong_hem_m": chieu_rong_hem,
        "Co_so_do": 1 if co_so_do == "Có" else 0, "Quan": quan,
    }])
    x = pd.get_dummies(row, columns=["Quan"]).reindex(columns=columns, fill_value=0)

    st.subheader("Kết quả dự đoán")
    cols = st.columns(len(models))
    for col, (name, model) in zip(cols, models.items()):
        col.metric(name, f"{model.predict(x)[0]:.2f} tỷ đồng")

with st.expander("Độ chính xác của mô hình trên tập test"):
    st.dataframe(metrics_df.round(3), hide_index=True)
