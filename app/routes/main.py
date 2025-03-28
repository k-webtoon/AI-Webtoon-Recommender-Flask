from flask import Blueprint, request, jsonify
import pickle
import psycopg2
import numpy as np

# Blueprint 설정
main_bp = Blueprint('main', __name__)

# 모델 로드 (pkl 파일 사용)
model_path = 'D:/KR-SBERT_vectorization_model.pkl'  # 모델 파일 경로 지정
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# PostgreSQL 데이터베이스 연결 설정
def connect_db():
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="rootmysql"
    )
    return conn

# 벡터화 함수
def vectorize(texts):
    return model.encode(texts, convert_to_numpy=True)

@main_bp.route('/process', methods=['POST'])
def predict():
    # 요청에서 텍스트 받기
    data = request.get_json()
    text = data.get('message', '')  # 'message' 필드로 텍스트 받기

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # 입력 텍스트 벡터화
    input_vec = vectorize([text])[0]  # 입력 문장 벡터화

    # 데이터베이스 연결
    conn = connect_db()
    cur = conn.cursor()

    # 데이터베이스에서 synop_vec 필드 가져오기
    cur.execute("SELECT id, title_name, synop_vec FROM webtoon")
    rows = cur.fetchall()

    # 유사도 계산
    similarities = []
    for row in rows:
        # synop_vec는 이미 리스트로 반환되므로, 이를 numpy 배열로 변환
        db_vec = np.array(row[2])  # 벡터 변환

        # 유사도 계산
        similarity = np.dot(input_vec, db_vec) / (np.linalg.norm(input_vec) * np.linalg.norm(db_vec))

        similarities.append((row[0], row[1], similarity))

    # 유사도가 높은 100개 웹툰 선택
    top_100 = sorted(similarities, key=lambda x: x[2], reverse=True)[:min(100, len(similarities))]

    # 결과 반환
    result = [
        {'id': item[0], 'title_name': item[1], 'similarity': item[2]}
        for item in top_100
    ]

    conn.close()  # 데이터베이스 연결 종료

    return jsonify({'result': result})

