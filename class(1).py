import random
# ============================================================
# 1. 데이터셋
#    정답: H(x) = 0.3x + 1  →  w=0.3, b=1
#    현재 데이터셋은 학습률 0.001로 해야 가능
# ============================================================
# x_data = [ 1, 2, 3, 4, 5 ... 50]
# y_data = [ 1.3, 1.6, 1.9, 2.2, 2.5 ... 16.0]
x_data = [count for count in range(1, 51)]
y_data = [0.3 * x + 1 for x in x_data]
num_of_sample = len(x_data)


# 2. 파라미터 초기화 (w, b)
w = random.random()
b = random.random()




# 3. 하이퍼파라미터 설정 (learning_rate, epochs)
epochs = 1000
learning_rate = 0.001


# 4. 학습 루프 (epoch 반복)
for epoch in range(epochs):
    

    # 4-1. 그래디언트 초기화
    w_grad = 0.0
    b_grad = 0.0
    loss = 0.0
    
    # 4-2. 전체 샘플 순회 (BGD: 모든 샘플을 다 본 뒤 업데이트)
    for x,y in zip(x_data , y_data):
        # (a) 예측값 계산: H(x) = wx + b
        y_pred = w*x+b

        # (b) 오차 계산: error = 예측값 - 실제값
        error = y_pred - y
        
        # (c) 그래디언트 누적: grad_w += 2 * x * error
        #                      grad_w += 2 * x * error

        w_grad += 2 * x * error
        b_grad += 2 * error
        loss += error ** 2

    # 4-3. 평균 그래디언트 계산 (누적값 / n)

    w_grad /= num_of_sample
    b_grad /= num_of_sample
    loss /= num_of_sample


    # 4-4. 파라미터 업데이트: w = w - lr * grad_w
    w = w - learning_rate * w_grad
    b  = b - learning_rate * b_grad



        # 4-5. 손실(loss) 출력 (학습 경과 확인)

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {loss:.4f}")
        


    # 5. 최종 결과 출력
    

print(f"w:{w}, b: {b}")
