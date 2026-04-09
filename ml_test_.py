import random
# ============================================================
# 1. 데이터셋
#    정답: H(x) = 0.3x + 1  →  w=0.3, b=1
#    동일함, 학습률 0.001로 해야 실행가능
#
#    아니면 데이터셋 그냥 숫자로 변경! (10 이하까지)
# ============================================================
# x_data = [ 1, 2, 3, 4, 5 ... 50]
# y_data = [ 1.3, 1.6, 1.9, 2.2, 2.5 ... 16.0]
x_data = [count for count in range(1, 51)]
y_data = [0.3 * x + 1 for x in x_data]
num_of_sample = len(x_data)


# 2. 파라미터 초기화 (w, b)

w = 0
b = 0


# 3. 하이퍼파라미터 설정 (learning_rate, epochs)
learning_rate = 0.001
epochs = 5000

# 4. 학습 루프 (epoch 반복)

for epoch in range(epochs) :

    # 4-1. 그래디언트 초기화
    grad_w = 0.0
    grad_b = 0.0
    loss = 0.0
    # 4-2. 전체 샘플 순회 (BGD: 모든 샘플을 다 본 뒤 업데이트)
    for x,y in zip(x_data , y_data):

        # (a) 예측값 계산: H(x) = wx + b
        pred_y = w * x + b

        # (b) 오차 계산: error = 예측값 - 실제값
        
        error = pred_y - y

        # (c) 그래디언트 누적: grad_w += 2 * x * error
        #                      grad_w += 2 * error

        grad_w += 2 * x * error
        grad_b += 2 * error 
        loss += error ** 2



    # 4-3. 평균 그래디언트 계산 (누적값 / n)

    grad_w /= num_of_sample
    grad_b /= num_of_sample
    loss /= num_of_sample


    # 4-4. 파라미터 업데이트: w = w - lr * grad_w


    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b



        # 4-5. 손실(loss) 출력 (학습 경과 확인)

    if epoch % 1000 == 0:
        print(F"epoch : {epoch} , loss:{loss : .4f}")


    # 5. 최종 결과 출력

    
print(f"w:{w} , b:{b}")
    

