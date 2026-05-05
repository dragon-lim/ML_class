import random

# ============================================================
# 데이터셋 (변경 금지) -- 정답: H(x) = 0.5x + 2  (w=0.5, b=2)
# ============================================================

random.seed(0)
# 랜덤값 이지만 고정을 한 랜덤값

x_data = [i for i in range(1, 21)]
y_data = [0.5 * x + 2 + random.uniform(-0.3, 0.3) for x in x_data]
num_of_sample = len(x_data)

# ============================================================
# 학생 구현 영역
# ============================================================


# 하이퍼파라미터 설정
epochs = 100
batch_size = 4

# 파라미터 업데이트 횟수 = step
# 아래 파라미터 업데이트 후 카운트중 ...
step = 0
# warmup_step을 계산하기 위해 total_step 계산  
total_steps = num_of_sample / batch_size * epochs

# learning rate 조절에 필요한 함수
min_lr = 0.001
max_lr = 0.005
# 현재 lr 이 max_lr 에 도달한 스탭수
# 즉 워밍업이 끝나 lr 가 지속되기 시작하는 스텝수
# 아래 step 계산 후 업데이트중
warmup_steps = 0



# 파라미터 설정
w = random.random()
b = random.random()

for epoch in range(1, epochs+1):
    indices = list(range(num_of_sample))
    random.shuffle(indices)
    #print(indices)

    # epoch 당 loss 률을 확인하기 위해 loss 변수 선언
    epoch_loss = 0.0

    for start in range(0, num_of_sample, batch_size):
        # 우리가 자르고자 하는 범위의 인덱스 번호
        # print(start)  
        
        # 배치만큼 자르기
        # 랜덤 인덱스 indices 안에 0 : 0 + 4 / 4 : 4 + 4 ...
        batch_indices = indices[start : batch_size + start]

        # 현재의 batch_size 크기 항상 숫자가 떨어지지 않기 때문에
        # 지금 batch size 가 얼마로 학습중인지 확인하기 위해
        # current_batch_size = len(batch_indices)

        # mini-batch 는 batch 만큼 학습하기 때문에
        # batch_size 크기 마다 기울기를 초기화 한다
        w_grad = 0.0
        b_grad = 0.0
        
        # batch 당 loss 확인해야 하니까
        batch_loss = 0.0

        for index in batch_indices:
            # x 안에는 인덱스 1,18,13,0 의 값이 저장됨
            x = x_data[index]
            y = y_data[index]

            # 예측값 , error 계산
            pred = w * x + b
            error = pred - y 

            # error 를 구한 후 
            # sample 마다 기울기 누적
            w_grad += 2 * x * error
            b_grad += 2 * error

            batch_loss += error ** 2


        #  현재의 batch_size 로 기울기 평균
        w_grad /= batch_size
        b_grad /= batch_size

        # batch loss 를 epoch loss 에 누적한다
        epoch_loss += batch_loss

        # 파라미터 업데이트 마다 step 업데이트
        step += step

        # 토탈 스탭중 0.05% 까지는 lr을 증가 시키고
        # 그 이후는 max_lr 유지한다
        warmup_steps = int(total_steps * 0.05)

        # lr 식 min 함수는 더 작은 수를 골라준다
        # step = 50 , 50 / 100 = 0.5
        # lr = 0.5 를 선택 
        lr = max_lr * min(1.0 , step / warmup_steps)

        # lr 이 min_lr 보다 작아지는걸 막기
        if lr < min_lr :
            lr = min_lr

        # batch 마다 파라미터 업데이트
        w = w - lr * w_grad
        b = b - lr * b_grad



    epoch_loss /= num_of_sample

    if epoch % 100 == 0:
        print(f"현재 epoch = {epoch} , epoch loss = {epoch_loss:.3f} , w = {w:.3f} , b = {b:.3f}")

print(f"학습 종료 결과 -> w : {w:.3f} , b : {b:.3f}")


























# # epochs for문
# for epoch in range(1, epochs+1):
#     # 데이터 셔플하기
#     # x_data 의 인덱스는 0-19 의 리스트를 이용해서 셔플
#     # random 을 이용해 섞기 -> random.shuffle
#     indices = list(range(num_of_sample))
#     random.shuffle(indices)
    
#     # 랜덤 중에서 batch_size 만큼 잘라주기
#     # batch for문

#     # 0 부터 샘플 갯수 까지 , batch_size 간격으로 ...
#     for batch in range(0, num_of_sample, batch_size):
#         # 섞은 리스트 1,18,13,0 배치 (0) 부터 0 + 4 간격 
#         # 즉, batch_size 간격으로 변수에 저장
#         batch_indices = indices[batch:batch+batch_size]

#         # sample 순회
#         # 이제 랜덤으로 섞인 리스트
#         # batch_indices 를 순회
#         for index in batch_indices:
            








# # for epochs
# for epochs in range(epochs+1):
#     # 셔플 돌아야함
#     # indiced = list(range(len(x_data)))
    

#     # for batch
#     # 현재 샘플 len 에서 10 만큼 잘라서 ㄱㄱ 해야하는데
#     for batchs in range(batch_size):        
#         list(len(x_data)) # 0부터 9까지 

#         w_grad = 0.0
#         b_grad = 0.0


#         # lr 조절
#         # lr = base_lr * min(1.0, global_step / warmup_steps)
        
        
#         # for sample
#         for x,y in zip(x_data , y_data):
#             predict = w * x + b
#             error = predict - y

#             w_grad += 2*x*error
#             b_grad += 2*error
#             loss = error**2

#     # parameter update
#     w = w - lr * w_grad
#     b = b - lr * b_grad

#     # numofsample 처럼 나눠야 하는데
#     # cost 값 계산 후 업데이트
    





