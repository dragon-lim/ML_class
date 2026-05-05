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
epochs = 1000
batch_size = 4

# 파라미터 업데이트 횟수 = step (1부터 시작)
global_step = 0

# total_steps 계산 후 warmup 구간 설정 (5%)
total_steps = (num_of_sample // batch_size) * epochs
warmup_steps = int(total_steps * 0.05)

# 학습률: 문제 수식에 따라 base_lr 하나만 사용
base_lr = 0.005

# 파라미터 초기화
w = random.random()
b = random.random()

for epoch in range(1, epochs + 1):
    indices = list(range(num_of_sample))
    random.shuffle(indices)

    epoch_loss = 0.0

    for start in range(0, num_of_sample, batch_size):
        batch_indices = indices[start : start + batch_size]
        current_batch_size = len(batch_indices)  # 마지막 배치 크기 대응

        w_grad = 0.0
        b_grad = 0.0
        batch_loss = 0.0

        for index in batch_indices:
            x = x_data[index]
            y = y_data[index]

            pred = w * x + b
            error = pred - y

            w_grad += 2 * x * error
            b_grad += 2 * error
            batch_loss += error ** 2

        # 실제 배치 크기로 나누어 기울기·loss 평균 계산
        w_grad /= current_batch_size
        b_grad /= current_batch_size
        batch_loss /= current_batch_size

        epoch_loss += batch_loss

        # step 을 먼저 증가시켜 1부터 시작하도록 보장
        global_step += 1

        # 문제 수식: lr = base_lr * min(1.0, global_step / warmup_steps)
        lr = base_lr * min(1.0, global_step / warmup_steps)

        w = w - lr * w_grad
        b = b - lr * b_grad

    epoch_loss /= (num_of_sample // batch_size)  # 배치 수로 epoch 평균 loss

    if epoch % 10 == 0 and epoch <= 70:
        print(f"현재 학습률 = {lr:.4f}")

    if epoch % 100 == 0:
        print(f"현재 epoch = {epoch} , epoch loss = {epoch_loss:.3f} , w = {w:.3f} , b = {b:.3f} ")
        print()

print(f"학습 종료 결과 -> w : {w:.3f} , b : {b:.3f}")
