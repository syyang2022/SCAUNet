_base_ = [
    'scaunet-s_256x256_500e_sirstaug.py'
]

model = dict(
    decode_head=dict(
        dim=32
    )
)
data = dict(train_batch=16)