_base_ = [
    'scaunet-s_512x512_1500e_nuaa.py'
]

model = dict(
    decode_head=dict(
        dim=32
    )
)
data = dict(train_batch=4)